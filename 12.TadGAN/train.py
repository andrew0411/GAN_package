"""TadGAN (Geiger et al., "TadGAN: Time Series Anomaly Detection Using Generative Adversarial Networks",
IEEE BigData 2020) 학습. 기준 구현: MIT sintel-dev/Orion `tadgan` 기본값.

방법 요약
    x → E → z → G → x̂ 순환으로 정상 window를 복원하도록 학습하고, critic 두 개로 분포를 맞춘다.
    - C_x: real window x vs G(z), z ~ N(0, I)    → G가 현실적인 window를 만들게 한다
    - C_z: prior z vs E(x)                        → E(x)가 N(0, I)를 따르게 한다
    - E·G: 두 adversarial 항 + cycle consistency λ_rec·MSE(x, G(E(x)))
    학습 후 G(E(x))는 정상 패턴만 잘 복원하므로 복원 오차와 C_x(x)가 이상 점수 재료가 된다 (detect.py).

손실 (Wasserstein + gradient penalty. critic은 Sigmoid 없는 실수 score)
    L_Cx = E[C_x(G(z))] − E[C_x(x)]  + λ_gp·GP(C_x; x, G(z))
    L_Cz = E[C_z(E(x))] − E[C_z(z)]  + λ_gp·GP(C_z; z, E(x))
    L_EG = −E[C_x(G(z))] − E[C_z(E(x))] + λ_rec·MSE(x, G(E(x)))
    batch마다 critic step(C_x 갱신 → C_z 갱신) 1번, n_critic번째 batch마다 E·G step 1번 (그 batch x를 다시 쓴다).
    1 epoch = 섞은 학습 window를 batch로 나눠 앞에서부터 (⌊N / batch⌋ // n_critic)·n_critic개 batch만 쓴다
    → E·G step 수 = ⌊N / batch⌋ // n_critic. 남는 batch(n_critic개 미만)는 E·G step 없이 critic만 갱신하게
    되므로 쓰지 않는다 (Orion의 epoch 정의와 같다).

기본값 (Orion): window 100, latent 20, batch 64, Adam lr 5e-4 (betas는 Keras 기본 (0.9, 0.999)),
λ_gp 10, λ_rec 10, n_critic 5, critic_z hidden 100. 네트워크 구조·Keras 대응은 model.py 참조.
epochs 35는 Orion `tadgan` pipeline 기본값이다 (v0.1.7 / v0.2.1 / v0.4.1 공통).
참고로 TadGAN primitive 자체의 기본값은 50, 논문은 "2000 iterations"로 적었다.

Orion·논문과 다른 점
- E·G step의 z는 새로 뽑는다 (Orion은 마지막 critic step의 z를 재사용한다). 기댓값은 같다
- 논문의 cycle 항은 L2 norm이지만 Orion을 따라 MSE(평균 제곱)를 쓴다
- Generator LSTM dropout·가중치 초기화의 Keras 대응은 model.py docstring 참조
- 비지도 학습: 이상이 섞인 신호 전체로 학습한다 (Orion 관례). 라벨은 학습에 쓰지 않는다

데이터 (data.py)
    --dataset synthetic : sine 합 + noise + 주입 이상 (다운로드 없음, 라벨 포함)
    --dataset csv       : --data_path `timestamp,value` CSV. 상대 경로가 현재 위치에 없으면 $DATA_ROOT 기준
                          예: $DATA_ROOT/tadgan/S-1.csv, 라벨 $DATA_ROOT/tadgan/S-1_anomalies.csv (`start,end`)
    --labels_path는 학습에 쓰지 않는다. 경로 검증만 하고 config에 남겨 detect.py가 기본값으로 쓴다.
    전처리(집계·평균 대치·MinMax [-1, 1]) 통계는 checkpoint의 `data_stats`에 저장되어 detect에서 재사용된다.

실행 (repo 루트에서 `pip install -e .` 후):
    cd 12.TadGAN
    python train.py --dataset synthetic
    python detect.py --checkpoint runs/TadGAN/<run>/checkpoints/last.pt
    python train.py --dataset csv --data_path tadgan/S-1.csv --labels_path tadgan/S-1_anomalies.csv
    python train.py --dataset csv --data_path my_signal.csv --interval 21600   # timestamp 단위 21600(초면 6시간) 평균 집계
이어서 학습 (--run_name 없이 --resume만 주면 원래 run 폴더에 이어 쓴다. --epochs는 총 epoch 수):
    python train.py --resume runs/TadGAN/<run>/checkpoints/last.pt --epochs 70
    python train.py --dataset csv --data_path tadgan/S-1.csv --interval 21600 \
        --resume runs/TadGAN/<run>/checkpoints/last.pt --epochs 70
    - 원래 run의 RESUME_KEYS(--dataset, --data_path, --interval, --window_size, --latent_dim, --critic_z_hidden,
      --synthetic_n, --synthetic_seed)를 같은 값으로 다시 줘야 한다. 기본값과 다르게 학습했다면 그 인자를 반복한다
    - 하나라도 다르면 load 전에 ValueError로 멈춘다 (다른 신호·구조로 이어 학습하면 checkpoint와 data_stats가 어긋난다)
    - RNG 상태는 복원하지 않으므로 끊지 않고 학습한 결과와 bit 단위로 같지는 않다
산출물: runs/TadGAN/<run_name|timestamp>/ 아래 config.json, tb/, samples/recon_*.png, samples/recon_final.png,
    checkpoints/last.pt (--keep_every N이면 epoch이 N의 배수일 때 ckpt_XXXXXXX.pt 사본도 남긴다)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.figure import Figure
from torch import Tensor

from gan_common.checkpoint import load_checkpoint, save_rolling_checkpoint
from gan_common.config import base_parser, positive_int
from gan_common.data import build_loader
from gan_common.logger import Logger
from gan_common.losses import GANLoss
from gan_common.regularizers import gradient_penalty
from gan_common.utils import count_params, get_device, resolve_run_dir, save_config, seed_everything

from data import DATASETS, TimeSeriesWindows, load_series, preprocess  # 같은 폴더의 data.py
from model import TadGAN  # 같은 폴더의 model.py

# resume 시 checkpoint와 같아야 하는 인자: 학습 신호(→ data_stats)와 모델 구조를 정한다
RESUME_KEYS = (
    "dataset",
    "data_path",
    "interval",
    "window_size",
    "latent_dim",
    "critic_z_hidden",
    "synthetic_n",
    "synthetic_seed",
)


def check_resume_config(ckpt_config: dict, args: argparse.Namespace) -> None:
    """checkpoint의 RESUME_KEYS가 이번 실행과 다르면 load_state_dict 전에 에러로 멈춘다.

    다른 신호로 이어 학습하면 checkpoint의 전처리 통계(data_stats)와 신호가 어긋나고,
    구조 인자가 다르면 state_dict를 읽을 수 없다. 조용히 섞이지 않도록 경고가 아니라 ValueError다.
    """
    diffs = {k: (ckpt_config.get(k), getattr(args, k)) for k in RESUME_KEYS if ckpt_config.get(k) != getattr(args, k)}
    if diffs:
        detail = ", ".join(f"--{k}: checkpoint {old} ≠ 지금 {new}" for k, (old, new) in diffs.items())
        raise ValueError(f"--resume checkpoint와 학습 신호·모델 구조 인자가 다릅니다 ({detail}). 원래 값으로 다시 실행하십시오.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = base_parser(
        "TadGAN: GAN-based time-series anomaly detection (Orion defaults)",
        dataset="synthetic",
        image_size=None,  # 이미지 전용 인자는 쓰지 않는다
        channels=None,
        download=None,
        batch_size=64,
        epochs=35,  # Orion tadgan pipeline 기본값 (v0.1.7/v0.2.1/v0.4.1). primitive 기본 50, 논문 "2000 iterations"
        lr=5e-4,
        beta1=0.9,  # Orion Keras `Adam(learning_rate)`의 기본 betas
        beta2=0.999,
        num_workers=0,  # 메모리 안의 신호를 slicing하므로 worker 프로세스 이득이 없다
        log_every=10,  # 단위: E·G step (기본 설정에서 epoch당 약 15 step)
        sample_every=100,  # 단위: E·G step. 복원 plot 저장 주기
    )
    g = p.add_argument_group("time series")
    g.add_argument(
        "--labels_path",
        type=str,
        default=None,
        help="--dataset csv의 이상 구간 CSV(`start,end`). 학습에는 쓰지 않고 detect.py 기본값으로 넘긴다",
    )
    g.add_argument(
        "--interval",
        type=positive_int,
        default=None,
        help="시간 집계 간격 (timestamp 단위). 주면 [t, t + interval) 평균으로 다시 샘플링한다. 없으면 집계 안 함",
    )
    g.add_argument("--window_size", type=positive_int, default=100, help="rolling window 길이 (짝수)")
    g.add_argument("--window_step", type=positive_int, default=1, help="학습 window 간격 (detect.py는 항상 1)")
    g.add_argument("--synthetic_n", type=positive_int, default=5000, help="--dataset synthetic 신호 길이")
    g.add_argument("--synthetic_seed", type=int, default=0, help="합성 신호 seed (학습 --seed와 별개)")

    g = p.add_argument_group("model")
    g.add_argument("--latent_dim", type=positive_int, default=20, help="latent z 길이 (z shape = (latent_dim, 1))")
    g.add_argument(
        "--critic_z_hidden",
        type=positive_int,
        default=100,
        help="Critic_z 은닉 폭. 100은 Orion v0.1.x·논문 시기 값, 현재 Orion master는 20",
    )
    g.add_argument("--n_critic", type=positive_int, default=5, help="E·G step 1번당 critic step 수")
    g.add_argument("--lambda_gp", type=float, default=10.0, help="두 critic의 gradient penalty 가중치")
    g.add_argument("--lambda_rec", type=float, default=10.0, help="cycle consistency MSE(x, G(E(x))) 가중치")
    g.add_argument("--n_plot", type=positive_int, default=4, help="복원 plot에 그릴 고정 window 수")

    args = p.parse_args(argv)
    if args.dataset not in DATASETS:
        p.error(f"--dataset은 {' | '.join(DATASETS)} 중 하나: {args.dataset!r}")
    if args.dataset == "csv" and args.data_path is None:
        p.error("--dataset csv에는 --data_path(`timestamp,value` CSV)가 필요합니다.")
    if args.window_size % 2 != 0:
        p.error(f"--window_size는 짝수여야 합니다 (Generator가 W/2 → Upsample×2 구조): {args.window_size}")
    return args


def critic_step(
    model: TadGAN,
    x: Tensor,
    opt_cx: torch.optim.Optimizer,
    opt_cz: torch.optim.Optimizer,
    wgan: GANLoss,
    lambda_gp: float,
) -> dict[str, float]:
    """C_x, C_z를 차례로 한 번씩 갱신한다. fake(G(z), E(x))는 no_grad로 만들어 E·G로 gradient가 가지 않게 한다."""
    model.critic_x.requires_grad_(True)
    model.critic_z.requires_grad_(True)
    z = model.sample_z(x.size(0), x.device)  # (B, d, 1) prior 샘플 = C_z의 real
    with torch.no_grad():
        x_fake = model.generator(z)  # (B, W, 1)
        z_fake = model.encoder(x)  # (B, d, 1)

    # C_x: real window x vs G(z). GP는 x와 G(z)를 샘플별 α로 섞은 점에서 ‖∇C_x‖ → 1
    cx_real = model.critic_x(x)  # (B,)
    cx_fake = model.critic_x(x_fake)
    gp_x = gradient_penalty(model.critic_x, x, x_fake)
    loss_cx = wgan.d_loss(cx_real, cx_fake) + lambda_gp * gp_x
    opt_cx.zero_grad(set_to_none=True)
    loss_cx.backward()
    opt_cx.step()

    # C_z: prior z vs E(x)
    cz_real = model.critic_z(z)  # (B,)
    cz_fake = model.critic_z(z_fake)
    gp_z = gradient_penalty(model.critic_z, z, z_fake)
    loss_cz = wgan.d_loss(cz_real, cz_fake) + lambda_gp * gp_z
    opt_cz.zero_grad(set_to_none=True)
    loss_cz.backward()
    opt_cz.step()

    return {
        "loss/critic_x": loss_cx.item(),
        "loss/critic_z": loss_cz.item(),
        "gp/x": gp_x.item(),
        "gp/z": gp_z.item(),
        "wdist/x": (cx_real.mean() - cx_fake.mean()).item(),  # Wasserstein 거리 추정 (x 공간)
        "wdist/z": (cz_real.mean() - cz_fake.mean()).item(),  # (latent 공간)
    }


def encoder_generator_step(
    model: TadGAN,
    x: Tensor,
    opt_eg: torch.optim.Optimizer,
    wgan: GANLoss,
    lambda_rec: float,
) -> dict[str, float]:
    """E·G를 함께 한 번 갱신한다. critic은 고정하고 입력 쪽 gradient만 E·G로 흘린다."""
    model.critic_x.requires_grad_(False)
    model.critic_z.requires_grad_(False)
    z = model.sample_z(x.size(0), x.device)  # (B, d, 1)
    x_fake = model.generator(z)  # (B, W, 1)
    z_fake = model.encoder(x)  # (B, d, 1)
    x_rec = model.generator(z_fake)  # (B, W, 1) = G(E(x))

    adv_x = wgan.g_loss(model.critic_x(x_fake))  # −E[C_x(G(z))]
    adv_z = wgan.g_loss(model.critic_z(z_fake))  # −E[C_z(E(x))]
    rec = F.mse_loss(x_rec, x)  # cycle consistency
    loss_eg = adv_x + adv_z + lambda_rec * rec
    opt_eg.zero_grad(set_to_none=True)
    loss_eg.backward()
    opt_eg.step()

    return {"loss/eg": loss_eg.item(), "loss/adv_x": adv_x.item(), "loss/adv_z": adv_z.item(), "loss/rec": rec.item()}


@torch.no_grad()
def plot_reconstructions(model: TadGAN, batch: Tensor, labels: list[str], path: Path, title: str) -> None:
    """고정 window들의 x와 G(E(x))를 겹쳐 그린 PNG를 저장한다. eval 모드(dropout 끔)로 복원한다."""
    model.eval()
    rec = model.reconstruct(batch)  # (n, W, 1)
    model.train()
    xs = batch.squeeze(-1).cpu().numpy()  # (n, W)
    rs = rec.squeeze(-1).cpu().numpy()

    fig = Figure(figsize=(10, 2.0 * len(xs)), layout="constrained")
    axes = fig.subplots(len(xs), 1, squeeze=False)[:, 0]
    for ax, x, r, label in zip(axes, xs, rs, labels):
        ax.plot(x, lw=1.0, label="x (scaled)")
        ax.plot(r, lw=1.0, label="G(E(x))")
        ax.set_ylabel(label)
    axes[0].set_title(title)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("time step in window")
    fig.savefig(path, dpi=100)


def _add(total: dict[str, float], values: dict[str, float]) -> None:
    for k, v in values.items():
        total[k] = total.get(k, 0.0) + v


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed)
    device = get_device(args.device)
    run_dir = resolve_run_dir(args, "TadGAN")
    ckpt = None
    if args.resume:
        ckpt = load_checkpoint(args.resume)
        check_resume_config(ckpt["config"], args)  # save_config 전에 검사해 원래 config.json을 덮어쓰지 않는다
    save_config(vars(args), run_dir)

    # 데이터: 1-D 신호 → (집계) → 평균 대치 → MinMax [-1, 1] → window (N, W, 1)
    timestamps, raw, anomalies = load_series(
        args.dataset,
        data_path=args.data_path,
        labels_path=args.labels_path,
        data_root=args.data_root,
        synthetic_n=args.synthetic_n,
        synthetic_seed=args.synthetic_seed,
    )
    stats = ckpt["data_stats"] if ckpt is not None else None  # resume 시 처음 fit한 통계를 그대로 쓴다
    timestamps, values, stats = preprocess(timestamps, raw, interval=args.interval, stats=stats)
    dataset = TimeSeriesWindows(values, args.window_size, args.window_step)
    loader = build_loader(dataset, args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    steps_per_epoch = len(loader) // args.n_critic  # E·G step 수
    batches_per_epoch = steps_per_epoch * args.n_critic  # 남는 batch는 E·G step 없이 critic만 갱신하므로 쓰지 않는다
    if steps_per_epoch == 0:
        raise ValueError(
            f"epoch당 batch 수({len(loader)})가 n_critic({args.n_critic})보다 적어 E·G step이 한 번도 일어나지 않습니다. "
            "신호를 늘리거나 --batch_size·--n_critic·--window_step을 줄이십시오."
        )

    model = TadGAN(args.window_size, args.latent_dim, args.critic_z_hidden).to(device)
    betas = (args.beta1, args.beta2)
    eg_params = list(model.encoder.parameters()) + list(model.generator.parameters())
    opt_eg = torch.optim.Adam(eg_params, lr=args.lr, betas=betas)
    opt_cx = torch.optim.Adam(model.critic_x.parameters(), lr=args.lr, betas=betas)
    opt_cz = torch.optim.Adam(model.critic_z.parameters(), lr=args.lr, betas=betas)
    wgan = GANLoss("wgan")

    start_epoch, global_step = 0, 0
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        opt_eg.load_state_dict(ckpt["opt_eg"])
        opt_cx.load_state_dict(ckpt["opt_cx"])
        opt_cz.load_state_dict(ckpt["opt_cz"])
        start_epoch, global_step = ckpt["epoch"], ckpt["step"]
        print(f"resume: {args.resume} (epoch {start_epoch}, step {global_step})", flush=True)

    # 복원 plot용 고정 window: 신호 전체에 고르게 n_plot개
    plot_idx = np.linspace(0, len(dataset) - 1, min(args.n_plot, len(dataset))).round().astype(int)
    plot_batch = torch.stack([dataset[int(i)] for i in plot_idx]).to(device)  # (n_plot, W, 1)
    plot_labels = [f"t0={timestamps[dataset.starts[i]]}" for i in plot_idx]

    n_labels = "없음" if anomalies is None else f"{len(anomalies)}개 (학습에 쓰지 않음)"
    print(
        f"device={device} | run_dir={run_dir}\n"
        f"signal L={len(values)} | windows N={len(dataset)} (W={args.window_size}, step={args.window_step}) | "
        f"batches/epoch={batches_per_epoch} (of {len(loader)}) | E·G steps/epoch={steps_per_epoch} | labels {n_labels}\n"
        f"params: E {count_params(model.encoder):,}, G {count_params(model.generator):,}, "
        f"C_x {count_params(model.critic_x):,}, C_z {count_params(model.critic_z):,}",
        flush=True,
    )

    with Logger(run_dir, vars(args), use_wandb=args.wandb, project=args.wandb_project, run_name=args.run_name) as logger:
        model.train()  # 학습 중에는 E·G·critic 모두 train 모드 (dropout on), Keras train_on_batch와 같다
        for epoch in range(start_epoch, args.epochs):
            c_sum: dict[str, float] = {}
            eg_sum: dict[str, float] = {}
            n_c = n_eg = 0
            for i, x in enumerate(loader):
                if i >= batches_per_epoch:  # 섞은 순서에서 앞쪽 batch만 쓴다 (Orion과 같다)
                    break
                x = x.to(device, non_blocking=True)  # (B, W, 1)
                c_stats = critic_step(model, x, opt_cx, opt_cz, wgan, args.lambda_gp)
                _add(c_sum, c_stats)
                n_c += 1
                if (i + 1) % args.n_critic != 0:
                    continue

                eg_stats = encoder_generator_step(model, x, opt_eg, wgan, args.lambda_rec)
                _add(eg_sum, eg_stats)
                n_eg += 1
                global_step += 1

                if global_step % args.log_every == 0:
                    scalars = {**c_stats, **eg_stats}  # 이번 step 값 (critic은 마지막 critic step)
                    logger.log_scalars(scalars, global_step)
                    print(
                        f"epoch {epoch + 1}/{args.epochs} step {global_step} | "
                        + " ".join(f"{k} {v:.4f}" for k, v in scalars.items()),
                        flush=True,
                    )
                if global_step % args.sample_every == 0:
                    plot_reconstructions(
                        model,
                        plot_batch,
                        plot_labels,
                        run_dir / "samples" / f"recon_{global_step:07d}.png",
                        f"step {global_step} (epoch {epoch + 1})",
                    )

            # epoch 평균 (Orion의 epoch별 cx/cz/eg loss 출력에 해당)
            summary = {k: v / max(n_c, 1) for k, v in c_sum.items() if k.startswith("loss/")}
            summary.update({k: v / max(n_eg, 1) for k, v in eg_sum.items() if k in ("loss/eg", "loss/rec")})
            print(
                f"[epoch {epoch + 1}/{args.epochs}] mean " + " ".join(f"{k} {v:.4f}" for k, v in summary.items()),
                flush=True,
            )

            if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
                save_rolling_checkpoint(
                    run_dir / "checkpoints",
                    epoch + 1,
                    args.keep_every,
                    model=model.state_dict(),
                    opt_eg=opt_eg.state_dict(),
                    opt_cx=opt_cx.state_dict(),
                    opt_cz=opt_cz.state_dict(),
                    epoch=epoch + 1,
                    step=global_step,
                    data_stats=stats,
                    config=vars(args),
                )

        plot_reconstructions(  # 학습 종료 시점의 복원 (주기 plot과 이름이 겹치지 않게 별도 파일)
            model,
            plot_batch,
            plot_labels,
            run_dir / "samples" / "recon_final.png",
            f"final: step {global_step} (epoch {args.epochs})",
        )


if __name__ == "__main__":
    main()
