"""StyleGAN2 (Karras et al., CVPR 2020) — style-based generator를 64px 얼굴 데이터로 학습한다 (config-f 축소판).

모델 (model.py, ops.py):
    G: mapping network(z → w, MLP 8층) + synthesis network(4×4 상수 → modulated conv + noise → ToRGB skip 누적)
    D: residual discriminator (blur-downsample ResBlock, minibatch stddev, logit 출력)
학습 (rosinality/stylegan2-pytorch train.py 기준):
    - loss: non-saturating logistic (GANLoss("vanilla") = softplus 형태)
    - lazy R1: D를 d_reg_every(16) step마다 real 이미지 gradient norm²으로 정규화. 가중치 γ/2 · d_reg_every
      γ(--r1) 기본값은 StyleGAN2-ADA 휴리스틱 γ = 0.0002·size²/batch_size로 자동 결정한다
      (64px·batch 16 → 0.0512). config-f의 γ = 10은 1024px·batch 32용이다. `--r1 10`처럼 명시하면 그 값을 쓴다
    - path length regularization: g_reg_every(4) step마다 ‖J_wᵀ y‖가 일정하도록 (latent → 이미지 변화량 균일화)
    - lazy regularization 보정: 정규화 step만큼 optimizer가 더 도므로 lr·c, betas^c (c = k/(k+1))
    - style mixing(0.9): 두 z의 w를 무작위 층에서 이어 붙여 층별 style이 서로 독립적이 되게 한다
    - G EMA: 샘플·평가는 G_ema로 한다. step마다 θ_ema ← β·θ_ema + (1-β)·θ, β = 0.5^(B_ref / (ema_kimg·1000)).
      과거 weight 비중이 절반이 되는 데 ema_kimg·1000 / B_ref step, 이미지로는 ema_kimg·1000 · batch_size / B_ref장이 걸린다
        기본 (--ema_batch_aware false): rosinality처럼 B_ref = 32 고정 → β = 0.99778.
          batch 16이면 half-life는 312.5 step = 5천 장 (ema_kimg 10천 장이 아니다)
        --ema_batch_aware true: 공식처럼 B_ref = batch_size → half-life가 batch와 무관하게 정확히 ema_kimg천 장
          (batch 16이면 β = 0.99889, 625 step)

원 논문(config-f, 1024px) 대비 단순화:
    - upfirdn2d·fused_leaky_relu를 custom CUDA 대신 순수 PyTorch로 구현 (같은 수학, 더 느림)
    - 해상도 64px·batch 16 기본 (12GB GPU용). 공식은 1024px·batch 32·8 GPU
    - ADA(적응형 augmentation), mixed precision(FP16), conv clamping, FID/PPL 측정, projector 없음
    - modulated conv는 항상 grouped conv(fused) 경로. 공식 ADA 코드는 학습 때 non-fused 경로를 쓴다
    - truncation 중심 w̄는 G 안의 running average(w_avg) 대신 샘플링 시점에 z 4096개로 추정 (rosinality 방식)
    - DDP 전용 트릭(`0 * real_pred[0]` 등) 없음 — 단일 GPU
    - R1 γ 기본값을 config-f 고정값 10 대신 해상도·batch에 맞춘 ADA 휴리스틱으로 정한다 (위 lazy R1 항목)

데이터 (center crop 후 --size로 resize, [-1, 1]):
    기본 celeba: $DATA_ROOT/celeba/img_align_celeba/*.jpg  (ImageFolder 구조)
    FFHQ 등 임의 폴더: --dataset folder --data_path <ImageFolder 루트> (예: $DATA_ROOT/ffhq → ffhq/<subdir>/*.png)

실행 예시 (repo 루트에서 `pip install -e .` 후):
    cd 8.StyleGAN2
    python train.py                                              # CelebA 64px, 100k iteration
    python train.py --dataset folder --data_path ffhq --size 128 --batch_size 8
    python train.py --dataset fake --iters 20 --log_every 5 --sample_every 10 --save_every 10 --num_workers 0
    python train.py --resume runs/StyleGAN2/<run>/checkpoints/last.pt            # 같은 run 폴더에 이어서
    python generate.py --checkpoint runs/StyleGAN2/<run>/checkpoints/last.pt --truncation 0.7

주기 인자 단위는 모두 iteration이다: --log_every, --sample_every, --save_every, --keep_every.
--keep_every는 --save_every의 배수여야 한다 (예: --save_every 5000 --keep_every 50000).
산출물: runs/StyleGAN2/<run_name|timestamp>/ 아래 config.json, tb/, samples/, checkpoints/last.pt (+ ckpt_XXXXXXX.pt)
    samples/real_*.png: real batch, samples/G_ema_*.png: 고정 z의 G_ema 샘플, samples/G_ema_final_*.png: 학습 종료 시점
"""

from __future__ import annotations

import argparse
import math
import random
import time
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from gan_common.checkpoint import load_checkpoint, save_rolling_checkpoint
from gan_common.config import base_parser, positive_int, str2bool
from gan_common.data import build_image_dataset, build_loader, infinite_batches
from gan_common.ema import EMA
from gan_common.logger import Logger
from gan_common.losses import GANLoss
from gan_common.regularizers import r1_penalty
from gan_common.utils import count_params, get_device, resolve_run_dir, save_config, seed_everything
from model import SIZES, Discriminator, Generator

MODEL_NAME = "StyleGAN2"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = base_parser(
        "StyleGAN2 (config-f 축소판, 순수 PyTorch ops)",
        dataset="celeba",
        channels=3,
        batch_size=16,
        lr=2e-3,
        beta1=0.0,
        beta2=0.99,
        log_every=100,
        sample_every=1000,
        save_every=5000,
        # iteration 기반이라 쓰지 않는다: 해상도는 --size, 학습 길이는 --iters
        epochs=None,
        image_size=None,
    )
    g = p.add_argument_group("StyleGAN2 model")
    g.add_argument("--size", type=int, default=64, help=f"학습 해상도 {SIZES}")
    g.add_argument("--style_dim", type=positive_int, default=512, help="z·w 차원")
    g.add_argument("--n_mlp", type=positive_int, default=8, help="mapping network 층 수")
    g.add_argument("--channel_multiplier", type=positive_int, default=2, help="64px 이상 채널 배율 (2 = config-f)")

    g = p.add_argument_group("StyleGAN2 training")
    g.add_argument("--iters", type=positive_int, default=100_000, help="총 iteration 수 (D step + G step = 1)")
    g.add_argument(
        "--r1",
        type=float,
        default=None,
        help="R1 가중치 γ. 없으면 0.0002·size²/batch_size (StyleGAN2-ADA 휴리스틱), 0이면 R1 끔",
    )
    g.add_argument("--d_reg_every", type=positive_int, default=16, help="lazy R1 주기 k_D")
    g.add_argument("--g_reg_every", type=positive_int, default=4, help="lazy path length 주기 k_G")
    g.add_argument("--path_regularize", type=float, default=2.0, help="path length 가중치 (0이면 끔)")
    g.add_argument("--path_batch_shrink", type=positive_int, default=2, help="path length 계산 batch = batch_size // 이 값")
    g.add_argument("--path_decay", type=float, default=0.01, help="path length 이동평균 decay")
    g.add_argument("--mixing", type=float, default=0.9, help="style mixing 확률")
    g.add_argument(
        "--ema_kimg",
        type=float,
        default=10.0,
        help="G EMA half-life(천 장). 정확히 이 값이 되는 것은 --ema_batch_aware true일 때이고, "
        "false면 실제 half-life = ema_kimg · batch_size / 32",
    )
    g.add_argument(
        "--ema_batch_aware",
        type=str2bool,
        default=False,
        help="true: β = 0.5^(batch_size/(ema_kimg·1000)) (공식). false: β = 0.5^(32/(ema_kimg·1000)) (rosinality)",
    )

    g = p.add_argument_group("StyleGAN2 sampling")
    g.add_argument("--n_samples", type=positive_int, default=64, help="고정 z 샘플 수 (로그용 grid)")
    g.add_argument("--truncation", type=float, default=1.0, help="로그 샘플 truncation ψ (1이면 끔)")
    g.add_argument("--truncation_mean", type=positive_int, default=4096, help="mean latent 추정에 쓸 z 개수")
    args = p.parse_args(argv)

    if args.epochs is not None:
        p.error("iteration 기반 학습입니다. --epochs 대신 --iters를 쓰십시오.")
    if args.image_size is not None and args.image_size != args.size:
        p.error("해상도는 --size로 정합니다 (--image_size는 쓰지 않습니다).")
    args.image_size = args.size
    if args.size not in SIZES:
        p.error(f"--size는 {SIZES} 중 하나여야 합니다: {args.size}")
    if not 0.0 <= args.mixing <= 1.0:
        p.error(f"--mixing은 [0, 1] 범위여야 합니다: {args.mixing}")
    if not 0.0 < args.truncation <= 1.0:
        p.error(f"--truncation은 (0, 1] 범위여야 합니다: {args.truncation}")
    if args.ema_kimg <= 0:
        p.error(f"--ema_kimg는 양수여야 합니다: {args.ema_kimg}")
    if args.batch_size < 1:
        p.error(f"--batch_size는 1 이상이어야 합니다: {args.batch_size}")
    if not 0.0 < args.path_decay <= 1.0:
        p.error(f"--path_decay는 (0, 1] 범위여야 합니다: {args.path_decay}")
    # R1 γ: 지정하지 않으면 StyleGAN2-ADA의 휴리스틱으로 정해 args에 기록한다 (config.json·checkpoint에 실제 값이 남는다)
    args.r1_auto = args.r1 is None
    if args.r1_auto:
        args.r1 = 0.0002 * args.size**2 / args.batch_size
    if args.r1 < 0 or args.path_regularize < 0:
        p.error("--r1, --path_regularize는 0 이상이어야 합니다.")
    if args.keep_every > 0 and args.keep_every % args.save_every != 0:
        p.error(f"--keep_every({args.keep_every})는 --save_every({args.save_every})의 배수여야 합니다 (단위: iteration).")
    return args


def build_generator(cfg: Mapping[str, Any]) -> Generator:
    """config(dict)로 G를 만든다. generate.py도 이 함수로 학습 때와 같은 구조를 만든다."""
    return Generator(cfg["size"], cfg["style_dim"], cfg["n_mlp"], cfg["channel_multiplier"], cfg["channels"])


def build_discriminator(cfg: Mapping[str, Any]) -> Discriminator:
    return Discriminator(cfg["size"], cfg["channel_multiplier"], cfg["channels"])


def ema_decay(args: argparse.Namespace) -> float:
    """step당 EMA decay β = 0.5^(B_ref / (ema_kimg·1000)). B_ref = batch_size(batch-aware) 또는 32.

    과거 weight 비중이 절반이 되는 데 ema_kimg·1000 / B_ref step = ema_kimg·1000 · batch_size / B_ref장이 걸린다.
    즉 half-life가 정확히 ema_kimg천 장인 것은 B_ref = batch_size일 때뿐이다.
    """
    ref_batch = args.batch_size if args.ema_batch_aware else 32
    return 0.5 ** (ref_batch / (args.ema_kimg * 1000.0))


def mixing_noise(n: int, style_dim: int, prob: float, device: torch.device) -> list[Tensor]:
    """확률 prob로 z 2개(style mixing), 아니면 z 1개의 list. 각 (n, style_dim)."""
    if prob > 0 and random.random() < prob:
        return list(torch.randn(2, n, style_dim, device=device).unbind(0))
    return [torch.randn(n, style_dim, device=device)]


def path_length_penalty(
    fake: Tensor, latents: Tensor, mean_path_length: Tensor, decay: float
) -> tuple[Tensor, Tensor, Tensor]:
    """Path length regularization (StyleGAN2 §3.2).

        y ~ N(0, I) / sqrt(H·W)            이미지 공간 무작위 방향 (해상도에 무관하도록 정규화)
        L_i = ‖J_wᵀ y‖₂ = ‖∇_w (g(w)·y)‖    w가 움직일 때 이미지가 변하는 정도
        a ← a + decay·(E[L] - a)            L의 이동평균
        penalty = E[(L - a)²]              모든 w에서 변화량이 같은 크기 a가 되도록 (J_w가 등척에 가깝게)

    `latents`는 G가 반환한 w+ (N, n_latent, style_dim). 층별 gradient norm²을 합한 뒤 층 평균을 내고 sqrt한다.
    penalty 자체를 G로 역전파해야 하므로 `create_graph=True` (2차 미분).
    a(path_mean)는 detach하지 않고 penalty에 쓴다 (rosinality·stylegan2-ada-pytorch와 같다. decay 0.01이라 영향은 작다).
    반환: (penalty, 새 이동평균 (detach), 샘플별 L (detach)).
    """
    _, _, h, w = fake.shape
    noise = torch.randn_like(fake) / math.sqrt(h * w)
    (grad,) = torch.autograd.grad(outputs=(fake * noise).sum(), inputs=latents, create_graph=True)
    path_lengths = grad.pow(2).sum(dim=2).mean(dim=1).sqrt()  # (N,)
    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)
    penalty = (path_lengths - path_mean).pow(2).mean()
    return penalty, path_mean.detach(), path_lengths.detach()


@torch.no_grad()
def sample_ema(
    G_ema: Generator, z: Tensor, truncation: float, truncation_mean: int, batch_size: int, seed: int
) -> Tensor:
    """고정 z → G_ema 이미지 (N, C, size, size). noise도 G 안의 고정 buffer를 써서 weight 변화만 비교한다.

    truncation < 1이면 w̄를 고정 seed의 torch.Generator로 뽑은 z로 추정한다: 샘플링 때마다 같은 z 집합을 써서
    w̄ 추정 잡음이 step 간 비교에 섞이지 않고, 전역 RNG(학습 난수 흐름)도 소비하지 않는다.
    """
    mean_w = None
    if truncation < 1:
        gen = torch.Generator(device=z.device).manual_seed(seed)
        mean_w = G_ema.mean_latent(truncation_mean, generator=gen)  # (1, style_dim)
    images = [
        G_ema([chunk], truncation=truncation, truncation_latent=mean_w, randomize_noise=False)[0]
        for chunk in z.split(batch_size)
    ]
    return torch.cat(images)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed)
    device = get_device(args.device)
    run_dir = resolve_run_dir(args, MODEL_NAME)
    config = vars(args)
    save_config(config, run_dir)

    # ---- data: [-1, 1], (N, C, size, size) ----
    dataset = build_image_dataset(
        args.dataset, args.size, args.channels, root=args.data_root, download=args.download, path=args.data_path
    )
    loader = build_loader(dataset, args.batch_size, num_workers=args.num_workers)
    data = infinite_batches(loader)  # epoch 경계 없이 (x, y) batch를 계속 낸다

    # ---- model: equalized lr 방식이라 별도 init_weights를 하지 않는다 (N(0, 1) + runtime scale) ----
    G = build_generator(config).to(device)
    D = build_discriminator(config).to(device)
    ema = EMA(G, decay=ema_decay(args))  # G.to(device) 다음에 만든다
    G_ema = ema.module

    # lazy regularization 보정: k step마다 정규화 step이 한 번 더 있으므로 lr과 momentum 시간상수를 c배로 맞춘다
    g_ratio = args.g_reg_every / (args.g_reg_every + 1) if args.path_regularize > 0 else 1.0
    d_ratio = args.d_reg_every / (args.d_reg_every + 1) if args.r1 > 0 else 1.0
    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr * g_ratio, betas=(args.beta1**g_ratio, args.beta2**g_ratio))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr * d_ratio, betas=(args.beta1**d_ratio, args.beta2**d_ratio))
    gan_loss = GANLoss("vanilla")  # D: softplus(-D(x)) + softplus(D(G(z))), G: softplus(-D(G(z)))

    sample_z = torch.randn(args.n_samples, args.style_dim, device=device)  # 학습 진행을 같은 z로 비교
    mean_path_length = torch.zeros((), device=device)  # path length 이동평균 a
    start_step = 0
    if args.resume:
        ckpt = load_checkpoint(args.resume)
        G.load_state_dict(ckpt["G"])
        D.load_state_dict(ckpt["D"])
        ema.load_state_dict(ckpt["G_ema"])
        opt_G.load_state_dict(ckpt["opt_G"])
        opt_D.load_state_dict(ckpt["opt_D"])
        start_step = ckpt["step"]
        mean_path_length = torch.tensor(ckpt["mean_path_length"], device=device)
        sample_z = ckpt["sample_z"].to(device)
        print(f"resume: {args.resume} (step {start_step})", flush=True)

    print(
        f"device={device} | {args.dataset} {len(dataset)}장 {args.size}px | "
        f"G {count_params(G):,} / D {count_params(D):,} params | EMA decay {ema.decay:.6f} | run_dir={run_dir}",
        flush=True,
    )
    r1_source = "자동: 0.0002·size²/batch_size" if args.r1_auto else "지정"
    print(f"R1 γ = {args.r1:g} ({r1_source})", flush=True)

    def sample_fixed() -> Tensor:
        return sample_ema(G_ema, sample_z, args.truncation, args.truncation_mean, args.batch_size, args.seed)

    def checkpoint_state(step: int) -> dict[str, Any]:
        return {
            "step": step,
            "G": G.state_dict(),
            "D": D.state_dict(),
            "G_ema": ema.state_dict(),
            "opt_G": opt_G.state_dict(),
            "opt_D": opt_D.state_dict(),
            "mean_path_length": float(mean_path_length),
            "sample_z": sample_z.cpu(),
            "config": config,
        }

    with Logger(run_dir, config, use_wandb=args.wandb, project=args.wandb_project, run_name=args.run_name) as logger:
        real_batch, _ = next(data)
        logger.log_images("real", real_batch[: args.n_samples], start_step)  # 비교용 real 이미지

        # 정규화 항은 주기적으로만 계산하므로 마지막 값을 들고 있다가 로깅한다
        last_r1 = torch.zeros((), device=device)
        last_path = torch.zeros((), device=device)
        last_path_length = torch.zeros((), device=device)
        step = start_step
        t0 = time.perf_counter()
        for step in range(start_step + 1, args.iters + 1):
            real, _ = next(data)
            real = real.to(device, non_blocking=True)  # (N, C, S, S)
            n = real.size(0)

            # (1) D step ---------------------------------------------------------------
            # fake는 no_grad로 만든다: D loss의 gradient가 G로 갈 필요가 없고 activation도 저장하지 않는다
            D.requires_grad_(True)
            with torch.no_grad():
                fake, _ = G(mixing_noise(n, args.style_dim, args.mixing, device))  # (N, C, S, S)
            real_logits = D(real)  # (N,)
            fake_logits = D(fake)
            loss_D = gan_loss.d_loss(real_logits, fake_logits)
            opt_D.zero_grad(set_to_none=True)
            loss_D.backward()
            opt_D.step()

            # (1b) lazy R1: real 이미지에서 ‖∇_x D(x)‖²를 줄여 real 주변에서 D를 평탄하게 한다.
            # d_reg_every step마다 한 번만 계산하고, 빠진 step 수만큼 d_reg_every를 곱해 보상한다
            if args.r1 > 0 and step % args.d_reg_every == 0:
                real_r1 = real.detach().requires_grad_(True)
                r1 = r1_penalty(D(real_r1), real_r1)
                opt_D.zero_grad(set_to_none=True)
                (args.r1 / 2 * r1 * args.d_reg_every).backward()
                opt_D.step()
                last_r1 = r1.detach()

            # (2) G step ---------------------------------------------------------------
            # D는 고정: D weight의 gradient 계산을 생략한다 (입력 fake로 가는 gradient는 그대로 흐른다)
            D.requires_grad_(False)
            fake, _ = G(mixing_noise(n, args.style_dim, args.mixing, device))
            loss_G = gan_loss.g_loss(D(fake))
            opt_G.zero_grad(set_to_none=True)
            loss_G.backward()
            opt_G.step()

            # (2b) lazy path length regularization (batch를 path_batch_shrink배 줄여 비용 절감)
            if args.path_regularize > 0 and step % args.g_reg_every == 0:
                path_n = max(1, n // args.path_batch_shrink)
                fake_pl, latents = G(mixing_noise(path_n, args.style_dim, args.mixing, device), return_latents=True)
                path_loss, mean_path_length, path_lengths = path_length_penalty(
                    fake_pl, latents, mean_path_length, args.path_decay
                )
                opt_G.zero_grad(set_to_none=True)
                (args.path_regularize * args.g_reg_every * path_loss).backward()
                opt_G.step()
                last_path = path_loss.detach()
                last_path_length = path_lengths.mean()

            ema.update(G)

            if step % args.log_every == 0:
                elapsed = time.perf_counter() - t0
                t0 = time.perf_counter()
                scalars = {
                    "loss/D": loss_D.item(),
                    "loss/G": loss_G.item(),
                    "score/real": real_logits.detach().mean().item(),  # D logit 평균
                    "score/fake": fake_logits.detach().mean().item(),
                    "reg/r1": last_r1.item(),
                    "reg/path": last_path.item(),
                    "reg/path_length": last_path_length.item(),
                    "reg/mean_path_length": mean_path_length.item(),
                    # 학습 step만의 속도: 샘플링·checkpoint 저장 시간은 아래에서 t0를 그만큼 밀어 제외한다.
                    # CUDA는 비동기라 경계가 정확하지 않은 근사치다
                    "perf/it_per_sec": args.log_every / elapsed,
                }
                logger.log_scalars(scalars, step)
                print(f"step {step}/{args.iters} | " + " ".join(f"{k} {v:.4f}" for k, v in scalars.items()), flush=True)
            if step % args.sample_every == 0:
                t_pause = time.perf_counter()
                logger.log_images("G_ema", sample_fixed(), step)
                t0 += time.perf_counter() - t_pause
            if step % args.save_every == 0 or step == args.iters:
                t_pause = time.perf_counter()
                save_rolling_checkpoint(run_dir / "checkpoints", step, args.keep_every, **checkpoint_state(step))
                t0 += time.perf_counter() - t_pause

        # 학습 종료 시점의 G_ema 샘플을 한 번 더 남긴다
        logger.log_images("G_ema_final", sample_fixed(), step)
    print(f"완료: step {step} | checkpoint {run_dir / 'checkpoints' / 'last.pt'}", flush=True)


if __name__ == "__main__":
    main()
