"""ProGAN (Karras 2018, Progressive Growing of GANs): 4x4에서 시작해 해상도를 2배씩 키우며 G·D를 함께 성장시킨다.

핵심 아이디어:
- 저해상도에서 큰 구조를 먼저 배우고, 새 해상도 layer는 fade-in(alpha 0 → 1)으로 서서히 끼워 넣어 고해상도 학습을 안정화한다
- equalized learning rate(runtime He scaling), G의 PixelNorm, D의 minibatch stddev로 BatchNorm 없이 안정성과 다양성을 얻는다
- loss는 WGAN-GP(λ 10) + drift ε·E[D(x)²]. G는 EMA(0.999) 사본으로 샘플을 뽑는다

학습 일정 (phase = fade-in 또는 stabilize 한 구간, 각각 D가 real 이미지 --images_per_phase장을 볼 때까지):
    4x4 stabilize → 8x8 fade-in → 8x8 stabilize → … → max_res fade-in → max_res stabilize
    기본값(max_res 128)은 11 phase × 600k = 6.6M 이미지다. 해상도마다 batch는 --batch_sizes 표를 따른다.
    공식 구현은 최종 해상도에서 total_kimg까지 계속 학습하지만, 여기서는 max_res stabilize도 --images_per_phase에서 끝난다.
    더 학습하려면 마지막 last.pt에서 --images_per_phase를 늘려 resume한다 (마지막 stabilize phase가 이어진다).

데이터 준비: CelebA(aligned)를 `<DATA_ROOT>/celeba/img_align_celeba/*.jpg`에 둔다 (ImageFolder 구조: 하위 폴더 1개 이상).
    `<DATA_ROOT>/celeba/list_eval_partition.txt`가 있으면 공식 train split만, 없으면 폴더 전체를 쓴다.
    논문의 CelebA-HQ 대신 aligned CelebA를 짧은 변 기준 resize + center crop해서 max_res로 읽고,
    학습 중 GPU에서 현재 해상도로 area downsample한다.
실행 (repo 루트에서 `pip install -e .` 후):
    cd 7.ProGAN && python train.py
    python train.py --max_res 64 --images_per_phase 200000
    python train.py --dataset fake --max_res 16 --images_per_phase 512 --log_every 5 --sample_every 20   # 파이프라인 확인
    python train.py --dataset folder --data_path ffhq --max_res 256 \
        --batch_sizes 4:64,8:64,16:32,32:16,64:16,128:8,256:4
산출물: runs/ProGAN/<run_name|timestamp>/ 아래 config.json, tb/, samples/, checkpoints/
checkpoint: --save_every iteration마다, 그리고 phase가 끝날 때마다 checkpoints/last.pt를 덮어쓴다.
--keep_every N이면 iteration이 N의 배수일 때 ckpt_XXXXXXX.pt 사본을 남긴다.
stabilize phase가 끝날 때마다 해상도별 사본 checkpoints/ckpt_<res>px.pt도 남긴다 (예: ckpt_64px.pt).

이어서 학습 (--run_name 없이 --resume만 주면 원래 run 폴더에 이어 쓴다):
    python train.py --resume runs/ProGAN/<run>/checkpoints/last.pt
    - checkpoint의 phase·images_in_phase부터 이어 간다 (중단된 phase의 Adam state도 복원)
    - --max_res·--z_dim·--channels는 원래 값과 같아야 한다 (모델 구조). 다르면 바로 중단한다
    - fixed noise는 checkpoint에 저장된 것을 쓰므로 --n_samples는 무시된다
    - RNG·데이터 순서는 복원하지 않으므로 끊지 않고 학습한 결과와 bit 단위로 같지는 않다
"""

from __future__ import annotations

import argparse
import functools
from collections.abc import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from gan_common.checkpoint import load_checkpoint, save_checkpoint, save_rolling_checkpoint
from gan_common.config import base_parser, positive_int
from gan_common.data import build_image_dataset, build_loader, infinite_batches
from gan_common.ema import EMA
from gan_common.logger import Logger
from gan_common.losses import GANLoss
from gan_common.regularizers import gradient_penalty
from gan_common.utils import count_params, get_device, resolve_run_dir, save_config, seed_everything
from model import RESOLUTIONS, Discriminator, Generator, depth_to_res, res_to_depth  # 같은 폴더의 model.py

DEFAULT_BATCH_SIZES = "4:64,8:64,16:32,32:16,64:16,128:8"
ARCH_KEYS = ("max_res", "z_dim", "channels")  # 모델 구조(파라미터 shape)를 바꾸는 인자. resume 시 checkpoint와 같아야 한다


def parse_batch_sizes(text: str) -> dict[int, int]:
    """'4:64,8:64,…' → {4: 64, 8: 64, …}."""
    table: dict[int, int] = {}
    for item in text.split(","):
        if not item.strip():
            continue
        res, _, bs = item.partition(":")
        try:
            r, b = int(res), int(bs)
        except ValueError:
            raise ValueError(f"--batch_sizes 항목은 'res:batch' 형식이어야 합니다: {item.strip()!r}") from None
        if b < 1:
            raise ValueError(f"--batch_sizes의 batch는 1 이상이어야 합니다: {item.strip()!r}")
        table[r] = b
    return table


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = base_parser(
        "ProGAN (progressive growing) on CelebA",
        dataset="celeba",
        channels=3,
        lr=1e-3,
        beta1=0.0,  # 논문 Adam(α=1e-3, β1=0, β2=0.99, ε=1e-8)
        beta2=0.99,
        sample_every=1000,
        save_every=10000,
        # 쓰지 않는 공통 인자: 해상도는 --max_res, batch는 해상도별 --batch_sizes, 길이는 --images_per_phase
        image_size=None,
        batch_size=None,
        epochs=None,
    )
    g = p.add_argument_group("model")
    g.add_argument("--max_res", type=int, default=128, choices=RESOLUTIONS, help="최종 해상도 (4x4에서 2배씩 성장)")
    g.add_argument("--z_dim", type=positive_int, default=512, help="latent z 차원")
    g.add_argument(
        "--images_per_phase",
        type=positive_int,
        default=600_000,
        help="phase(fade-in 또는 stabilize) 하나에서 D가 보는 real 이미지 수. 논문 600k",
    )
    g.add_argument(
        "--batch_sizes",
        type=str,
        default=DEFAULT_BATCH_SIZES,
        help="해상도별 batch 크기 'res:batch,…'. max_res까지 모든 해상도가 있어야 한다",
    )
    g.add_argument("--n_critic", type=positive_int, default=1, help="G 1회 갱신당 D 갱신 횟수. 논문은 1 (번갈아 갱신)")
    g.add_argument("--lambda_gp", type=float, default=10.0, help="gradient penalty 계수 λ")
    g.add_argument("--drift", type=float, default=0.001, help="drift 계수 ε: loss_D += ε·E[D(x)²]")
    g.add_argument("--ema_decay", type=float, default=0.999, help="G EMA decay (샘플은 EMA G로 뽑는다)")
    g.add_argument(
        "--n_samples",
        type=positive_int,
        default=32,
        help="fixed noise 샘플 수 (로그용 grid). --resume 시에는 checkpoint의 fixed noise를 쓰므로 무시된다",
    )
    args = p.parse_args(argv)
    try:
        table = parse_batch_sizes(args.batch_sizes)
    except ValueError as e:
        p.error(str(e))
    missing = [r for r in RESOLUTIONS if r <= args.max_res and r not in table]
    if missing:
        p.error(f"--batch_sizes에 해상도 {missing}의 batch가 없습니다 (예: '{missing[0]}:4'를 추가).")
    if not 0.0 <= args.ema_decay < 1.0:
        p.error(f"--ema_decay는 0 이상 1 미만이어야 합니다: {args.ema_decay}")
    if args.lambda_gp < 0:
        p.error(f"--lambda_gp는 0 이상이어야 합니다: {args.lambda_gp}")
    if args.drift < 0:
        p.error(f"--drift는 0 이상이어야 합니다: {args.drift}")
    return args


def build_phases(max_depth: int) -> list[tuple[int, bool]]:
    """[(depth, fade_in), …]: 4x4 stabilize → (8x8 fade-in, 8x8 stabilize) → … → max_res stabilize."""
    phases = [(0, False)]  # 4x4는 섞을 이전 해상도가 없으므로 fade-in이 없다
    for d in range(1, max_depth + 1):
        phases += [(d, True), (d, False)]
    return phases


def fade_alpha(images_in_phase: int, images_per_phase: int, fade_in: bool) -> float:
    """fade-in이면 phase 진행률(0 → 1)에 비례해 선형 증가, stabilize면 1 (새 layer만 쓴다)."""
    return min(1.0, images_in_phase / images_per_phase) if fade_in else 1.0


def prepare_reals(x: Tensor, res: int, alpha: float, fade_in: bool) -> Tensor:
    """max_res real batch (N, C, R, R) → 현재 해상도 (N, C, res, res).

    fade-in 중 G 출력은 "이전 해상도 이미지의 upsample"과 "새 해상도 출력"의 alpha 혼합이다.
    real도 같은 방식으로 섞어야 D가 선명도 차이만 보고 real/fake를 가르지 못한다.
    """
    if x.size(-1) != res:
        x = F.interpolate(x, size=(res, res), mode="area")  # 정수배 축소면 블록 평균과 같다
    if fade_in and alpha < 1.0:
        low = F.interpolate(F.avg_pool2d(x, 2), scale_factor=2.0, mode="nearest")  # 이전 해상도 → nearest upsample
        x = torch.lerp(low, x, alpha)  # (1 - alpha)·low + alpha·x
    return x


def to_display(x: Tensor, max_res: int) -> Tensor:
    """해상도가 달라도 grid 크기가 같도록 max_res로 nearest upsample한다 (로그용)."""
    if x.size(-1) == max_res:
        return x
    return F.interpolate(x, size=(max_res, max_res), mode="nearest")


def take_reals(data: Iterator, n: int) -> Tensor:
    """infinite_batches에서 batch를 이어 꺼내 real 이미지 n장을 모은다 (로그용: batch가 n보다 작은 해상도 대비)."""
    xs: list[Tensor] = []
    count = 0
    while count < n:
        x, _ = next(data)  # batch = (이미지, label)
        xs.append(x)
        count += x.size(0)
    return torch.cat(xs)[:n]  # (n, C, max_res, max_res)


def check_resume_config(ckpt_config: dict, args: argparse.Namespace) -> None:
    """checkpoint의 모델 구조 인자(ARCH_KEYS)가 이번 실행과 다르면 load_state_dict 전에 알기 쉬운 에러로 멈춘다."""
    diffs = {k: (ckpt_config.get(k), getattr(args, k)) for k in ARCH_KEYS if ckpt_config.get(k) != getattr(args, k)}
    if diffs:
        detail = ", ".join(f"--{k}: checkpoint {old} ≠ 지금 {new}" for k, (old, new) in diffs.items())
        raise ValueError(f"--resume checkpoint와 모델 구조 인자가 다릅니다 ({detail}). 원래 값으로 다시 실행하십시오.")


def make_optimizers(
    G: nn.Module, D: nn.Module, args: argparse.Namespace
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    return opt_G, opt_D


@torch.no_grad()
def log_samples(
    logger: Logger, G: nn.Module, fixed_noise: Tensor, alpha: float, depth: int, max_res: int, step: int
) -> None:
    """fixed noise 샘플 grid (G는 EMA 사본을 넘긴다)."""
    logger.log_images("fake", to_display(G(fixed_noise, alpha, depth), max_res), step)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed)
    device = get_device(args.device)
    run_dir = resolve_run_dir(args, "ProGAN")

    batch_sizes = parse_batch_sizes(args.batch_sizes)
    max_depth = res_to_depth(args.max_res)
    phases = build_phases(max_depth)

    # 데이터: [-1, 1]로 정규화된 (N, C, max_res, max_res). 현재 해상도로 줄이는 것은 prepare_reals가 GPU에서 한다
    dataset = build_image_dataset(
        args.dataset,
        args.max_res,
        args.channels,
        root=args.data_root,
        download=args.download,
        path=args.data_path,
    )

    # 모델: 모든 해상도의 layer를 미리 만든다. Equalized layer가 N(0, 1)로 스스로 초기화한다
    G = Generator(args.z_dim, args.channels, args.max_res).to(device)
    D = Discriminator(args.channels, args.max_res).to(device)
    G_ema = EMA(G, args.ema_decay)  # G.to(device) 다음에 만든다 (deepcopy가 같은 device에 생기도록)
    gan_loss = GANLoss("wgan")

    fixed_noise = torch.randn(args.n_samples, args.z_dim).to(device)  # 학습 진행을 같은 z로 비교
    start_phase, start_images, step = 0, 0, 0
    resume_opt: tuple[dict, dict] | None = None
    if args.resume:
        ckpt = load_checkpoint(args.resume)
        check_resume_config(ckpt["config"], args)
        G.load_state_dict(ckpt["G"])
        D.load_state_dict(ckpt["D"])
        G_ema.load_state_dict(ckpt["G_ema"])
        start_phase, start_images, step = ckpt["phase"], ckpt["images_in_phase"], ckpt["step"]
        resume_opt = (ckpt["opt_G"], ckpt["opt_D"])
        fixed_noise = ckpt["fixed_noise"].to(device)
        print(
            f"resume: {args.resume} (phase {start_phase + 1}/{len(phases)}, {depth_to_res(ckpt['depth'])}px, "
            f"alpha {ckpt['alpha']:.3f}, images {start_images:,}, step {step})",
            flush=True,
        )
    save_config(vars(args), run_dir)  # resume 구조 검사를 통과한 뒤에 기록한다

    print(
        f"device={device} | run_dir={run_dir} | G {count_params(G):,} params, D {count_params(D):,} params | "
        f"{len(phases)} phases × {args.images_per_phase:,} images",
        flush=True,
    )

    with Logger(run_dir, vars(args), use_wandb=args.wandb, project=args.wandb_project, run_name=args.run_name) as logger:
        loader: DataLoader | None = None
        data: Iterator | None = None  # infinite_batches: (이미지, label) batch를 끝없이 낸다
        for phase in range(start_phase, len(phases)):
            depth, fade_in = phases[phase]
            res = depth_to_res(depth)
            batch_size = batch_sizes[res]
            images_in_phase = start_images if phase == start_phase else 0
            if images_in_phase >= args.images_per_phase:
                continue  # phase 끝에서 저장된 checkpoint로 resume한 경우: 다음 phase부터

            # phase가 바뀔 때마다(fade-in 시작, stabilize 시작) Adam moment를 초기화한다
            # (공식 구현 reset_opt_for_new_lod=True). 새 layer가 들어오면 이전 moment가 새 loss 지형과 맞지 않는다.
            # 단, resume으로 phase 중간부터 시작하면 저장된 moment를 이어 쓴다
            opt_G, opt_D = make_optimizers(G, D, args)
            if phase == start_phase and resume_opt is not None:
                opt_G.load_state_dict(resume_opt[0])
                opt_D.load_state_dict(resume_opt[1])
            if loader is None or loader.batch_size != batch_size:  # batch 크기가 바뀔 때만 DataLoader를 새로 만든다
                loader = build_loader(dataset, batch_size, num_workers=args.num_workers)
                data = infinite_batches(loader)

            print(
                f"phase {phase + 1}/{len(phases)}: {res}x{res} {'fade-in' if fade_in else 'stabilize'} | "
                f"batch {batch_size} | images {images_in_phase:,}/{args.images_per_phase:,}",
                flush=True,
            )
            real_vis = prepare_reals(take_reals(data, args.n_samples).to(device), res, 1.0, False)
            logger.log_images("real", to_display(real_vis, args.max_res), step)  # 이 해상도에서 D가 보는 real

            while images_in_phase < args.images_per_phase:
                alpha = fade_alpha(images_in_phase, args.images_per_phase, fade_in)
                critic = functools.partial(D, alpha=alpha, depth=depth)  # 현재 alpha·depth를 고정한 critic_fn

                # ---- Train D (critic): WGAN-GP + drift
                #   L_D = E[D(G(z))] - E[D(x)] + λ·E[(||∇_x̂ D(x̂)||_2 - 1)^2] + ε·E[D(x)^2]
                #   GP: real·fake 사이 보간점 x̂에서 gradient norm을 1로 당겨 1-Lipschitz에 가깝게 만든다
                #   drift: WGAN loss는 D 출력에 상수를 더해도 값이 같아 출력이 한없이 떠내려갈 수 있으므로 0 근처에 묶는다
                D.requires_grad_(True)
                for _ in range(args.n_critic):
                    x, _ = next(data)  # label은 쓰지 않는다 (unsupervised)
                    real = prepare_reals(x.to(device, non_blocking=True), res, alpha, fade_in)
                    z = torch.randn(real.size(0), args.z_dim, device=device)
                    with torch.no_grad():  # D step에서는 G로 gradient를 보내지 않는다
                        fake = G(z, alpha, depth)  # (N, C, res, res)
                    real_logits = critic(real)  # (N,)
                    fake_logits = critic(fake)
                    loss_w = gan_loss.d_loss(real_logits, fake_logits)  # Wasserstein 항
                    gp = gradient_penalty(critic, real, fake)
                    drift = real_logits.pow(2).mean()
                    loss_D = loss_w + args.lambda_gp * gp + args.drift * drift
                    opt_D.zero_grad()
                    loss_D.backward()
                    opt_D.step()

                # ---- Train G: L_G = -E[D(G(z))]. D weight gradient는 필요 없으므로 고정한다
                D.requires_grad_(False)
                z = torch.randn(batch_size, args.z_dim, device=device)
                loss_G = gan_loss.g_loss(critic(G(z, alpha, depth)))
                opt_G.zero_grad()
                loss_G.backward()
                opt_G.step()
                G_ema.update(G)  # θ_ema ← decay·θ_ema + (1 - decay)·θ

                step += 1
                images_in_phase += batch_size * args.n_critic  # D가 본 real 이미지 수로 진행을 센다
                phase_done = images_in_phase >= args.images_per_phase

                if step % args.log_every == 0:
                    losses = {
                        "loss/D": loss_D.item(),
                        "loss/G": loss_G.item(),
                        # E[D(x)] - E[D(G(z))]: Wasserstein 거리 추정치
                        "wasserstein": -loss_w.item(),
                        "gp": gp.item(),
                        "drift": drift.item(),
                    }
                    logger.log_scalars({**losses, "progress/alpha": alpha, "progress/resolution": res}, step)
                    print(
                        f"step {step} | {res}px {'fade-in' if fade_in else 'stable'} alpha {alpha:.3f} "
                        f"{images_in_phase:,}/{args.images_per_phase:,} | "
                        + " ".join(f"{k} {v:.4f}" for k, v in losses.items()),
                        flush=True,
                    )
                if step % args.sample_every == 0 or phase_done:  # phase 끝마다 해상도별 결과를 남긴다
                    log_samples(logger, G_ema.module, fixed_noise, alpha, depth, args.max_res, step)
                if step % args.save_every == 0 or phase_done:
                    state = {
                        "G": G.state_dict(),
                        "D": D.state_dict(),
                        "G_ema": G_ema.state_dict(),
                        "opt_G": opt_G.state_dict(),
                        "opt_D": opt_D.state_dict(),
                        "phase": phase,
                        "depth": depth,
                        "fade_in": fade_in,
                        "alpha": fade_alpha(images_in_phase, args.images_per_phase, fade_in),
                        "images_in_phase": images_in_phase,
                        "step": step,
                        "fixed_noise": fixed_noise.cpu(),
                        "config": vars(args),
                    }
                    save_rolling_checkpoint(run_dir / "checkpoints", step, args.keep_every, **state)
                    if phase_done and not fade_in:  # 해상도별 최종(stabilize 끝) 상태를 따로 보존한다
                        save_checkpoint(run_dir / "checkpoints" / f"ckpt_{res}px.pt", **state)

        # 학습 종료 시점 샘플: 마지막 phase는 max_res stabilize(alpha 1)
        log_samples(logger, G_ema.module, fixed_noise, 1.0, max_depth, args.max_res, step)


if __name__ == "__main__":
    main()
