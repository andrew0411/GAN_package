"""SNGAN (Miyato 2018): spectral normalization으로 D를 안정화한 GAN. CIFAR-10 32px ResNet 설정.

핵심 아이디어:
- D의 모든 conv/linear weight를 최대 특이값으로 나눈다 (W / σ(W)). σ는 forward마다 power iteration 1회로 싸게 추정한다
- layer마다 Lipschitz 상수가 1로 묶여 D의 gradient가 폭주하지 않는다. WGAN-GP처럼 입력 gradient를
  따로 계산하지 않고, weight clipping처럼 weight 분포를 뭉개지도 않는다
- loss는 hinge, G 1회 갱신당 D를 n_dis(5)회 갱신한다. 학습 길이는 epoch이 아니라 G step 수(`--iters`)다

원 구현(pfnet-research/sngan_projection)과의 관계:
- `--lr_decay linear`(기본): G step마다 lr_t = lr·(1 - (step - 1)/iters)로 2e-4 → 0 선형 감소 (D·G 모두)
  추측: 원 repo의 CIFAR-10 설정은 Adam alpha를 LinearShift로 첫 iteration부터(iteration_decay_start 0) 끝까지
  0으로 줄인다. 원 repo의 configs/sn_cifar10.yml·train.py로 확정할 수 있다
- `--g_batch_size`: G step batch 크기 (기본 = --batch_size 64).
  추측: 원 설정은 G step에 128을 썼다. 원 repo의 configs/sn_cifar10.yml·updater.py로 확정할 수 있다

데이터 준비: CIFAR-10은 torchvision이 `<DATA_ROOT>/cifar-10-batches-py`에 둔다 (없으면 `--download`).
실행 (repo 루트에서 `pip install -e .` 후):
    cd 6.SNGAN && python train.py --download
    python train.py --dataset fake --iters 20 --n_dis 1 --log_every 5 --sample_every 10   # 데이터 없이 파이프라인 확인
산출물: runs/SNGAN/<run_name|timestamp>/ 아래 config.json, tb/, samples/, checkpoints/
checkpoint: --save_every iteration마다 checkpoints/last.pt를 덮어쓰고, --keep_every N이면 iteration이 N의
배수일 때 ckpt_XXXXXXX.pt 사본을 남긴다.

이어서 학습 (--run_name 없이 --resume만 주면 원래 run 폴더에 이어 쓴다):
    python train.py --resume runs/SNGAN/<run>/checkpoints/last.pt --iters 100000
    - --iters는 총 G step 수다 (추가할 step 수가 아니다)
    - lr은 step과 --iters로 매 step 다시 계산하므로 resume해도 이어진다. 단 --iters를 바꾸면 감소 기울기도 바뀐다
    - fixed noise는 checkpoint에 저장된 것을 쓰므로 --n_samples는 무시된다
    - RNG·데이터 순서는 복원하지 않으므로 끊지 않고 학습한 결과와 bit 단위로 같지는 않다
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable

import torch
import torch.nn as nn
from torch import Tensor

from gan_common.checkpoint import load_checkpoint, save_rolling_checkpoint
from gan_common.config import base_parser, positive_int
from gan_common.data import build_image_dataset, build_loader, infinite_batches
from gan_common.logger import Logger
from gan_common.losses import GANLoss
from gan_common.utils import count_params, get_device, resolve_run_dir, save_config, seed_everything
from model import Discriminator, Generator  # 같은 폴더의 model.py


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = base_parser(
        "SNGAN (spectral normalization) on CIFAR-10 32px",
        dataset="cifar10",
        image_size=32,
        channels=3,
        batch_size=64,
        lr=2e-4,
        beta1=0.0,  # 논문 CIFAR-10 ResNet 설정: Adam(α=2e-4, β1=0, β2=0.9), n_dis=5
        beta2=0.9,
        sample_every=1000,
        save_every=5000,
        epochs=None,  # 쓰지 않는 공통 인자: 학습 길이는 --iters (G step 수)
    )
    g = p.add_argument_group("model")
    g.add_argument("--iters", type=positive_int, default=50000, help="총 G 갱신 횟수 (log·sample·save 주기의 단위)")
    g.add_argument("--n_dis", type=positive_int, default=5, help="G 1회 갱신당 D 갱신 횟수. D step마다 새 real batch")
    g.add_argument(
        "--lr_decay",
        type=str,
        default="linear",
        choices=("linear", "none"),
        help="linear: D·G lr을 --iters에 걸쳐 0까지 선형 감소 (원 구현). none: 고정",
    )
    g.add_argument(
        "--g_batch_size",
        type=positive_int,
        default=None,
        help="G step의 batch 크기. 없으면 --batch_size (추측: 원 설정은 128)",
    )
    g.add_argument("--z_dim", type=positive_int, default=128, help="latent z 차원")
    g.add_argument("--g_ch", type=positive_int, default=256, help="G ResBlock 채널 수")
    g.add_argument("--d_ch", type=positive_int, default=128, help="D ResBlock 채널 수")
    g.add_argument(
        "--n_samples",
        type=positive_int,
        default=64,
        help="fixed noise 샘플 수 (로그용 grid). --resume 시에는 checkpoint의 fixed noise를 쓰므로 무시된다",
    )
    return p.parse_args(argv)


def lr_at(step: int, args: argparse.Namespace) -> float:
    """step(1부터)의 학습률. linear면 lr·(1 - (step - 1)/iters): 첫 step은 lr, 마지막 step은 lr/iters."""
    if args.lr_decay == "none":
        return args.lr
    return args.lr * (1.0 - (step - 1) / args.iters)


def set_lr(optimizers: Iterable[torch.optim.Optimizer], lr: float) -> None:
    """optimizer의 모든 param_group lr을 바꾼다 (resume 시 checkpoint에 저장된 lr도 덮어쓴다)."""
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = lr


@torch.no_grad()
def log_samples(logger: Logger, G: nn.Module, fixed_noise: Tensor, step: int) -> None:
    """fixed noise로 만든 샘플 grid를 기록한다. G의 BatchNorm은 running stats(eval 모드)로 쓴다."""
    G.eval()
    logger.log_images("fake", G(fixed_noise), step)
    G.train()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed)
    device = get_device(args.device)
    run_dir = resolve_run_dir(args, "SNGAN")
    save_config(vars(args), run_dir)

    # 데이터: [-1, 1]로 정규화된 (N, channels, image_size, image_size) 이미지. batch는 (x, label)
    dataset = build_image_dataset(
        args.dataset,
        args.image_size,
        args.channels,
        root=args.data_root,
        download=args.download,
        path=args.data_path,
    )
    loader = build_loader(dataset, args.batch_size, num_workers=args.num_workers)
    data = infinite_batches(loader)

    # 모델: 초기화는 model.py 안에서 원 구현대로 끝난다 (D는 init → spectral_norm 순서)
    G = Generator(args.z_dim, args.channels, args.g_ch, args.image_size).to(device)
    D = Discriminator(args.channels, args.d_ch).to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    gan_loss = GANLoss("hinge")
    g_batch_size = args.g_batch_size or args.batch_size

    fixed_noise = torch.randn(args.n_samples, args.z_dim).to(device)  # 학습 진행을 같은 z로 비교
    start_step = 0
    if args.resume:
        ckpt = load_checkpoint(args.resume)
        G.load_state_dict(ckpt["G"])
        D.load_state_dict(ckpt["D"])  # spectral_norm의 power iteration 벡터(u, v)도 buffer로 함께 복원된다
        opt_G.load_state_dict(ckpt["opt_G"])
        opt_D.load_state_dict(ckpt["opt_D"])
        start_step = ckpt["step"]
        fixed_noise = ckpt["fixed_noise"].to(device)
        print(f"resume: {args.resume} (step {start_step})", flush=True)

    print(
        f"device={device} | run_dir={run_dir} | G {count_params(G):,} params, D {count_params(D):,} params",
        flush=True,
    )

    with Logger(run_dir, vars(args), use_wandb=args.wandb, project=args.wandb_project, run_name=args.run_name) as logger:
        logger.log_images("real", next(data)[0][: args.n_samples], start_step)  # 비교용 real 이미지

        G.train()
        D.train()
        for step in range(start_step + 1, args.iters + 1):
            lr = lr_at(step, args)
            set_lr((opt_G, opt_D), lr)

            # ---- Train D (n_dis회): hinge loss
            #   L_D = E[relu(1 - D(x))] + E[relu(1 + D(G(z)))]
            #   margin 1 밖으로 이미 잘 분류된 샘플은 gradient 0 → D가 쉬운 샘플에 과적합하지 않는다
            D.requires_grad_(True)
            for _ in range(args.n_dis):
                real, _ = next(data)  # label은 쓰지 않는다 (unsupervised)
                real = real.to(device, non_blocking=True)
                z = torch.randn(real.size(0), args.z_dim, device=device)
                with torch.no_grad():  # D step에서는 G로 gradient를 보내지 않는다
                    fake = G(z)
                real_logits = D(real)  # (N,)
                fake_logits = D(fake)
                loss_D = gan_loss.d_loss(real_logits, fake_logits)
                opt_D.zero_grad()
                loss_D.backward()
                opt_D.step()

            # ---- Train G: L_G = -E[D(G(z))]
            # D weight gradient는 필요 없다. D는 train 모드이므로 SN power iteration은 계속 갱신된다 (원 구현과 같음)
            D.requires_grad_(False)
            z = torch.randn(g_batch_size, args.z_dim, device=device)
            loss_G = gan_loss.g_loss(D(G(z)))
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            if step % args.log_every == 0:
                scalars = {
                    "loss/D": loss_D.item(),
                    "loss/G": loss_G.item(),
                    "logit/D_real": real_logits.detach().mean().item(),  # hinge에서는 +1 근처가 목표
                    "logit/D_fake": fake_logits.detach().mean().item(),  # -1 근처가 목표
                    "lr": lr,
                }
                logger.log_scalars(scalars, step)
                print(
                    f"step {step}/{args.iters} | " + " ".join(f"{k} {v:.4g}" for k, v in scalars.items()),
                    flush=True,
                )
            if step % args.sample_every == 0:
                log_samples(logger, G, fixed_noise, step)
            if step % args.save_every == 0 or step == args.iters:
                save_rolling_checkpoint(
                    run_dir / "checkpoints",
                    step,
                    args.keep_every,
                    G=G.state_dict(),
                    D=D.state_dict(),
                    opt_G=opt_G.state_dict(),
                    opt_D=opt_D.state_dict(),
                    step=step,
                    fixed_noise=fixed_noise.cpu(),
                    config=vars(args),
                )

        log_samples(logger, G, fixed_noise, max(start_step, args.iters))  # 학습 종료 시점 샘플


if __name__ == "__main__":
    main()
