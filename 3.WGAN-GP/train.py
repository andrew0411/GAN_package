"""WGAN-GP (Gulrajani 2017): weight clipping 대신 gradient penalty로 critic의 Lipschitz 제약을 건다.

네트워크는 gan_common.networks.dcgan. critic은 BatchNorm 대신 InstanceNorm(affine)을 쓴다
(GP는 샘플별 gradient를 보므로 batch 안 샘플을 섞는 BatchNorm은 맞지 않는다).
실행 (repo 루트에서 `pip install -e .` 후):
    cd 3.WGAN-GP && python train.py --download
    python train.py --dataset celeba --channels 3   # <DATA_ROOT>/celeba/img_align_celeba/*.jpg
산출물: runs/WGAN-GP/<run_name|timestamp>/ 아래 config.json, tb/, samples/, checkpoints/
checkpoint: 저장할 때마다 checkpoints/last.pt를 덮어쓰고, --keep_every N이면 epoch이 N의 배수일 때
ckpt_XXXXXXX.pt 사본을 남긴다.

이어서 학습 (--run_name 없이 --resume만 주면 원래 run 폴더에 이어 쓴다):
    python train.py --resume runs/WGAN-GP/<run>/checkpoints/last.pt
    - --epochs는 총 epoch 수다 (추가할 epoch 수가 아니다)
    - fixed noise는 checkpoint에 저장된 것을 쓰므로 --n_samples는 무시된다
    - RNG 상태는 복원하지 않으므로 끊지 않고 학습한 결과와 bit 단위로 같지는 않다
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from torch import Tensor

from gan_common.checkpoint import load_checkpoint, save_rolling_checkpoint
from gan_common.config import base_parser, positive_int
from gan_common.data import build_image_dataset, build_loader
from gan_common.logger import Logger
from gan_common.losses import GANLoss
from gan_common.networks.dcgan import DCGANDiscriminator, DCGANGenerator
from gan_common.regularizers import gradient_penalty
from gan_common.utils import count_params, get_device, resolve_run_dir, save_config, seed_everything
from gan_common.weights import init_weights


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = base_parser(
        "WGAN-GP on MNIST 64px",
        dataset="mnist",
        image_size=64,
        channels=1,
        batch_size=64,
        epochs=100,
        lr=1e-4,
        beta1=0.0,  # WGAN-GP 논문 Adam(α=1e-4, β1=0, β2=0.9)
        beta2=0.9,
    )
    g = p.add_argument_group("model")
    g.add_argument("--z_dim", type=int, default=100, help="latent z 차원")
    g.add_argument("--features", type=int, default=16, help="G·critic 기본 채널 수")
    g.add_argument("--n_critic", type=positive_int, default=5, help="G 1회 갱신당 critic 갱신 횟수")
    g.add_argument("--lambda_gp", type=float, default=10.0, help="gradient penalty 계수 λ")
    g.add_argument(
        "--n_samples",
        type=positive_int,
        default=32,
        help="fixed noise 샘플 수 (로그용 grid). --resume 시에는 checkpoint의 fixed noise를 쓰므로 무시된다",
    )
    return p.parse_args(argv)


@torch.no_grad()
def log_samples(logger: Logger, G: nn.Module, fixed_noise: Tensor, step: int) -> None:
    """fixed noise로 만든 샘플 grid를 기록한다. BatchNorm은 running stats(eval 모드)로 써서
    batch 구성에 따라 결과가 바뀌지 않게 한다."""
    G.eval()
    logger.log_images("fake", G(fixed_noise), step)
    G.train()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed)
    device = get_device(args.device)
    run_dir = resolve_run_dir(args, "WGAN-GP")
    save_config(vars(args), run_dir)

    # 데이터: [-1, 1]로 정규화된 (N, C, 64, 64) 이미지
    dataset = build_image_dataset(
        args.dataset,
        args.image_size,
        args.channels,
        root=args.data_root,
        download=args.download,
        path=args.data_path,
    )
    loader = build_loader(dataset, args.batch_size, num_workers=args.num_workers)

    # 모델: critic은 확률이 아니라 실수 score를 내므로 discriminator 대신 critic이라 부른다
    G = DCGANGenerator(args.z_dim, args.channels, args.features, args.image_size)
    critic = DCGANDiscriminator(args.channels, args.features, args.image_size, norm="instance")
    init_weights(G, "normal", 0.02)
    init_weights(critic, "normal", 0.02)
    G, critic = G.to(device), critic.to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    opt_critic = torch.optim.Adam(critic.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    gan_loss = GANLoss("wgan")

    fixed_noise = torch.randn(args.n_samples, args.z_dim).to(device)  # 학습 진행을 같은 z로 비교
    start_epoch, global_step = 0, 0
    if args.resume:
        ckpt = load_checkpoint(args.resume)
        G.load_state_dict(ckpt["G"])
        critic.load_state_dict(ckpt["critic"])
        opt_G.load_state_dict(ckpt["opt_G"])
        opt_critic.load_state_dict(ckpt["opt_critic"])
        start_epoch, global_step = ckpt["epoch"], ckpt["step"]
        fixed_noise = ckpt["fixed_noise"].to(device)
        print(f"resume: {args.resume} (epoch {start_epoch}, step {global_step})", flush=True)

    print(
        f"device={device} | run_dir={run_dir} | "
        f"G {count_params(G):,} params, critic {count_params(critic):,} params",
        flush=True,
    )
    try:
        real_batch, _ = next(iter(loader))
    except StopIteration:
        raise RuntimeError(
            f"데이터셋 크기({len(dataset)})가 batch_size({args.batch_size})보다 작아 batch가 없습니다 "
            "(drop_last=True). --batch_size를 줄이십시오."
        ) from None

    with Logger(run_dir, vars(args), use_wandb=args.wandb, project=args.wandb_project, run_name=args.run_name) as logger:
        logger.log_images("real", real_batch[: args.n_samples], global_step)  # 비교용 real 이미지

        G.train()
        critic.train()
        for epoch in range(start_epoch, args.epochs):
            for real, _ in loader:  # label은 쓰지 않는다 (unsupervised)
                real = real.to(device, non_blocking=True)
                n = real.size(0)

                # ---- Train critic: min E[D(G(z))] - E[D(x)] + λ·E[(||∇_x̂ D(x̂)||_2 - 1)^2]
                # x̂는 real과 fake 사이 직선 위의 점. 최적 critic은 그 위에서 gradient norm이 1이다
                # 원 구현을 따라 같은 real batch로 n_critic번 갱신한다 (논문은 매번 새 real batch를 뽑는다)
                critic.requires_grad_(True)
                for _ in range(args.n_critic):
                    noise = torch.randn(n, args.z_dim, device=device)
                    with torch.no_grad():  # critic step에서는 G로 gradient를 보내지 않는다
                        fake = G(noise)
                    loss_w = gan_loss.d_loss(critic(real), critic(fake))  # Wasserstein 항
                    gp = gradient_penalty(critic, real, fake)
                    loss_critic = loss_w + args.lambda_gp * gp
                    opt_critic.zero_grad()
                    loss_critic.backward()
                    opt_critic.step()

                # ---- Train G: max E[D(G(z))]  ⇔  min -E[D(G(z))]. 새 noise로 fake를 만든다
                critic.requires_grad_(False)  # critic은 고정. G step에서 critic weight gradient는 필요 없다
                noise = torch.randn(n, args.z_dim, device=device)
                loss_G = gan_loss.g_loss(critic(G(noise)))
                opt_G.zero_grad()
                loss_G.backward()
                opt_G.step()

                global_step += 1
                if global_step % args.log_every == 0:
                    scalars = {
                        "loss/critic": loss_critic.item(),
                        "loss/G": loss_G.item(),
                        "gp": gp.item(),
                        # E[D(x)] - E[D(G(z))]: Wasserstein 거리 추정치. 줄어들수록 G 분포가 real에 가깝다
                        "wasserstein": -loss_w.item(),
                    }
                    logger.log_scalars(scalars, global_step)
                    print(
                        f"epoch {epoch + 1}/{args.epochs} step {global_step} | "
                        + " ".join(f"{k} {v:.4f}" for k, v in scalars.items()),
                        flush=True,
                    )
                if global_step % args.sample_every == 0:
                    log_samples(logger, G, fixed_noise, global_step)

            if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
                save_rolling_checkpoint(
                    run_dir / "checkpoints",
                    epoch + 1,
                    args.keep_every,
                    G=G.state_dict(),
                    critic=critic.state_dict(),
                    opt_G=opt_G.state_dict(),
                    opt_critic=opt_critic.state_dict(),
                    epoch=epoch + 1,
                    step=global_step,
                    fixed_noise=fixed_noise.cpu(),
                    config=vars(args),
                )

        log_samples(logger, G, fixed_noise, global_step)  # 학습 종료 시점 샘플


if __name__ == "__main__":
    main()
