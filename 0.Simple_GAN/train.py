"""Simple GAN (Goodfellow 2014): MLP Generator·Discriminator로 MNIST 28px 숫자를 생성한다.

실행 (repo 루트에서 `pip install -e .` 후):
    cd 0.Simple_GAN && python train.py --download
    python train.py --data_root dataset            # 예전 로컬 복사본 dataset/MNIST 를 그대로 쓸 때
산출물: runs/SimpleGAN/<run_name|timestamp>/ 아래 config.json, tb/, samples/, checkpoints/
checkpoint: 저장할 때마다 checkpoints/last.pt를 덮어쓰고, --keep_every N이면 epoch이 N의 배수일 때
ckpt_XXXXXXX.pt 사본을 남긴다.

이어서 학습 (--run_name 없이 --resume만 주면 원래 run 폴더에 이어 쓴다):
    python train.py --resume runs/SimpleGAN/<run>/checkpoints/last.pt --epochs 60
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
from gan_common.utils import count_params, get_device, resolve_run_dir, save_config, seed_everything
from model import Discriminator, Generator  # 같은 폴더의 model.py


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = base_parser(
        "Simple GAN (MLP) on MNIST",
        dataset="mnist",
        image_size=28,
        channels=1,
        batch_size=32,
        epochs=50,
        lr=3e-4,
        beta1=0.9,  # Adam 기본 betas (0.9, 0.999)
        beta2=0.999,
    )
    g = p.add_argument_group("model")
    g.add_argument("--z_dim", type=int, default=64, help="latent z 차원")
    g.add_argument(
        "--n_samples",
        type=positive_int,
        default=32,
        help="fixed noise 샘플 수 (로그용 grid). --resume 시에는 checkpoint의 fixed noise를 쓰므로 무시된다",
    )
    return p.parse_args(argv)


@torch.no_grad()
def log_samples(logger: Logger, G: nn.Module, fixed_noise: Tensor, step: int) -> None:
    """fixed noise로 만든 샘플 grid를 기록한다. eval 모드로 샘플링한다 (BatchNorm·Dropout이 있을 때 대비)."""
    G.eval()
    logger.log_images("fake", G(fixed_noise), step)
    G.train()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed)
    device = get_device(args.device)
    run_dir = resolve_run_dir(args, "SimpleGAN")
    save_config(vars(args), run_dir)

    # 데이터: [-1, 1]로 정규화된 (N, 1, 28, 28) 이미지
    dataset = build_image_dataset(
        args.dataset,
        args.image_size,
        args.channels,
        root=args.data_root,
        download=args.download,
        path=args.data_path,
    )
    loader = build_loader(dataset, args.batch_size, num_workers=args.num_workers)

    # 모델: 초기화는 PyTorch 기본값 (원 구현과 동일)
    img_shape = (args.channels, args.image_size, args.image_size)
    G = Generator(args.z_dim, img_shape).to(device)
    D = Discriminator(img_shape).to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    gan_loss = GANLoss("vanilla")

    fixed_noise = torch.randn(args.n_samples, args.z_dim).to(device)  # 학습 진행을 같은 z로 비교
    start_epoch, global_step = 0, 0
    if args.resume:
        ckpt = load_checkpoint(args.resume)
        G.load_state_dict(ckpt["G"])
        D.load_state_dict(ckpt["D"])
        opt_G.load_state_dict(ckpt["opt_G"])
        opt_D.load_state_dict(ckpt["opt_D"])
        start_epoch, global_step = ckpt["epoch"], ckpt["step"]
        fixed_noise = ckpt["fixed_noise"].to(device)
        print(f"resume: {args.resume} (epoch {start_epoch}, step {global_step})", flush=True)

    print(
        f"device={device} | run_dir={run_dir} | G {count_params(G):,} params, D {count_params(D):,} params",
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
        D.train()
        for epoch in range(start_epoch, args.epochs):
            for real, _ in loader:
                real = real.to(device, non_blocking=True)
                n = real.size(0)

                # ---- Train D: max log D(x) + log(1 - D(G(z)))
                D.requires_grad_(True)
                noise = torch.randn(n, args.z_dim, device=device)
                fake = G(noise)  # G step에서 다시 쓰므로 G graph를 남겨 둔다
                real_logits = D(real)
                fake_logits = D(fake.detach())  # detach: D step의 gradient가 G로 흘러가지 않게 끊는다
                loss_D = 0.5 * gan_loss.d_loss(real_logits, fake_logits)  # 원 구현처럼 (real + fake) / 2
                opt_D.zero_grad()
                loss_D.backward()
                opt_D.step()

                # ---- Train G: min log(1 - D(G(z))) 대신 max log D(G(z)) (non-saturating loss)
                # 학습 초반 D가 fake를 쉽게 구분하면 log(1 - D(G(z)))는 gradient가 거의 0이 되지만,
                # non-saturating 형태는 그때도 gradient가 충분히 남는다
                D.requires_grad_(False)  # D는 고정. G step에서 D weight gradient는 필요 없다
                loss_G = gan_loss.g_loss(D(fake))  # D step에서 만든 fake를 갱신된 D에 다시 넣는다
                opt_G.zero_grad()
                loss_G.backward()
                opt_G.step()

                global_step += 1
                if global_step % args.log_every == 0:
                    scalars = {
                        "loss/D": loss_D.item(),
                        "loss/G": loss_G.item(),
                        "prob/D_real": torch.sigmoid(real_logits.detach()).mean().item(),  # D(x), 1에 가까울수록 real
                        "prob/D_fake": torch.sigmoid(fake_logits.detach()).mean().item(),  # D(G(z))
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
                    D=D.state_dict(),
                    opt_G=opt_G.state_dict(),
                    opt_D=opt_D.state_dict(),
                    epoch=epoch + 1,
                    step=global_step,
                    fixed_noise=fixed_noise.cpu(),
                    config=vars(args),
                )

        log_samples(logger, G, fixed_noise, global_step)  # 학습 종료 시점 샘플


if __name__ == "__main__":
    main()
