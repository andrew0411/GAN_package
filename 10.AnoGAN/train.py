"""AnoGAN 1단계 — 정상 데이터만으로 DCGAN을 학습한다 (Schlegl et al., IPMI 2017).

방법 요약:
    AnoGAN은 정상 데이터의 manifold를 GAN으로 배우고, 새 이미지가 그 manifold에서 얼마나 먼지를 anomaly score로 쓴다.
    1) train.py : 정상 클래스 이미지만으로 DCGAN(G, D)을 학습한다 → G(z)는 정상 이미지만 만들 줄 안다
    2) detect.py: test 이미지 x마다 G·D를 고정하고 z만 최적화해 G(z)를 x에 최대한 가깝게 만든다
        L(z) = (1 − λ)·Σ|x − G(z)| + λ·Σ|f(x) − f(G(z))|,   f = D.features (D 중간 feature), λ = 0.1
        anomaly score A(x) = L(z*)   (z* = n_iters번 최적화한 z)
       정상 x는 G가 잘 재현해 A(x)가 작고, G가 본 적 없는 이상 x는 A(x)가 커진다.

프로토콜 (10.AnoGAN · 11.GANomaly 공통):
    MNIST에서 --anomaly_class k(기본 0)를 이상, 나머지 9개 숫자를 정상으로 둔다.
    학습: train split의 정상 클래스만 (`build_image_dataset(classes=[...])`로 Subset)
    평가(detect.py): test split 전체, label = (y == k) (1 = anomaly). AUROC·AP 보고
    train/test split이 있는 torchvision 데이터셋(mnist, fashion_mnist, cifar10)을 전제로 한다.

네트워크: gan_common.networks.dcgan (1.DCGAN과 같음). vanilla(non-saturating) GAN loss, Adam(2e-4, 0.5, 0.999).

논문 DCGAN 설정과 다른 점 (여기서는 repo 공통 DCGAN을 그대로 쓴다):
    - z: 논문은 균등분포 z, 여기는 N(0, I). 그래서 detect.py도 z를 N(0, I)에서 시작한다
    - filter: 논문 5×5, 여기 4×4 (Conv/ConvT(4, 2, 1))
    - 채널: 논문 G 512-256-128-64. 여기 64px에서 `--features 32`의 G와 같고, 기본값 `--features 64`는 2배 넓다
    - 학습 길이: 논문 20 epoch → 기본값 `--epochs 20`
    - detect.py의 z clamp(`--z_clamp`)는 논문에 없는 옵션이다.
      추측: 논문이 인용한 DCGAN image completion 코드에서 온 관행으로 보인다. 그 코드를 확인해야 확정된다

실행 예시 (repo 루트에서 `pip install -e .` 후):
    cd 10.AnoGAN && python train.py --download
    python train.py --anomaly_class 3 --run_name digit3
    python train.py --resume runs/AnoGAN/digit3/checkpoints/last.pt --anomaly_class 3 --epochs 40
    python detect.py --checkpoint runs/AnoGAN/<run>/checkpoints/last.pt
산출물: runs/AnoGAN/<run_name|timestamp>/ 아래 config.json, tb/, samples/, checkpoints/(last.pt, ckpt_*.pt)

이어서 학습 (--run_name 없이 --resume만 주면 원래 run 폴더에 이어 쓴다):
    - 원래 run의 RESUME_KEYS(--dataset, --anomaly_class, --image_size, --channels, --z_dim, --features)를
      같은 값으로 다시 줘야 한다. 기본값과 다르게 학습했다면(예: --anomaly_class 3) 그 인자를 반드시 반복한다
    - 하나라도 다르면 load 전에 ValueError로 멈춘다. 특히 anomaly_class가 바뀌면 이상 클래스가 학습에
      섞여 되돌릴 수 없으므로 경고가 아니라 에러다
"""

from __future__ import annotations

import argparse

import torch

from gan_common.checkpoint import load_checkpoint, save_rolling_checkpoint
from gan_common.config import base_parser, positive_int
from gan_common.data import build_image_dataset, build_loader
from gan_common.logger import Logger
from gan_common.losses import GANLoss
from gan_common.networks.dcgan import DCGANDiscriminator, DCGANGenerator
from gan_common.utils import count_params, get_device, resolve_run_dir, save_config, seed_everything
from gan_common.weights import init_weights

NUM_CLASSES = 10  # MNIST · Fashion-MNIST · CIFAR-10
SPLIT_DATASETS = ("mnist", "fashion_mnist", "cifar10")  # train/test split이 따로 있는 데이터셋
# resume 시 checkpoint와 같아야 하는 인자: 데이터 분할(정상/이상)과 모델 구조(파라미터 shape)를 정한다
RESUME_KEYS = ("dataset", "anomaly_class", "image_size", "channels", "z_dim", "features")


def normal_classes(anomaly_class: int) -> list[int]:
    """이상 클래스 k를 뺀 나머지 = 정상 클래스 (학습에 쓰는 클래스)."""
    return [c for c in range(NUM_CLASSES) if c != anomaly_class]


def check_resume_config(ckpt_config: dict, args: argparse.Namespace) -> None:
    """checkpoint의 RESUME_KEYS가 이번 실행과 다르면 load_state_dict 전에 에러로 멈춘다.

    anomaly_class·dataset이 바뀌면 이상 클래스가 학습 데이터에 섞여(정상/이상 분할 오염) 되돌릴 수 없으므로
    경고가 아니라 ValueError다.
    """
    diffs = {k: (ckpt_config.get(k), getattr(args, k)) for k in RESUME_KEYS if ckpt_config.get(k) != getattr(args, k)}
    if diffs:
        detail = ", ".join(f"--{k}: checkpoint {old} ≠ 지금 {new}" for k, (old, new) in diffs.items())
        raise ValueError(
            f"--resume checkpoint와 데이터 분할·모델 구조 인자가 다릅니다 ({detail}). 원래 값으로 다시 실행하십시오."
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = base_parser(
        "AnoGAN step 1: DCGAN on normal classes only",
        dataset="mnist",
        image_size=64,
        channels=1,
        batch_size=128,
        epochs=20,  # 논문 20 epoch
        lr=2e-4,
        beta1=0.5,  # DCGAN 관례: beta1 0.9는 학습이 진동해 0.5로 낮춘다
        beta2=0.999,
    )
    g = p.add_argument_group("anomaly detection")
    g.add_argument(
        "--anomaly_class",
        type=int,
        default=0,
        choices=range(NUM_CLASSES),
        help="이상으로 둘 클래스 k. 학습에서 제외되고 detect.py에서 label 1이 된다",
    )
    g = p.add_argument_group("model")
    g.add_argument("--z_dim", type=positive_int, default=100, help="latent z 차원")
    g.add_argument("--features", type=positive_int, default=64, help="G·D 기본 채널 수")
    g.add_argument("--n_samples", type=positive_int, default=64, help="fixed noise 샘플 수 (로그용 grid)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed)
    device = get_device(args.device)
    run_dir = resolve_run_dir(args, "AnoGAN")
    ckpt = None
    if args.resume:
        ckpt = load_checkpoint(args.resume)
        check_resume_config(ckpt["config"], args)  # save_config 전에 검사해 원래 config.json을 덮어쓰지 않는다
    save_config(vars(args), run_dir)
    if args.dataset not in SPLIT_DATASETS:
        print(
            f"경고: dataset={args.dataset!r}은 train/test split이 없어 detect.py가 학습 데이터로 평가하게 됩니다.",
            flush=True,
        )

    # 데이터: train split에서 정상 클래스만 남긴 Subset. [-1, 1] (N, C, S, S)
    normal = normal_classes(args.anomaly_class)
    dataset = build_image_dataset(
        args.dataset,
        args.image_size,
        args.channels,
        train=True,
        root=args.data_root,
        download=args.download,
        path=args.data_path,
        classes=normal,
    )
    loader = build_loader(dataset, args.batch_size, num_workers=args.num_workers)

    # 모델: DCGAN 관례대로 Conv weight N(0, 0.02), BatchNorm weight N(1, 0.02)
    G = DCGANGenerator(args.z_dim, args.channels, args.features, args.image_size)
    D = DCGANDiscriminator(args.channels, args.features, args.image_size, norm="batch")
    init_weights(G, "normal", 0.02)
    init_weights(D, "normal", 0.02)
    G, D = G.to(device), D.to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    gan_loss = GANLoss("vanilla")

    fixed_noise = torch.randn(args.n_samples, args.z_dim, device=device)  # 학습 진행을 같은 z로 비교
    start_epoch, global_step = 0, 0
    if ckpt is not None:
        G.load_state_dict(ckpt["G"])
        D.load_state_dict(ckpt["D"])
        opt_G.load_state_dict(ckpt["opt_G"])
        opt_D.load_state_dict(ckpt["opt_D"])
        start_epoch, global_step = ckpt["epoch"], ckpt["step"]
        fixed_noise = ckpt["fixed_noise"].to(device)
        print(f"resume: {args.resume} (epoch {start_epoch}, step {global_step})", flush=True)

    print(
        f"device={device} | run_dir={run_dir} | G {count_params(G):,} params, D {count_params(D):,} params\n"
        f"정상 클래스 {normal} ({len(dataset):,}장으로 학습), 이상 클래스 {args.anomaly_class}",
        flush=True,
    )

    with Logger(
        run_dir, vars(args), use_wandb=args.wandb, project=args.wandb_project, run_name=args.run_name
    ) as logger:
        real_batch, _ = next(iter(loader))
        logger.log_images("real", real_batch[: args.n_samples], global_step)  # 학습에 쓰는 정상 이미지

        G.train()
        D.train()
        for epoch in range(start_epoch, args.epochs):
            for real, _ in loader:
                real = real.to(device, non_blocking=True)
                n = real.size(0)

                # ---- D step: max log D(x) + log(1 − D(G(z)))
                # fake.detach(): D step의 gradient가 G로 흘러가지 않게 끊는다
                D.requires_grad_(True)
                noise = torch.randn(n, args.z_dim, device=device)
                fake = G(noise)  # (N, C, S, S)
                real_logits = D(real)  # (N,)
                fake_logits = D(fake.detach())
                loss_D = 0.5 * gan_loss.d_loss(real_logits, fake_logits)  # (real + fake) / 2
                opt_D.zero_grad(set_to_none=True)
                loss_D.backward()
                opt_D.step()

                # ---- G step: max log D(G(z)) (non-saturating). D는 고정해 D 파라미터 grad를 만들지 않는다
                D.requires_grad_(False)
                loss_G = gan_loss.g_loss(D(fake))
                opt_G.zero_grad(set_to_none=True)
                loss_G.backward()
                opt_G.step()

                global_step += 1
                if global_step % args.log_every == 0:
                    scalars = {
                        "loss/D": loss_D.item(),
                        "loss/G": loss_G.item(),
                        "prob/D_real": torch.sigmoid(real_logits.detach()).mean().item(),  # D(x)
                        "prob/D_fake": torch.sigmoid(fake_logits.detach()).mean().item(),  # D(G(z))
                    }
                    logger.log_scalars(scalars, global_step)
                    print(
                        f"epoch {epoch + 1}/{args.epochs} step {global_step} | "
                        + " ".join(f"{k} {v:.4f}" for k, v in scalars.items()),
                        flush=True,
                    )
                if global_step % args.sample_every == 0:
                    G.eval()  # BatchNorm running stats로 샘플링 (batch 구성에 따라 결과가 바뀌지 않게)
                    with torch.no_grad():
                        logger.log_images("fake", G(fixed_noise), global_step)
                    G.train()

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

        # 학습 종료 후 fixed noise 샘플을 한 번 더 기록 (sample_every와 어긋나도 최종 상태가 남게)
        G.eval()
        with torch.no_grad():
            logger.log_images("fake", G(fixed_noise), global_step)

    print(f"완료: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
