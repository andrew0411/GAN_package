"""CycleGAN (Zhu et al., ICCV 2017) — 짝이 없는(unpaired) 두 image domain A, B 사이의 번역을 학습한다.

A → B 번역 G_AB와 B → A 번역 G_BA를 함께 학습한다. 각 domain의 D(D_A, D_B)를 속이는 adversarial loss에
"A → B → A로 돌아오면 원래 이미지여야 한다"는 cycle consistency loss를 더해, 짝 데이터 없이도
입력의 내용(구조)은 유지한 채 스타일만 바꾸게 한다. loss 구성과 네트워크 이름은 model.py를 참조
(junyanz/pytorch-CycleGAN-and-pix2pix의 cycle_gan 모델 기준).

데이터 폴더 구조:
    $DATA_ROOT/cyclegan/horse2zebra/
        trainA/*.jpg    trainB/*.jpg
        testA/*.jpg     testB/*.jpg
    데이터 위치는 DATA_ROOT 환경변수(또는 --data_root), 데이터셋 이름은 --dataset으로 정한다.
    데이터는 junyanz repo의 `datasets/download_cyclegan_dataset.sh horse2zebra`로 받을 수 있다.

실행 예시 (repo 루트에서 `pip install -e .` 후, 전체 옵션은 `python train.py --help`):
    cd 5.CycleGAN
    python train.py --run_name h2z                                  # horse2zebra, 100 + 100 epoch
    python train.py --dataset monet2photo --run_name monet --keep_every 50
    python train.py --resume runs/CycleGAN/h2z/checkpoints/last.pt  # 같은 run 폴더에 이어서 기록
    python test.py --checkpoint runs/CycleGAN/h2z/checkpoints/last.pt

산출물: runs/CycleGAN/<run_name|timestamp>/ 아래 config.json, tb/, samples/, checkpoints/
    tb/: loss scalar (TensorBoard). --wandb를 주면 W&B에도 기록한다
    checkpoints/last.pt: --save_every epoch마다 덮어쓰는 최신 학습 상태 (--keep_every면 ckpt_{epoch:07d}.pt 사본도)
    samples/fixed_*.png: epoch마다 고정 test 샘플 번역 결과. fixed_final_*.png는 학습 종료 후
        한 행 = [a | G_AB(a) | G_BA(G_AB(a)) | b | G_BA(b) | G_AB(G_BA(b))]
    samples/train_*.png: --sample_every step마다 현재 학습 batch (행 배열은 위와 같음)
"""

from __future__ import annotations

import argparse
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from gan_common.checkpoint import load_checkpoint, save_rolling_checkpoint
from gan_common.config import base_parser, nonneg_int, positive_int, str2bool
from gan_common.data import UnpairedImageDataset, build_loader, get_data_root
from gan_common.logger import Logger
from gan_common.utils import count_params, get_device, resolve_run_dir, save_config, seed_everything
from model import VISUAL_KEYS, CycleGAN, check_crop_size

MODEL_NAME = "CycleGAN"
# resume 시 checkpoint config와 달라지면 경고할 구조·데이터 key
RESUME_CHECK_KEYS = ("dataset", "norm", "ngf", "ndf", "n_blocks", "n_layers_D", "use_dropout", "crop_size")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = base_parser(
        "CycleGAN: unpaired image-to-image translation (junyanz cycle_gan 기준)",
        dataset="horse2zebra",
        batch_size=1,
        lr=2e-4,
        beta1=0.5,
        beta2=0.999,
        # 쓰지 않는 공통 인자. epochs·image_size·channels는 아래에서 n_epochs + n_epochs_decay, crop_size, 3으로 채운다
        epochs=None,
        image_size=None,
        channels=None,
        data_path=None,
        download=None,
    )
    g = p.add_argument_group("cyclegan")
    g.add_argument("--load_size", type=positive_int, default=286, help="train: 이 크기로 resize한 뒤 crop_size로 random crop")
    g.add_argument("--crop_size", type=positive_int, default=256, help="학습 해상도. 4의 배수여야 한다")
    g.add_argument("--flip", type=str2bool, default=True, help="train random horizontal flip")
    g.add_argument("--serial", type=str2bool, default=False, help="true면 B를 무작위 대신 index 순서로 짝짓고 shuffle 안 함")
    g.add_argument("--ngf", type=positive_int, default=64, help="G 첫 conv 채널 수")
    g.add_argument("--ndf", type=positive_int, default=64, help="D 첫 conv 채널 수")
    g.add_argument("--n_blocks", type=nonneg_int, default=9, help="ResNet G의 residual block 수 (256px: 9, 128px: 6)")
    g.add_argument("--n_layers_D", type=positive_int, default=3, help="PatchGAN stride-2 conv 수 (3이면 70x70 patch)")
    g.add_argument("--norm", type=str, default="instance", choices=("batch", "instance", "none"))
    g.add_argument("--use_dropout", type=str2bool, default=False, help="ResNet block 안 Dropout(0.5)")
    g.add_argument("--init_type", type=str, default="normal", choices=("normal", "xavier", "kaiming", "orthogonal"))
    g.add_argument("--init_gain", type=float, default=0.02)
    g.add_argument("--gan_mode", type=str, default="lsgan", choices=("vanilla", "lsgan"))
    g.add_argument("--lambda_A", type=float, default=10.0, help="cycle loss 가중치 (A → B → A)")
    g.add_argument("--lambda_B", type=float, default=10.0, help="cycle loss 가중치 (B → A → B)")
    g.add_argument(
        "--lambda_identity", type=float, default=0.5, help="identity loss 가중치 (해당 domain의 λ_A 또는 λ_B에 곱함). 0이면 끔"
    )
    g.add_argument("--pool_size", type=nonneg_int, default=50, help="ImagePool 크기 (0이면 pool 없이 최신 fake만)")
    g.add_argument("--n_epochs", type=nonneg_int, default=100, help="초기 lr을 유지하는 epoch 수")
    g.add_argument("--n_epochs_decay", type=nonneg_int, default=100, help="lr을 0을 향해 선형으로 줄이는 epoch 수")
    g.add_argument("--num_samples", type=nonneg_int, default=4, help="epoch마다 번역해 보는 고정 샘플 수 (0이면 생략)")
    g.add_argument("--sample_phase", type=str, default="test", help="고정 샘플을 뽑을 split ({phase}A, {phase}B 폴더)")
    args = p.parse_args(argv)

    total_epochs = args.n_epochs + args.n_epochs_decay
    if total_epochs < 1:
        p.error("n_epochs + n_epochs_decay는 1 이상이어야 합니다.")
    for name, value in (("epochs", total_epochs), ("image_size", args.crop_size), ("channels", 3)):
        given = getattr(args, name)
        if given is not None and given != value:
            p.error(
                f"--{name}은 이 스크립트에서 쓰지 않습니다 (자동으로 {value}). "
                "epoch 수는 --n_epochs/--n_epochs_decay, 해상도는 --crop_size로 정합니다."
            )
        setattr(args, name, value)
    try:
        check_crop_size(args.crop_size)
    except ValueError as e:
        p.error(str(e))
    return args


def warn_config_mismatch(saved: Mapping[str, Any], args: argparse.Namespace, keys: tuple[str, ...]) -> None:
    """resume checkpoint의 학습 config와 지금 CLI 값이 다른 구조·데이터 key를 경고한다 (CLI 값을 쓴다)."""
    diffs = [
        f"{k}: checkpoint={saved.get(k)!r}, 지금={getattr(args, k)!r}" for k in keys if saved.get(k) != getattr(args, k)
    ]
    if diffs:
        warnings.warn("resume checkpoint와 설정이 다릅니다 → " + "; ".join(diffs), stacklevel=2)


def interleave(*columns: Tensor) -> Tensor:
    """(N, C, H, W) 텐서 k개 → (N·k, C, H, W). grid를 nrow=k로 그리면 한 행이 한 샘플이 된다."""
    return torch.stack(columns, dim=1).flatten(0, 1)


def load_fixed_samples(root: Path, args: argparse.Namespace) -> tuple[Tensor, Tensor] | None:
    """`sample_phase` split의 A·B 앞쪽 `num_samples`장.

    serial=True, `augment=False`라 split이 train이어도 random crop·flip 없이 resize만 하므로 매번 같은 이미지다.
    """
    if args.num_samples <= 0:
        return None
    ds = UnpairedImageDataset(
        root,
        phase=args.sample_phase,
        load_size=args.load_size,
        crop_size=args.crop_size,
        serial=True,
        augment=False,
    )
    items = [ds[i] for i in range(min(args.num_samples, len(ds)))]
    return torch.stack([it["A"] for it in items]), torch.stack([it["B"] for it in items])


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed)
    device = get_device(args.device)
    run_dir = resolve_run_dir(args, MODEL_NAME)  # --resume만 주면 원래 run 폴더를 이어 쓴다
    config = vars(args)
    save_config(config, run_dir)

    # ---- data ----
    root = get_data_root(args.data_root) / "cyclegan" / args.dataset
    train_ds = UnpairedImageDataset(
        root, phase="train", load_size=args.load_size, crop_size=args.crop_size, flip=args.flip, serial=args.serial
    )
    loader = build_loader(train_ds, args.batch_size, shuffle=not args.serial, num_workers=args.num_workers)
    fixed = load_fixed_samples(root, args)
    if fixed is not None:
        fixed = (fixed[0].to(device), fixed[1].to(device))

    # ---- model ----
    model = CycleGAN(
        device,
        ngf=args.ngf,
        ndf=args.ndf,
        n_blocks=args.n_blocks,
        n_layers_D=args.n_layers_D,
        norm=args.norm,
        use_dropout=args.use_dropout,
        init_type=args.init_type,
        init_gain=args.init_gain,
        gan_mode=args.gan_mode,
        lambda_A=args.lambda_A,
        lambda_B=args.lambda_B,
        lambda_identity=args.lambda_identity,
        pool_size=args.pool_size,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        n_epochs=args.n_epochs,
        n_epochs_decay=args.n_epochs_decay,
    )

    start_epoch, step = 1, 0
    if args.resume:
        ckpt = load_checkpoint(args.resume)
        warn_config_mismatch(ckpt["config"], args, RESUME_CHECK_KEYS)
        model.load_state_dict(ckpt["model"])
        start_epoch, step = ckpt["epoch"] + 1, ckpt["step"]
        print(
            f"resume: {args.resume} (epoch {ckpt['epoch']}까지 완료, step {step}). ImagePool은 빈 상태로 다시 시작",
            flush=True,
        )

    n_params = " / ".join(f"{name} {count_params(net):,}" for name, net in model.nets.items())
    print(
        f"device={device} | trainA {len(train_ds.paths_a)}장, trainB {len(train_ds.paths_b)}장 | {n_params} params "
        f"| run_dir={run_dir}",
        flush=True,
    )
    with Logger(run_dir, config, use_wandb=args.wandb, project=args.wandb_project, run_name=args.run_name) as logger:
        for epoch in range(start_epoch, args.epochs + 1):
            t0 = time.time()
            for batch in loader:
                losses = model.train_step(batch)
                step += 1
                if step % args.log_every == 0:
                    logger.log_scalars({**losses, "lr": model.lr}, step)
                    print(
                        f"[epoch {epoch}/{args.epochs}] step {step} | D_A {losses['loss/D_A']:.4f} "
                        f"D_B {losses['loss/D_B']:.4f} | G {losses['loss/G']:.4f} "
                        f"(cycle_A {losses['loss/cycle_A']:.4f}, cycle_B {losses['loss/cycle_B']:.4f}) "
                        f"| lr {model.lr:.2e}",
                        flush=True,
                    )
                if step % args.sample_every == 0:
                    v = model.visuals
                    logger.log_images("train", interleave(*(v[k] for k in VISUAL_KEYS)), step, nrow=len(VISUAL_KEYS))

            # epoch이 끝날 때 lr schedule을 한 칸 진행한다 (다음 epoch부터 적용, model.linear_decay_rule 참조)
            next_lr = model.update_learning_rate()
            if fixed is not None:
                v = model.translate(*fixed)
                logger.log_images("fixed", interleave(*(v[k] for k in VISUAL_KEYS)), step, nrow=len(VISUAL_KEYS))
            if epoch % args.save_every == 0 or epoch == args.epochs:
                save_rolling_checkpoint(
                    run_dir / "checkpoints",
                    epoch,
                    args.keep_every,
                    epoch=epoch,
                    step=step,
                    model=model.state_dict(),
                    config=config,
                )
            print(f"epoch {epoch}/{args.epochs} 완료 ({time.time() - t0:.0f}s) | 다음 epoch lr {next_lr:.2e}", flush=True)

        # 학습 루프가 끝난 뒤 최종 G로 고정 샘플을 한 번 더 기록한다
        if fixed is not None:
            v = model.translate(*fixed)
            logger.log_images("fixed_final", interleave(*(v[k] for k in VISUAL_KEYS)), step, nrow=len(VISUAL_KEYS))
    print(f"학습 종료: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
