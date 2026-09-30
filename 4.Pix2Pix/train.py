"""Pix2Pix (Isola et al., CVPR 2017) — 조건부 GAN(cGAN)으로 짝이 있는(paired) 이미지 번역 A → B를 학습한다.

G(U-Net)는 입력 A를 받아 B를 만들고, D(70x70 PatchGAN)는 [A, B] 쌍을 채널로 붙여 받아
patch마다 "A와 짝이 맞는 진짜 B인가"를 판별한다. G는 D를 속이는 adversarial loss와
정답 B에 가까워지는 L1 loss를 함께 줄인다. 구현은 junyanz/pytorch-CycleGAN-and-pix2pix의 pix2pix 모델을 따른다.

데이터 폴더 구조 (각 이미지는 A | B를 좌우로 붙인 한 장):
    $DATA_ROOT/pix2pix/facades/
        train/*.jpg
        val/*.jpg
        test/*.jpg
    facades는 왼쪽(A)이 건물 사진, 오른쪽(B)이 label map이다. 기본값 --direction BtoA는 label → 사진.
    데이터는 junyanz repo의 `datasets/download_pix2pix_dataset.sh facades`로 받을 수 있다.

실행 예시 (repo 루트에서 `pip install -e .` 후):
    cd 4.Pix2Pix
    python train.py --run_name facades                           # facades, BtoA, 100 + 100 epoch
    python train.py --dataset maps --direction AtoB --sample_phase val
    python train.py --resume runs/Pix2Pix/facades/checkpoints/last.pt   # 같은 run 폴더에 이어서 기록
    python train.py --keep_every 20                              # 20 epoch마다 번호 붙은 checkpoint 사본 보존
    python test.py --checkpoint runs/Pix2Pix/facades/checkpoints/last.pt

산출물: runs/Pix2Pix/<run_name|timestamp>/ 아래 config.json, tb/, samples/, checkpoints/
    checkpoints/last.pt: --save_every epoch마다 덮어쓰는 최신 학습 상태 (--keep_every면 ckpt_{epoch:07d}.pt 사본도)
    samples/fixed_*.png: epoch마다 고정 샘플 번역 결과, 한 행 = [A | G(A) | B]. fixed_final_*.png는 학습 종료 후
    samples/train_*.png: --sample_every step마다 현재 학습 batch (행 배열은 위와 같음)
"""

from __future__ import annotations

import argparse
import contextlib
import time
import warnings
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from gan_common.checkpoint import load_checkpoint, save_rolling_checkpoint
from gan_common.config import base_parser, nonneg_int, positive_int, str2bool
from gan_common.data import PairedImageDataset, build_loader, get_data_root
from gan_common.logger import Logger
from gan_common.losses import GANLoss
from gan_common.networks.image2image import NLayerDiscriminator, UnetGenerator
from gan_common.utils import count_params, get_device, resolve_run_dir, save_config, seed_everything
from gan_common.weights import init_weights

MODEL_NAME = "Pix2Pix"
# resume 시 checkpoint config와 달라지면 경고할 구조·데이터 key
RESUME_CHECK_KEYS = (
    "dataset", "direction", "norm", "ngf", "ndf", "num_downs", "n_layers_D", "use_dropout", "crop_size",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = base_parser(
        "Pix2Pix: paired image-to-image translation (junyanz pix2pix 기준)",
        dataset="facades",
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
    g = p.add_argument_group("pix2pix")
    g.add_argument("--direction", type=str, default="BtoA", choices=("AtoB", "BtoA"), help="번역 방향. facades는 BtoA")
    g.add_argument("--load_size", type=positive_int, default=286, help="train: 이 크기로 resize한 뒤 crop_size로 random crop")
    g.add_argument("--crop_size", type=positive_int, default=256, help="학습 해상도. 2^num_downs의 배수여야 한다")
    g.add_argument("--flip", type=str2bool, default=True, help="train random horizontal flip (A·B 동시)")
    g.add_argument("--ngf", type=positive_int, default=64, help="G 첫 conv 채널 수")
    g.add_argument("--ndf", type=positive_int, default=64, help="D 첫 conv 채널 수")
    g.add_argument("--num_downs", type=positive_int, default=8, help="U-Net downsample 횟수 (256px: 8, 128px: 7)")
    g.add_argument("--n_layers_D", type=positive_int, default=3, help="PatchGAN stride-2 conv 수 (3이면 70x70 patch)")
    g.add_argument("--norm", type=str, default="batch", choices=("batch", "instance", "none"))
    g.add_argument("--use_dropout", type=str2bool, default=True, help="U-Net 안쪽 decoder block의 Dropout(0.5)")
    g.add_argument("--init_type", type=str, default="normal", choices=("normal", "xavier", "kaiming", "orthogonal"))
    g.add_argument("--init_gain", type=float, default=0.02)
    g.add_argument("--gan_mode", type=str, default="vanilla", choices=("vanilla", "lsgan"))
    g.add_argument("--lambda_l1", type=float, default=100.0, help="L1 loss 가중치 λ")
    g.add_argument("--n_epochs", type=nonneg_int, default=100, help="초기 lr을 유지하는 epoch 수")
    g.add_argument("--n_epochs_decay", type=nonneg_int, default=100, help="lr을 0을 향해 선형으로 줄이는 epoch 수")
    g.add_argument("--num_samples", type=nonneg_int, default=4, help="epoch마다 번역해 보는 고정 샘플 수 (0이면 생략)")
    g.add_argument("--sample_phase", type=str, default="test", help="고정 샘플을 뽑을 split (test가 없는 데이터셋은 val)")
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
        check_crop_size(args.crop_size, args.num_downs)
    except ValueError as e:
        p.error(str(e))
    return args


def check_crop_size(crop_size: int, num_downs: int) -> None:
    """U-Net은 num_downs번 절반으로 줄였다가 되돌리므로 crop_size가 2^num_downs의 배수여야 skip 크기가 맞는다."""
    if crop_size % (2**num_downs) != 0:
        raise ValueError(f"crop_size({crop_size})는 2^num_downs({2**num_downs})의 배수여야 합니다 (U-Net skip 크기).")


def linear_decay_rule(n_epochs: int, n_epochs_decay: int) -> Callable[[int], float]:
    """LambdaLR용 lr 배율. epoch e(1부터)의 배율 = 1 - max(0, e - n_epochs) / (n_epochs_decay + 1).

    처음 n_epochs 동안 초기 lr을 유지하고, 이후 n_epochs_decay 동안 0을 향해 선형으로 줄인다
    (junyanz lr_policy="linear", epoch_count=1의 공식). LambdaLR의 last_epoch는 0부터 세므로 e = last_epoch + 1.
    scheduler.step()은 PyTorch 권장 순서(optimizer.step() 뒤)대로 epoch이 끝날 때 부른다.
    junyanz train.py는 update_learning_rate()를 epoch 시작에서 부르는 버전이 있어, 그 경우 schedule이
    한 epoch 앞당겨지고 마지막 epoch의 lr이 0이 된다. 여기서는 마지막 epoch lr = lr / (n_epochs_decay + 1).
    """

    def rule(epoch_idx: int) -> float:
        return 1.0 - max(0, epoch_idx + 1 - n_epochs) / float(n_epochs_decay + 1)

    return rule


def build_generator(cfg: Mapping[str, Any]) -> UnetGenerator:
    """config(dict)로 U-Net G를 만든다. test.py도 이 함수로 학습 때와 같은 구조를 만든다."""
    return UnetGenerator(
        3, 3, num_downs=cfg["num_downs"], ngf=cfg["ngf"], norm=cfg["norm"], use_dropout=cfg["use_dropout"]
    )


def interleave(*columns: Tensor) -> Tensor:
    """(N, C, H, W) 텐서 k개 → (N·k, C, H, W). grid를 nrow=k로 그리면 한 행이 한 샘플이 된다."""
    return torch.stack(columns, dim=1).flatten(0, 1)


def resolve_sample_phase(root: Path, phase: str) -> str:
    """고정 샘플 split. `root/phase`가 없고 `root/val`이 있으면 val로 바꾼다.

    maps·cityscapes·edges2shoes 등은 test split 없이 train/val만 있다.
    """
    if (root / phase).is_dir():
        return phase
    if (root / "val").is_dir():
        print(f"'{root / phase}'가 없어 고정 샘플은 val split에서 뽑습니다 (--sample_phase val).", flush=True)
        return "val"
    raise FileNotFoundError(
        f"고정 샘플 split '{root / phase}'와 '{root / 'val'}'이 모두 없습니다. "
        "--sample_phase train 또는 --num_samples 0을 쓰십시오."
    )


def warn_config_mismatch(saved: Mapping[str, Any], args: argparse.Namespace, keys: tuple[str, ...]) -> None:
    """resume checkpoint의 학습 config와 지금 CLI 값이 다른 구조·데이터 key를 경고한다 (CLI 값을 쓴다)."""
    diffs = [
        f"{k}: checkpoint={saved.get(k)!r}, 지금={getattr(args, k)!r}" for k in keys if saved.get(k) != getattr(args, k)
    ]
    if diffs:
        warnings.warn("resume checkpoint와 설정이 다릅니다 → " + "; ".join(diffs), stacklevel=2)


@contextlib.contextmanager
def preserved_buffers(*nets: nn.Module) -> Iterator[None]:
    """블록 안의 forward가 buffer(BN running stats 등)를 바꿔도, 끝나면 원래 값으로 되돌린다."""
    saved = [[b.clone() for b in net.buffers()] for net in nets]
    try:
        yield
    finally:
        with torch.no_grad():
            for net, bufs in zip(nets, saved):
                for b, s in zip(net.buffers(), bufs):
                    b.copy_(s)


def load_fixed_samples(root: Path, args: argparse.Namespace) -> tuple[Tensor, Tensor] | None:
    """`sample_phase` split 앞쪽 `num_samples`장의 (A, B).

    `augment=False`라 split이 train이어도 random crop·flip 없이 resize만 하므로 매번 같은 이미지다.
    """
    if args.num_samples <= 0:
        return None
    ds = PairedImageDataset(
        root,
        phase=args.sample_phase,
        direction=args.direction,
        load_size=args.load_size,
        crop_size=args.crop_size,
        augment=False,
    )
    items = [ds[i] for i in range(min(args.num_samples, len(ds)))]
    return torch.stack([it["A"] for it in items]), torch.stack([it["B"] for it in items])


@torch.no_grad()
def translate_fixed(G: nn.Module, real_a: Tensor) -> Tensor:
    """고정 샘플을 한 장씩(학습과 같은 batch 1 조건) 번역한다.

    G는 train 모드 그대로 둔다: test.py 기본값(--eval_mode false)·junyanz 관행과 같이 dropout을 켜고
    BatchNorm은 이미지 자체의 통계를 쓴다. 이 forward가 BN running stats를 샘플 이미지로 바꾸지 않도록
    buffer는 원래 값으로 되돌린다.
    """
    with preserved_buffers(G):
        return torch.cat([G(real_a[i : i + 1]) for i in range(real_a.size(0))])  # (N, 3, H, W)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed)
    device = get_device(args.device)
    run_dir = resolve_run_dir(args, MODEL_NAME)  # --resume만 주면 원래 run 폴더를 이어 쓴다
    config = vars(args)

    # ---- data ----
    root = get_data_root(args.data_root) / "pix2pix" / args.dataset
    train_ds = PairedImageDataset(
        root,
        phase="train",
        direction=args.direction,
        load_size=args.load_size,
        crop_size=args.crop_size,
        flip=args.flip,
    )
    loader = build_loader(train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers)
    if args.num_samples > 0:
        args.sample_phase = resolve_sample_phase(root, args.sample_phase)
    save_config(config, run_dir)  # sample_phase 보정 뒤에 저장 (config는 vars(args)라 같이 바뀐다)
    fixed = load_fixed_samples(root, args)
    if fixed is not None:
        fixed = (fixed[0].to(device), fixed[1].to(device))

    # ---- model ----
    G = build_generator(config)
    # 조건부 D: 입력은 [A, B]를 채널로 붙인 6채널. 출력은 patch별 logits map (256px → (N, 1, 30, 30))
    D = NLayerDiscriminator(6, ndf=args.ndf, n_layers=args.n_layers_D, norm=args.norm)
    init_weights(G, args.init_type, args.init_gain)
    init_weights(D, args.init_type, args.init_gain)
    G.to(device)
    D.to(device)

    gan_loss = GANLoss(args.gan_mode)
    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    rule = linear_decay_rule(args.n_epochs, args.n_epochs_decay)
    sched_G = torch.optim.lr_scheduler.LambdaLR(opt_G, rule)
    sched_D = torch.optim.lr_scheduler.LambdaLR(opt_D, rule)

    start_epoch, step = 1, 0
    if args.resume:
        ckpt = load_checkpoint(args.resume)
        warn_config_mismatch(ckpt["config"], args, RESUME_CHECK_KEYS)
        G.load_state_dict(ckpt["G"])
        D.load_state_dict(ckpt["D"])
        opt_G.load_state_dict(ckpt["opt_G"])
        opt_D.load_state_dict(ckpt["opt_D"])
        sched_G.load_state_dict(ckpt["sched_G"])
        sched_D.load_state_dict(ckpt["sched_D"])
        start_epoch, step = ckpt["epoch"] + 1, ckpt["step"]
        print(f"resume: {args.resume} (epoch {ckpt['epoch']}까지 완료, step {step})", flush=True)

    print(
        f"device={device} | train {len(train_ds)}장 | G {count_params(G):,} / D {count_params(D):,} params "
        f"| run_dir={run_dir}",
        flush=True,
    )
    with Logger(run_dir, config, use_wandb=args.wandb, project=args.wandb_project, run_name=args.run_name) as logger:
        for epoch in range(start_epoch, args.epochs + 1):
            t0 = time.time()
            for batch in loader:
                real_a = batch["A"].to(device, non_blocking=True)  # (N, 3, H, W) 조건 입력
                real_b = batch["B"].to(device, non_blocking=True)  # (N, 3, H, W) 정답
                fake_b = G(real_a)  # (N, 3, H, W) G(A)

                # (1) D update -----------------------------------------------------------
                # 진짜 쌍 [A, B] → 1(real), 가짜 쌍 [A, G(A)] → 0(fake)으로 patch마다 판별하도록 학습한다.
                # fake_b.detach(): D loss의 gradient가 G로 흘러가지 않게 끊는다.
                # ×0.5: D objective를 절반으로 줄여 D가 G보다 빨리 학습하는 것을 늦춘다 (pix2pix 논문 관행).
                D.requires_grad_(True)
                opt_D.zero_grad()
                pred_fake = D(torch.cat([real_a, fake_b.detach()], dim=1))  # (N, 1, 30, 30) patch logits
                pred_real = D(torch.cat([real_a, real_b], dim=1))
                loss_d = 0.5 * gan_loss.d_loss(pred_real, pred_fake)
                loss_d.backward()
                opt_D.step()

                # (2) G update -----------------------------------------------------------
                # L_G = L_GAN + λ·L_L1
                #   L_GAN: D가 가짜 쌍 [A, G(A)]를 진짜(1)로 보게 만든다 (vanilla: -log σ(D(A, G(A))), non-saturating)
                #   L_L1 : |B - G(A)|₁. 정답과 픽셀 단위로 가깝게 한다. 전체 구조·색(저주파)은 L1이,
                #          선명한 질감(고주파)은 patch 단위 GAN 항이 맡는다 (λ = 100)
                # D.requires_grad_(False): 이 단계에서 D는 고정이므로 D weight의 gradient 계산을 생략한다.
                D.requires_grad_(False)
                opt_G.zero_grad()
                loss_g_gan = gan_loss.g_loss(D(torch.cat([real_a, fake_b], dim=1)))
                loss_g_l1 = F.l1_loss(fake_b, real_b) * args.lambda_l1
                loss_g = loss_g_gan + loss_g_l1
                loss_g.backward()
                opt_G.step()

                step += 1
                if step % args.log_every == 0:
                    scalars = {
                        "loss/D": loss_d.item(),
                        "loss/G_GAN": loss_g_gan.item(),
                        "loss/G_L1": loss_g_l1.item(),
                        "loss/G": loss_g.item(),
                        "lr": opt_G.param_groups[0]["lr"],
                    }
                    logger.log_scalars(scalars, step)
                    print(
                        f"[epoch {epoch}/{args.epochs}] step {step} | D {scalars['loss/D']:.4f} | "
                        f"G_GAN {scalars['loss/G_GAN']:.4f} | G_L1 {scalars['loss/G_L1']:.4f} | lr {scalars['lr']:.2e}",
                        flush=True,
                    )
                if step % args.sample_every == 0:
                    logger.log_images("train", interleave(real_a, fake_b.detach(), real_b), step, nrow=3)

            # epoch이 끝날 때 lr schedule을 한 칸 진행한다 (다음 epoch부터 적용, linear_decay_rule 참조)
            sched_G.step()
            sched_D.step()
            if fixed is not None:
                fixed_a, fixed_b = fixed
                logger.log_images("fixed", interleave(fixed_a, translate_fixed(G, fixed_a), fixed_b), step, nrow=3)
            if epoch % args.save_every == 0 or epoch == args.epochs:
                save_rolling_checkpoint(
                    run_dir / "checkpoints",
                    epoch,
                    args.keep_every,
                    epoch=epoch,
                    step=step,
                    G=G.state_dict(),
                    D=D.state_dict(),
                    opt_G=opt_G.state_dict(),
                    opt_D=opt_D.state_dict(),
                    sched_G=sched_G.state_dict(),
                    sched_D=sched_D.state_dict(),
                    config=config,
                )
            next_lr = opt_G.param_groups[0]["lr"]
            print(f"epoch {epoch}/{args.epochs} 완료 ({time.time() - t0:.0f}s) | 다음 epoch lr {next_lr:.2e}", flush=True)

        # 학습 루프가 끝난 뒤 최종 G로 고정 샘플을 한 번 더 기록한다
        if fixed is not None:
            fixed_a, fixed_b = fixed
            logger.log_images("fixed_final", interleave(fixed_a, translate_fixed(G, fixed_a), fixed_b), step, nrow=3)
    print(f"학습 종료: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
