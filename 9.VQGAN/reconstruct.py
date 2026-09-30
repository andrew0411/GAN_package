"""VQGAN 복원 평가 — checkpoint로 이미지를 encode → quantize → decode해 [입력 | 복원] grid를 저장하고 평균 L1·PSNR을 낸다.

데이터: 학습 config의 dataset을 `train=False`로 읽는다. mnist·fashion_mnist·cifar10은 test split이다.
celeba는 $DATA_ROOT/celeba/list_eval_partition.txt가 있으면 공식 test split(partition 2)을 쓰고, 없으면 폴더 전체를
쓴다 (학습 이미지와 같아지며 gan_common이 warning을 낸다). folder에는 split이 없어 학습 때와 같은 폴더를 읽으므로,
학습에 쓰지 않은 이미지로 보려면 `--dataset folder --data_path <폴더>`로 따로 둔 폴더를 준다
(ImageFolder 구조: <폴더>/<하위폴더>/*.jpg).
앞에서부터 --num_images장을 순서대로(shuffle 없이) 평가하고, grid에는 첫 batch만 그린다.
모델 구조·해상도는 checkpoint에 저장된 학습 config를 따른다.

지표 (이미지별로 구해 평균):
    L1   = mean |x − x̂|             [-1, 1] 스케일 (학습 loss의 L1 항과 같은 값)
    PSNR = 10 · log10(1 / MSE)      x, x̂를 [0, 1]로 옮기고 clamp한 뒤의 MSE (저장되는 이미지 기준)

실행 예시 (repo 루트에서 `pip install -e .` 후):
    cd 9.VQGAN
    python reconstruct.py --checkpoint runs/VQGAN/<run>/checkpoints/last.pt
    python reconstruct.py --checkpoint runs/VQGAN/<run>/checkpoints/last.pt --dataset folder --data_path celeba_val
산출물: <run_dir>/reconstruct/recon_<dataset>_<step>.png (한 행에 [입력 | 복원] 4쌍). run_dir = checkpoints/의 상위 폴더
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import Subset

from gan_common.checkpoint import load_checkpoint
from gan_common.config import nonneg_int, positive_int
from gan_common.data import build_image_dataset, build_loader
from gan_common.utils import get_device
from gan_common.viz import denorm, save_grid
from model import build_vqmodel
from train import PAIRS_PER_ROW, interleave


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="VQGAN reconstruction: checkpoint로 [입력 | 복원] grid 저장 + 평균 L1·PSNR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=str, required=True, help="train.py가 저장한 checkpoints/last.pt")
    p.add_argument("--data_root", type=str, default=None, help="없으면 학습 config 값 → DATA_ROOT env → ~/data")
    p.add_argument("--dataset", type=str, default=None, help="없으면 학습 config의 dataset")
    p.add_argument("--data_path", type=str, default=None, help="--dataset folder일 때 ImageFolder 루트 (없으면 학습 config 값)")
    p.add_argument("--download", action="store_true", help="torchvision 데이터셋이 없으면 내려받는다")
    p.add_argument("--num_images", type=positive_int, default=64, help="평가할 이미지 수 (데이터셋 앞에서부터)")
    p.add_argument("--batch_size", type=positive_int, default=16, help="grid에는 첫 batch만 그린다")
    p.add_argument("--out", type=str, default=None, help="grid PNG 경로. 없으면 <run_dir>/reconstruct/ 아래")
    p.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda | cuda:N")
    p.add_argument("--num_workers", type=nonneg_int, default=4)
    return p.parse_args(argv)


def run_dir_of(ckpt_path: Path) -> Path:
    """`<run_dir>/checkpoints/last.pt` → `<run_dir>`. checkpoints/ 밖에 있으면 파일이 든 폴더."""
    parent = ckpt_path.parent
    return parent.parent if parent.name == "checkpoints" else parent


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = get_device(args.device)
    ckpt_path = Path(args.checkpoint).expanduser()
    ckpt = load_checkpoint(ckpt_path)
    cfg = ckpt["config"]

    model = build_vqmodel(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dataset_name = args.dataset or cfg["dataset"]
    ds = build_image_dataset(
        dataset_name,
        cfg["image_size"],
        cfg["channels"],
        train=False,
        root=args.data_root or cfg.get("data_root"),
        download=args.download,
        path=args.data_path or cfg.get("data_path"),
    )
    ds = Subset(ds, range(min(args.num_images, len(ds))))
    # batch가 0개면 build_loader가 ValueError를 내므로, 아래 루프는 최소 1 batch를 돈다
    loader = build_loader(ds, args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

    l1_sum, psnr_sum, n = 0.0, 0.0, 0
    grid = None
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)  # (B, 3, S, S) in [-1, 1]
            x_rec, _, _ = model(x)  # (B, 3, S, S)
            l1 = (x - x_rec).abs().flatten(1).mean(1)  # (B,)
            mse = (denorm(x) - denorm(x_rec)).pow(2).flatten(1).mean(1)  # (B,) [0, 1] 스케일
            psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-10))  # (B,) dB, 완전 일치면 100dB로 상한
            l1_sum += l1.sum().item()
            psnr_sum += psnr.sum().item()
            n += x.size(0)
            if grid is None:
                grid = interleave(x, x_rec).cpu()  # (2B, 3, S, S): 샘플마다 [입력 | 복원]

    step = ckpt["step"]
    out = (
        Path(args.out).expanduser()
        if args.out
        else run_dir_of(ckpt_path) / "reconstruct" / f"recon_{dataset_name}_{step:07d}.png"
    )
    save_grid(grid, out, nrow=2 * PAIRS_PER_ROW)
    print(
        f"{n}장 복원 | {dataset_name} (train=False) | checkpoint step {step} | "
        f"mean L1 {l1_sum / n:.4f} ([-1, 1]) | mean PSNR {psnr_sum / n:.2f} dB ([0, 1]) | grid → {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
