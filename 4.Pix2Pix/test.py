"""Pix2Pix 테스트 — 학습한 checkpoint로 test split을 번역해 [A | G(A) | B] 이미지를 한 장씩 저장한다.

G 모드: 기본값 --eval_mode false는 G를 train 모드로 둔다 (junyanz 관행). pix2pix는 test 때도 dropout을
켜 두어 확률성을 주고, batch 1 BatchNorm은 이미지 자체의 통계를 쓴다 (학습 때와 같은 조건).
--eval_mode true면 dropout을 끄고 BN running stats를 쓴다. train 모드에서는 BN 통계가 이미지끼리
섞이지 않도록 --batch_size 1을 권장한다.

방향(direction)·해상도·네트워크 구조는 checkpoint에 저장된 학습 config를 따른다.

데이터 폴더 구조: train.py와 같다.
    $DATA_ROOT/pix2pix/facades/{train,val,test}/*.jpg   (각 이미지는 A | B를 좌우로 붙인 한 장)

실행 예시:
    cd 4.Pix2Pix
    python test.py --checkpoint runs/Pix2Pix/<run>/checkpoints/last.pt
    python test.py --checkpoint runs/Pix2Pix/<run>/checkpoints/last.pt --phase val --eval_mode true
    python test.py --checkpoint runs/Pix2Pix/<run>/checkpoints/ckpt_0000100.pt   # --keep_every 사본

산출물: <run_dir>/test_results/<원본 파일명>.png  (run_dir = checkpoints/ 폴더의 상위 폴더)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from gan_common.checkpoint import load_checkpoint
from gan_common.config import nonneg_int, positive_int, str2bool
from gan_common.data import PairedImageDataset, build_loader, get_data_root
from gan_common.utils import get_device, seed_everything
from gan_common.viz import save_grid
from train import build_generator, check_crop_size


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pix2Pix test: checkpoint로 split을 번역해 [A | G(A) | B] 저장",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=str, required=True, help="train.py가 저장한 checkpoints/last.pt")
    p.add_argument("--data_root", type=str, default=None, help="없으면 학습 config 값 → DATA_ROOT env → ~/data")
    p.add_argument("--dataset", type=str, default=None, help="없으면 학습 config의 dataset")
    p.add_argument("--phase", type=str, default="test", help="번역할 split (test가 없는 데이터셋은 val)")
    p.add_argument("--eval_mode", type=str2bool, default=False, help="true: G.eval() (dropout 끔, BN running stats)")
    p.add_argument("--batch_size", type=positive_int, default=1)
    p.add_argument("--num_test", type=positive_int, default=None, help="번역할 최대 이미지 수 (없으면 전부)")
    p.add_argument("--results_dir", type=str, default=None, help="없으면 <run_dir>/test_results")
    p.add_argument("--seed", type=int, default=0, help="dropout 난수 고정")
    p.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda | cuda:N")
    p.add_argument("--num_workers", type=nonneg_int, default=4)
    return p


def run_dir_of(ckpt_path: Path) -> Path:
    """`<run_dir>/checkpoints/last.pt` → `<run_dir>`. checkpoints/ 밖에 있으면 파일이 든 폴더."""
    parent = ckpt_path.parent
    return parent.parent if parent.name == "checkpoints" else parent


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed)
    device = get_device(args.device)
    ckpt_path = Path(args.checkpoint).expanduser().resolve()  # 상대 경로여도 run_dir을 바르게 구한다
    ckpt = load_checkpoint(ckpt_path)
    cfg = ckpt["config"]
    check_crop_size(cfg["crop_size"], cfg["num_downs"])

    G = build_generator(cfg).to(device)
    G.load_state_dict(ckpt["G"])
    G.train(not args.eval_mode)  # 기본은 train 모드 (위 docstring 참조)

    dataset = args.dataset or cfg["dataset"]
    root = get_data_root(args.data_root or cfg.get("data_root")) / "pix2pix" / dataset
    if not (root / args.phase).is_dir():
        hint = (
            "test split이 없는 데이터셋(maps, cityscapes, edges2shoes 등)은 --phase val을 쓰십시오."
            if root.is_dir()
            else "--data_root 또는 DATA_ROOT 환경변수를 확인하십시오."
        )
        build_parser().error(f"split 폴더가 없습니다: '{root / args.phase}'. {hint}")
    # augment=False: --phase train이어도 random crop·flip 없이 crop_size로 resize만 한다
    ds = PairedImageDataset(
        root,
        phase=args.phase,
        direction=cfg["direction"],
        load_size=cfg["load_size"],
        crop_size=cfg["crop_size"],
        augment=False,
    )
    loader = build_loader(ds, args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    out_dir = Path(args.results_dir).expanduser() if args.results_dir else run_dir_of(ckpt_path) / "test_results"
    limit = len(ds) if args.num_test is None else min(args.num_test, len(ds))

    n = 0
    with torch.no_grad():
        for batch in loader:
            if n >= limit:
                break
            real_a = batch["A"].to(device)  # (N, 3, H, W)
            real_b = batch["B"].to(device)
            fake_b = G(real_a)
            for i in range(real_a.size(0)):
                if n >= limit:
                    break
                name = Path(batch["A_path"][i]).stem
                save_grid(torch.stack([real_a[i], fake_b[i], real_b[i]]), out_dir / f"{name}.png", nrow=3)
                n += 1

    print(
        f"{n}장 번역 → {out_dir} | {dataset}/{args.phase}, direction {cfg['direction']}, "
        f"checkpoint epoch {ckpt['epoch']}, eval_mode={args.eval_mode}",
        flush=True,
    )


if __name__ == "__main__":
    main()
