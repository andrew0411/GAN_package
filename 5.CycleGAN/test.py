"""CycleGAN 테스트 — checkpoint의 G_AB, G_BA로 {phase}A는 A → B, {phase}B는 B → A로 번역해 저장한다.

저장 이미지: 입력 | 번역 | 되돌린 이미지(cycle)를 가로로 붙인 한 장
    <run_dir>/test_results/AtoB/<원본 파일명>.png   [a | G_AB(a) | G_BA(G_AB(a))]
    <run_dir>/test_results/BtoA/<원본 파일명>.png   [b | G_BA(b) | G_AB(G_BA(b))]
    (run_dir = checkpoints/ 폴더의 상위 폴더)

G 모드: --eval_mode 기본값 false는 G를 train 모드로 둔다 (junyanz 관행, Pix2Pix test.py와 같은 인터페이스).
기본 설정(instance norm은 running stats 없음, dropout 없음)에서는 train/eval 모드 결과가 같다.
해상도·네트워크 구조는 checkpoint에 저장된 학습 config를 따른다.

데이터 폴더 구조: train.py와 같다.
    $DATA_ROOT/cyclegan/horse2zebra/{trainA,trainB,testA,testB}/*.jpg

실행 예시:
    cd 5.CycleGAN
    python test.py --checkpoint runs/CycleGAN/<run>/checkpoints/last.pt
    python test.py --checkpoint runs/CycleGAN/<run>/checkpoints/last.pt --num_test 50
    python test.py --checkpoint runs/CycleGAN/<run>/checkpoints/ckpt_0000100.pt   # --keep_every 사본
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import Tensor

from gan_common.checkpoint import load_checkpoint
from gan_common.config import nonneg_int, positive_int, str2bool
from gan_common.data import UnpairedImageDataset, build_loader, get_data_root
from gan_common.utils import get_device, seed_everything
from gan_common.viz import save_grid
from model import build_generator, check_crop_size


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CycleGAN test: checkpoint로 A → B, B → A 번역 결과 저장",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=str, required=True, help="train.py가 저장한 checkpoints/last.pt")
    p.add_argument("--data_root", type=str, default=None, help="없으면 학습 config 값 → DATA_ROOT env → ~/data")
    p.add_argument("--dataset", type=str, default=None, help="없으면 학습 config의 dataset")
    p.add_argument("--phase", type=str, default="test", help="번역할 split ({phase}A, {phase}B 폴더)")
    p.add_argument("--eval_mode", type=str2bool, default=False, help="true: G.eval() (dropout 끔)")
    p.add_argument("--batch_size", type=positive_int, default=1)
    p.add_argument("--num_test", type=positive_int, default=None, help="방향별 최대 번역 이미지 수 (없으면 전부)")
    p.add_argument("--results_dir", type=str, default=None, help="없으면 <run_dir>/test_results")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda | cuda:N")
    p.add_argument("--num_workers", type=nonneg_int, default=4)
    return p.parse_args(argv)


def run_dir_of(ckpt_path: Path) -> Path:
    """`<run_dir>/checkpoints/last.pt` → `<run_dir>`. checkpoints/ 밖에 있으면 파일이 든 폴더."""
    parent = ckpt_path.parent
    return parent.parent if parent.name == "checkpoints" else parent


def save_once(done: set[str], src_path: str, images: Tensor, out_dir: Path, limit: int | None) -> None:
    """원본 이미지마다 한 번만 저장한다.

    UnpairedImageDataset의 길이는 max(|A|, |B|)라서 적은 쪽 domain의 이미지는 한 epoch 안에서 반복된다.
    """
    if src_path in done or (limit is not None and len(done) >= limit):
        return
    save_grid(images, out_dir / f"{Path(src_path).stem}.png", nrow=3)
    done.add(src_path)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed)
    device = get_device(args.device)
    ckpt_path = Path(args.checkpoint).expanduser().resolve()  # 상대 경로여도 run_dir을 바르게 구한다
    ckpt = load_checkpoint(ckpt_path)
    cfg = ckpt["config"]
    check_crop_size(cfg["crop_size"])

    G_AB = build_generator(cfg["ngf"], cfg["norm"], cfg["use_dropout"], cfg["n_blocks"]).to(device)  # A → B
    G_BA = build_generator(cfg["ngf"], cfg["norm"], cfg["use_dropout"], cfg["n_blocks"]).to(device)  # B → A
    G_AB.load_state_dict(ckpt["model"]["G_AB"])
    G_BA.load_state_dict(ckpt["model"]["G_BA"])
    G_AB.train(not args.eval_mode)
    G_BA.train(not args.eval_mode)

    dataset = args.dataset or cfg["dataset"]
    root = get_data_root(args.data_root or cfg.get("data_root")) / "cyclegan" / dataset
    # serial=True: A·B 모두 index 순서. augment=False: --phase train이어도 random crop·flip 없이 resize만
    ds = UnpairedImageDataset(
        root, phase=args.phase, load_size=cfg["load_size"], crop_size=cfg["crop_size"], serial=True, augment=False
    )
    loader = build_loader(ds, args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    out_dir = Path(args.results_dir).expanduser() if args.results_dir else run_dir_of(ckpt_path) / "test_results"

    done_a: set[str] = set()
    done_b: set[str] = set()
    with torch.no_grad():
        for batch in loader:
            real_a = batch["A"].to(device)  # (N, 3, H, W)
            real_b = batch["B"].to(device)
            fake_b = G_AB(real_a)
            rec_a = G_BA(fake_b)
            fake_a = G_BA(real_b)
            rec_b = G_AB(fake_a)
            for i in range(real_a.size(0)):
                row_a = torch.stack([real_a[i], fake_b[i], rec_a[i]])  # (3, 3, H, W)
                row_b = torch.stack([real_b[i], fake_a[i], rec_b[i]])
                save_once(done_a, batch["A_path"][i], row_a, out_dir / "AtoB", args.num_test)
                save_once(done_b, batch["B_path"][i], row_b, out_dir / "BtoA", args.num_test)
            if args.num_test is not None and len(done_a) >= args.num_test and len(done_b) >= args.num_test:
                break

    print(
        f"A → B {len(done_a)}장, B → A {len(done_b)}장 → {out_dir} | {dataset}/{args.phase}, "
        f"checkpoint epoch {ckpt['epoch']}, eval_mode={args.eval_mode}",
        flush=True,
    )


if __name__ == "__main__":
    main()
