"""StyleGAN2 샘플 생성 — checkpoint의 G_ema로 N장을 만들어 grid PNG 한 장으로 저장한다.

truncation trick: w ← w̄ + ψ(w - w̄). ψ < 1이면 평균 얼굴 쪽으로 당겨 품질이 오르고 다양성은 줄어든다
(ψ = 1이면 끔, 0이면 모두 평균 얼굴). w̄는 z `--truncation_mean`개를 mapping한 평균이다.
네트워크 구조(size, channel_multiplier 등)는 checkpoint에 저장된 학습 config를 따른다.

실행 예시:
    cd 8.StyleGAN2
    python generate.py --checkpoint runs/StyleGAN2/<run>/checkpoints/last.pt
    python generate.py --checkpoint runs/StyleGAN2/<run>/checkpoints/last.pt --n_samples 16 --nrow 4 --truncation 1.0

산출물: 기본 <run_dir>/generated/step{step}_psi{ψ}_seed{seed}.png  (run_dir = checkpoints/ 폴더의 상위 폴더)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from gan_common.checkpoint import load_checkpoint
from gan_common.config import positive_int
from gan_common.utils import get_device, seed_everything
from gan_common.viz import save_grid
from train import build_generator


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="StyleGAN2: checkpoint의 G_ema로 샘플 grid 생성",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=str, required=True, help="train.py가 저장한 checkpoints/last.pt")
    p.add_argument("--n_samples", type=positive_int, default=64, help="생성할 이미지 수")
    p.add_argument("--truncation", type=float, default=0.7, help="truncation ψ (0, 1]. 1이면 끔")
    p.add_argument("--truncation_mean", type=positive_int, default=4096, help="mean latent 추정에 쓸 z 개수")
    p.add_argument("--batch_size", type=positive_int, default=16, help="한 번에 생성할 이미지 수 (메모리)")
    p.add_argument("--nrow", type=positive_int, default=8, help="grid 한 행의 이미지 수")
    p.add_argument("--seed", type=int, default=0, help="z·noise 난수 고정")
    p.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda | cuda:N")
    p.add_argument("--out", type=str, default=None, help="저장 경로 (.png). 없으면 <run_dir>/generated/ 아래")
    args = p.parse_args(argv)
    if not 0.0 < args.truncation <= 1.0:
        p.error(f"--truncation은 (0, 1] 범위여야 합니다: {args.truncation}")
    return args


def run_dir_of(ckpt_path: Path) -> Path:
    """`<run_dir>/checkpoints/last.pt` → `<run_dir>`. checkpoints/ 밖에 있으면 파일이 든 폴더."""
    parent = ckpt_path.parent
    return parent.parent if parent.name == "checkpoints" else parent


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed)
    device = get_device(args.device)
    ckpt_path = Path(args.checkpoint).expanduser()
    ckpt = load_checkpoint(ckpt_path)
    cfg = ckpt["config"]

    G = build_generator(cfg).to(device)
    G.load_state_dict(ckpt["G_ema"])  # 샘플은 EMA weight로 만든다
    G.eval()

    with torch.no_grad():
        mean_w = G.mean_latent(args.truncation_mean) if args.truncation < 1 else None  # (1, style_dim)
        chunks = []
        for start in range(0, args.n_samples, args.batch_size):
            n = min(args.batch_size, args.n_samples - start)
            z = torch.randn(n, cfg["style_dim"], device=device)
            img, _ = G([z], truncation=args.truncation, truncation_latent=mean_w)  # (n, C, size, size)
            chunks.append(img)
        images = torch.cat(chunks)  # (n_samples, C, size, size)

    if args.out:
        out_path = Path(args.out).expanduser()
    else:
        name = f"step{ckpt['step']:07d}_psi{args.truncation:g}_seed{args.seed}.png"
        out_path = run_dir_of(ckpt_path) / "generated" / name
    save_grid(images, out_path, nrow=args.nrow)
    print(
        f"{args.n_samples}장 생성 → {out_path} | {cfg['size']}px, ψ={args.truncation:g}, checkpoint step {ckpt['step']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
