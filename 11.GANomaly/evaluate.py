"""GANomaly 평가 — checkpoint의 NetG로 test split의 anomaly score·AUROC·AP를 계산해 scores.csv로 저장한다.

score (train.py의 epoch 평가와 같은 계산):
    A(x) = mean_j (z_j − ẑ_j)²,  z = E1(x), ẑ = E2(Decoder(z))   → test 전체에서 min-max scaling해 [0, 1]
    min-max scaling은 순서를 바꾸지 않으므로 AUROC·AP는 raw score로 구한 값과 같다.
프로토콜: test split 전체, label = (y == anomaly_class) (1 = anomaly). NetG는 eval 모드 (BN running stats).
모델 구조·anomaly_class·dataset은 checkpoint의 학습 config를 따른다.

산출물: <output_dir>/scores.csv (기본 run_dir = checkpoints/ 폴더의 상위 폴더)
    index(test split index), label, score(min-max), score_raw

실행 예시:
    cd 11.GANomaly
    python evaluate.py --checkpoint runs/GANomaly/<run>/checkpoints/best.pt
    python evaluate.py --checkpoint runs/GANomaly/<run>/checkpoints/last.pt --output_dir runs/GANomaly/<run>/eval_last
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from gan_common.checkpoint import load_checkpoint
from gan_common.config import nonneg_int, positive_int
from gan_common.data import build_image_dataset, build_loader
from gan_common.metrics import average_precision, roc_auc
from gan_common.utils import get_device, seed_everything
from model import NetG
from train import compute_scores, has_both_classes, min_max_scale


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GANomaly evaluate: test split anomaly score, AUROC, AP → scores.csv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=str, required=True, help="train.py가 저장한 checkpoints/best.pt 또는 last.pt")
    p.add_argument("--data_root", type=str, default=None, help="없으면 학습 config 값 → DATA_ROOT env → ~/data")
    p.add_argument("--download", action="store_true", help="test split이 없으면 내려받는다")
    p.add_argument("--batch_size", type=positive_int, default=256, help="평가 batch 크기 (eval 모드라 결과와 무관)")
    p.add_argument("--output_dir", type=str, default=None, help="없으면 <run_dir> (checkpoints/의 상위 폴더)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda | cuda:N")
    p.add_argument("--num_workers", type=nonneg_int, default=4)
    return p.parse_args(argv)


def run_dir_of(ckpt_path: Path) -> Path:
    """`<run_dir>/checkpoints/best.pt` → `<run_dir>`. checkpoints/ 밖에 있으면 파일이 든 폴더."""
    parent = ckpt_path.parent
    return parent.parent if parent.name == "checkpoints" else parent


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed)
    device = get_device(args.device)
    ckpt_path = Path(args.checkpoint).expanduser()
    ckpt = load_checkpoint(ckpt_path)
    cfg = ckpt["config"]
    anomaly_class = cfg["anomaly_class"]

    netG = NetG(cfg["image_size"], cfg["nz"], cfg["channels"], cfg["ngf"]).to(device)
    netG.load_state_dict(ckpt["netG"])

    test_set = build_image_dataset(
        cfg["dataset"],
        cfg["image_size"],
        cfg["channels"],
        train=False,
        root=args.data_root or cfg.get("data_root"),
        download=args.download,
        path=cfg.get("data_path"),
    )
    # 평가 loader: 모든 샘플을 순서대로 한 번씩 (shuffle=False, drop_last=False) → index = test split index
    loader = build_loader(test_set, args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    raw, labels = compute_scores(netG, loader, device, anomaly_class)  # compute_scores가 netG.eval()을 건다
    scores = min_max_scale(raw)

    out_dir = Path(args.output_dir).expanduser() if args.output_dir else run_dir_of(ckpt_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "label", "score", "score_raw"])
        writer.writerows(zip(range(len(scores)), labels.tolist(), scores.tolist(), raw.tolist()))

    summary = (
        f"checkpoint epoch {ckpt['epoch']} | anomaly_class {anomaly_class} | "
        f"test {len(labels):,}장 (anomaly {int(labels.sum()):,})"
    )
    if has_both_classes(labels):
        auroc = roc_auc(labels, scores)
        ap = average_precision(labels, scores)
        print(f"{summary} | AUROC {auroc:.4f} | AP {ap:.4f}", flush=True)
    else:
        print(f"{summary} | 경고: label이 한 종류뿐이라 AUROC·AP를 계산하지 않습니다.", flush=True)
    print(f"저장: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
