"""AnoGAN 2단계 — 학습한 G·D로 test 이미지마다 latent z를 역추정해 anomaly score를 구한다.

방법 요약 (Schlegl et al., IPMI 2017, "Mapping new images to the latent space"):
    G는 정상 데이터만 배웠으므로 정상 x에는 G(z) ≈ x인 z가 있고, 이상 x에는 없다.
    x마다 z를 N(0, I)에서 뽑고, G·D는 고정한 채 z만 Adam으로 n_iters번 갱신해 아래 loss를 줄인다.
        L_R(z) = Σ|x − G(z)|                  residual loss (픽셀 L1 합)
        L_D(z) = Σ|f(x) − f(G(z))|            discrimination loss (f = D.features, feature matching)
        L(z)   = (1 − λ)·L_R(z) + λ·L_D(z)    λ = --lam (논문 0.1)
    anomaly score A(x) = L(z*) (마지막 step 뒤의 z*). R(x) = L_R(z*), D(x) = L_D(z*)도 함께 저장한다.
    L_D는 픽셀 오차만으로는 놓치는 "D가 보기에 정상다운가"를 더한다.

batch 병렬: loss는 샘플별 항의 합이고 z_i는 자기 항에만 영향을 준다 (G·D가 eval 모드라 샘플끼리 섞이지 않음).
Adam은 원소별로 갱신하므로 batch로 한꺼번에 최적화해도 샘플 하나씩 최적화한 것과 같다.

z 범위: 원 논문의 DCGAN은 균등분포 z를 썼고, 재구현들은 최적화 중 z를 [-1, 1]로 자르기도 한다.
여기 G는 N(0, I)로 학습했으므로 z도 N(0, I)에서 시작하고 기본은 자르지 않는다 (학습 prior와 일치).
--z_clamp 1.0을 주면 초기값과 매 step 뒤 z를 [-1, 1]로 자른다.

프로토콜 (train.py와 같음):
    test split 전체, label = (y == anomaly_class) (1 = anomaly). --max_test를 주면 seed로 무작위 부분집합.
    모델 구조·anomaly_class·dataset은 checkpoint의 학습 config를 따른다.

산출물 (<output_dir>, 기본값 run_dir = checkpoints/ 폴더의 상위 폴더):
    scores.csv         index(test split 원래 index), label, score, residual, discrimination
    detect_top.png     score 상위 --n_show개. 행마다 [x | G(z*) | |x − G(z*)|]
    detect_bottom.png  score 하위 --n_show개 (가장 정상다운 샘플)

실행 예시:
    cd 10.AnoGAN
    python detect.py --checkpoint runs/AnoGAN/<run>/checkpoints/last.pt
    python detect.py --checkpoint runs/AnoGAN/<run>/checkpoints/last.pt --max_test 1000 --n_iters 200   # 빠른 확인
    python detect.py --checkpoint runs/AnoGAN/<run>/checkpoints/last.pt --lam 0.0 --output_dir runs/AnoGAN/<run>/lam0
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, Subset

from gan_common.checkpoint import load_checkpoint
from gan_common.config import nonneg_int, positive_int
from gan_common.data import build_image_dataset, build_loader
from gan_common.metrics import average_precision, roc_auc
from gan_common.networks.dcgan import DCGANDiscriminator, DCGANGenerator
from gan_common.utils import get_device, seed_everything
from gan_common.viz import save_grid


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AnoGAN detect: test 이미지마다 latent z를 최적화해 anomaly score 계산",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=str, required=True, help="train.py가 저장한 checkpoints/last.pt (G, D 포함)")
    p.add_argument("--data_root", type=str, default=None, help="없으면 학습 config 값 → DATA_ROOT env → ~/data")
    p.add_argument("--download", action="store_true", help="test split이 없으면 내려받는다")
    p.add_argument("--n_iters", type=positive_int, default=500, help="샘플별 z 최적화 step 수 (논문 500)")
    p.add_argument("--z_lr", type=float, default=0.01, help="z를 갱신하는 Adam learning rate")
    p.add_argument("--lam", type=float, default=0.1, help="λ: discrimination loss 가중치, [0, 1] (논문 0.1)")
    p.add_argument(
        "--z_clamp",
        type=float,
        default=0.0,
        help="0보다 크면 z를 [-c, c]로 자른다 (재구현 관행 1.0). 0이면 끈다",
    )
    p.add_argument(
        "--max_test",
        type=positive_int,
        default=None,
        help="평가할 test 샘플 수 (없으면 전체, MNIST 10,000장). seed로 무작위 추출. "
        "비용 = 샘플 수 × n_iters번의 G·D forward/backward. 전체 × 500은 샘플-step 5,000,000회로, "
        "정상 train 데이터(약 54,000장)로 학습을 약 90 epoch 도는 것과 같은 샘플 수라 오래 걸린다",
    )
    p.add_argument("--batch_size", type=positive_int, default=256, help="한 번에 병렬 최적화할 샘플 수 (속도·메모리)")
    p.add_argument("--n_show", type=positive_int, default=8, help="grid에 넣을 score 상위/하위 샘플 수")
    p.add_argument("--output_dir", type=str, default=None, help="없으면 <run_dir> (checkpoints/의 상위 폴더)")
    p.add_argument("--seed", type=int, default=0, help="z 초기값과 --max_test 부분집합 난수")
    p.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda | cuda:N")
    p.add_argument("--num_workers", type=nonneg_int, default=4)
    args = p.parse_args(argv)
    if args.z_lr <= 0:
        p.error(f"--z_lr는 0보다 커야 합니다: {args.z_lr}")
    if not 0.0 <= args.lam <= 1.0:
        p.error(f"--lam은 [0, 1] 범위여야 합니다: {args.lam}")
    if args.z_clamp < 0:
        p.error(f"--z_clamp는 0 이상이어야 합니다: {args.z_clamp}")
    return args


def run_dir_of(ckpt_path: Path) -> Path:
    """`<run_dir>/checkpoints/last.pt` → `<run_dir>`. checkpoints/ 밖에 있으면 파일이 든 폴더."""
    parent = ckpt_path.parent
    return parent.parent if parent.name == "checkpoints" else parent


def residual_and_discrimination(
    G: nn.Module, D: DCGANDiscriminator, x: Tensor, feat_x: Tensor, z: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """G(z)와 샘플별 L_R = Σ|x − G(z)|, L_D = Σ|f(x) − f(G(z))|. 합은 샘플 안에서만 (batch 축은 유지)."""
    x_gen = G(z)  # (N, C, S, S)
    residual = (x - x_gen).abs().flatten(1).sum(1)  # (N,)
    discrimination = (feat_x - D.features(x_gen)).abs().flatten(1).sum(1)  # (N,)
    return x_gen, residual, discrimination


def optimize_latent(
    G: DCGANGenerator,
    D: DCGANDiscriminator,
    x: Tensor,
    *,
    n_iters: int,
    lr: float,
    lam: float,
    z_clamp: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """batch x의 샘플마다 z를 최적화한다. 반환: z* (N, z_dim), score L(z*), L_R(z*), L_D(z*) 각 (N,).

    전제: G·D는 eval 모드이고 파라미터 requires_grad=False (z만 학습 대상).
    """
    with torch.no_grad():
        feat_x = D.features(x)  # (N, C', 4, 4) 목표 feature. x는 고정이므로 한 번만 계산
    z = torch.randn(x.size(0), G.z_dim, device=x.device)  # G 학습 prior N(0, I)에서 시작
    if z_clamp > 0:
        z.clamp_(-z_clamp, z_clamp)
    z.requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)

    for _ in range(n_iters):
        _, residual, discrimination = residual_and_discrimination(G, D, x, feat_x, z)
        loss = (1.0 - lam) * residual + lam * discrimination  # (N,)
        opt.zero_grad(set_to_none=True)
        # mean이 아니라 sum: z_i의 gradient가 자기 샘플 loss의 gradient 그대로가 된다 (batch 크기와 무관)
        loss.sum().backward()
        opt.step()
        if z_clamp > 0:
            with torch.no_grad():
                z.clamp_(-z_clamp, z_clamp)

    # score는 마지막 갱신 뒤의 z*로 다시 계산한다 (grid의 G(z*)와 같은 z)
    with torch.no_grad():
        _, residual, discrimination = residual_and_discrimination(G, D, x, feat_x, z)
        score = (1.0 - lam) * residual + lam * discrimination
    return z.detach(), score, residual, discrimination


@torch.no_grad()
def save_triplets(
    G: DCGANGenerator, dataset: Dataset, z_star: Tensor, positions: np.ndarray, path: Path, device: torch.device
) -> None:
    """선택한 샘플마다 [x | G(z*) | |x − G(z*)|]를 한 행으로 저장한다. positions는 dataset 안 위치."""
    idx = torch.as_tensor(positions, dtype=torch.long)
    x = torch.stack([dataset[int(i)][0] for i in positions]).to(device)  # (K, C, S, S)
    x_gen = G(z_star[idx].to(device))  # (K, C, S, S)
    # |x − G(z*)| ∈ [0, 2]. save_grid는 [-1, 1]을 받아 (v + 1) / 2로 옮기므로 1을 빼면 residual / 2로 보인다
    residual = (x - x_gen).abs() - 1.0
    rows = torch.stack([x, x_gen, residual], dim=1).flatten(0, 1)  # (K·3, C, S, S): 행마다 x | G(z*) | residual
    save_grid(rows, path, nrow=3)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed)
    device = get_device(args.device)
    ckpt_path = Path(args.checkpoint).expanduser()
    ckpt = load_checkpoint(ckpt_path)
    cfg = ckpt["config"]
    anomaly_class = cfg["anomaly_class"]

    G = DCGANGenerator(cfg["z_dim"], cfg["channels"], cfg["features"], cfg["image_size"]).to(device)
    D = DCGANDiscriminator(cfg["channels"], cfg["features"], cfg["image_size"], norm="batch").to(device)
    G.load_state_dict(ckpt["G"])
    D.load_state_dict(ckpt["D"])
    # eval 모드: BatchNorm이 학습 때 쌓은 running stats를 쓴다. train 모드면 batch 통계로 정규화되어
    # 샘플 i의 G(z_i)·f(x_i)가 같은 batch의 다른 샘플에 의존하고(score가 batch 구성에 따라 바뀜),
    # forward마다 running stats가 test 데이터로 덮어써진다. eval이면 샘플마다 독립이라 batch 병렬 최적화가 정당하다.
    # (D의 running stats는 학습 중 real·fake batch가 섞여 쌓인 값이지만 f(x)와 f(G(z))에 같은 정규화가 적용된다)
    G.eval()
    D.eval()
    G.requires_grad_(False)  # 모델은 고정, gradient는 z로만 흐른다
    D.requires_grad_(False)

    test_set: Dataset = build_image_dataset(
        cfg["dataset"],
        cfg["image_size"],
        cfg["channels"],
        train=False,
        root=args.data_root or cfg.get("data_root"),
        download=args.download,
        path=cfg.get("data_path"),
    )
    indices = list(range(len(test_set)))  # scores.csv의 index = test split 원래 index
    if args.max_test is not None and args.max_test < len(test_set):
        gen = torch.Generator().manual_seed(args.seed)
        indices = torch.randperm(len(test_set), generator=gen)[: args.max_test].sort().values.tolist()
        test_set = Subset(test_set, indices)
    # 평가 loader: 모든 샘플을 순서대로 한 번씩 (shuffle=False, drop_last=False)
    loader = build_loader(test_set, args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    out_dir = Path(args.output_dir).expanduser() if args.output_dir else run_dir_of(ckpt_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"device={device} | checkpoint epoch {ckpt['epoch']} | anomaly_class {anomaly_class} | "
        f"test {len(test_set):,}장 × n_iters {args.n_iters}, λ {args.lam}, z_lr {args.z_lr}, z_clamp {args.z_clamp}",
        flush=True,
    )

    labels, scores, residuals, discriminations, z_stars = [], [], [], [], []
    start = time.perf_counter()
    for b, (x, y) in enumerate(loader, 1):
        x = x.to(device, non_blocking=True)
        z_star, score, residual, discrimination = optimize_latent(
            G, D, x, n_iters=args.n_iters, lr=args.z_lr, lam=args.lam, z_clamp=args.z_clamp
        )
        labels.append((y == anomaly_class).long())  # (N,) 1 = anomaly
        scores.append(score.cpu())
        residuals.append(residual.cpu())
        discriminations.append(discrimination.cpu())
        z_stars.append(z_star.cpu())
        print(
            f"batch {b}/{len(loader)} | mean score {score.mean().item():.2f} | {time.perf_counter() - start:.0f}s",
            flush=True,
        )

    label_np = torch.cat(labels).numpy()
    score_np = torch.cat(scores).numpy()
    residual_np = torch.cat(residuals).numpy()
    discrimination_np = torch.cat(discriminations).numpy()
    z_star_all = torch.cat(z_stars)  # (N, z_dim)

    csv_path = out_dir / "scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "label", "score", "residual", "discrimination"])
        writer.writerows(
            zip(indices, label_np.tolist(), score_np.tolist(), residual_np.tolist(), discrimination_np.tolist())
        )

    # 행마다 [x | G(z*) | residual]: score가 가장 큰(이상다운) 샘플과 가장 작은(정상다운) 샘플
    order = np.argsort(score_np, kind="stable")  # 오름차순
    top = order[::-1][: args.n_show].copy()  # copy: 음수 stride view는 torch.as_tensor가 받지 않는다
    bottom = order[: args.n_show]
    save_triplets(G, test_set, z_star_all, top, out_dir / "detect_top.png", device)
    save_triplets(G, test_set, z_star_all, bottom, out_dir / "detect_bottom.png", device)

    n_pos = int(label_np.sum())
    if 0 < n_pos < len(label_np):
        auroc = roc_auc(label_np, score_np)
        ap = average_precision(label_np, score_np)
        mean_normal = float(score_np[label_np == 0].mean())
        mean_anomaly = float(score_np[label_np == 1].mean())
        print(
            f"AUROC {auroc:.4f} | AP {ap:.4f} | mean score normal {mean_normal:.2f}, anomaly {mean_anomaly:.2f} "
            f"(anomaly {n_pos}/{len(label_np)})",
            flush=True,
        )
    else:
        print(f"경고: label이 한 종류뿐이라(anomaly {n_pos}/{len(label_np)}) AUROC·AP를 계산하지 않습니다.", flush=True)
    print(f"저장: {csv_path}, {out_dir / 'detect_top.png'}, {out_dir / 'detect_bottom.png'}", flush=True)


if __name__ == "__main__":
    main()
