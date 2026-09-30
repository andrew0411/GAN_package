"""GANomaly (Akcay et al., ACCV 2018) — encoder-decoder-encoder GAN을 정상 데이터만으로 학습해 이상을 탐지한다.

방법 요약:
    G = E1 → Decoder → E2.  z = E1(x),  x̂ = Decoder(z),  ẑ = E2(x̂)   (model.py)
    정상 데이터만 학습하면 G는 정상 이미지의 압축·복원만 잘하게 된다. 이상 이미지는 x̂가 '정상처럼' 복원되어
    E1(x)과 E2(x̂)의 latent가 어긋난다. 이 차이가 anomaly score다:
        A(x) = mean_j (z_j − ẑ_j)²   → test 전체에서 min-max scaling해 [0, 1]
    G loss = w_adv·L_adv + w_con·L_con + w_enc·L_enc
        L_adv = mean (f(x) − f(x̂))²   D 중간 feature의 feature matching (L2)   w_adv 1
        L_con = mean |x − x̂|           contextual(복원) loss (L1)              w_con 50
        L_enc = mean (z − ẑ)²          encoder loss (L2)                        w_enc 1
    D loss = 0.5·[BCE(D(x), 1) + BCE(D(x̂), 0)] (vanilla, logits). D loss < 1e-5면 D를 재초기화한다 (공식 구현).
    한 iteration은 공식 구현 순서대로 G step → D step이다.

프로토콜 (10.AnoGAN과 공통):
    MNIST에서 --anomaly_class k(기본 0)를 이상, 나머지 9개 숫자를 정상으로 둔다.
    학습: train split의 정상 클래스만. 평가: test split 전체, label = (y == k) (1 = anomaly)
    매 epoch 끝에 test split에서 AUROC·AP를 계산하고, AUROC가 가장 높은 epoch를 checkpoints/best.pt로 남긴다.
    주의: test split으로 best를 고르는 것은 공식 구현 관행이다. 엄밀한 평가라면 별도 validation split으로 골라야 한다.
    평가 시 NetG는 eval 모드(BN running stats)라 score가 샘플마다 독립이고 batch 구성과 무관하다.
    평가 label이 한 종류뿐이면(split 없는 데이터셋 등) AUROC·AP를 건너뛰고 best.pt 없이 last.pt만 저장한다.

공식 구현(samet-akcay/ganomaly)과 다른 점:
    (1) test set 구성: 공식 lib/data.py는 train split의 이상 클래스 샘플을 모두 test로 옮긴다
        (test = cat(정상 test, 이상 train, 이상 test)). 여기는 test split만 쓴다. 그래서 이상 비율이 다르다:
        여기 k=0이면 980/10,000 ≈ 9.8%.
        추측: 공식은 약 43%다 (표준 MNIST 클래스 수로 계산하면 (5,923 + 980) / (9,020 + 5,923 + 980)).
        AP는 이상 비율에 따라 달라지므로 논문 수치와 비교할 수 없다. AUROC는 비율의 영향을 덜 받는다
    (2) 공식은 평가 때 BN을 train 모드(batch 통계)로 두고 MNIST test loader를 shuffle=True로 돌린다.
        여기는 eval 모드 + shuffle=False라 score가 샘플마다 결정적이다
    (3) 공식 MNIST 정규화는 Normalize(0.1307, 0.3081). 여기는 Decoder 출력 Tanh에 맞춰 [-1, 1]
    (4) 공식 NetD는 ngf로 만든다 (--ndf는 쓰이지 않음). 여기는 --ndf를 쓴다. 기본값이 둘 다 64라 결과는 같다
    그 밖에 D logits + BCEWithLogits(공식: Sigmoid + BCELoss), Decoder 마지막 ConvT bias는 model.py docstring 참조.

실행 예시 (repo 루트에서 `pip install -e .` 후):
    cd 11.GANomaly && python train.py --download
    python train.py --anomaly_class 3 --run_name digit3
    python train.py --resume runs/GANomaly/digit3/checkpoints/last.pt --anomaly_class 3 --epochs 30
    python evaluate.py --checkpoint runs/GANomaly/<run>/checkpoints/best.pt
산출물: runs/GANomaly/<run_name|timestamp>/ 아래 config.json, tb/, samples/, checkpoints/(last.pt, best.pt, ckpt_*.pt)

이어서 학습 (--run_name 없이 --resume만 주면 원래 run 폴더에 이어 쓴다):
    - 원래 run의 RESUME_KEYS(--dataset, --anomaly_class, --image_size, --channels, --nz, --ngf, --ndf)를
      같은 값으로 다시 줘야 한다. 기본값과 다르게 학습했다면(예: --anomaly_class 3) 그 인자를 반드시 반복한다
    - 하나라도 다르면 load 전에 ValueError로 멈춘다. 특히 anomaly_class가 바뀌면 이상 클래스가 학습에
      섞여 되돌릴 수 없으므로 경고가 아니라 에러다
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gan_common.checkpoint import load_checkpoint, save_checkpoint, save_rolling_checkpoint
from gan_common.config import base_parser, positive_int
from gan_common.data import build_image_dataset, build_loader
from gan_common.logger import Logger
from gan_common.losses import GANLoss
from gan_common.metrics import average_precision, roc_auc
from gan_common.utils import count_params, get_device, resolve_run_dir, save_config, seed_everything
from gan_common.weights import init_weights
from model import NetD, NetG, latent_score

NUM_CLASSES = 10  # MNIST · Fashion-MNIST · CIFAR-10
SPLIT_DATASETS = ("mnist", "fashion_mnist", "cifar10")  # train/test split이 따로 있는 데이터셋
D_REINIT_THRESHOLD = 1e-5  # 공식 구현: D loss가 이보다 작으면 D를 재초기화
# resume 시 checkpoint와 같아야 하는 인자: 데이터 분할(정상/이상)과 모델 구조(파라미터 shape)를 정한다
RESUME_KEYS = ("dataset", "anomaly_class", "image_size", "channels", "nz", "ngf", "ndf")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = base_parser(
        "GANomaly on MNIST 32px (normal classes only)",
        dataset="mnist",
        image_size=32,
        channels=1,
        batch_size=64,
        epochs=15,
        lr=2e-4,
        beta1=0.5,
        beta2=0.999,
    )
    g = p.add_argument_group("anomaly detection")
    g.add_argument(
        "--anomaly_class",
        type=int,
        default=0,
        choices=range(NUM_CLASSES),
        help="이상으로 둘 클래스 k. 학습에서 제외되고 평가에서 label 1이 된다",
    )
    g = p.add_argument_group("model")
    g.add_argument("--nz", type=positive_int, default=100, help="latent z 차원")
    g.add_argument("--ngf", type=positive_int, default=64, help="NetG(E1, Decoder, E2) 기본 채널 수")
    g.add_argument("--ndf", type=positive_int, default=64, help="NetD 기본 채널 수")
    g.add_argument("--w_adv", type=float, default=1.0, help="adversarial(feature matching) loss 가중치")
    g.add_argument("--w_con", type=float, default=50.0, help="contextual(L1 복원) loss 가중치")
    g.add_argument("--w_enc", type=float, default=1.0, help="encoder(latent L2) loss 가중치")
    g.add_argument("--n_samples", type=positive_int, default=16, help="복원 grid에 넣을 고정 test 이미지 수")
    return p.parse_args(argv)


@torch.no_grad()
def compute_scores(
    netG: NetG, loader: DataLoader, device: torch.device, anomaly_class: int
) -> tuple[np.ndarray, np.ndarray]:
    """loader 순서대로 raw score A(x) = mean((z − ẑ)²)와 label(1 = anomaly)을 모은다. 각 (N_test,).

    loader는 `shuffle=False, drop_last=False`여야 모든 샘플이 순서대로 한 번씩 들어간다.
    netG를 eval 모드로 돌린 뒤 원래 모드로 되돌린다.
    """
    was_training = netG.training
    netG.eval()
    scores, labels = [], []
    for x, y in loader:
        _, z, z_hat = netG(x.to(device, non_blocking=True))
        scores.append(latent_score(z, z_hat).cpu())  # (N,)
        labels.append((y == anomaly_class).long())  # (N,) 1 = anomaly
    netG.train(was_training)
    return torch.cat(scores).numpy(), torch.cat(labels).numpy()


def min_max_scale(scores: np.ndarray) -> np.ndarray:
    """[0, 1]로 선형 변환 (공식 구현). 순서를 바꾸지 않으므로 AUROC·AP는 그대로다. 값이 모두 같으면 0."""
    s = scores.astype(np.float64)
    lo, hi = s.min(), s.max()
    if hi <= lo:
        return np.zeros_like(s)
    return (s - lo) / (hi - lo)


def has_both_classes(labels: np.ndarray) -> bool:
    """AUROC·AP는 정상(0)과 이상(1)이 모두 있어야 정의된다."""
    n_pos = int(labels.sum())
    return 0 < n_pos < len(labels)


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


@torch.no_grad()
def log_recon(logger: Logger, netG: NetG, fixed_x: torch.Tensor, step: int) -> None:
    """고정 test 배치의 입력(윗줄)과 복원 x̂(아랫줄). 이상 클래스가 '정상처럼' 복원되는지 본다."""
    was_training = netG.training
    netG.eval()
    x_hat, _, _ = netG(fixed_x)  # (K, C, S, S)
    netG.train(was_training)
    logger.log_images("recon", torch.cat([fixed_x, x_hat]), step, nrow=fixed_x.size(0))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed)
    device = get_device(args.device)
    run_dir = resolve_run_dir(args, "GANomaly")
    ckpt_dir = run_dir / "checkpoints"
    ckpt = None
    if args.resume:
        ckpt = load_checkpoint(args.resume)
        check_resume_config(ckpt["config"], args)  # save_config 전에 검사해 원래 config.json을 덮어쓰지 않는다
    save_config(vars(args), run_dir)
    if args.dataset not in SPLIT_DATASETS:
        print(f"경고: dataset={args.dataset!r}은 train/test split이 없어 학습 데이터로 평가하게 됩니다.", flush=True)

    # 데이터: 학습 = train split 정상 클래스 Subset, 평가 = test split 전체. [-1, 1] (N, C, 32, 32)
    normal = [c for c in range(NUM_CLASSES) if c != args.anomaly_class]
    data_kw = {"root": args.data_root, "download": args.download, "path": args.data_path}
    train_set = build_image_dataset(
        args.dataset, args.image_size, args.channels, train=True, classes=normal, **data_kw
    )
    test_set = build_image_dataset(args.dataset, args.image_size, args.channels, train=False, **data_kw)
    train_loader = build_loader(train_set, args.batch_size, num_workers=args.num_workers)
    test_loader = build_loader(test_set, args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

    # 모델: Conv weight N(0, 0.02), BatchNorm weight N(1, 0.02) (공식 weights_init과 같음)
    netG = NetG(args.image_size, args.nz, args.channels, args.ngf)
    netD = NetD(args.image_size, args.channels, args.ndf)
    init_weights(netG, "normal", 0.02)
    init_weights(netD, "normal", 0.02)
    netG, netD = netG.to(device), netD.to(device)
    opt_G = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    opt_D = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    gan_loss = GANLoss("vanilla")

    # 고정 test 배치 (앞 n_samples장, 정상·이상 섞임): 입력 | 복원 비교용
    fixed_x = torch.stack([test_set[i][0] for i in range(min(args.n_samples, len(test_set)))]).to(device)

    start_epoch, global_step, best_auroc = 0, 0, float("-inf")
    if ckpt is not None:
        netG.load_state_dict(ckpt["netG"])
        netD.load_state_dict(ckpt["netD"])
        opt_G.load_state_dict(ckpt["opt_G"])
        opt_D.load_state_dict(ckpt["opt_D"])
        start_epoch, global_step, best_auroc = ckpt["epoch"], ckpt["step"], ckpt["best_auroc"]
        print(
            f"resume: {args.resume} (epoch {start_epoch}, step {global_step}, best AUROC {best_auroc:.4f})",
            flush=True,
        )

    print(
        f"device={device} | run_dir={run_dir} | "
        f"NetG {count_params(netG):,} params, NetD {count_params(netD):,} params\n"
        f"정상 클래스 {normal} ({len(train_set):,}장으로 학습), 이상 클래스 {args.anomaly_class}, "
        f"test {len(test_set):,}장",
        flush=True,
    )

    with Logger(
        run_dir, vars(args), use_wandb=args.wandb, project=args.wandb_project, run_name=args.run_name
    ) as logger:
        for epoch in range(start_epoch, args.epochs):
            netG.train()
            netD.train()
            for real, _ in train_loader:
                real = real.to(device, non_blocking=True)  # (N, C, S, S)

                # ---- G step: D는 고정하고 feature 추출기로만 쓴다
                netD.requires_grad_(False)
                x_hat, z, z_hat = netG(real)  # (N, C, S, S), (N, nz), (N, nz)
                with torch.no_grad():
                    _, feat_real = netD(real)  # feature matching 목표 f(x) (N, C', 4, 4)
                _, feat_fake = netD(x_hat)  # f(x̂): gradient가 D를 거쳐 G로 간다
                loss_adv = F.mse_loss(feat_fake, feat_real)
                loss_con = F.l1_loss(x_hat, real)
                loss_enc = F.mse_loss(z_hat, z)
                loss_G = args.w_adv * loss_adv + args.w_con * loss_con + args.w_enc * loss_enc
                opt_G.zero_grad(set_to_none=True)
                loss_G.backward()
                opt_G.step()

                # ---- D step: real → 1, x̂ → 0. x̂.detach()로 G까지 gradient가 가지 않게 끊는다
                netD.requires_grad_(True)
                real_logits, _ = netD(real)  # (N,)
                fake_logits, _ = netD(x_hat.detach())
                loss_D = 0.5 * gan_loss.d_loss(real_logits, fake_logits)
                opt_D.zero_grad(set_to_none=True)
                loss_D.backward()
                opt_D.step()

                # D loss가 0에 붙으면(D가 완벽히 구분) G가 받는 신호가 사라진다 → 공식 구현처럼 D weight만 재초기화
                # (optimizer 상태·BN running stats는 공식 구현과 같이 그대로 둔다)
                if loss_D.item() < D_REINIT_THRESHOLD:
                    init_weights(netD, "normal", 0.02)
                    print(f"step {global_step + 1}: loss/D < {D_REINIT_THRESHOLD:g} → NetD 재초기화", flush=True)

                global_step += 1
                if global_step % args.log_every == 0:
                    scalars = {
                        "loss/G": loss_G.item(),
                        "loss/G_adv": loss_adv.item(),
                        "loss/G_con": loss_con.item(),
                        "loss/G_enc": loss_enc.item(),
                        "loss/D": loss_D.item(),
                        "prob/D_real": torch.sigmoid(real_logits.detach()).mean().item(),
                        "prob/D_fake": torch.sigmoid(fake_logits.detach()).mean().item(),
                    }
                    logger.log_scalars(scalars, global_step)
                    print(
                        f"epoch {epoch + 1}/{args.epochs} step {global_step} | "
                        + " ".join(f"{k} {v:.4f}" for k, v in scalars.items()),
                        flush=True,
                    )
                if global_step % args.sample_every == 0:
                    log_recon(logger, netG, fixed_x, global_step)

            # ---- epoch 끝: test split 평가 → best.pt(AUROC 최고), 주기 저장(last.pt)
            raw, labels = compute_scores(netG, test_loader, device, args.anomaly_class)
            auroc: float | None = None
            ap: float | None = None
            is_best = False
            if has_both_classes(labels):
                scores = min_max_scale(raw)
                auroc = roc_auc(labels, scores)
                ap = average_precision(labels, scores)
                is_best = auroc > best_auroc
                if is_best:
                    best_auroc = auroc
                logger.log_scalars({"eval/auroc": auroc, "eval/ap": ap, "eval/best_auroc": best_auroc}, global_step)
                print(
                    f"epoch {epoch + 1}/{args.epochs} eval | AUROC {auroc:.4f} AP {ap:.4f} | "
                    f"best AUROC {best_auroc:.4f}" + (" (new best)" if is_best else ""),
                    flush=True,
                )
            else:
                print(
                    f"경고: epoch {epoch + 1} 평가 label이 한 종류뿐이라(anomaly {int(labels.sum())}/{len(labels)}) "
                    "AUROC·AP와 best.pt를 건너뜁니다.",
                    flush=True,
                )

            state = {
                "netG": netG.state_dict(),
                "netD": netD.state_dict(),
                "opt_G": opt_G.state_dict(),
                "opt_D": opt_D.state_dict(),
                "epoch": epoch + 1,
                "step": global_step,
                "auroc": auroc,
                "ap": ap,
                "best_auroc": best_auroc,
                "config": vars(args),
            }
            if is_best:
                save_checkpoint(ckpt_dir / "best.pt", **state)
            if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
                save_rolling_checkpoint(ckpt_dir, epoch + 1, args.keep_every, **state)

        # 학습 종료 후 고정 test 배치 복원을 한 번 더 기록
        log_recon(logger, netG, fixed_x, global_step)

    print(f"완료: {run_dir} | best AUROC {best_auroc:.4f}", flush=True)


if __name__ == "__main__":
    main()
