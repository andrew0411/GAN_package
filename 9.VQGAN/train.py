"""VQGAN first stage (Esser et al., "Taming Transformers", CVPR 2021) — VQ autoencoder를 L1 + perceptual + PatchGAN으로 학습한다.

모델(model.py): Encoder → VectorQuantizer(codebook 1024 × 64) → Decoder. 128px 이미지 → 16x16 code 격자 (f = 8).
D: PatchGAN NLayerDiscriminator(3, 64, 3, "batch") (gan_common), hinge loss. stage 2 transformer는 범위 밖이다.

왜 adversarial + perceptual loss인가:
    L1/L2 복원 loss만 쓰면 decoder는 그럴듯한 출력들의 평균을 내게 되어 질감·경계(고주파)가 뭉개진 흐린 이미지가 된다.
    압축률 f가 클수록 심해진다. VQGAN은 (1) VGG feature 공간에서 가깝게 만드는 perceptual loss로 지각적 구조를,
    (2) patch 단위 D가 "진짜 같은 국소 질감"을 요구하는 adversarial loss로 선명함을 더해 f = 8~16으로 강하게
    압축해도 선명하게 복원되는 code를 얻는다. code 격자가 짧아서 stage 2 transformer가 고해상도 이미지를 다룰 수 있다.
    이 recipe(taming Encoder/Decoder + L1 + LPIPS + PatchGAN + adaptive weight)는 Latent Diffusion(Rombach 2022)의
    autoencoder, 즉 Stable Diffusion VAE(KL-f8)가 그대로 이어받았다. latent 정규화가 VQ 대신 약한 KL이라는 점이 주된 차이다.

loss (taming VQLPIPSWithDiscriminator):
    rec    = |x − x̂| + perceptual_weight · P(x, x̂)            픽셀별 (P는 샘플별 값을 broadcast)
    nll    = mean(rec)
    g_loss = −mean(D(x̂))
    λ      = ‖∇_L nll‖ / (‖∇_L g_loss‖ + 1e-4), clamp [0, 1e4], × disc_weight     L = decoder 마지막 conv weight
    L_AE   = nll + λ · disc_factor · g_loss + codebook_weight · q_loss
    L_D    = disc_factor · 0.5 · (mean relu(1 − D(x)) + mean relu(1 + D(sg[x̂])))   hinge
    disc_factor는 완료한 iteration 수가 --disc_start보다 작으면 0: 먼저 복원을 배운 뒤 D를 켠다 (그 전에는 λ 계산 생략).
    λ는 GAN 항 gradient의 크기를 복원 항과 같은 규모로 맞춰 준다 (손으로 GAN 가중치를 고르지 않아도 된다).

학습률: lr = base_lr × batch_size (4.5e-6 × 8 = 3.6e-5), Adam betas (0.5, 0.9). --lr을 주면 그 값을 그대로 쓴다.
optimizer는 둘이다: autoencoder(encoder·decoder·codebook·quant_conv·post_quant_conv)와 D.

VGG 다운로드: --perceptual_weight > 0(기본 1.0)이면 첫 실행 때 torchvision VGG16 ImageNet 가중치(약 528MB)를
    $TORCH_HOME/hub/checkpoints (기본 ~/.cache/torch/hub/checkpoints)로 내려받는다. 0이면 VGG를 만들지 않는다.
    perceptual 항은 학습된 lin 층이 없는 LPIPS-lite다 (perceptual.py 참조). lin 가중치 없이 채널을 그대로 합하므로
    값이 실제 LPIPS의 몇 배가 될 수 있고, 기본 --perceptual_weight 1.0(taming 값)에서는 L1 항을 압도할 수 있다.
    `loss/perceptual_over_l1`(= perceptual_weight · mean P / mean L1, rec 안에서 perceptual 항이 L1 항의 몇 배인지)
    로그를 보고 --perceptual_weight를 조정한다.

데이터: 기본 celeba = ImageFolder($DATA_ROOT/celeba), 예: $DATA_ROOT/celeba/img_align_celeba/*.jpg
    (짧은 변을 image_size로 resize → center crop). $DATA_ROOT/celeba/list_eval_partition.txt가 있으면 학습에는
    공식 train split(partition 0)만 쓴다 (없으면 폴더 전체). 흑백 데이터(mnist 등)도 3채널로 읽는다.

실행 (repo 루트에서 `pip install -e .` 후):
    cd 9.VQGAN && python train.py                                  # CelebA 128px, batch 8, 100k iter, D는 10k iter 이후
    python train.py --perceptual_weight 0                          # VGG 없이 L1 + GAN (다운로드 없음)
    python train.py --dataset cifar10 --image_size 32 --ch_mult 1 2 2 --download   # 32px → 8x8 code
    python train.py --dataset fake --iters 20 --disc_start 10 --perceptual_weight 0 --log_every 5 --sample_every 10
    python reconstruct.py --checkpoint runs/VQGAN/<run>/checkpoints/last.pt
산출물: runs/VQGAN/<run_name|timestamp>/ 아래 config.json, tb/, samples/recon_*.png, checkpoints/
    samples/recon_*.png: 고정 샘플의 [입력 | 복원] 쌍, 한 행에 4쌍
    --log_every·--sample_every·--save_every·--keep_every는 모두 iteration 단위다. checkpoint는 last.pt를 덮어쓰고,
    --keep_every N이면 iteration이 N의 배수일 때 ckpt_XXXXXXX.pt 사본을 남긴다.

이어서 학습 (--run_name 없이 --resume만 주면 원래 run 폴더에 이어 쓴다):
    python train.py --resume runs/VQGAN/<run>/checkpoints/last.pt --iters 200000
    - --iters는 총 iteration 수다 (추가할 수가 아니다). 모델 구조 인자는 처음 학습과 같게 준다
    - RNG·데이터 순서는 복원하지 않으므로 끊지 않고 학습한 결과와 bit 단위로 같지는 않다
"""

from __future__ import annotations

import argparse
import warnings

import torch
from torch import Tensor

from gan_common.checkpoint import load_checkpoint, save_rolling_checkpoint
from gan_common.config import base_parser, nonneg_int, positive_int
from gan_common.data import build_image_dataset, build_loader, infinite_batches
from gan_common.logger import Logger
from gan_common.losses import GANLoss
from gan_common.networks.image2image import NLayerDiscriminator
from gan_common.utils import count_params, get_device, resolve_run_dir, save_config, seed_everything
from gan_common.weights import init_weights
from model import VQModel, build_vqmodel  # 같은 폴더의 model.py
from perceptual import VGGPerceptualLoss

MODEL_NAME = "VQGAN"
PAIRS_PER_ROW = 4  # 샘플 grid 한 행에 놓는 [입력 | 복원] 쌍 수


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = base_parser(
        "VQGAN first stage: VQ autoencoder + PatchGAN + perceptual loss (taming-transformers 기준)",
        dataset="celeba",
        image_size=128,
        channels=3,
        batch_size=8,
        lr=None,  # 없으면 base_lr × batch_size
        beta1=0.5,  # taming: Adam betas (0.5, 0.9), autoencoder·D 공통
        beta2=0.9,
        sample_every=1000,
        save_every=5000,
        epochs=None,  # 쓰지 않는 공통 인자: 학습 길이는 --iters
    )
    g = p.add_argument_group(
        "VQGAN model",
        "--lr을 주지 않으면 base_lr × batch_size. --log_every·--sample_every·--save_every·--keep_every는 iteration 단위",
    )
    g.add_argument("--iters", type=positive_int, default=100000, help="총 iteration 수 (autoencoder·D 각 1회 갱신 = 1)")
    g.add_argument("--base_lr", type=float, default=4.5e-6, help="batch 1개 샘플당 학습률. lr = base_lr × batch_size")
    g.add_argument("--ch", type=positive_int, default=64, help="Encoder/Decoder 기본 채널 수")
    g.add_argument("--ch_mult", type=positive_int, nargs="+", default=[1, 2, 2, 4], help="level별 채널 배수. f = 2^(개수-1)")
    g.add_argument("--num_res_blocks", type=positive_int, default=2, help="Encoder level당 ResnetBlock 수 (Decoder는 +1)")
    g.add_argument(
        "--attn_resolutions",
        type=positive_int,
        nargs="*",
        default=None,
        help="AttnBlock을 넣을 해상도. 없으면 최저 해상도(image_size / f). 값 없이 주면 level attention 없음 (mid는 항상)",
    )
    g.add_argument("--dropout", type=float, default=0.0, help="ResnetBlock dropout (taming VQGAN 설정은 0)")
    g.add_argument("--z_channels", type=positive_int, default=64, help="Encoder 출력 / Decoder 입력 채널")
    g.add_argument("--embed_dim", type=positive_int, default=64, help="codebook 벡터 차원")
    g.add_argument("--n_embed", type=positive_int, default=1024, help="codebook 크기 (code 개수)")
    g.add_argument("--beta", type=float, default=0.25, help="commitment loss 가중치 β")

    g = p.add_argument_group("VQGAN loss")
    g.add_argument("--codebook_weight", type=float, default=1.0, help="q_loss(codebook + β·commitment) 가중치")
    g.add_argument(
        "--perceptual_weight",
        type=float,
        default=1.0,
        help="perceptual 가중치 (taming 값). LPIPS-lite는 실제 LPIPS보다 값이 몇 배 클 수 있어 L1을 압도할 수 있다 — "
        "loss/perceptual_over_l1 로그를 보고 조정. 0이면 VGG를 만들지 않는다",
    )
    g.add_argument("--disc_weight", type=float, default=0.8, help="adaptive weight λ에 곱하는 값")
    g.add_argument("--disc_factor", type=float, default=1.0, help="D를 켠 뒤 GAN 항·D loss에 곱하는 값")
    g.add_argument("--disc_start", type=nonneg_int, default=10000, help="이 iteration 수를 마친 뒤부터 D를 켠다")
    g.add_argument("--ndf", type=positive_int, default=64, help="D 첫 conv 채널 수")
    g.add_argument("--n_layers_D", type=positive_int, default=3, help="PatchGAN stride-2 conv 수")
    g.add_argument("--n_samples", type=positive_int, default=8, help="복원 grid에 쓰는 고정 샘플 수")
    args = p.parse_args(argv)

    if args.epochs is not None:
        p.error("--epochs는 쓰지 않습니다. 학습 길이는 --iters(총 iteration 수)로 정합니다.")
    if args.channels != 3:
        p.error("VQGAN은 --channels 3만 지원합니다 (VGG perceptual loss). 흑백 데이터도 3채널로 읽습니다.")
    if args.attn_resolutions is None:
        args.attn_resolutions = [args.image_size // 2 ** (len(args.ch_mult) - 1)]  # 최저 해상도
    if args.lr is None:
        args.lr = args.base_lr * args.batch_size
    return args


def interleave(*columns: Tensor) -> Tensor:
    """(N, C, H, W) 텐서 k개 → (N·k, C, H, W). 샘플 i의 k장이 연속으로 놓인다 (grid에서 나란히 보인다)."""
    return torch.stack(columns, dim=1).flatten(0, 1)


def adaptive_weight(nll: Tensor, g_loss: Tensor, last_layer: Tensor, disc_weight: float) -> Tensor:
    """λ = ‖∇_L nll‖ / (‖∇_L g_loss‖ + 1e-4), clamp [0, 1e4], × disc_weight (taming calculate_adaptive_weight).

    두 loss가 decoder 마지막 층 L에 주는 gradient 크기의 비율이다. GAN 항의 gradient가 복원 항과 같은
    규모가 되도록 맞춰, 학습 단계마다 GAN 가중치가 자동으로 조정된다. λ는 상수로 쓰므로 detach한다.
    retain_graph=True: 같은 그래프로 뒤에서 L_AE.backward()를 한 번 더 해야 한다.
    """
    (nll_grad,) = torch.autograd.grad(nll, last_layer, retain_graph=True)
    (g_grad,) = torch.autograd.grad(g_loss, last_layer, retain_graph=True)
    d_weight = nll_grad.norm() / (g_grad.norm() + 1e-4)
    return d_weight.clamp(0.0, 1e4).detach() * disc_weight


@torch.no_grad()
def log_reconstructions(logger: Logger, model: VQModel, fixed: Tensor, step: int) -> None:
    """고정 샘플의 [입력 | 복원] grid를 기록한다 (한 행에 PAIRS_PER_ROW쌍)."""
    model.eval()  # GroupNorm·dropout 0이라 결과는 train 모드와 같지만 평가 관례를 따른다
    x_rec, _, _ = model(fixed)
    model.train()
    logger.log_images("recon", interleave(fixed, x_rec), step, nrow=2 * PAIRS_PER_ROW)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed)
    device = get_device(args.device)
    run_dir = resolve_run_dir(args, MODEL_NAME)
    config = vars(args)
    save_config(config, run_dir)

    # ---- data: [-1, 1]로 정규화된 (N, 3, S, S) 이미지
    dataset = build_image_dataset(
        args.dataset,
        args.image_size,
        args.channels,
        root=args.data_root,
        download=args.download,
        path=args.data_path,
    )
    loader = build_loader(dataset, args.batch_size, num_workers=args.num_workers)
    data = infinite_batches(loader)  # (x, label) batch를 끝없이 낸다. pass마다 다시 shuffle
    n_fixed = min(args.n_samples, len(dataset))
    fixed = torch.stack([dataset[i][0] for i in range(n_fixed)]).to(device)  # (n, 3, S, S) 매번 같은 앞쪽 이미지

    # ---- model: Encoder/Decoder는 PyTorch 기본 초기화 (taming과 같음)
    # D: conv weight N(0, 0.02), BatchNorm weight N(1, 0.02)·bias 0은 taming weights_init과 같다.
    # 차이 하나: init_weights는 bias가 있는 conv(첫 conv, 마지막 logits conv)의 bias도 0으로 두지만
    # taming은 conv bias를 PyTorch 기본 초기화로 남긴다 (repo 공통 init을 따르고, 영향은 미미하다고 본다)
    model = build_vqmodel(config).to(device)
    D = NLayerDiscriminator(args.channels, ndf=args.ndf, n_layers=args.n_layers_D, norm="batch")
    init_weights(D, "normal", 0.02)
    D.to(device)
    perceptual = VGGPerceptualLoss().to(device) if args.perceptual_weight > 0 else None
    gan_loss = GANLoss("hinge")
    opt_ae = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    start_step = 0
    if args.resume:
        ckpt = load_checkpoint(args.resume)
        model.load_state_dict(ckpt["model"])
        D.load_state_dict(ckpt["D"])
        opt_ae.load_state_dict(ckpt["opt_ae"])
        opt_d.load_state_dict(ckpt["opt_d"])
        # optimizer state에는 이전 실행의 lr이 들어 있다. 이번 설정(--lr 또는 base_lr × batch_size)과 다르면 이번 값으로 덮어쓴다
        old_lr = opt_ae.param_groups[0]["lr"]
        if old_lr != args.lr:
            warnings.warn(
                f"checkpoint의 lr {old_lr:.3e}를 이번 설정 lr {args.lr:.3e}로 덮어씁니다 (autoencoder·D 모두).",
                stacklevel=1,
            )
            for opt in (opt_ae, opt_d):
                for group in opt.param_groups:
                    group["lr"] = args.lr
        start_step = ckpt["step"]
        print(f"resume: {args.resume} (step {start_step})", flush=True)

    print(
        f"device={device} | run_dir={run_dir} | lr {args.lr:.2e} | VQModel {count_params(model):,} params, "
        f"D {count_params(D):,} params | perceptual {'VGG16' if perceptual is not None else 'off'}",
        flush=True,
    )

    usage = torch.zeros(args.n_embed, dtype=torch.long, device=device)  # log 구간 동안 code별 선택 횟수
    with Logger(run_dir, config, use_wandb=args.wandb, project=args.wandb_project, run_name=args.run_name) as logger:
        model.train()
        D.train()
        for step in range(start_step + 1, args.iters + 1):
            real, _ = next(data)
            real = real.to(device, non_blocking=True)  # (N, 3, S, S)
            # 완료한 iteration 수(step - 1, taming의 global_step)가 disc_start 미만이면 GAN 항을 끈다
            disc_factor = args.disc_factor if step - 1 >= args.disc_start else 0.0

            # ---- (1) autoencoder step: L_AE = nll + λ·disc_factor·g_loss + codebook_weight·q_loss
            # D weight gradient는 필요 없다 (x̂로 가는 gradient만 D를 통과한다)
            D.requires_grad_(False)
            x_rec, q_loss, info = model(real)  # (N, 3, S, S), scalar, {"indices": (N, S/f, S/f), "perplexity"}
            l1 = (real - x_rec).abs()  # (N, 3, S, S)
            if perceptual is not None:
                p_loss = perceptual(real, x_rec)  # (N, 1, 1, 1)
                rec = l1 + args.perceptual_weight * p_loss  # broadcast → (N, 3, S, S)
            else:
                p_loss = None
                rec = l1
            nll = rec.mean()
            if disc_factor > 0:
                g_loss = gan_loss.g_loss(D(x_rec))  # hinge G: −mean(D(x̂)), D는 (N, 1, h, w) patch logits
                d_weight = adaptive_weight(nll, g_loss, model.last_layer, args.disc_weight)
                loss_ae = nll + d_weight * disc_factor * g_loss + args.codebook_weight * q_loss
            else:
                # D를 켜기 전: GAN 항 계수가 0이므로 λ의 gradient 계산과 D 역전파를 생략한다. g_loss는 로깅용
                with torch.no_grad():
                    g_loss = gan_loss.g_loss(D(x_rec))
                d_weight = torch.zeros((), device=device)
                loss_ae = nll + args.codebook_weight * q_loss
            opt_ae.zero_grad()
            loss_ae.backward()
            opt_ae.step()

            # ---- (2) D step: hinge, x̂는 detach (D loss의 gradient가 autoencoder로 가지 않게)
            # taming(Lightning 2-pass)은 AE 갱신 뒤 forward를 다시 해 새 x̂로 D를 학습한다. 여기서는 AE 갱신 전의
            # x̂를 재사용한다 (1 step 오래된 값, iteration마다 AE forward 1회 절약).
            # taming hinge_d_loss는 0.5·(real 항 + fake 항). GANLoss.d_loss는 합이므로 0.5를 여기서 곱한다.
            # disc_factor 0이면 loss도 0이라 D weight는 변하지 않는다 (taming과 같이 step은 그대로 진행)
            D.requires_grad_(True)
            logits_real = D(real)  # (N, 1, h, w)
            logits_fake = D(x_rec.detach())
            loss_d = disc_factor * 0.5 * gan_loss.d_loss(logits_real, logits_fake)
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            usage += torch.bincount(info["indices"].flatten(), minlength=args.n_embed)
            if step % args.log_every == 0:
                scalars = {
                    "loss/l1": l1.mean().item(),
                    "loss/rec": nll.item(),  # mean(L1 + perceptual_weight·P) = nll
                    "loss/q": q_loss.item(),
                    "loss/g": g_loss.item(),
                    "loss/ae": loss_ae.item(),
                    "loss/d": loss_d.item(),
                    "logit/D_real": logits_real.detach().mean().item(),
                    "logit/D_fake": logits_fake.detach().mean().item(),
                    "weight/d_weight": d_weight.item(),
                    "weight/disc_factor": disc_factor,
                    "codebook/perplexity": info["perplexity"].item(),  # 현재 batch의 code 분포 perplexity
                    "codebook/usage": (usage > 0).float().mean().item(),  # 최근 log_every iter 동안 쓰인 code 비율
                }
                if p_loss is not None:
                    scalars["loss/perceptual"] = p_loss.mean().item()
                    # rec 안에서 perceptual 항이 L1 항의 몇 배인지. 1보다 훨씬 크면 --perceptual_weight를 낮추는 것을 고려
                    scalars["loss/perceptual_over_l1"] = (
                        args.perceptual_weight * scalars["loss/perceptual"] / max(scalars["loss/l1"], 1e-12)
                    )
                usage.zero_()
                logger.log_scalars(scalars, step)
                print(
                    f"step {step}/{args.iters} | " + " ".join(f"{k} {v:.4f}" for k, v in scalars.items()),
                    flush=True,
                )
            if step % args.sample_every == 0:
                log_reconstructions(logger, model, fixed, step)
            if step % args.save_every == 0 or step == args.iters:
                save_rolling_checkpoint(
                    run_dir / "checkpoints",
                    step,
                    args.keep_every,
                    model=model.state_dict(),
                    D=D.state_dict(),
                    opt_ae=opt_ae.state_dict(),
                    opt_d=opt_d.state_dict(),
                    step=step,
                    config=config,
                )

        log_reconstructions(logger, model, fixed, max(start_step, args.iters))  # 학습 종료 시점 복원


if __name__ == "__main__":
    main()
