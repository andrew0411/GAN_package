# GAN_package

PyTorch로 구현한 GAN 학습용 코드 모음이다. 2021–2023년에 직접 구현한 기본 모델(0–5)을 2026년 기준 PyTorch 2.6 / torchvision 0.21로 현대화하고,
모든 폴더가 공유하는 공통 모듈 `gan_common`을 만든 뒤 이미지 생성 모델(6–9)과 GAN 기반 이상탐지 모델(10–12)을 추가했다.
기본값은 0–3이 원래 코드의 하이퍼파라미터를 유지하고, 4–12는 원 논문과 기준 구현(junyanz, pfnet, rosinality, taming-transformers, Orion 등)을 따른다.

> **주의: 모든 코드는 정적 검수(code review)만 거쳤고, 실제 학습 실행 검증은 아직 하지 않았다.**
> 이 repo에서 얻은 학습 결과, FID 등 수치, 생성 샘플 이미지는 없다. `resources/`의 그림은 원래 README의 모델 설명용 그림이다.

## 설치

Linux 또는 WSL2의 CUDA GPU 환경을 전제로 한다. 명령은 repo 루트에서 실행한다.

`environment.yml`은 conda env `gan`을 만들고 Python 3.11, torch 2.6.0 + torchvision 0.21.0(pip CUDA 12.4 wheel),
tensorboard·wandb·matplotlib·pandas 등을 설치한 뒤 마지막에 `pip install -e .`로 `gan_common`을 editable 설치한다.

```bash
mamba env create -f environment.yml
```

```bash
conda activate gan
```

이미 torch·torchvision이 있는 env를 쓸 때는 `gan_common`만 설치한다. `pyproject.toml`에는 런타임 의존성이 적혀 있지 않으므로
tensorboard(모든 train 스크립트), matplotlib·pandas(12.TadGAN), wandb(`--wandb` 사용 시)는 `environment.yml`을 보고 직접 맞춘다.

```bash
pip install -e .
```

## 데이터

데이터 경로는 코드에 하드코딩하지 않는다. 모든 스크립트는 `--data_root` 인자 → `DATA_ROOT` 환경변수 → `~/data` 순서로 데이터 루트를 정한다.
데이터셋은 repo에 커밋하지 않는다 (`.gitignore`).

```bash
export DATA_ROOT=~/data
```

| 데이터셋 | 쓰는 폴더 | 기대 구조 (`$DATA_ROOT` 기준) | 준비 |
|---|---|---|---|
| MNIST, Fashion-MNIST | 0–3, 10, 11 | torchvision 기본 구조 (`MNIST/`, `FashionMNIST/`) | `--download` |
| CIFAR-10 | 6 (1 선택) | `cifar-10-batches-py/` | `--download` |
| CelebA (aligned) | 7, 8, 9 (2, 3 선택) | `celeba/img_align_celeba/*.jpg` + 선택 `celeba/list_eval_partition.txt` | 직접 준비. partition 파일이 있으면 train(0)·test(2) split을 쓰고, 없으면 폴더 전체를 쓴다 |
| 임의 이미지 폴더 | 0–3, 6–9 | `--dataset folder --data_path <root>` → `<root>/<하위 폴더>/*.jpg` (ImageFolder). 상대 경로가 현재 위치에 없으면 `$DATA_ROOT` 기준 | 예: `--data_path ffhq` |
| pix2pix | 4 | `pix2pix/<name>/{train,val,test}/*.jpg`. 각 이미지는 A \| B를 좌우로 붙인 한 장 | junyanz repo `datasets/download_pix2pix_dataset.sh facades` |
| CycleGAN | 5 | `cyclegan/<name>/{trainA,trainB,testA,testB}/*.jpg` | junyanz repo `datasets/download_cyclegan_dataset.sh horse2zebra` |
| 시계열 CSV | 12 | 신호 `timestamp,value` (예: `tadgan/S-1.csv`) + 선택 이상 구간 `start,end` (예: `tadgan/S-1_anomalies.csv`) | `--dataset csv --data_path ... --labels_path ...` |
| fake / synthetic | 0–3, 6–9 / 12 | 없음 | `--dataset fake`(torchvision `FakeData`), 12는 `--dataset synthetic`(sine 합 + noise + 주입 이상, 라벨 포함). 데이터 없이 파이프라인을 점검할 때 쓴다 |

## 폴더 구조

| 폴더 | 모델 (논문, 연도) | 핵심 아이디어 | 기본 데이터 |
|---|---|---|---|
| `0.Simple_GAN` | GAN (Goodfellow et al., 2014) | MLP G·D의 adversarial training, non-saturating G loss | MNIST 28px |
| `1.DCGAN` | DCGAN (Radford et al., 2015) | transposed-conv G + conv D, BatchNorm | MNIST 64px |
| `2.WGAN` | WGAN (Arjovsky et al., 2017) | Wasserstein-1 critic + weight clipping, RMSprop | MNIST 64px |
| `3.WGAN-GP` | WGAN-GP (Gulrajani et al., 2017) | weight clipping 대신 gradient penalty, InstanceNorm critic | MNIST 64px |
| `4.Pix2Pix` | Pix2Pix (Isola et al., 2017) | U-Net G + 70x70 PatchGAN, cGAN loss + L1 | pix2pix `facades` (BtoA) |
| `5.CycleGAN` | CycleGAN (Zhu et al., 2017) | G·D 2쌍, cycle consistency + identity loss, LSGAN | cyclegan `horse2zebra` |
| `6.SNGAN` | SNGAN (Miyato et al., 2018) | D의 모든 층에 spectral normalization, hinge loss | CIFAR-10 32px |
| `7.ProGAN` | ProGAN (Karras et al., 2018) | 4x4부터 해상도를 키우는 progressive growing, fade-in | CelebA, 최종 128px |
| `8.StyleGAN2` | StyleGAN2 (Karras et al., 2020) | mapping network + modulated conv, lazy R1, path length reg | CelebA 64px |
| `9.VQGAN` | VQGAN first stage (Esser et al., 2021) | VQ autoencoder + perceptual + PatchGAN, adaptive weight | CelebA 128px |
| `10.AnoGAN` | AnoGAN (Schlegl et al., 2017) | 정상 데이터 DCGAN + test 시 latent z 역최적화로 anomaly score | MNIST 64px, 이상 클래스 0 |
| `11.GANomaly` | GANomaly (Akcay et al., 2018) | encoder-decoder-encoder G, latent 차이로 anomaly score | MNIST 32px, 이상 클래스 0 |
| `12.TadGAN` | TadGAN (Geiger et al., 2020) | 시계열 window의 cycle 복원 GAN (BiLSTM), critic 2개 | synthetic 신호 |

그 밖에 `gan_common/`(공통 모듈), `docs/recent/`(최신 논문 학습 노트), `resources/`(README 그림), `environment.yml`, `pyproject.toml`이 있다.

## 공통 모듈 `gan_common`

| 모듈 | 내용 |
|---|---|
| `config.py` | `base_parser`(모든 train 스크립트의 공통 CLI 인자), `str2bool`, `positive_int`, `nonneg_int` |
| `data.py` | `get_data_root`(DATA_ROOT), `image_transform`([-1, 1] 정규화), `build_image_dataset`(mnist·fashion_mnist·cifar10·celeba·folder·fake, `classes` 필터), `build_loader`, `infinite_batches`, `PairedImageDataset`, `UnpairedImageDataset` |
| `utils.py` | `seed_everything`, `get_device`, `count_params`, `make_run_dir`·`resolve_run_dir`(run 폴더), `save_config` |
| `checkpoint.py` | `save_checkpoint`(weights_only 호환 타입 검사 + 원자적 저장), `save_rolling_checkpoint`(`last.pt` + `--keep_every` 사본), `load_checkpoint`(`weights_only=True`) |
| `logger.py` | `Logger`: TensorBoard(`tb/`) + `samples/*.png` + 선택적 W&B. context manager |
| `losses.py` | `GANLoss(mode)`: vanilla(BCEWithLogits, non-saturating G) · lsgan · hinge · wgan. D는 항상 logits 입력 |
| `regularizers.py` | `gradient_penalty`(WGAN-GP), `r1_penalty`(R1), `clip_weights_`(WGAN) |
| `layers.py` | `spectral_norm`, `EqualizedConv2d`·`EqualizedLinear`(equalized lr), `PixelNorm`, `MinibatchStdDev` |
| `weights.py` | `init_weights`: normal·xavier·kaiming·orthogonal 초기화 (DCGAN·pix2pix 관례 N(0, 0.02)) |
| `ema.py` | `EMA`: generator weight의 exponential moving average (ProGAN·StyleGAN) |
| `viz.py` | `denorm`([-1, 1] → [0, 1]), `save_grid`, `make_gif` |
| `metrics.py` | numpy 전용 `roc_auc`, `average_precision`, `precision_recall_f1` (label 1 = anomaly) |
| `networks/dcgan.py` | `DCGANGenerator`, `DCGANDiscriminator`(norm 선택, 중간 feature용 `features()`). 1–3, 10, 11이 사용 |
| `networks/image2image.py` | `UnetGenerator`, `ResnetGenerator`, `NLayerDiscriminator`(PatchGAN), `PixelDiscriminator`, `get_norm_layer`. 4, 5, 9가 사용 |

## 실행 방법

모델 폴더로 이동해 `train.py`를 실행한다. 스크립트가 같은 폴더의 모듈(`model.py`, `train.py` 등)을 import하고 산출물도 실행 위치 기준 `runs/`에 쓰므로 반드시 해당 폴더 안에서 실행한다.
전체 인자와 기본값은 `python train.py --help`로 확인한다.

```bash
cd 1.DCGAN && python train.py --download
```

- **산출물**: `--out_dir`(기본 `runs`, 실행 위치 기준) 아래 `runs/<Model>/<run_name 또는 timestamp>/`에 저장된다.
  - `config.json` — 실행 인자 전체. `--resume`으로 같은 run에 이어 쓰면 기존 파일을 덮어쓰지 않고 `config_resume_<시각>.json`을 남긴다
  - `tb/` — TensorBoard scalar·이미지
  - `samples/` — 샘플 grid PNG
  - `checkpoints/last.pt` — `--save_every`마다 덮어쓰는 최신 학습 상태. `--keep_every N`이면 저장 index(epoch 또는 iteration)가 N의 배수일 때 `ckpt_XXXXXXX.pt` 사본도 남긴다
- **저장 주기 단위**: epoch 기반(0–5, 10–12)은 `--save_every`가 epoch, iteration 기반(6–9)은 iteration이다. `--log_every`·`--sample_every`는 step 단위다.
- **학습 길이**: 0–3, 10–12는 `--epochs`, 4·5는 `--n_epochs` + `--n_epochs_decay`(100 + 100, 뒤 구간 lr 선형 감쇠), 6·8·9는 `--iters`, 7은 `--images_per_phase`로 정한다.
- **이어서 학습**: `--resume runs/<Model>/<run>/checkpoints/last.pt`. `--run_name` 없이 주면 원래 run 폴더에 이어 쓴다.
  `--epochs`·`--iters`는 추가분이 아니라 총량이다. RNG 상태는 복원하지 않으므로 끊지 않은 학습과 bit 단위로 같지는 않다.
- **공통 인자**: `--device auto|cpu|cuda|cuda:N`, `--seed`(기본 0), `--num_workers`, `--data_root`, `--run_name`.
- **로깅**: TensorBoard는 모델 폴더에서 아래처럼 연다. W&B는 `--wandb`를 주면 함께 기록한다 (project 기본 `GAN_package`, `--wandb_project`로 변경).

```bash
tensorboard --logdir runs
```

폴더별 최소 실행 예시 (모두 각 스크립트 docstring에서 가져왔다. `<run>`은 실제 run 폴더 이름으로 바꾼다).

```bash
cd 0.Simple_GAN
python train.py --download
```

```bash
cd 1.DCGAN
python train.py --download
python train.py --dataset cifar10 --channels 3 --download
```

```bash
cd 2.WGAN
python train.py --download
python train.py --dataset celeba --channels 3
```

```bash
cd 3.WGAN-GP
python train.py --download
```

```bash
cd 4.Pix2Pix
python train.py --run_name facades
python train.py --dataset maps --direction AtoB --sample_phase val
python test.py --checkpoint runs/Pix2Pix/facades/checkpoints/last.pt
```

```bash
cd 5.CycleGAN
python train.py --run_name h2z
python test.py --checkpoint runs/CycleGAN/h2z/checkpoints/last.pt
```

```bash
cd 6.SNGAN
python train.py --download
python train.py --dataset fake --iters 20 --n_dis 1 --log_every 5 --sample_every 10
```

```bash
cd 7.ProGAN
python train.py
python train.py --max_res 64 --images_per_phase 200000
python train.py --dataset fake --max_res 16 --images_per_phase 512 --log_every 5 --sample_every 20
```

```bash
cd 8.StyleGAN2
python train.py
python train.py --dataset fake --iters 20 --log_every 5 --sample_every 10 --save_every 10 --num_workers 0
python generate.py --checkpoint runs/StyleGAN2/<run>/checkpoints/last.pt --truncation 0.7
```

```bash
cd 9.VQGAN
python train.py
python train.py --dataset fake --iters 20 --disc_start 10 --perceptual_weight 0 --log_every 5 --sample_every 10
python reconstruct.py --checkpoint runs/VQGAN/<run>/checkpoints/last.pt
```

```bash
cd 10.AnoGAN
python train.py --download
python detect.py --checkpoint runs/AnoGAN/<run>/checkpoints/last.pt --max_test 1000 --n_iters 200
```

```bash
cd 11.GANomaly
python train.py --download
python evaluate.py --checkpoint runs/GANomaly/<run>/checkpoints/best.pt
```

```bash
cd 12.TadGAN
python train.py --dataset synthetic
python detect.py --checkpoint runs/TadGAN/<run>/checkpoints/last.pt
python train.py --dataset csv --data_path tadgan/S-1.csv --labels_path tadgan/S-1_anomalies.csv
```

학습 후 스크립트의 산출물 위치 (`<run_dir>` = checkpoint가 든 `checkpoints/`의 상위 폴더):

| 스크립트 | 산출물 |
|---|---|
| `4.Pix2Pix/test.py` | `<run_dir>/test_results/<파일명>.png` — 한 장에 [A \| G(A) \| B] |
| `5.CycleGAN/test.py` | `<run_dir>/test_results/{AtoB,BtoA}/<파일명>.png` — [입력 \| 번역 \| cycle 복원] |
| `8.StyleGAN2/generate.py` | `<run_dir>/generated/step<step>_psi<ψ>_seed<seed>.png` (G_ema 샘플 grid) |
| `9.VQGAN/reconstruct.py` | `<run_dir>/reconstruct/recon_<dataset>_<step>.png` + 평균 L1·PSNR 출력 |
| `10.AnoGAN/detect.py` | `<run_dir>/scores.csv`, `detect_top.png`, `detect_bottom.png` + AUROC·AP 출력 |
| `11.GANomaly/evaluate.py` | `<run_dir>/scores.csv` + AUROC·AP 출력 |
| `12.TadGAN/detect.py` | `<run_dir>/detect/<신호>_<rec_error>_<comb>/` 아래 `anomalies.csv`, `scores.csv`, `detection.png`, `summary.json` |

## 모델 설명

### 0. 가장 기본적인 Generative Adversarial Model

 - 생성자(Generator)와 구분자(Discriminator)로 구성된 기본적인 형태
 - Adversarial training을 기반으로 서로 다른 목적 함수를 가지고 minmax game을 수행

### 1. DCGAN: Deep Convolutional GAN
- [Radford, A., Metz, L., & Chintala, S. (2015). Unsupervised representation learning with deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434.]

 - GAN의 일종으로, generator와 discriminator가 CNN으로 구성되었다는 특징 가짐
 - Image generation, image super-resolution, text2img synthesis 등의 task를 수행할 수 있게 해줌

<img src="./resources/DCGAN_1.PNG"  width="1000" height="500">

### 2. WGAN: Wasserstein GAN
- [Arjovsky, M., Chintala, S., & Bottou, L. (2017, July). Wasserstein generative adversarial networks. In International conference on machine learning (pp. 214-223). PMLR.]

<img src="./resources/WGAN_1.PNG">

 - Loss function이 Wasserstein distance로 구성되어 있는 GAN
 - WGAN의 특징은 다음과 같이 나타낼 수 있음

  1. 향상된 안정성 : 전통적인 GAN의 loss function은 수식 전개하면 결국 real data distribution과 generated data distribution 사이의 JS divergence를 줄이는 것임. 이는 최적화하기 어려운데 WGAN의 Wasserstein distance는 더 안정적이고 쉽게 최적화 가능함.

  2. *립시츠 제약* : WGAN은 Discriminator가 립시츠 제약을 만족하도록 유도하는데, 이를 통해 과적합을 낮추거나 모델이 너무 복잡해지는 것을 방지해줄 수 있음

  3. 샘플의 질적 향상 : Image generation에서 특히 매우 질적인 향상을 보임

  4. Application : WGAN은 다양한 task(이미지 생성, 이미지 super-resolution, T2I 등)에 적용해도 좋은 결과 보임

### 3. WGAN-GP: Wasserstein GAN - Gradient Penalty를 이용하여 학습 안정성 높인 버전
- [Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., & Courville, A. C. (2017). Improved training of wasserstein gans. Advances in neural information processing systems, 30.]

 - Weight clipping을 사용하여 립시츠 제약을 유도한 기존의 WGAN과는 다르게 GP(gradient penalty)를 통해 립시츠 제약을 유도함.

 - 기존의 WGAN보다 더 안정적인 학습을 할 수 있고 좋은 성능을 보인다는 것을 보여줌

<img src="./resources/WGAN_gp_1.PNG"  >

<img src="./resources/WGAN_gp_2.PNG"  >

### 4. Pix2Pix: Style Transfer하는 데 이용되는 구조
- [Isola, P., Zhu, J. Y., Zhou, T., & Efros, A. A. (2017). Image-to-image translation with conditional adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1125-1134).]

 - Image-to-image translation task를 위해 제안된 구조

 - Generator는 conditional 하게 특정 input image가 주어졌을 때 desired output image를 만들도록 학습됨.

<img src="./resources/pix2pix_1.PNG"  >

<img src="./resources/pix2pix_2.PNG"  >

### 5. CycleGAN
- [Zhu, J. Y., Park, T., Isola, P., & Efros, A. A. (2017). Unpaired image-to-image translation using cycle-consistent adversarial networks. In Proceedings of the IEEE international conference on computer vision (pp. 2223-2232).]

 - Image-to-image translation task를 위해 제안된 구조

 - pix2pix는 하나의 generator, discriminator를 가지는 반면 CycleGAN은 2개씩 가짐.

 - 또, pix2pix와 달리 paired image 대신 set of unpaired images를 통해 unsupervised training으로 학습함.

<img src="./resources/cycleGAN_1.PNG"  >

<img src="./resources/cycleGAN_2.PNG"  >

### 6. SNGAN: Spectral Normalization GAN
- [Miyato, T., Kataoka, T., Koyama, M., & Yoshida, Y. (2018). Spectral normalization for generative adversarial networks. In International Conference on Learning Representations.]

 - Discriminator의 모든 conv/linear weight를 최대 특이값으로 나눔 (W / σ(W)). σ는 forward마다 power iteration 1회로 싸게 추정함

 - 층마다 Lipschitz 상수가 1로 묶여 D의 gradient가 폭주하지 않음. WGAN-GP처럼 입력 gradient를 따로 계산하지 않고, weight clipping처럼 weight 분포를 뭉개지도 않음

 - 구현: CIFAR-10 32px ResNet G·D, hinge loss, G 1회당 D 5회(`--n_dis`), Adam(2e-4, 0, 0.9), 50k G step에 걸쳐 lr 선형 감소

### 7. ProGAN: Progressive Growing of GANs
- [Karras, T., Aila, T., Laine, S., & Lehtinen, J. (2018). Progressive growing of GANs for improved quality, stability, and variation. In International Conference on Learning Representations.]

 - 4x4에서 시작해 해상도를 2배씩 키우며 G·D를 함께 성장시킴. 저해상도에서 큰 구조를 먼저 배우고, 새 해상도 layer는 fade-in(alpha 0 → 1)으로 서서히 끼워 넣음

 - equalized learning rate(runtime He scaling), G의 PixelNorm, D의 minibatch stddev로 BatchNorm 없이 안정성과 다양성을 얻음

 - 구현: WGAN-GP(λ 10) + drift 0.001·E[D(x)²], G EMA(0.999), phase(fade-in 또는 stabilize)당 real 이미지 600k장, 해상도별 batch 표(`--batch_sizes`). 논문의 CelebA-HQ 대신 aligned CelebA를 `--max_res`(기본 128)로 읽음

### 8. StyleGAN2: Analyzing and Improving the Image Quality of StyleGAN
- [Karras, T., Laine, S., Aittala, M., Hellsten, J., Lehtinen, J., & Aila, T. (2020). Analyzing and improving the image quality of StyleGAN. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition.]

 - mapping network(z → w, MLP 8층)가 만든 style로 conv weight를 modulate·demodulate함. StyleGAN1의 AdaIN이 만들던 물방울(droplet) artifact를 없애려고 정규화를 feature가 아닌 weight로 옮긴 것

 - progressive growing 대신 해상도별 ToRGB 출력을 누적하는 skip G와 residual D를 씀. 층마다 noise를 더해 확률적 세부를 만듦

 - 학습: non-saturating logistic loss, lazy R1(16 step마다), path length regularization(4 step마다), style mixing 0.9, G EMA. 샘플링 시 truncation trick

 - 구현: rosinality/stylegan2-pytorch 기준 config-f를 12GB GPU용으로 축소(64px, batch 16). upfirdn2d·fused LeakyReLU는 custom CUDA 대신 순수 PyTorch. R1 γ 기본값은 StyleGAN2-ADA 휴리스틱 0.0002·size²/batch. ADA·FP16·FID 측정은 없음

### 9. VQGAN: Taming Transformers (first stage)
- [Esser, P., Rombach, R., & Ommer, B. (2021). Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition.]

 - 이미지를 f배 작은 격자의 discrete code(codebook index)로 압축했다가 복원하는 VQ autoencoder. 이 폴더는 first stage만 다루고, code 위의 transformer prior(stage 2)는 범위 밖임

 - L1/L2 복원 loss만 쓰면 결과가 흐려지므로 VGG perceptual loss와 PatchGAN adversarial loss(hinge)를 더함. GAN 항 가중치는 decoder 마지막 층에서 두 loss의 gradient 크기 비율로 정하는 adaptive weight λ

 - `--disc_start`(기본 10k iteration) 전에는 D를 끄고 복원부터 학습함. 이 recipe는 Latent Diffusion(Stable Diffusion)의 autoencoder로 이어짐

 - 구현: 128px, f = 8(16x16 code), codebook 1024 × 64. perceptual 항은 학습된 lin 층이 없는 LPIPS-lite(VGG16)라 실제 LPIPS보다 값이 클 수 있어 `loss/perceptual_over_l1` 로그를 보고 `--perceptual_weight`를 조정함

### 10. AnoGAN: Unsupervised Anomaly Detection with GANs
- [Schlegl, T., Seeböck, P., Waldstein, S. M., Schmidt-Erfurth, U., & Langs, G. (2017). Unsupervised anomaly detection with generative adversarial networks to guide marker discovery. In International conference on information processing in medical imaging.]

 - 정상 데이터만으로 DCGAN을 학습해 G(z)가 정상 manifold만 표현하게 함 (`train.py`)

 - test 이미지 x마다 G·D를 고정하고 z만 Adam으로 최적화함 (`detect.py`, 기본 500 step, `--z_lr` 0.01):
   L(z) = (1 − λ)·Σ|x − G(z)| + λ·Σ|f(x) − f(G(z))|, f = D 중간 feature, λ = `--lam` 0.1

 - anomaly score A(x) = L(z*). 정상 x는 G가 잘 재현해 작고, G가 본 적 없는 이상 x는 커짐. 샘플마다 z를 최적화하므로 평가 비용이 큼 (`--max_test`로 개수 제한)

 - 평가 프로토콜 (10·11 공통): MNIST에서 `--anomaly_class k`(기본 0)를 이상, 나머지 9개 숫자를 정상으로 둠. 학습은 train split의 정상 클래스만, 평가는 test split 전체(label = y == k, 1 = anomaly)이고 AUROC·AP를 보고함

### 11. GANomaly: Semi-supervised Anomaly Detection via Adversarial Training
- [Akcay, S., Atapour-Abarghouei, A., & Breckon, T. P. (2018). GANomaly: Semi-supervised anomaly detection via adversarial training. In Asian conference on computer vision.]

 - G = E1 → Decoder → E2 (encoder-decoder-encoder). z = E1(x), x̂ = Decoder(z), ẑ = E2(x̂)

 - G loss = w_adv·L_adv(D feature matching L2, 1) + w_con·L_con(L1 복원, 50) + w_enc·L_enc(latent L2, 1). D loss가 1e-5보다 작아지면 D를 재초기화함 (공식 구현)

 - anomaly score A(x) = mean((z − ẑ)²)를 test 전체에서 min-max scaling함. 이상 x는 '정상처럼' 복원되어 E1(x)와 E2(x̂)가 어긋남. AnoGAN과 달리 test 때 최적화가 없음

 - 평가 프로토콜: 10과 같음. 매 epoch 끝에 test split AUROC·AP를 계산하고 AUROC가 가장 높은 epoch를 `checkpoints/best.pt`로 남김. test split으로 best를 고르는 것은 공식 구현 관행이며, 엄밀한 평가라면 별도 validation split으로 골라야 함

### 12. TadGAN: Time Series Anomaly Detection Using GANs
- [Geiger, A., Liu, D., Alnegheimish, S., Cuesta-Infante, A., & Veeramachaneni, K. (2020). TadGAN: Time series anomaly detection using generative adversarial networks. In 2020 IEEE International Conference on Big Data (Big Data).]

 - 신호를 길이 100 window로 잘라 E(BiLSTM)·G(BiLSTM)로 x → z → x̂ 순환 복원을 학습함. 기본값은 기준 구현 MIT Orion `tadgan`을 따름

 - critic 2개: C_x는 real window vs G(z), C_z는 prior z vs E(x)를 구분함. 둘 다 Wasserstein + gradient penalty(λ 10), E·G에는 cycle consistency MSE(x, G(E(x)))(λ 10)를 더함

 - 비지도 학습: 이상이 섞인 신호 전체로 학습하고 라벨은 평가에만 씀 (Orion 관례)

 - scoring pipeline (`detect.py`):
   1. 창별 복원 G(E(x_i)) → 시점을 덮는 창들의 median으로 복원 신호 x̂(t)
   2. 복원 오차 `--rec_error` dtw(기본) | point | area → 평활 → z-score (0 이하는 중립)
   3. critic score: 창별 C_x(x_i) → 시점별 평균 → 분위 평균 중심의 양방향 z-score → 평활
   4. 결합 `--comb` mult(기본, 곱) | sum(α 0.5)
   5. 슬라이딩 창마다 fixed threshold mean + 4·std → 앞뒤 padding → 구간화·pruning → 창 간 구간 병합
   6. 라벨이 있으면 overlapping segment precision·recall·F1 (+ 보조로 point-wise P/R/F1)

 - Orion 대비 단순화: critic 시점 집계를 KDE 최빈값 대신 평균, fastdtw 대신 numpy 정확 DTW, dynamic threshold 미구현. 상세는 `detect.py` docstring

## 최신 동향

diffusion model이 주류가 된 뒤 GAN과 adversarial loss가 쓰이는 방식을 다룬 최근 논문 5편의 학습 노트다. 구현은 없고 문서만 있다.
비교표와 추천 읽기 순서는 [docs/recent/README.md](docs/recent/README.md)에 있다.

- [DDGAN](docs/recent/DDGAN.md) — denoising 한 스텝을 conditional GAN으로 모델링해 4 step으로 샘플링 (ICLR 2022)
- [Diffusion-GAN](docs/recent/Diffusion-GAN.md) — forward diffusion을 D 입력의 instance noise로 쓰고 최대 step T를 적응 조절 (ICLR 2023)
- [GigaGAN](docs/recent/GigaGAN.md) — StyleGAN 계열을 1B parameter text-to-image GAN으로 확장, GAN upsampler (CVPR 2023)
- [ADD / LADD](docs/recent/ADD_LADD.md) — 사전학습 diffusion model을 adversarial distillation으로 1–4 step 생성기로 줄임 (ADD arXiv 2023-11, LADD arXiv 2024-03)

## 변경 이력 요약

2026 현대화 (`modernize` 브랜치):

- 로깅: Pix2Pix의 TensorFlow 1 logger(`tf.summary.FileWriter`)를 `torch.utils.tensorboard` + 선택적 W&B(`gan_common.logger`)로 교체
- 폐기된 API 제거: `transforms.Scale` → `Resize`, `loss.data[0]` → `.item()`, `Variable` 제거, `.cuda()` 하드코딩 → `--device`, matplotlib `set_adjustable('box-forced')` 제거, imageio → PIL(`gan_common.viz`)
- CycleGAN 완성: 학습 모델과 test 스크립트가 없던 `data/ models/ options/` 구조를 `model.py`(ImagePool, CycleGAN), `train.py`, `test.py`로 완성
- D 출력 통일: Sigmoid + `BCELoss` 대신 D는 logits를 반환하고 loss는 `BCEWithLogits` 계열(`GANLoss`)을 씀
- 데이터: MNIST 원본과 TensorFlow event 로그를 git 이력에서 제거하고 `.gitignore`에 등록. 경로는 `DATA_ROOT`로 추상화
- 공통 모듈 `gan_common`, `pyproject.toml`, `environment.yml` 추가. 모든 폴더가 같은 CLI 인자·산출물 구조·checkpoint 규약(`weights_only=True`)을 따름
- 새 모델 6–12와 `docs/recent/` 학습 노트 추가
