# 최근 GAN 관련 논문 5편 — 학습 노트

이 폴더는 이 repo의 고전 GAN 구현(0–12, `gan_common`)을 공부한 뒤 읽을 최근 논문 5편의 정리다. 구현은 없고 문서만 있다.

## 지금 GAN이 쓰이는 자리

2022년 이후 이미지 생성의 주류는 diffusion model이 되었고, GAN이 단독 생성기로 쓰이는 경우는 줄었다.
대신 adversarial loss는 다른 생성 모델 안의 구성 요소로 자리를 옮겼다 (이 단락은 다섯 논문을 읽고 정리한 해석이다).
첫째, few-step diffusion distillation이다. 사전학습 diffusion model을 1–4 step 생성기로 줄이면 회귀·distillation loss만으로는 결과가 흐려지는데,
ADD와 LADD는 판별기로 생성 결과의 분포를 목표 분포 쪽으로 끌어당겨 이를 막는다.
기준 분포는 서로 다르다. ADD의 판별기는 real 이미지와 비교하고, LADD의 판별기는 teacher가 만든 synthetic 분포와 비교한다 (text-to-image 기준, LADD §3, §4.2. inpainting은 synthetic data를 쓰지 않는다).
둘째, GAN과 diffusion의 결합이다. DDGAN은 diffusion의 각 denoising 스텝 안에 conditional GAN을 넣었고, Diffusion-GAN은 반대로 GAN 학습 안에 forward diffusion을 넣어 판별기를 안정시켰다.
셋째, 대규모 text-to-image GAN이다. GigaGAN은 StyleGAN 계열을 1B parameter까지 키울 수 있음을 보였고, 특히 한 번의 forward로 끝나는 GAN upsampler의 속도 이점을 보여 주었다.
세 방향 모두 이 repo에서 다루는 non-saturating·hinge loss, R1 penalty, StyleGAN2 backbone, PatchGAN discriminator 위에 서 있다.
여기에 repo 밖의 개념인 projected discriminator(frozen 사전학습 특징망 + 학습형 head, Projected GAN 계열)가 더해진다. GigaGAN의 vision-aided D와 ADD·LADD의 판별기가 이 계열이다.

## 비교표

| 논문 | 연도·학회 | 핵심 아이디어 | adversarial loss의 역할 | 샘플링 step | discriminator 설계 |
|---|---|---|---|---|---|
| [DDGAN](DDGAN.md) | arXiv 2021-12, ICLR 2022 | denoising 한 스텝을 latent $z$ 조건 conditional GAN으로 모델링. $x_0$를 예측한 뒤 Gaussian posterior로 $x_{t-1}$ 샘플 | 유일한 학습 목적. 스텝별 Gaussian KL을 대체 (non-saturating + R1) | 4 (T=4. T=2도 FID 4.08) | 시간 조건부 ResNet D. 입력은 $x_{t-1}$과 $x_t$의 concat, minibatch std |
| [Diffusion-GAN](Diffusion-GAN.md) | arXiv 2022-06, ICLR 2023 | forward diffusion을 D 입력의 instance noise(augmentation)로 쓰고 최대 step $T$를 적응 조절 | GAN의 주 목적 그대로. diffusion은 D 입력만 바꾼다 | 1 | backbone D + timestep 조건 (StyleGAN2 D의 mapping 입력을 $t$로). ProjectedGAN 변형은 $t$ 무시 |
| [GigaGAN](GigaGAN.md) | arXiv 2023-03, CVPR 2023 | StyleGAN 확장: sample-adaptive kernel, attention, multi-scale I/O D, matching-aware loss, GAN upsampler | 주 목적. CLIP contrastive loss와 vision-aided D가 보조 (upsampler는 LPIPS 추가) | 1 (base와 upsampler 각 1회 forward) | 텍스트 조건 multi-scale input/output D (15개 예측) + frozen CLIP 기반 vision-aided D |
| [ADD](ADD_LADD.md) | arXiv 2023-11 (학회 미확인, 추측: ECCV 2024) | 사전학습 diffusion model을 adversarial + score distillation으로 1–4 step 생성기로 fine-tuning | 저자들은 adversarial·distillation 두 loss가 모두 필수라고 본다. adv 단독 FID 20.8, distill 단독 315.6, 결합 20.6 (Table 1d) | 1–4 | frozen DINOv2 ViT-S 특징 + 경량 head (Projected GAN 계열), text·image 조건, hinge + R1 |
| [LADD](ADD_LADD.md) | arXiv 2024-03 (학회 미확인, 추측: SIGGRAPH Asia 2024) | 판별기를 teacher의 생성 특징(latent)으로 대체하고 teacher의 synthetic data로 학습 | text-to-image에서는 유일한 loss | 1–4 | frozen teacher(MMDiT) attention block별 token 특징 + 2D conv head, noise level·pooled CLIP 조건 |

## 추천 읽기 순서

폴더 6–12와 `gan_common/`은 현대화 계획(`progress.md`)에 따라 추가될 예정이다. 아래 링크 중 아직 없는 폴더가 있을 수 있다.

| 단계 | 읽을 것 | 먼저 볼 repo 폴더 | 초점 |
|---|---|---|---|
| 0 | 선수 지식 | [0.Simple_GAN](../../0.Simple_GAN/), [1.DCGAN](../../1.DCGAN/), [3.WGAN-GP](../../3.WGAN-GP/), [6.SNGAN](../../6.SNGAN/), [8.StyleGAN2](../../8.StyleGAN2/) | non-saturating·hinge loss, gradient penalty와 R1, mapping network, EMA |
| 1 | [Diffusion-GAN](Diffusion-GAN.md) | 1.DCGAN, 6.SNGAN, 8.StyleGAN2 | 기존 GAN에 가장 적은 코드로 붙일 수 있다. D 입력 diffusion과 $T$ 적응 규칙 |
| 2 | [DDGAN](DDGAN.md) | [4.Pix2Pix](../../4.Pix2Pix/) (조건부 D), 8.StyleGAN2 (mapping, R1) | DDPM의 forward process·posterior를 먼저 익혀야 한다 (repo 밖) |
| 3 | [GigaGAN](GigaGAN.md) | 8.StyleGAN2, [7.ProGAN](../../7.ProGAN/) (multi-scale), 4.Pix2Pix | 대규모 조건부 GAN의 안정화 기법과 GAN upsampler |
| 4 | [ADD → LADD](ADD_LADD.md) | 6.SNGAN (hinge), 8.StyleGAN2 (R1), [9.VQGAN](../../9.VQGAN/) (latent 공간) | 사전학습 diffusion model의 adversarial distillation, projected discriminator (repo 밖 개념) |

`gan_common`에서 자주 등장하는 조각: `GANLoss("vanilla" | "hinge")`, `r1_penalty`, `spectral_norm`, `EMA`, `MinibatchStdDev`, `networks.image2image.NLayerDiscriminator`(PatchGAN).

## 표기와 근거

- 각 문서의 수치는 해당 논문의 arXiv HTML(버전 명시)에서 직접 확인한 것이며, 표·그림 번호를 함께 적었다.
- 다섯 논문 모두 본문과 부록 전체를 읽었다. 참고문헌 목록은 필요한 항목만 확인했다. 그림에만 있는 수치(user study 막대, 곡선)는 인용하지 않았다.
- `미확인:` 확인하지 못한 사실과 무엇으로 확정되는지. `추측:` 논문 밖 기억에 기댄 미확인 정보(학회 게재처 등). `추정:` 노트 작성자의 판단(특히 RTX 4070 Super 12GB 재현 가능성). `해석` 논문 내용에 대한 노트 작성자의 해석. 논문 저자의 주장은 "저자 주장", "저자들은 … 서술한다"로 구분했다.
- 논문 문장을 옮기지 않고 요약했다.
