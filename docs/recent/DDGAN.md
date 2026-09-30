# DDGAN — Denoising Diffusion GANs

> 근거: arXiv 2112.07804v2 HTML의 본문과 부록 A–G 전체를 읽었다 (참고문헌 목록 제외).
> 표·그림 번호는 HTML v2 캡션 기준이다. 본문 일부가 표 번호를 다르게 부르는 곳이 있어 캡션을 따랐다.

## 1. 서지 정보

| 항목 | 내용 |
|---|---|
| 제목 | Tackling the Generative Learning Trilemma with Denoising Diffusion GANs |
| 저자 | Zhisheng Xiao, Karsten Kreis, Arash Vahdat |
| 학회/연도 | ICLR 2022 (Spotlight, arXiv comment 기준). arXiv v1 2021-12-15 |
| arXiv | https://arxiv.org/abs/2112.07804 |
| 공식 코드 | https://github.com/NVlabs/denoising-diffusion-gan (HEAD 200 확인) |
| 프로젝트 페이지 | https://nvlabs.github.io/denoising-diffusion-gan (HEAD 200 확인) |

## 2. 한 줄 요약

Diffusion model의 역과정 한 스텝을 Gaussian 대신 latent $`z`$를 받는 conditional GAN으로 모델링하여,
T=4 같은 큰 step에서도 다봉(multimodal) denoising 분포를 표현하게 만든 모델이다.

## 3. 문제의식

- 저자들이 말하는 generative learning trilemma: 샘플 품질, mode coverage, 빠른 샘플링을 동시에 만족하기 어렵다.
  GAN은 빠르고 품질이 좋지만 coverage가 약하고, diffusion은 품질과 coverage가 좋지만 수백–수천 NFE가 든다.
- 진단: diffusion이 느린 근본 원인은 denoising 분포 $`q(x_{t-1}\mid x_t)`$를 Gaussian으로 근사하는 데 있다.
  이 근사는 step이 매우 작거나 데이터 marginal 자체가 Gaussian일 때만 정당하다 (§3.1).
- step을 키우면 참 denoising 분포가 복잡하고 다봉적으로 변한다 (Fig. 2의 1D 예시).
  같은 noisy 이미지에 대응하는 그럴듯한 clean 이미지가 여러 개이기 때문이다.

## 4. 핵심 아이디어

Forward process는 DDPM과 같되, T를 작게(T ≤ 8, 실험은 T=4) 두고 각 $`\beta_t`$를 크게 잡는다.

```math
q(x_t\mid x_{t-1})=\mathcal N\big(x_t;\sqrt{1-\beta_t}\,x_{t-1},\,\beta_t I\big)
```

DDPM의 ELBO가 스텝별 KL의 합이라면, DDGAN은 이를 스텝별 adversarial divergence로 바꾼다.

```math
\min_\theta \sum_{t\ge 1}\mathbb E_{q(x_t)}\Big[D_{\mathrm{adv}}\big(q(x_{t-1}\mid x_t)\,\|\,p_\theta(x_{t-1}\mid x_t)\big)\Big]
```

non-saturating GAN을 쓰므로 $`D_{\mathrm{adv}}`$는 softened reverse KL에 해당한다. 원래 diffusion 학습의 forward KL과 방향이 다르다.

가장 중요한 설계는 parameterization이다. Generator가 $`x_{t-1}`$을 직접 내지 않고 $`x_0`$를 예측한 뒤, Gaussian posterior로 $`x_{t-1}`$을 뽑는다.

```math
p_\theta(x_{t-1}\mid x_t):=\int p(z)\,q\big(x_{t-1}\mid x_t,\,x_0=G_\theta(x_t,z,t)\big)\,dz,\qquad z\sim\mathcal N(0,I)
```

- $`q(x_{t-1}\mid x_t,x_0)`$는 step 크기와 무관하게 항상 Gaussian이다 (Appendix A의 $`\tilde\mu_t,\tilde\beta_t`$).
- DDPM도 같은 꼴로 해석되지만 $`x_0`$가 결정적($`f_\theta(x_t,t)`$)이다 (Appendix B). DDGAN은 $`z`$ 때문에 $`p_\theta(x_{t-1}\mid x_t)`$가 implicit 다봉 분포가 된다.
- real 쌍 $`(x_{t-1},x_t)`$는 $`q(x_0)\,q(x_{t-1}\mid x_0)\,q(x_t\mid x_{t-1})`$ 순서의 ancestral sampling으로 만든다. $`q(x_{t-1}\mid x_t)`$를 직접 뽑을 필요가 없다.

저자들의 설명(§3.2, Appendix D): 각 스텝은 $`x_t`$에 강하게 conditioning된 쉬운 문제이고, D가 noise로 smoothing된 분포를 보므로 overfitting이 줄어 학습이 안정적이다.

## 5. 학습 목적 함수

Discriminator $`D_\phi(x_{t-1},x_t,t)\in[0,1]`$ (Eq. 5, Appendix C.2):

```math
\min_\phi\sum_{t}\mathbb E_{q(x_0)q(x_{t-1}\mid x_0)q(x_t\mid x_{t-1})}\Big[-\log D_\phi(x_{t-1},x_t,t)+\mathbb E_{p_\theta(x_{t-1}\mid x_t)}\big[-\log\big(1-D_\phi(x_{t-1},x_t,t)\big)\big]\Big]
```

Generator (non-saturating):

```math
\max_\theta\sum_t\mathbb E_{q(x_t)}\,\mathbb E_{p_\theta(x_{t-1}\mid x_t)}\big[\log D_\phi(x_{t-1},x_t,t)\big]
```

Regularizer — real $`x_{t-1}`$에 대한 R1:

```math
R_1(\phi)=\frac{\gamma}{2}\,\mathbb E\Big[\big\|\nabla_{x_{t-1}}D_\phi(x_{t-1},x_t,t)\big\|^2\Big],\qquad \gamma=0.05\ (\text{CIFAR-10}),\ 1\ (\text{CelebA-HQ, LSUN Church})
```

학습 설정 (Appendix C.2, Table 8):

- $`\beta_t`$: VP SDE를 T=4로 등간격 이산화, $`\beta_{\min}=0.1`$, $`\beta_{\max}=20`$. $`t\in\{1,2,3,4\}`$를 데이터마다 무작위 샘플 (Ho et al.와 동일 방식)
- Adam $`(\beta_1,\beta_2)=(0.5,0.9)`$, lr D $`10^{-4}`$ / G $`1.6\times10^{-4}`$ (LSUN은 $`2\times10^{-4}`$), cosine decay
- G EMA 0.9999 (CIFAR-10), 0.999 (256px). 저자들은 EMA가 성능에 결정적이라고 서술
- batch 128 / 32 / 64, 400k / 750k / 600k iterations (CIFAR-10 / CelebA-HQ / LSUN Church)
- CIFAR-10은 V100 4장으로 약 48시간, 256px 두 데이터셋은 V100 8장으로 약 180시간

## 6. 구조

Generator (Appendix C.1, Table 6)

- NCSN++ U-Net (ResNet block + attention, FIR up/down-sampling, skip을 $`1/\sqrt2`$로 rescale). 입력은 $`x_t`$, 시간은 sinusoidal embedding
- $`z`$ 주입: StyleGAN식 mapping network(FC 3층) → $`w`$ → 모든 GroupNorm을 AdaGN으로 바꿔 채널별 shift·scale을 예측. mapping과 AdaGN은 $`t`$와 무관
- latent 차원 256 (CIFAR-10) / 100 (256px), embedding 512 / 256. Dropout은 쓰지 않음

Discriminator (Table 7)

- ResNet conv D. $`x_{t-1}`$과 $`x_t`$를 channel concat해서 넣고, $`t`$는 sinusoidal embedding으로 조건화
- LeakyReLU(0.2), 마지막 ResBlock 뒤 minibatch std layer → global sum pooling → FC scalar
- CIFAR-10: 1x1 conv 128 → ResBlock 128 → down 256 → down 512 → down 512. 256px는 down ResBlock 6개

## 7. 주요 결과

CIFAR-10 unconditional (Table 1, 시간은 V100 1장에서 100장 생성):

| Model | IS | FID | Recall | NFE | Time (s) |
|---|---|---|---|---|---|
| DDGAN, T=4 | 9.63 | 3.75 | 0.57 | 4 | 0.21 |
| DDPM | 9.46 | 3.21 | 0.57 | 1000 | 80.5 |
| Score SDE (VP) | 9.68 | 2.41 | 0.59 | 2000 | 421.5 |
| DDIM, T=50 | 8.78 | 4.67 | 0.53 | 50 | 4.01 |
| StyleGAN2 w/ ADA | 9.83 | 2.92 | 0.49 | 1 | 0.04 |
| SNGAN | 8.22 | 21.7 | 0.44 | 1 | - |

- 저자 주장: Song et al.의 predictor-corrector 샘플링 대비 약 2000배, FastDDPM(T=50) 대비 약 20배 빠르다.
- FID는 StyleGAN2-ADA보다 나쁘지만, Table 1의 GAN들은 Recall이 모두 0.5 미만인 반면 DDGAN은 0.57이다.

Ablation (Table 2, CIFAR-10):

| 변형 | IS | FID | Recall |
|---|---|---|---|
| T=1 (사실상 unconditional GAN) | 8.93 | 14.6 | 0.19 |
| T=2 | 9.80 | 4.08 | 0.54 |
| T=4 | 9.63 | 3.75 | 0.57 |
| T=8 | 9.43 | 4.36 | 0.56 |
| one-shot GAN + diffusion을 augmentation으로 | 8.96 | 13.2 | 0.25 |
| direct denoising ($`x_{t-1}`$ 직접 출력) | 9.10 | 6.03 | 0.53 |
| noise generation ($`\epsilon`$ 출력) | 8.79 | 8.04 | 0.52 |
| latent $`z`$ 제거 | 8.37 | 20.6 | 0.42 |

- 확인된 사실: 스텝 분할(T>1), $`x_0`$ parameterization, latent $`z`$ 셋 다 성능에 크게 기여한다. diffusion을 단순 augmentation으로 쓰는 것과는 다르다.
- StackedMNIST (Table 3): 1000개 mode 전부 포착, KL 0.071 (StyleGAN2는 940개, 0.424).
- CelebA-HQ 256 (Table 4): FID 7.64. LSUN Church 256 (Table 5): FID 5.25 (DDPM 7.89, StyleGAN2 3.86).
- Stroke-based synthesis: 256px 1장 0.16s, 비교 대상 Meng et al.(2021b)은 181s (약 1100배).

## 8. 한계

- 최고 FID는 여전히 최상위 diffusion(LSGM 2.10, Score SDE VE 2.20)과 StyleGAN2-ADA(2.92)에 못 미친다 (Table 1).
- 원래 diffusion보다도 FID가 나쁘다. DDGAN 3.75는 DDPM 3.21, Probability Flow (VP) 3.08, FastDDPM (T=50) 3.41보다 높다 (Table 1). 해석: 속도를 얻는 대신 FID를 일부 내준 것이다.
- T=8에서 오히려 성능이 약간 떨어진다. 저자들은 스텝마다 conditional GAN이 필요해 용량이 더 필요할 것이라고 가설만 제시한다.
- 학습 비용이 작지 않다 (CIFAR-10에도 V100 4장 약 48시간).
- 해석: GAN 학습이라 R1 $`\gamma`$, lr, EMA 같은 GAN 하이퍼파라미터 부담이 그대로 남고, likelihood 추정은 제공되지 않는다.
- 확인된 사실: 논문에 텍스트 조건부 실험이나 대규모 데이터 확장 실험은 없다 (가장 큰 설정은 256px CelebA-HQ·LSUN Church).
- 미확인: 이 방식이 텍스트 조건부·대규모로 확장되는지는 이 논문으로 판단할 수 없다. 후속 연구를 읽어야 한다.

## 9. 이 repo와의 연결

| DDGAN 구성 요소 | repo 대응 |
|---|---|
| non-saturating G loss + BCE D loss | `gan_common` `GANLoss("vanilla")` |
| real $`x_{t-1}`$에 대한 R1 | `gan_common` `r1_penalty` ($`\gamma/2`$ 곱은 호출자). 미분 대상은 $`x_{t-1}`$만이고 $`x_t`$는 조건 입력 |
| mapping network → AdaGN | 8.StyleGAN2의 mapping network 아이디어 (AdaIN 대신 AdaGN) |
| minibatch std layer | `gan_common` `MinibatchStdDev` (7.ProGAN, 8.StyleGAN2에서 사용) |
| G EMA | `gan_common` `EMA` |
| 입력 concat 조건부 D | 4.Pix2Pix의 `cat(A, B)` 조건부 D와 같은 방식 |

RTX 4070 Super 12GB에서의 소규모 재현 (모두 추정):

- 추정: 25-Gaussians toy(Appendix C.5 설정: G·D 각 FC 3층 512 hidden, batch 512, 50k iter)는 수 분 안에 끝난다.
- 추정: CIFAR-10 T=4를 채널을 줄인 U-Net(예: base 64)으로 학습하는 것은 12GB에 들어간다. 원 설정(base 128, batch 128, 400k iter)을 GPU 1장으로 끝까지 돌리면 수일 이상 걸린다.
- 추정: 학습 목적으로는 MNIST/CIFAR-10 소형 모델로 Table 2의 T=1 vs T=4, latent 유무 비교를 재현해 Recall 차이를 보는 것이 비용 대비 효과가 크다.
- 사전 지식: 이 repo에는 diffusion 폴더가 없으므로 DDPM의 forward process와 posterior $`q(x_{t-1}\mid x_t,x_0)`$를 먼저 익혀야 한다.

## 10. 참고

- Z. Xiao, K. Kreis, A. Vahdat. Tackling the Generative Learning Trilemma with Denoising Diffusion GANs. ICLR 2022. arXiv:2112.07804
- 논문이 기반으로 삼는 연구 (본문 인용 기준): Ho et al. 2020 (DDPM), Song et al. 2021 (Score SDE, NCSN++), Mescheder et al. 2018 (R1), Karras et al. 2019/2020 (StyleGAN, StyleGAN2, ADA), Goodfellow et al. 2014 (non-saturating GAN)
- 공식 구현: https://github.com/NVlabs/denoising-diffusion-gan
