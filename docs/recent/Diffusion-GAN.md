# Diffusion-GAN — Training GANs with Diffusion

> 근거: arXiv 2206.02262v4 HTML의 본문과 부록 A–L 전체를 읽었다 (참고문헌 목록 제외).

## 1. 서지 정보

| 항목 | 내용 |
|---|---|
| 제목 | Diffusion-GAN: Training GANs with Diffusion |
| 저자 | Zhendong Wang, Huangjie Zheng, Pengcheng He, Weizhu Chen, Mingyuan Zhou |
| 학회/연도 | ICLR 2023 (arXiv comment: camera ready version). arXiv v1 2022-06-05 |
| arXiv | https://arxiv.org/abs/2206.02262 |
| 공식 코드 | https://github.com/Zhendong-Wang/Diffusion-GAN (HEAD 200 확인) |

## 2. 한 줄 요약

Forward diffusion을 D 입력용 instance noise(미분 가능한 augmentation)로 쓰고, D를 timestep-conditional로 만들며,
diffusion 최대 길이 $`T`$를 D의 overfitting 정도에 따라 자동 조절하는 GAN 학습법이다. reverse chain이 없으므로 샘플링은 보통 GAN과 같은 1회 forward다.

## 3. 문제의식

- real/fake 분포의 support가 겹치지 않으면 JS divergence가 상수($`\log 2`$)가 되어 G에 쓸모 있는 gradient가 없다 (Appendix D의 toy: $`x=(0,z)`$ vs $`x_g=(\theta,z)`$).
- instance noise는 이 문제의 이론적 처방으로 알려져 있었지만, 고차원 이미지에서 적절한 noise 분포를 찾기 어려워 실제 성공 사례가 없었다는 것이 저자들의 문제 제기다.
- ADA·DiffAug 같은 augmentation은 도메인 지식이 필요하고 leakage 위험이 있으며, 데이터가 충분히 큰 경우 오히려 성능을 해치기도 한다 (Table 1의 LSUN 결과).

## 4. 핵심 아이디어

Diffusion-induced mixture로 real $`x`$와 fake $`x_g=G(z)`$를 똑같이 흐린다.

```math
q(y\mid x)=\sum_{t=1}^{T}\pi_t\,q(y\mid x,t),\qquad q(y\mid x,t)=\mathcal N\big(y;\ \sqrt{\bar\alpha_t}\,x,\ (1-\bar\alpha_t)\,\sigma^2 I\big)
```

reparameterization $`y_g=\sqrt{\bar\alpha_t}\,G_\theta(z)+\sqrt{1-\bar\alpha_t}\,\sigma\epsilon`$ 덕분에 D의 gradient가 diffusion을 거쳐 G까지 흐른다.

Adaptive diffusion (Eq. 5): D가 real을 과신하면 $`T`$를 늘려 과제를 어렵게 만든다.

```math
r_d=\mathbb E_{y,t}\big[\mathrm{sign}\big(D_\phi(y,t)-0.5\big)\big],\qquad T\leftarrow T+\mathrm{sign}(r_d-d_{\mathrm{target}})\cdot C
```

- $`r_d`$는 ADA(Karras et al. 2020a)의 overfitting 지표와 같다. 4 minibatch마다 갱신하고 $`[T_{\min},T_{\max}]`$로 자른다.
- $`t`$ 분포 $`p_\pi`$ (Eq. 6): uniform $`1/T`$ 또는 priority $`t/\sum_{s=1}^{T}s`$ (큰 $`t`$에 가중).
- exploration list $`t_{epl}`$: 64칸 = 0이 32개 + $`p_\pi`$에서 뽑은 32개. $`T`$를 갱신할 때만 다시 뽑아 급격한 변화를 막는다 (Algorithm 1).

이론 (§3.4, Appendix B):

- Theorem 1: 모든 $`t`$에서 noisy real/fake 분포 사이의 f-divergence가 $`\theta`$에 대해 연속·미분 가능하다.
- Theorem 2 (non-leaking): $`y=f(x)+h(\epsilon)`$ 꼴이고 $`f,h`$가 일대일이면 $`p(y)=p_g(y)\Leftrightarrow p(x)=p_g(x)`$. Gaussian diffusion이 이 조건을 만족하므로 augmentation이 생성 분포로 새지 않는다.
- Eq. 4: $`(y,t)`$ 결합 분포의 JSD는 $`t`$에 대한 조건부 JSD의 기댓값과 같다 (Appendix C).
- 해석: Theorem 1은 gradient의 존재만 보장하고 크기는 보장하지 않는다. 저자들도 toy에서 특정 $`t`$(예: $`t=200`$)에 평평한 구간이 있음을 보이고, 여러 $`t`$의 mixture로 이를 피한다고 설명한다 (Fig. 2).

## 5. 학습 목적 함수

```math
V(G,D)=\mathbb E_{x\sim p(x),\,t\sim p_\pi,\,y\sim q(y\mid x,t)}\big[\log D_\phi(y,t)\big]+\mathbb E_{z,\,t\sim p_\pi,\,y_g\sim q(y\mid G_\theta(z),t)}\big[\log\big(1-D_\phi(y_g,t)\big)\big]
```

- D는 $`V`$를 최대화, G는 최소화한다 (Algorithm 1). 논문이 밝히는 것은 backbone의 네트워크 구조와 학습 하이퍼파라미터(config)를 그대로 상속한다는 것뿐이다 (§4, Appendix G).
  미확인: 실제 G loss 형태(Eq. 3의 saturating 형태인지, backbone의 non-saturating 형태인지)는 본문에 없다. 공식 코드로 확인해야 한다.
- 추가 하이퍼파라미터는 4개: $`\sigma`$, $`T_{\max}`$, $`d_{\mathrm{target}}`$, $`p_\pi`$. 새 데이터셋 권장 시작점은 $`\sigma=0.05`$, $`T_{\max}=500`$, $`d_{\mathrm{target}}=0.6`$, uniform.
- pixel 주입 (Table 5): $`\beta`$ 선형 $`10^{-4}\to0.02`$, $`T\in[5,1000]`$ (priority) 또는 $`[5,500]`$ (uniform), $`\sigma=0.05`$ (픽셀 범위 $`[-1,1]`$).
- feature 주입 (ProjectedGAN): $`\beta_T=0.01`$, $`T\in[5,500]`$, $`\sigma=0.5`$. $`d_{\mathrm{target}}`$은 데이터셋별 0.2–0.6 (Table 4).
- StyleGAN2 계열 $`d_{\mathrm{target}}=0.6`$. 0.8은 Diffusion StyleGAN2의 FFHQ에만 쓰고, Diffusion InsGen의 FFHQ는 0.6이다 (Table 6).

## 6. 구조

- 네트워크 구조는 backbone(StyleGAN2, ProjectedGAN, InsGen)을 그대로 쓴다. 새로 넣는 것은 diffusion sampling 파이프라인과 $`t`$ 조건뿐이다 (Appendix H).
- Diffusion StyleGAN2 / Diffusion InsGen: D 안의 class-label용 mapping network 입력을 $`t`$로 바꿔 $`D_\phi(y,t)`$를 만든다. unconditional 학습에서는 원래 쓰이지 않던 경로다.
- Diffusion ProjectedGAN: 구현 단순화를 위해 $`t`$를 무시한 $`D_\phi(y)`$. §4.2와 Table 5는 noise를 픽셀이 아니라 EfficientNet 특징 벡터에 주입한다고 쓴다 (domain-agnostic 실험).
  논문 내부 불일치: Appendix H는 같은 모델에서 D가 diffuse된 이미지 $`y`$를 입력받는다고 쓴다. 미확인: 실제 주입 위치는 공식 코드로 확인해야 한다.
- 25-Gaussians: G·D 모두 128-unit hidden layer 2개 + LeakyReLU인 MLP.

## 7. 주요 결과

Table 1 (FID / Recall, 25M images 학습. CIFAR-10의 비교군 수치는 기존 논문 인용):

| Dataset | StyleGAN2 | + DiffAug | + ADA | Diffusion StyleGAN2 |
|---|---|---|---|---|
| CIFAR-10 (32) | 8.32 / 0.41 | 5.79 / 0.42 | 2.92 / 0.49 | 3.19 / 0.58 |
| CelebA (64) | 2.32 / 0.55 | 2.75 / 0.52 | 2.49 / 0.53 | 1.69 / 0.67 |
| STL-10 (64) | 11.70 / 0.44 | 12.97 / 0.39 | 13.72 / 0.36 | 11.43 / 0.45 |
| LSUN-Bedroom (256) | 3.98 / 0.32 | 4.25 / 0.19 | 7.89 / 0.05 | 3.65 / 0.32 |
| LSUN-Church (256) | 3.93 / 0.39 | 4.66 / 0.33 | 4.12 / 0.18 | 3.17 / 0.42 |
| FFHQ (1024) | 4.41 / 0.42 | 4.46 / 0.41 | 4.47 / 0.41 | 2.83 / 0.49 |

- FID는 6개 중 5개에서 최고이고, CIFAR-10에서만 ADA(2.92)가 더 낫다. Recall은 모두 최고이거나 동률이다 (LSUN-Bedroom은 StyleGAN2와 같은 0.32).
- Table 2 (ProjectedGAN → Diffusion ProjectedGAN, FID): CIFAR-10 3.10→2.54, STL-10 7.76→6.91, LSUN-Bedroom 2.25→1.43, LSUN-Church 3.42→1.85.
- Table 3 (적은 데이터, InsGen → Diffusion InsGen, FID): FFHQ-200 102.58→63.34, FFHQ-1k 34.90→30.91, FFHQ-5k 9.89→8.48, AFHQ-Cat 2.60→2.40, Dog 5.44→4.83, Wild 1.77→1.51.
- Table 8 (Appendix J, CIFAR-10 FID): DCGAN 28.65→24.67, SNGAN 20.76→17.23.
- Table 9: Diffusion StyleGAN2의 CIFAR-10 IS 9.94, NFE 1.
- 비용 (§4.1, CIFAR-10, V100 4장, 이미지 4k장당): StyleGAN2 8.0s, StyleGAN2-ADA 9.8s, Diffusion StyleGAN2 9.5s. 추론 비용 증가는 없다.
- Table 7 (mixing ablation, FID): priority vs uniform = CIFAR-10 3.19 vs 3.44, STL-10 11.43 vs 11.75, FFHQ 3.22 vs 2.83.
- Fig. 7 (Appendix I): 적응형 $`T`$가 고정 $`T`$보다 FID가 빨리, 더 낮게 수렴한다 (곡선 수치는 텍스트로 제공되지 않음).

## 8. 한계

- 샘플링은 GAN만큼 빠르지만, 품질 상한은 backbone GAN에 묶이고 diffusion model식 iterative refinement는 없다.
- 최적 mixing 방식이 데이터셋마다 다르다 (Table 7). 추가 하이퍼파라미터 4개를 새로 튜닝해야 한다.
- ProjectedGAN 변형은 $`t`$를 무시하므로 이론이 가정하는 timestep-dependent D와 다르다.
- 논문 내부 불일치: Fig. 10 캡션의 FFHQ 수치(FID 3.71, Recall 0.43)가 Table 1(2.83, 0.49)과 다르다. 본문에 설명이 없다.
  3.71은 Table 7의 priority(3.22)와 uniform(2.83) 어느 쪽과도 맞지 않는다. Fig. 8·9 캡션 수치는 Table 2·3과 일치하므로 어긋나는 것은 Fig. 10뿐이다.
  미확인: 어느 설정의 결과인지는 공식 repo의 설정 파일이나 로그로 확인해야 한다.
- 해석: Theorem 2는 분포 일치의 동치 관계만 말하고, 유한 데이터·유한 용량에서의 수렴 속도는 다루지 않는다.

## 9. 이 repo와의 연결

다섯 논문 중 가장 적은 코드로 기존 폴더에 붙일 수 있다. 필요한 것은 (1) D 직전에 real·fake를 같은 $`t`$ 분포로 diffuse, (2) D에 $`t`$ 전달, (3) $`r_d`$ 기반 $`T`$ 갱신 세 가지다.

| 대상 | 적용 방법 |
|---|---|
| 1.DCGAN, 6.SNGAN | Appendix J에서 직접 검증된 backbone. 추정: D에 $`t`$ embedding(예: projection 또는 채널 bias)을 추가하면 된다. Appendix J는 이 두 모델에 $`t`$를 어떻게 넣었는지 설명하지 않는다 |
| 8.StyleGAN2 | 논문 방식대로 D의 mapping network 입력을 $`t`$로 바꾼다 |
| `gan_common` `GANLoss` | Eq. 3은 `"vanilla"` 형태. `"hinge"`와의 조합은 가능하지만 논문에서 검증되지 않았다 |
| `gan_common` `r1_penalty` | 추정: StyleGAN2 계열 실험은 R1을 쓴다. Table 6 캡션이 'stl' config의 gamma를 0.01로 바꿨다고 쓰는데, StyleGAN2-ADA config의 gamma는 R1 계수다. Appendix H는 최적화 전반에서 $`x`$ 대신 $`(y,t)`$를 쓴다고만 서술한다. 미확인: R1의 입력이 $`y`$인지는 공식 코드로 확인해야 한다 |
| 10.AnoGAN, 12.TadGAN | 저자들은 저차원 벡터·특징 공간에도 적용 가능하다고 주장한다 (25-Gaussians, feature 주입). 시계열 critic에 적용하는 것은 추정이며 논문 범위 밖이다 |

RTX 4070 Super 12GB에서의 소규모 재현 (모두 추정):

- 추정: CIFAR-10 32px에서 DCGAN/SNGAN 대 Diffusion 변형 비교(Table 8 재현)는 12GB로 충분하고 수 시간 단위다.
- 추정: 25-Gaussians toy는 CPU로도 가능하며 mode collapse 억제를 눈으로 확인하기 좋다.
- 추정: 8.StyleGAN2 64px Diffusion 변형도 메모리상 가능하지만, 논문처럼 25M images를 보려면 수일이 걸린다.

## 10. 참고

- Z. Wang, H. Zheng, P. He, W. Chen, M. Zhou. Diffusion-GAN: Training GANs with Diffusion. ICLR 2023. arXiv:2206.02262
- 논문이 기반으로 삼는 연구 (본문 인용 기준): Arjovsky & Bottou 2017, Sønderby et al. 2017 (instance noise), Roth et al. 2017, Mescheder et al. 2018 (gradient penalty·R1), Karras et al. 2020a (ADA), Zhao et al. 2020 (DiffAug), Sauer et al. 2021 (ProjectedGAN), Yang et al. 2021 (InsGen)
- 공식 구현: https://github.com/Zhendong-Wang/Diffusion-GAN
