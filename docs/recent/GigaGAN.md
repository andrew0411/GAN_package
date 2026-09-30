# GigaGAN — Scaling up GANs for Text-to-Image Synthesis

> 근거: arXiv 2303.05511v2 HTML의 본문과 부록 A–C 전체를 읽었다 (참고문헌 목록 제외).
> 그림 안에만 있는 수치(Fig. A3의 FID–CLIP 곡선 등)는 텍스트로 제공되지 않아 인용하지 않았다.

## 1. 서지 정보

| 항목 | 내용 |
|---|---|
| 제목 | Scaling up GANs for Text-to-Image Synthesis |
| 저자 | Minguk Kang, Jun-Yan Zhu, Richard Zhang, Jaesik Park, Eli Shechtman, Sylvain Paris, Taesung Park |
| 학회/연도 | CVPR 2023 (arXiv comment 기준). arXiv v1 2023-03-09 |
| arXiv | https://arxiv.org/abs/2303.05511 |
| 프로젝트 페이지 | https://mingukkang.github.io/GigaGAN/ (HEAD 200 확인) |
| 공식 코드 | 미확인: 논문과 arXiv 페이지에 코드 링크가 없다. 프로젝트 페이지 본문은 열람하지 않았으므로 학습 코드·가중치 공개 여부는 그 페이지를 읽어야 확정된다 |

## 2. 한 줄 요약

StyleGAN2를 단순히 키우면 불안정해지는 문제를 sample-adaptive kernel selection, attention, multi-scale input/output discriminator, matching-aware loss 등으로 풀어,
1B parameter GAN을 LAION 규모 text-to-image에 학습했다. 저자 주장: 수십억 장의 인터넷 이미지로 billion-scale GAN을 학습한 첫 사례다 (§1). 512px 이미지를 0.13초에 만든다.

## 3. 문제의식

- diffusion·autoregressive 모델은 iterative inference라 느리다. GAN은 single forward라 빠르지만 open-world 대규모 데이터로 확장된 적이 없다.
- StyleGAN2의 폭을 키워 파라미터를 5.7배(27.8M→158.9M)로 늘리면 FID가 오히려 나빠진다 (Table 1: 29.91 → 34.07).
- 저자 가설: 같은 conv filter가 모든 텍스트 조건과 모든 위치를 담당해야 하는 표현력 한계가 병목이다.
- 추가 관찰: 모델이 커지면 저해상도 초기 층이 거의 쓰이지 않아(dynamic range가 작음) 전역 구조와 image-text alignment가 약해진다 (§3.3).

## 4. 핵심 아이디어

(a) Sample-adaptive kernel selection (Eq. 1–2): filter bank $`N`$개를 style $`w`$로 섞은 뒤 StyleGAN2의 modulation/demodulation을 적용한다.

```math
K=\sum_{i=1}^{N}K_i\cdot\mathrm{softmax}\big(W_{\mathrm{filter}}^\top w+b_{\mathrm{filter}}\big)_i,\qquad g_{\mathrm{adaconv}}(f,w)=\big((W_{\mathrm{mod}}^\top w+b_{\mathrm{mod}})\otimes K\big)*f
```

선택은 층마다 한 번이라 연산량이 해상도와 분리된다.

(b) Attention과 conv의 교차 배치 (Eq. 5):

```math
f_{\ell+1}=g^{\ell}_{\mathrm{xa}}\Big(g^{\ell}_{\mathrm{attn}}\big(g^{\ell}_{\mathrm{adaconv}}(f_\ell,w),\,w\big),\ t_{\mathrm{local}}\Big)
```

- self-attention은 dot product 대신 L2 distance를 logit으로 써서 Lipschitz 성질을 돕는다. key·query 행렬 공유, weight decay, equalized lr, 작은 residual gain을 함께 쓴다. 단순히 attention을 넣으면 학습이 붕괴한다고 저자들은 서술한다.
- cross-attention은 feature를 query, 단어 임베딩을 key·value로 쓴다.

(c) 텍스트 조건 (Eq. 3–4): frozen CLIP ViT-L/14의 penultimate feature(77 token × 768) → 학습형 attention layer $`T`$ → $`t_{\mathrm{local}}`$(EOT 제외 단어들), $`t_{\mathrm{global}}`$(EOT). $`w=M(z,t_{\mathrm{global}})`$, $`z\in\mathbb R^{128}`$, $`x=\tilde G(w,t_{\mathrm{local}})`$.

(d) G가 image pyramid $`L=5`$ (64, 32, 16, 8, 4 px)를 출력하고 레벨마다 독립적으로 GAN loss를 받는다.

(e) 2-stage: 64px base generator + GAN upsampler(64→512, 또는 128→1024의 8배 모델).

## 5. 학습 목적 함수

Multi-scale input, multi-scale output loss (Eq. 6):

```math
\mathcal V_{\mathrm{MS\text{-}I/O}}(G,D)=\sum_{i=0}^{L-1}\sum_{j=i+1}^{L}\Big[\mathcal V_{\mathrm{GAN}}(G_i,D_{ij})+\mathcal V_{\mathrm{match}}(G_i,D_{ij})\Big]
```

- $`\mathcal V_{\mathrm{GAN}}`$: non-saturating loss (Table A2의 표기는 Logistic).
- Matching-aware loss (Eq. 8): real 이미지 $`x`$와 무작위 캡션 $`\hat c`$의 쌍, $`G(c)`$와 $`\hat c`$의 쌍을 모두 fake로 취급한다. G 쪽에도 거는 것이 핵심이다 (Table 1: D에만 27.29, G·D 모두 21.66).
- CLIP contrastive loss (Eq. 9):

```math
\mathcal L_{\mathrm{CLIP}}=\mathbb E_{\{c_n\}}\Big[-\log\frac{\exp\big(E_{\mathrm{img}}(G(c_0))^\top E_{\mathrm{txt}}(c_0)\big)}{\sum_n\exp\big(E_{\mathrm{img}}(G(c_0))^\top E_{\mathrm{txt}}(c_n)\big)}\Big]
```

- Vision-aided adversarial loss $`\mathcal L_{\mathrm{Vision}}`$: frozen CLIP image encoder(Table A2: ViT-B/32)의 중간 특징 + 3x3 conv head. 조건은 modulation으로 넣고, Projected GAN의 고정 random projection을 추가한다.
- 최종 목적: $`\mathcal V=\mathcal V_{\mathrm{MS\text{-}I/O}}+\mathcal L_{\mathrm{CLIP}}+\mathcal L_{\mathrm{Vision}}`$.
- Regularizer·최적화 (Table A2, T2I 64px): R1 strength 0.2048–2.048, R1 interval 16 (lazy), attention weight decay 0.01, AdamW lr 0.0025, $`\beta=(0,0.99)`$, G EMA 0.9999, batch 512–1024, 1350k iterations, A100 96–128장.
- Upsampler: 같은 loss에 LPIPS 추가 (64→512는 weight 10), vision-aided D는 쓰지 않고, real·생성 입력 간 차이를 줄이려 Gaussian noise augmentation을 건다.

## 6. 구조

Generator: mapping $`M`$ (4층) + synthesis network (해상도별로 adaconv → self-attn → cross-attn 블록을 여러 개 쌓음). style mixing과 path length regularization은 끈다 (StyleGAN-XL을 따름). T2I 64px base G는 652.5M.

Discriminator (§3.3, Fig. 5):

- text branch: CLIP + 학습형 attention, global descriptor $`t_D`$만 쓴다.
- image branch $`\phi`$: 층마다 self-attention + stride-2 conv, 출력 해상도 32, 16, 8, 4, 1.
- pyramid 레벨 $`x_i`$를 $`\phi`$ 중간으로 넣고(late entry) 이후 모든 스케일 $`j>i`$에서 예측한다(early exit). 총 $`L(L+1)/2=15`$개 예측이다.
- 예측 함수 (Eq. 7): $`D_{ij}(x,c)=\psi_j\big(\phi_{i\to j}(x_i),t_D\big)+\mathrm{Conv}_{1\times1}\big(\phi_{i\to j}(x_i)\big)`$. $`\psi_j`$는 4층 1x1 modulated conv, $`\mathrm{Conv}_{1\times1}`$은 무조건부 예측 skip이다.
- T2I 64px D 크기 381.4M (Table A2).

Upsampler: asymmetric U-Net (down residual block 3개 + attention을 가진 up block 6개, 같은 해상도 skip). 359.1M. base와 합쳐 약 1.0B.

## 7. 주요 결과

Table 1 (64px ablation, FID-10k, CLIP score는 ViT-B/32, Scale-up 행 외에는 100k steps·batch 256):

| 모델 | FID-10k | CLIP | # Param |
|---|---|---|---|
| StyleGAN2 | 29.91 | 0.222 | 27.8M |
| + Larger (5.7×) | 34.07 | 0.223 | 158.9M |
| + Tuned | 28.11 | 0.228 | 26.2M |
| + Attention | 23.87 | 0.235 | 59.0M |
| + Matching-aware D | 27.29 | 0.250 | 59.0M |
| + Matching-aware G and D | 21.66 | 0.254 | 59.0M |
| + Adaptive convolution | 19.97 | 0.261 | 80.2M |
| + Deeper | 19.18 | 0.263 | 161.9M |
| + CLIP loss | 14.88 | 0.280 | 161.9M |
| + Multi-scale training | 14.92 | 0.300 | 164.0M |
| + Vision-aided GAN | 13.67 | 0.287 | 164.0M |
| + Scale-up (GigaGAN) | 9.18 | 0.307 | 652.5M |

Table 2 (COCO2014 zero-shot FID-30k, GigaGAN은 512px 생성 후 256px로 줄여 평가):

| 모델 | 종류 | # Param | FID-30k | 추론 시간 |
|---|---|---|---|---|
| GigaGAN | GAN | 1.0B | 9.09 | 0.13s |
| SD-v1.5 (저자 재평가) | Diffusion | 0.9B | 9.62 | 2.9s |
| DALL·E 2 | Diffusion | 5.5B | 10.39 | - |
| Imagen | Diffusion | 3.0B | 7.27 | 9.1s |
| Muse-3B | AR | 3.0B | 7.88 | 1.3s |
| Parti-750M | AR | 750M | 10.71 | - |

- 학습 비용 (Table 2 캡션): GigaGAN 4,783 A100 GPU-days, SD-v1.5 6,250 A100 GPU-days.
- Table 3 (COCO2017, FID-5k): GigaGAN 1 step 21.1 / CLIP 0.32 / 0.13s. SD-distilled-4는 26.0 / 0.30 / 0.33s.
- Table 4 (text-conditioned 128→1024, LAION 10k장): FID-10k 1.54, 0.13s, 693M. SD Upscaler 9.39 (7.75s), Real-ESRGAN 8.60.
- Table 5 (ImageNet unconditional 64→256): FID-50k 1.2, IS 191.5, 1 step, 359M. SR3 5.2 (625M, 100 steps), LDM-4 2.8 (169M) / 2.4 (552M), LDM-4-G 4.4 (183M, 50 steps).
- Table A1 (ImageNet 256 class-conditional): FID 3.45, IS 225.52, 569M. StyleGAN-XL 2.32, ADM-G-U 4.01.
- 속도 (초록): 512px 0.13s, 16-megapixel(4096px) 3.66s.

## 8. 한계

- 저자들이 직접 DALL·E 2 수준의 사실감·compositionality에는 못 미친다고 밝힌다 (Fig. 9, Fig. A9–A14의 구조 오류: 침대 다리 수, 꽃병 대칭 등).
- 낮은 zero-shot COCO FID가 더 나은 시각 품질을 뜻하지 않을 수 있다고 저자들이 경고한다. ADD 논문 Appendix C도 같은 점을 지적한다.
- 해석: 보조 신호(CLIP loss, vision-aided D, matching loss)에 크게 기대며, 항목 간 trade-off가 있다 (Table 1: vision-aided D 추가 시 FID 14.92→13.67, CLIP 0.300→0.287).
- prompt 기반 style mixing은 단일·단순 객체에서만 동작한다 (Fig. A7 캡션).
- 학습 규모가 커서 학계 재현이 어렵다 (A100 96–128장, Table A2).

## 9. 이 repo와의 연결

| GigaGAN 구성 요소 | repo 대응 |
|---|---|
| mapping network, modulated conv, lazy R1, G EMA, truncation | 8.StyleGAN2, `gan_common` `r1_penalty` · `EMA` |
| minibatch std (upsampler와 ImageNet config만. T2I 64px base D는 끔, Table A2) | `gan_common` `MinibatchStdDev` |
| image pyramid 출력, 저해상도 입력을 중간 층에 주입 | 7.ProGAN의 toRGB/fromRGB·해상도별 블록 구조 |
| 조건부 D, matching-aware loss | 4.Pix2Pix의 조건부 D 개념 (GigaGAN은 concat이 아니라 modulation·projection으로 조건화) |
| vision-aided D, fixed random projection | repo에 없음. ADD·LADD의 projected discriminator와 같은 계열 |
| GAN upsampler + LPIPS | 4.Pix2Pix(image-conditional, GAN + L1)와 유사한 학습, 9.VQGAN의 perceptual loss |
| 텍스트 조건 truncation (Appendix C.1) | 8.StyleGAN2 truncation을 전역 평균과 조건 평균 두 번 lerp하도록 확장 |

RTX 4070 Super 12GB에서의 소규모 재현 (모두 추정):

- 추정: 1B 원 모델 재현은 불가능하다.
- 추정: 8.StyleGAN2 64px에 adaptive kernel selection($`N`$=4–8)과 L2 self-attention을 넣고 CIFAR-10 class-conditional로 Table 1식 ablation을 하는 것은 수십 M parameter 규모로 12GB에서 가능하다.
- 추정: MS-I/O D는 추가 예측이 주로 저해상도라 메모리 부담이 크지 않다. 이는 논문 §3.3의 서술에서 추론한 것이다.
- 추정: CLIP loss와 vision-aided D는 frozen CLIP을 추가로 올려야 하므로 batch를 줄여야 할 수 있다.

## 10. 참고

- M. Kang, J.-Y. Zhu, R. Zhang, J. Park, E. Shechtman, S. Paris, T. Park. Scaling up GANs for Text-to-Image Synthesis. CVPR 2023. arXiv:2303.05511
- 논문이 기반으로 삼는 연구 (본문 인용 기준): Karras et al. 2020 (StyleGAN2), Sauer et al. 2022 (StyleGAN-XL), Kumari et al. 2022 (Vision-aided GAN), Sauer et al. 2021 (Projected GAN), Karnewar & Wang 2020 (MSG-GAN), Kim et al. 2021 (L2 self-attention의 Lipschitz 성질), Miyato & Koyama 2018 (projection discriminator)
- 동시기 관련 연구: StyleGAN-T (Sauer et al. 2023), GALIP
