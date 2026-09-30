# ADD / LADD — Adversarial Diffusion Distillation과 그 latent 버전

> 근거: ADD는 arXiv 2311.17042v1 HTML, LADD는 arXiv 2403.12015v1 HTML의 본문과 부록을 전부 읽었다 (참고문헌 목록 제외).
> 두 논문의 user study 결과(ELO, 승률 막대)와 LADD의 CLIP score 곡선은 그림으로만 제공되어 수치를 인용하지 않았다.

## 1. 서지 정보

| 항목 | ADD | LADD |
|---|---|---|
| 제목 | Adversarial Diffusion Distillation | Fast High-Resolution Image Synthesis with Latent Adversarial Diffusion Distillation |
| 저자 | Axel Sauer, Dominik Lorenz, Andreas Blattmann, Robin Rombach | Axel Sauer, Frederic Boesel, Tim Dockhorn, Andreas Blattmann, Patrick Esser, Robin Rombach |
| 소속 | Stability AI | Stability AI |
| arXiv | https://arxiv.org/abs/2311.17042 (2023-11-28) | https://arxiv.org/abs/2403.12015 (2024-03-18) |
| 학회 | 미확인: arXiv 페이지에 journal ref·comment가 없다. 추측: ECCV 2024. DBLP나 학회 proceedings로 확인해야 한다 | 미확인: 위와 같다. 추측: SIGGRAPH Asia 2024. 같은 방법으로 확인해야 한다 |
| 코드·가중치 | 논문 명시: https://github.com/Stability-AI/generative-models, https://huggingface.co/stabilityai/ (둘 다 HEAD 200) | 논문은 공개 예정이라고만 쓰고 URL을 주지 않는다. 미확인 |
| 산출 모델 | ADD-M (860M), ADD-XL (3.1B, SDXL 기반) | SD3-Turbo (Stable Diffusion 3 8B 기반) |

미확인: ADD-XL과 공개 체크포인트 SDXL-Turbo(https://huggingface.co/stabilityai/sdxl-turbo, HEAD 200)의 대응은 ADD 본문에 적혀 있지 않다. LADD Fig. 10이 SDXL-Turbo를 4-step 비교군으로 쓸 뿐이다. 모델 카드를 읽어야 확정된다.

## 2. 한 줄 요약

- ADD: 사전학습 diffusion model을 student로 초기화하고, adversarial loss(frozen DINOv2 특징 위 판별기)와 score distillation loss(frozen teacher)를 함께 걸어 1–4 step 생성기로 만든다.
- LADD: ADD의 판별기를 teacher 자신의 생성 특징(latent 공간)으로 바꾸고 teacher가 만든 synthetic data로 학습해, distillation loss 없이 adversarial loss만으로 megapixel·multi-aspect 1–4 step 모델(SD3-Turbo)을 얻는다.

## 3. 문제의식

- ADD: 기존 few-step 방법(progressive distillation, LCM, InstaFlow 등)은 4 step 근처에서 흐릿함과 artifact가 생기고, step을 더 줄이면 악화된다. 반대로 순수 GAN(StyleGAN-T, GigaGAN)은 빠르지만 품질이 diffusion에 못 미치고 규모를 키우면 성능이 포화된다 (§2, §4.1).
- LADD가 지적하는 ADD의 한계 (§1):
  - frozen DINOv2 때문에 판별기 학습 해상도가 518×518로 묶인다.
  - 판별기 피드백을 전역 형태 대 국소 질감 쪽으로 조절할 수단이 없다.
  - latent diffusion을 distill할 때도 판별기가 RGB에서 동작하므로 decode가 필요하고, 이것이 $512^2$ 초과 학습을 가로막는다.

## 4. 핵심 아이디어

### 4.1 ADD

세 네트워크: student $\hat x_\theta$ (사전학습 DM으로 초기화), 학습형 판별기 head $\mathcal D_{\phi,k}$, frozen teacher $\psi$ (§3.1, Fig. 2).

- 입력은 real 이미지를 forward diffusion한 $x_s=\alpha_s x_0+\sigma_s\epsilon$. $s$는 student timestep 집합 $T_{\mathrm{student}}=\{\tau_1,\dots,\tau_N\}$ ($N=4$)에서 균등 샘플하고, $\tau_N=1000$과 zero-terminal SNR을 강제해 추론 시 순수 noise에서 출발할 수 있게 한다.
- 전체 목적 (Eq. 1): $\mathcal L=\mathcal L^{G}_{\mathrm{adv}}\big(\hat x_\theta(x_s,s),\phi\big)+\lambda\,\mathcal L_{\mathrm{distill}}\big(\hat x_\theta(x_s,s),\psi\big)$, $\lambda=2.5$.
- 추론 시 classifier-free guidance를 쓰지 않는다. 여러 step으로 결과를 다듬는 능력은 유지된다 (Fig. 4). 미확인: multi-step 추론의 재노이즈 절차는 본문에 명시되지 않았다.

### 4.2 LADD

- 판별기와 teacher를 하나로 합친다 (§3): 생성 latent를 $\hat t\sim\pi(t;m,s)$ (logit-normal)에서 뽑은 noise level로 재노이즈하고, teacher(MMDiT)에 통과시켜 각 attention block 뒤의 token sequence를 특징으로 쓴다. 각 특징에 독립 head를 붙인다.
- noise level이 판별기 성격을 정한다: 높은 noise는 전역 구조, 낮은 noise는 질감에 대한 피드백이 된다. $\pi(t;m=1,s=1)$을 채택 (§4.1, Fig. 4).
- synthetic data: teacher가 고정 CFG로 만든 latent를 "real"로 쓴다. 이 경우 distillation loss를 더해도 이득이 없어 adversarial loss만 남긴다 (§4.2, Fig. 5).
- 전 과정이 latent 공간이라 decode·encode가 필요 없다.
- 해석(논문에 식이 없어 서술을 옮긴 것): $\hat t=\operatorname{sigmoid}(m+s\,u)$, $u\sim\mathcal N(0,1)$; $\hat x_{\hat t}=(1-\hat t)\,\hat x_\theta+\hat t\,\varepsilon'$ (rectified flow, §2.1); 판별기 출력은 $\sum_k\mathcal D_{\phi,k}\big(F^{\psi}_k(\hat x_{\hat t},\hat t);\ \hat t,\ c_{\mathrm{pool}}\big)$.
  미확인: adversarial loss의 형태(hinge 여부)와 R1 사용 여부는 본문에 없다. 판별기 설계는 StyleGAN-T와 ADD를 대부분 따른다고만 서술한다. real(합성) latent도 같은 방식으로 재노이즈하는지도 명시되지 않았다.

### 4.3 ADD vs LADD 비교

| 항목 | ADD | LADD |
|---|---|---|
| student / teacher | SD1.5·SD2.1·SDXL U-Net / frozen DM | SD3 MMDiT / 같은 계열 frozen MMDiT (data generator 겸용) |
| 학습 공간 | pixel 기준 (distill loss를 pixel에서 계산, 판별기도 RGB) | latent만 |
| 판별기 특징망 | frozen DINOv2 ViT-S (discriminative 특징) | frozen teacher (generative 특징, attention block마다) |
| head 구조 | StyleGAN-T 설계를 따름 (LADD 서술상 1D conv) | token을 공간 배치로 되돌린 뒤 2D conv (multi-aspect 대응) |
| 판별기 조건 | $c_{\mathrm{text}}$ (CLIP ViT-g-14 text) + $c_{\mathrm{img}}$ (DINOv2 ViT-L CLS) | noise level + pooled CLIP embedding |
| 학습 데이터 | real 이미지 | teacher의 synthetic latent (고정 CFG) |
| 보조 loss | score distillation ($\lambda=2.5$, 최종 모델은 NFSD weighting) | T2I에서는 없음. synthetic data를 못 쓰는 inpainting에서만 distill loss 추가 |
| adversarial loss | hinge + R1 ($\gamma=10^{-5}$, head 입력 특징에 대해) | 미확인 |
| 판별기 성격 조절 | 수단 없음 (LADD의 지적) | teacher noise 분포 $\pi(t;m,s)$ |
| 해상도 | $512^2$ 평가 (DINOv2 제약 518) | 최대 $1024^2$, multi-aspect |
| 추론 | 1–4 step, CFG 없음 | 1–4 step, unguided, 2·4 step은 consistency sampler |
| 대표 결과 | ADD-XL 4 step이 SDXL 50 step을 user study 다수 비교에서 이김 | SD3-Turbo 4 step이 SD3 50 step과 이미지 품질 동급, prompt alignment는 약간 낮음 |

## 5. 학습 목적 함수

ADD (Eq. 2–4). $F_k$는 frozen 특징망의 $k$번째 층 특징, $\mathrm{sg}$는 stop-gradient.

$$\mathcal L^{G}_{\mathrm{adv}}=-\mathbb E_{s,\epsilon,x_0}\Big[\sum_k\mathcal D_{\phi,k}\big(F_k(\hat x_\theta(x_s,s))\big)\Big]$$

$$\mathcal L^{D}_{\mathrm{adv}}=\mathbb E_{x_0}\Big[\sum_k\max\big(0,1-\mathcal D_{\phi,k}(F_k(x_0))\big)+\gamma R_1(\phi)\Big]+\mathbb E_{\hat x_\theta}\Big[\sum_k\max\big(0,1+\mathcal D_{\phi,k}(F_k(\hat x_\theta))\big)\Big]$$

$$\mathcal L_{\mathrm{distill}}=\mathbb E_{t,\epsilon'}\Big[c(t)\,\big\|\hat x_\theta-\hat x_\psi\big(\mathrm{sg}(\hat x_{\theta,t});t\big)\big\|_2^2\Big],\qquad \hat x_{\theta,t}=\alpha_t\hat x_\theta+\sigma_t\epsilon',\quad \hat x_\psi=\frac{\hat x_{\theta,t}-\sigma_t\hat\epsilon_\psi(\hat x_{\theta,t},t)}{\alpha_t}$$

- teacher에는 student 출력을 그대로 넣지 않고 다시 diffuse해서 넣는다. 깨끗한 입력은 teacher에게 분포 밖이기 때문이다.
- $c(t)$: exponential $c(t)=\alpha_t$, SDS weighting, NFSD 세 가지를 비교. $c(t)=\frac{\alpha_t}{2\sigma_t}w(t)$로 두면 이 loss의 gradient가 SDS와 같아진다 (Appendix A).
- R1은 픽셀이 아니라 각 head의 입력 특징에 대해 계산하며, $128^2$ 초과 해상도에서 특히 도움이 된다고 서술한다.
- LADD의 목적 함수는 §4.2 참조 (식 미제시).

## 6. 구조

- ADD student: SD2.1(ablation), SD1.5(비교 실험) 기반 ADD-M 860M, SDXL 기반 ADD-XL 3.1B. 모든 평가는 $512^2$.
- ADD 판별기: Projected GAN 계열 (LADD §3이 ADD가 Projected GAN paradigm을 쓴다고 명시하며, ADD §3.2의 구조도 이와 같다). frozen ViT 특징망 + 여러 층의 경량 head. projection 방식으로 텍스트·이미지 embedding을 조건화한다. $x_0$ 정보가 들어 있는 $\tau<1000$ 입력에서는 이미지 조건이 student가 입력을 활용하도록 유도한다.
- LADD 학습 기본값 (§4): student·teacher·data generator 모두 MMDiT depth 24 (약 2B), 10k iterations. 최종 모델은 8B.
- LADD 학습 timestep (§5): $t\in\{1,0.75,0.5,0.25\}$. $512^2$ 초과 해상도는 처음 500 iteration 동안 낮은 noise만 쓰고($p=[0,0,0.5,0.5]$) 이후 $p=[0.7,0.1,0.1,0.1]$로 바꾼다. multi-aspect는 bucketing(binning).
- LADD + DPO (§4.5): teacher에 rank 256 LoRA를 붙여 Diffusion-DPO로 3k iteration 미세조정 → 이 모델로 LADD → student에 같은 DPO-LoRA를 다시 적용.

## 7. 주요 결과

ADD ablation (Table 1, COCO zero-shot FID-5k / CLIP score, student 1 step, 4000 iterations, batch 128):

| 실험 | 설정 → FID / CS |
|---|---|
| (a) 특징망 | DINOv2 ViT-S 20.6 / 0.319 (최고), DINOv1 ViT-S 21.5 / 0.312, DINOv2 ViT-L 24.0 / 0.302, CLIP ViT-L 23.3 / 0.308 |
| (b) 판별기 조건 | 없음 21.2 / 0.302, text 21.2 / 0.307, image 21.1 / 0.316, 둘 다 20.6 / 0.319 |
| (c) student 초기화 | random 293.6 / 0.065, pretrained 20.6 / 0.319 |
| (d) loss | adv만 20.8 / 0.315, distill만 315.6 / 0.076, adv+exp 20.6 / 0.319, adv+SDS 22.3 / 0.325, adv+NFSD 21.8 / 0.327 |
| (e) student / teacher | SD2.1/SD2.1 20.6 / 0.319, SD2.1/SDXL 21.3 / 0.321, SDXL/SD2.1 29.3 / 0.314, SDXL/SDXL 28.41 / 0.325 |
| (f) teacher step 수 | 1: 20.6 / 0.319, 2: 20.8 / 0.321, 4: 20.3 / 0.317 |

- 저자 입장 (§4.1): 두 loss가 모두 필수다. distillation loss 단독은 효과가 없고(315.6 / 0.076), adversarial loss와 결합하면 개선된다(adv 단독 20.8 / 0.315 → 결합 20.6 / 0.319).
- 해석: 수치상 FID를 끌어내리는 것은 대부분 adversarial loss이므로 품질의 주 신호로 볼 수 있다. distillation loss의 기여는 FID보다 CLIP score 쪽에서 더 보인다 (SDS·NFSD weighting 행).
- 확인된 사실: 사전학습 초기화 없이는 학습이 되지 않는다 (Table 1c).
- Table 2 (SD1.5 기반, COCO FID-5k): ADD-M 1 step 0.09s, FID 19.7, CLIP 0.326. InstaFlow-0.9B 23.4 / 0.304, UFOGen 22.5 / 0.311, Progressive Distillation 1 step 37.2 / 0.275, DPM Solver 25 step 20.1 / 0.318 (0.88s).
- user study (Fig. 5–7, PartiPrompts 100개): ADD-XL 1 step은 SDXL을 제외한 비교군을 이기고 LCM-XL 4 step보다 낫다. ADD-XL 4 step은 Fig. 6 캡션 기준 teacher SDXL(50 step)을 포함한 모든 비교군을 이기며, 본문은 SDXL 대비 다수 비교에서 이긴다고 조금 더 약하게 표현한다.
- LADD (§4): synthetic data가 real data보다 image-text alignment가 확연히 좋다. 동일 student에서 LCM보다 크게 낫고, LCM은 하이퍼파라미터에 매우 민감했다. scaling에서는 student 크기가 teacher나 data generator 크기보다 훨씬 중요하다.
- LADD의 synthetic data 근거 (§3): COCO 실제 이미지의 평균 CLIP score 0.29, SD3가 COCO prompt로 만든 이미지 0.35.
- LADD DPO: DPO-LoRA 재적용 student가 비DPO student에 대해 1 step human preference 승률 56%.
- LADD user study (Fig. 9–10, PartiPrompts 128개, 모델당 4장): SD3-Turbo 1 step이 모든 비교군보다 낫고, 4 step은 SD3와 이미지 품질이 같고 Midjourney v6도 이긴다.
- LADD inpainting (Fig. 13 오른쪽, COCO, FID / LPIPS): SD3-inpainting Turbo 1 step 9.44 / 0.3416, teacher SD3-inpainting 50 step 8.94 / 0.3465, SD1.5-inpainting 10.29 / 0.3879, LaMa 27.21 / 0.3137.

## 8. 한계

- ADD: 샘플 다양성이 teacher보다 낮다 (Fig. 8 캡션). student가 teacher 성향을 물려받아, SDXL 기반은 FID가 높게 나온다 (Table 1e). 최적 $c(t)$ 선택은 열린 문제로 남긴다.
- ADD 구조적 제약 (LADD의 지적): 판별기 해상도 518 제한, RGB decode 필요.
- LADD: prompt alignment가 teacher보다 떨어진다. 객체 병합·중복, 세밀한 공간 배치, 부정문 처리에서 실패한다 (§6, Fig. 15).
- LADD 편집 모델은 이미지·텍스트 guidance 강도를 조절할 수 없고, 입력에 지나치게 붙어 큰 변경이 어렵다.
- 두 논문 모두 주 비교를 user study로 하며 결과 수치가 그림에만 있다. 해석: 독립 재현·비교가 어렵다.
- 공통 전제: 강력한 사전학습 diffusion model이 필요하다. 해석: 처음부터 GAN을 학습하는 문제를 푼 것이 아니라, 사전학습 DM을 adversarial fine-tuning하는 문제로 바꾼 것이다.

## 9. 이 repo와의 연결

| 구성 요소 | repo 대응 |
|---|---|
| hinge loss (ADD) | `gan_common` `GANLoss("hinge")`, 6.SNGAN. head 여러 개면 head별 loss를 합산 |
| R1 on head 입력 특징 (ADD) | `gan_common` `r1_penalty(real_logits, real_images)`에 이미지 대신 `requires_grad_(True)`인 특징 텐서를 넘기면 된다 |
| 여러 층의 경량 head가 공간별 logits를 냄 | 4.Pix2Pix PatchGAN(`NLayerDiscriminator`)이 logits map을 내는 것과 유사 (해석). `GANLoss`는 임의 shape logits를 받는다 |
| projection 조건화 | repo에 없음 (Miyato & Koyama 2018) |
| latent 공간 학습 (LADD) | 9.VQGAN의 first-stage autoencoder가 latent 공간 개념의 출발점 |
| 사전학습 generator 초기화 | repo에 사전학습 diffusion model이 없으므로 teacher를 먼저 만들어야 한다 |

RTX 4070 Super 12GB에서의 소규모 재현 (모두 추정):

- 추정: 논문 설정(ADD-M 860M student + 같은 크기 frozen teacher + DINOv2 ViT-S/ViT-L + CLIP ViT-g text encoder, $512^2$, batch 128)은 12GB 단일 GPU에서 불가능에 가깝다. LoRA·gradient checkpointing·작은 batch로 흉내 낼 수는 있으나 논문과 다른 실험이 된다.
- 추정: 학습용 경로는 MNIST/CIFAR-10 32px에서 (1) 소형 pixel DDPM teacher를 학습하고 (2) 같은 가중치로 student를 초기화한 뒤 (3) 작은 frozen 특징망 + 경량 head, hinge + R1, $\lambda=2.5$, student timestep 4개($\tau_N$=최대 t)로 1·2·4 step을 비교하는 것이다. 12GB에서 충분하다.
- 추정: LADD 흉내는 9.VQGAN latent 위에 소형 latent diffusion teacher를 먼저 학습해야 해서 선행 비용이 크다. teacher 중간 특징에 head를 붙이는 부분 자체는 가볍다.

## 10. 참고

- A. Sauer, D. Lorenz, A. Blattmann, R. Rombach. Adversarial Diffusion Distillation. arXiv:2311.17042, 2023
- A. Sauer, F. Boesel, T. Dockhorn, A. Blattmann, P. Esser, R. Rombach. Fast High-Resolution Image Synthesis with Latent Adversarial Diffusion Distillation. arXiv:2403.12015, 2024
- 두 논문이 기반으로 삼는 연구 (본문 인용 기준): Sauer et al. 2021 (Projected GAN), Sauer et al. 2023 (StyleGAN-T), Mescheder et al. 2018 (R1), Lim & Ye 2017 (hinge loss), Poole et al. 2022 (SDS, DreamFusion), Katzir et al. 2023 (NFSD), Lin et al. 2023 (zero-terminal SNR), Oquab et al. (DINOv2), Luo et al. 2023 (LCM), Song et al. 2023 (consistency models), Esser et al. 2024 (SD3, rectified flow transformer), Wallace et al. 2023 (Diffusion-DPO)
