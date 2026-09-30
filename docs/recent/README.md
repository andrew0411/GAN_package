# Recent Adversarial Methods: Study Notes

Notes on five recent papers that build on the classic GAN techniques implemented in this repository (folders 0–12 and `gan_common`). The notes themselves are written in Korean; there is no code in this folder.

## Where adversarial training is used today

Since 2022, diffusion models have become the dominant approach to image generation, and GANs are rarely used as stand-alone generators. Adversarial losses now appear mainly as components of other generative models:

- **Few-step diffusion distillation.** When a pretrained diffusion model is distilled into a 1–4 step generator, regression or distillation losses alone produce blurry samples. ADD and LADD add a discriminator that pulls the student's outputs toward the target distribution. The reference distribution differs: ADD compares against real images, while LADD compares against synthetic samples produced by the teacher (for text-to-image; LADD §3, §4.2).
- **Hybrids of GANs and diffusion.** DDGAN places a conditional GAN inside each denoising step, while Diffusion-GAN places forward diffusion inside GAN training to stabilize the discriminator.
- **Large-scale text-to-image GANs.** GigaGAN scales the StyleGAN family to one billion parameters and shows the speed advantage of a single-forward-pass GAN upsampler.

All three directions rely on techniques implemented here: non-saturating and hinge losses, the R1 penalty, the StyleGAN2 backbone, and PatchGAN discriminators. They also use projected discriminators (a frozen pretrained feature network with trainable heads, from Projected GAN), which are not implemented in this repository; GigaGAN's vision-aided discriminator and the ADD/LADD discriminators belong to this family.

## Comparison

| Paper | Year / venue | Core idea | Role of the adversarial loss | Sampling steps | Discriminator |
|---|---|---|---|---|---|
| [DDGAN](DDGAN.md) | arXiv 2021-12, ICLR 2022 | Models each denoising step as a conditional GAN with latent $`z`$; predicts $`x_0`$ and samples $`x_{t-1}`$ from the Gaussian posterior | Sole training objective, replacing the per-step Gaussian KL (non-saturating loss + R1) | 4 ($`T = 4`$; FID 4.08 with $`T = 2`$) | Time-conditioned ResNet on the concatenation of $`x_{t-1}`$ and $`x_t`$, minibatch std |
| [Diffusion-GAN](Diffusion-GAN.md) | arXiv 2022-06, ICLR 2023 | Uses forward diffusion as instance noise on discriminator inputs and adapts the maximum step $`T`$ | Unchanged GAN objective; diffusion only alters the discriminator input | 1 | Backbone discriminator conditioned on timestep $`t`$ (fed to the StyleGAN2 D mapping input); the Projected GAN variant ignores $`t`$ |
| [GigaGAN](GigaGAN.md) | arXiv 2023-03, CVPR 2023 | Extends StyleGAN with sample-adaptive kernels, attention, a multi-scale input/output discriminator, a matching-aware loss, and a GAN upsampler | Main objective, with a CLIP contrastive loss and a vision-aided discriminator as auxiliaries (LPIPS added for the upsampler) | 1 (one forward pass each for the base model and the upsampler) | Text-conditioned multi-scale input/output discriminator (15 predictions) plus a frozen-CLIP vision-aided discriminator |
| [ADD](ADD_LADD.md) | arXiv 2023-11 | Fine-tunes a pretrained diffusion model into a 1–4 step generator with adversarial and score-distillation losses | The authors consider both losses essential: FID 20.8 with the adversarial loss alone, 315.6 with distillation alone, 20.6 combined (Table 1d) | 1–4 | Frozen DINOv2 ViT-S features with lightweight heads (Projected GAN style), text and image conditioning, hinge loss + R1 |
| [LADD](ADD_LADD.md) | arXiv 2024-03 | Replaces the discriminator features with the teacher's generative latent features and trains on teacher-generated data | Sole loss for text-to-image | 1–4 | Token features from each attention block of the frozen teacher (MMDiT) with 2D convolutional heads, conditioned on noise level and pooled CLIP embedding |

## Suggested reading order

| Step | Read | Related folders | Focus |
|---|---|---|---|
| 0 | Prerequisites | [0.Simple_GAN](../../0.Simple_GAN/), [1.DCGAN](../../1.DCGAN/), [3.WGAN-GP](../../3.WGAN-GP/), [6.SNGAN](../../6.SNGAN/), [8.StyleGAN2](../../8.StyleGAN2/) | Non-saturating and hinge losses, gradient penalty and R1, mapping network, EMA |
| 1 | [Diffusion-GAN](Diffusion-GAN.md) | 1.DCGAN, 6.SNGAN, 8.StyleGAN2 | Smallest change to an existing GAN: diffused discriminator inputs and the adaptive $`T`$ rule |
| 2 | [DDGAN](DDGAN.md) | [4.Pix2Pix](../../4.Pix2Pix/) (conditional D), 8.StyleGAN2 (mapping, R1) | Requires the DDPM forward process and posterior (not covered in this repository) |
| 3 | [GigaGAN](GigaGAN.md) | 8.StyleGAN2, [7.ProGAN](../../7.ProGAN/) (multi-scale), 4.Pix2Pix | Stabilizing large conditional GANs, GAN upsampler |
| 4 | [ADD → LADD](ADD_LADD.md) | 6.SNGAN (hinge), 8.StyleGAN2 (R1), [9.VQGAN](../../9.VQGAN/) (latent space) | Adversarial distillation of pretrained diffusion models, projected discriminators |

Relevant pieces of `gan_common`: `GANLoss("vanilla" | "hinge")`, `r1_penalty`, `spectral_norm`, `EMA`, `MinibatchStdDev`, and `networks.image2image.NLayerDiscriminator` (PatchGAN).

## Conventions in the notes

- Numbers are taken from the arXiv HTML version stated in each note, with the table or figure number. Values shown only in plots (user-study bars, curves) are not quoted.
- Labels: `미확인` marks an unverified fact together with what would confirm it; `추측` marks unverified information recalled from outside the paper (e.g. publication venue); `추정` marks the note author's estimate (e.g. feasibility on a 12 GB GPU); `해석` marks the note author's interpretation. Claims made by the paper's authors are attributed explicitly.
- Papers are summarized, not quoted.
