# GAN_package
 Base codes for generative models

### 0. 가장 기본적인 Generative Adversarial Model

 - 생성자(Generator)와 구분자 (Discriminator)로 구성된 기본적인 형태
 - Adversarial training을 기반으로 서로 다른 목적 함수를 가지고 minmax game 을 수행

### 1. DCGAN: Deep Convolution GAN
- [Radford, A., Metz, L., & Chintala, S. (2015). Unsupervised representation learning with deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434.]

 - GAN의 일종으로, generator와 discriminator가 CNN으로 구성되었다는 특징 가짐
 - Image generation, image super-resolution, text2img synthesis 등의 task 를 수행할 수 있게 해줌

<img src="./resources/DCGAN_1.PNG"  width="1000" height="500">

### 2. WGAN: Wasserstein GAN
- [Arjovsky, M., Chintala, S., & Bottou, L. (2017, July). Wasserstein generative adversarial networks. In International conference on machine learning (pp. 214-223). PMLR.]

<img src="./resources/WGAN_1.PNG">

 - Loss function 이 Wasserstein distance로 구성되어 있는 GAN
 - WGAN의 특징은 다음과 같이 나타낼 수 있음
 
  1. 향상된 안정성 : 전통적인 GAN의 loss function은 수식 전개하면 결국 real data distribution 과 generated data distribution 사이의 JS divergence를 줄이는 것임. 이는 최적화하기 어려운데 WGAN의 Wasserstein distance는 더 안정적이고 쉽게 최적화 가능함.
  
  2. *립시츠 제약* : WGAN은 Discriminator가 립시츠 제약을 만족하도록 유도하는데, 이를 통해 과적합을 낮추거나 모델이 너무 복잡해지는 것을 방지해줄 수 있음
  
  3. 샘플의 질적 향상 : Image generation 에서 특히 매우 질적인 향상을 보임
  
  4. Application : WGAN은 다양한 task (이미지 생성, 이미지 super-resolution, T2I 등)에 적용해도 좋은 결과 보임

### 3. WGAN-GP: Wasserstein GAN - Gradient Penalty를 이용하여 학습 안정성 높인 버전
- [Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., & Courville, A. C. (2017). Improved training of wasserstein gans. Advances in neural information processing systems, 30.]

 - Weight clipping 을 사용하여 립시츠 제약을 유도한 기존의 WGAN과는 다르게 GP(gradient penalty)를 통해 립시츠 제약을 유도함.
  
 - 기존의 WGAN보다 더 안정적인 학습을 할 수 있고 좋은 성능을 보인다는 것을 보여줌
  
<img src="./resources/WGAN_gp_1.PNG"  >

<img src="./resources/WGAN_gp_2.PNG"  >

### 4. Pix2Pix: Style Transfer 하는데 이용되는 구조
- [Isola, P., Zhu, J. Y., Zhou, T., & Efros, A. A. (2017). Image-to-image translation with conditional adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1125-1134).]

 - Image-to-image translation task를 위해 제안된 구조
 
 - Generator는 conditional 하게 특정 input image가 주어졌을 때 desired output image를 만들도록 학습됨.

<img src="./resources/pix2pix_1.PNG"  >

<img src="./resources/pix2pix_2.PNG"  >

### 5. CycleGAN:
- [Zhu, J. Y., Park, T., Isola, P., & Efros, A. A. (2017). Unpaired image-to-image translation using cycle-consistent adversarial networks. In Proceedings of the IEEE international conference on computer vision (pp. 2223-2232).]

 - Image-to-image translation task를 위해 제안된 구조
 
 - pix2pix는 하나의 generator, discriminator를 가지는 반면 CycleGAN은 2개씩 가짐.
 
 - 또, pix2pix와 달리 paired image 대신 set of unpaired images를 통해 unsupervised training으로 

<img src="./resources/cycleGAN_1.PNG"  >

<img src="./resources/cycleGAN_2.PNG"  >
