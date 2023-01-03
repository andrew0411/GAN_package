'''
Various Models (option --model : pix2pix, cyclegan, colorization)
Different Datasets (option --dataset_mode : aligned, unaligned, single, colorization)
에 대해 적용할 수 있으며,

Specify 해야하는 정보들은 다음과 같다.
dataset 경로 (--dataroot), 실험 이름 (--name), 모델 (--model)

Training 과정에서 
image들을 visualize하거나 save 할 수 있고, loss plot을 그리거나 save 할 수 있고, model 도 save 가능

또한 --continue_train 을 통해 previous training을 재개 할 수 있음

EX
python train.py --dataroot ./datasets/maps --name maps_cyclegan_1 --model cycle_gan

더 많은 training option은 options/base_options.py 또는 options/train_options.py 에서 볼 수 있음
'''

import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer