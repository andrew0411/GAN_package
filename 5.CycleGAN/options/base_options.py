import argparse
import os
from util import util
import torch
import models
import data

class BaseOptions():
    '''
    Training 과 Test에 모두 사용되는 common options 정의
    '''

    def __init__(self):
        '''Class reset'''
        self.initialized = False

    def initialize(self, parser):
        '''Common options 정의'''
        # Basic parameters
        parser.add_argument('--dataroot', required=True, help='path to images')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--use_wandb', action='store_true', help='use wandb')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids : 0 0,1,2 0 2. -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models will be saved here')

        # Model parameters
        parser.add_argument('--model', type=str, default='cycle_gan', help='choose which model to use. [cycle_gan | pix2pix | test | colorization]')
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        parser.add_argument('--ngf', type=int, default=64, help='# of generator filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discriminator filters in the first conv layer')
        parser.add_argument('--netD', type=str, default='basic', help='specify discriminator architecture [basic | n_layers | pixel')
        parser.add_argument('--netG', type=str, default='resnet_9blocks', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD == n_layers')
        parser.add_argument('--norm', type=str, default='instance', help='[instance | batch | none]')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier, orthogonal')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')

        # dataset parameters
        parser.add_argument('--dataset_mode', type=str, default='unaligned', help='choose how datasets are loaded. [unaligned | aligned | single | colorization]')
        parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in orger to make batches, otherwise takes them randomly')
        parser.add_argument('--num_threads', type=int, default=4, help='# of threads for loading data')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
        parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
        parser.add_argument('--max_dataset_size', type=int, default=float('inf'), help='Maximum number of samples allowed per dataset. If dataset directory contains more than this, only a subset is loaded')
        parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='[resize_and_crop | crop | scale_width | scale_width_and_crop | none')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')

        # additional parameters
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load?')
        parser.add_argument('--load_iter', type=int, default=0, helpt='which iteration to load? if load_iter > 0, it will load models by iter_[load_iter]; otherwise, it will load models by [epoch]')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', type=str, default='', help='customized suffix ')
        self.initialized = True
        return parser

    def gather_options(self):
        '''
        Basic options를 통해 parser를 initialize.
        그후 additional model-specific, dataset-specific options 설정
        '''

        if not self.initialized: # Initialize가 되었는지 확인
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter) # formatter_class는 도움말 출력을 정의
            parser = self.initialize(parser)                                                         # ArgumentDefaultHelpFormatter은 default 값을 자동으로 추가

        # basic options 가져오기
        opt, _ = parser.parse_known_args() # 명령행 인자 중 일부를 파싱

        # 모델 관련된 parser option 변경
        model_name = opt.model 
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()

        # dataset 관련된 option 
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        self.parser = parser
        return parser.parse_args

    def print_options(self, opt):
        '''
        출력과 저장과 관련된 option
        '''
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # 저장
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        '''
        모든 option을 parse하고 checkpoint directory 생성
        '''
        opt = self.gather_options()
        opt.isTrain = self.isTrain

        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
