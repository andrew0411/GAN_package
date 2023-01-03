'''
Data loading 과 전처리와 관련된 모듈
'''

import importlib
import torch.utils.data
from data.base_dataset import BaseDataset

def find_dataset_using_name(dataset_name):
    '''
    Import the module 'data/[dataset_name]_dataset.py'
    '''
    dataset_filename = 'data' + dataset_name + '_dataset'
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
            and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

def get_option_setter(dataset_name):
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options

def create_dataset(opt):
    '''
    주어진 option에 맞게 dataset 생성

    ex)
    from data import create_dataset
    dataset = create_dataset(opt)
    '''
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset

class CustomDatasetDataLoader():

    def __init__(self, opt):
        '''
        Initialize the class

        Step 1 : create a dataset instance given the name [dataset_mode]
        Step 2 : create a multi-threaded data loader
        '''
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        self.dataset = dataset_class(opt)
        print("dataset [%s] was created" % type(self.dataset).__name__)
        self.dataloader = torch.util.data.DataLoader(
            self.dataset,
            batch_size = opt.batch_size,
            shuffle = not opt.serial_batches,
            num_workers = int(opt.num_threads)
        )

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data
