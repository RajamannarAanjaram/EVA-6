from torchvision.datasets import CIFAR10
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import cv2
from torch.utils.data import DataLoader,random_split
import torchvision.transforms as tf
import numpy as np
from utils.helper import get_mean_std


def datasplit(dataset, trainVol, valVol):
    '''
    datasplit => Random split of data as training data and test data

    keyword Arguments:
    dataset   -> Downloaded dataset to be split
    trainVol  -> length of split to be generated for training
    valVol    -> length of split to be generated for testing/Validation

    return:
    subsetA -> new dataset with specified length
    subsetB -> new dataset with specified length
    '''
    lengths = [int(len(dataset)*trainVol), int(len(dataset)*valVol)]
    subsetA, subsetB = random_split(dataset, lengths)
    return subsetA, subsetB

def mean_std():
    '''
    mean_std => Computing mean and std of dataset

    Keyword Arguments:
    NA

    return:
    get_mean_std(train_loader, num_channels=3) -> invoked to calculate mean and std of data
    '''
    print('======> Computing mean and std of dataset')
    data = CIFAR10(root='./data', train=True, download=True, transform=tf.ToTensor())
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.manual_seed(40)
        kwargs = {'batch_size': 128, 'pin_memory': True, 'num_workers': 4}
    else:
        torch.manual_seed(40)
        kwargs = {'batch_size': 32}

    train_loader = DataLoader(data, shuffle=True,
                              drop_last=True, **kwargs)

    return get_mean_std(train_loader, num_channels=3)



class Transforms:
    ''' The class Transforms defined for common image transformations'''
    def __init__(self, transforms: A.Compose):
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        '''
        Keword Arguments:
        img  -> image to be transformed

        return:
        Image converted into Numpy Array in Height, Width, Channel format
        '''
        return self.transforms(image=np.array(img))


class Loader:
    '''The class Loader is defined for loading the data for the model'''
    def __init__(self,batch_size):

        self.text = 'This class loads the data for the model'
        self.batch_size = batch_size

    def transform(self):
        '''
        transform  => image transformations such as shifting, scaling, rotating and albumnations etc

        Keyword Arguments:
        NA

        return:
        trainTransform   -> The series of data augmentation used for composing the final transform
        simpleTransform  -> general normalization of data.
        '''
        DATA_MEAN, DATA_STD = mean_std()
        trainTransform = A.Compose([A.PadIfNeeded(min_height=40, min_width=40, always_apply=True),
                                      A.RandomCrop(width=32, height=32,p=1),
                                      A.Rotate(limit=5),
                                      #A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=5, p=0.25),
                                      A.CoarseDropout(max_holes=1,min_holes = 1, max_height=16, max_width=16, p=0.5,fill_value=tuple([x * 255.0 for x in DATA_MEAN]),
                                      min_height=16, min_width=16),
                                      A.Normalize(mean=DATA_MEAN, std=DATA_STD,always_apply=True),
                                      ToTensorV2()
                                    ])
        simpleTransform = A.Compose([A.Normalize(mean=DATA_MEAN, std=DATA_STD, always_apply=True),
                                 ToTensorV2()])
        # print(trainTransform,simpleTransform)
        return Transforms(trainTransform), Transforms(simpleTransform)



    def Loader(self, trainTransform, simpleTransform, cuda: bool = True):
        '''
        Loader => For downloading the dataset required

        Keyword Arguments:
        trainTransform  -> image transformations defined
        simpleTransform -> image transformations defined
        cuda: bool      -> boolean to use CUDA

        return:
        train_loader -> Dataloader for Training dataset
        test_loader -> Dataloader for Testing/Validation dataset
        '''
        seed = 42
        if cuda:
            torch.cuda.manual_seed(seed)
            kwargs = {'batch_size': self.batch_size,
                      'pin_memory': True, 'num_workers': 4}
        else:
            torch.manual_seed(seed)
            kwargs = {'batch_size': self.batch_size}

        train = CIFAR10(root='./data', train=True,
                        download=True, transform=trainTransform)
        test = CIFAR10(root='./data', download=True, transform=simpleTransform)

        train_loader = DataLoader(train, shuffle=True, **kwargs)
        test_loader = DataLoader(test, shuffle=True, **kwargs)

        return train_loader, test_loader


class DeNorm:
    def __init__(self):
        self.mean ,self.std = mean_std()


    def __call__(self, tensor):
        '''
        UnNormalizes an image given its mean and standard deviation
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        '''
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
