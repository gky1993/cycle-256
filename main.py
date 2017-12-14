#!/usr/bin/env python
# encoding: utf-8

import os
import argparse
import datetime
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import transforms
from model import single_DR_GAN_model as single_model
from model import multiple_DR_GAN_model as multi_model
from util.create_randomdata import create_randomdata
from util.DataAugmentation import RandomCrop
from util.DataAugmentation import Resize as Rsz
from train_single_DRGAN import train_single_DRGAN
from train_multiple_DRGAN import train_multiple_DRGAN
from Generate_Image import Generate_Image
import pdb

import glob

from skimage import io, transform
from matplotlib import pylab as plt

from tqdm import tqdm


class Resize(object):
    #  assume image  as H x W x C numpy array
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        if h > w:
            new_h, new_w = self.output_size, int(self.output_size * w / h)
        else:
            new_h, new_w = int(self.output_size * h / w), self.output_size

            
        
        resized_image = transform.resize(image, (new_h, new_w))
        # print(resized_image.shape)

        if h > w:
            diff = self.output_size - new_w
            if diff % 2 == 0:
                pad_l = int(diff / 2)
                pad_s = int(diff / 2)
            else:
                pad_l = int(diff / 2) + 1
                pad_s = int(diff / 2)

            padded_image = np.lib.pad(resized_image, ((0, 0), (pad_l, pad_s), (0, 0)), 'edge')

        else:
            diff = self.output_size - new_h
            if diff % 2 == 0:
                pad_l = int(diff / 2)
                pad_s = int(diff / 2)
            else:
                pad_l = int(diff / 2) + 1
                pad_s = int(diff / 2)

            padded_image = np.lib.pad(resized_image, ((pad_l, pad_s), (0, 0), (0, 0)), 'edge')
        return padded_image


def DataLoader(data_place):
    image_dir = data_place
    rsz = Resize(256)

    Indv_dir = []
    for x in os.listdir(image_dir):
        if os.path.isdir(os.path.join(image_dir, x)):
            Indv_dir.append(x)

    Indv_dir = np.sort(Indv_dir)

    images = np.zeros((7000, 256, 256, 3))
    id_labels = np.zeros(7000)
    pose_labels = np.zeros(7000)
    count = 0
    gray_count = 0
    for i in tqdm(range(len(Indv_dir))):
        Frontal_dir = os.path.join(image_dir, Indv_dir[i], 'frontal')
        Profile_dir = os.path.join(image_dir, Indv_dir[i], 'profile')

        front_img_files = os.listdir(Frontal_dir)
        prof_img_files = os.listdir(Profile_dir)

        for img_file in front_img_files:
            img = io.imread(os.path.join(Frontal_dir, img_file))

            if len(img.shape) == 2:
                gray_count = gray_count + 1
                continue
            img_rsz = rsz(img)

            images[count] = img_rsz
            id_labels[count] = i
            pose_labels[count] = 0
            count = count + 1

        for img_file in prof_img_files:
            img = io.imread(os.path.join(Profile_dir, img_file))
            if len(img.shape) == 2:
                gray_count = gray_count + 1
                continue
            img_rsz = rsz(img)
            images[count] = img_rsz
            id_labels[count] = i
            pose_labels[count] = 1
            count = count + 1

    id_labels = id_labels.astype('int64')
    pose_labels = pose_labels.astype('int64')

    # [0,255] -> [-1,1]
    images = images * 2 - 1

    # RGB -> BGR
    images = images[:, :, :, [2, 1, 0]]
    # B x H x W x C-> B x C x H x W
    images = images.transpose(0, 3, 1, 2)

    # ç™½é»’ç”»åƒãƒ‡ãƒ¼ã‚¿ã‚’å–ã‚Šé™¤ã
    images = images[:gray_count * -1]
    id_labels = id_labels[:gray_count * -1]
    pose_labels = pose_labels[:gray_count * -1]
    Np = int(pose_labels.max() + 1)
    Nd = int(id_labels.max() + 1)
    Nz = 50
    channel_num = 3

    return [images, id_labels, pose_labels, Nd, Np, Nz, channel_num]
class mydata(Dataset):
    def __init__(self):
        super(mydata, self).__init__()
        datasetpath = '/home/kaiyu/data/CAS-PEAL-R1'
        posepath='/POSE'
        poseimgdir = datasetpath+posepath

        Indv_dir = []
        self.imagepaths = []
        self.poses = []
        self.IDs=[]
        # self.transform=transforms.Compose([Resize(110), RandomCrop((96,96))])
        self.transform = Resize(256)

        for x in os.listdir(poseimgdir):
            if os.path.isdir(os.path.join(poseimgdir, x)):
                Indv_dir.append(os.path.join(poseimgdir,x))
        for i in tqdm(range(len(Indv_dir))):
            for file in os.listdir(Indv_dir[i]):
                if not file.split('.')[1]=='tif':
                    continue
                self.imagepaths.append(os.path.join(Indv_dir[i], file))
                self.IDs.append(i)
                namearray = file.split('_')
                filterarray = list(namearray[3])
                if filterarray[1]=='D':
                    if filterarray[2]=='+':
                        self.poses.append(int(filterarray[3]))
                    elif filterarray[2]=='-':
                        self.poses.append(int(filterarray[3])-1)
                elif filterarray[1]=='M':
                    if filterarray[2] == '+':
                        self.poses.append(int(filterarray[3])+7)
                    elif filterarray[2] == '-':
                        self.poses.append(int(filterarray[3]) - 1+7)
                elif filterarray[1]=='U':
                    if filterarray[2] == '+':
                        self.poses.append(int(filterarray[3])+14)
                    elif filterarray[2] == '-':
                        self.poses.append(int(filterarray[3]) - 1+14)
        self.Nd = len(Indv_dir)
        self.Np = 21

    def __len__(self):
        return len(self.imagepaths)

    def __getitem__(self, idx):
        path = self.imagepaths[idx]

        image = io.imread(path)
        image1 = np.append(image,image)
        image1 = np.append(image1,image)
        image1 = np.reshape(image1,np.append(3,image.shape))
        image1 = image1.transpose(1,2,0)
        img_rsz = self.transform(image1)
        img_rsz = img_rsz*2-1
        img_rsz = img_rsz.transpose(2,0,1)
        img_ = Rsz((256,256))(img_rsz)
        #img = RandomCrop((96,96))(img_)
        ID = self.IDs[idx]
        pose = self.poses[idx]



        return [img, ID, pose]




if __name__=="__main__":

    parser = argparse.ArgumentParser(description='DR_GAN')
    # learning & saving parameterss
    parser.add_argument('-lr', type=float, default=0.0002, help='initial learning rate [default: 0.0002]')
    parser.add_argument('-beta1', type=float, default=0.5, help='adam optimizer parameter [default: 0.5]')
    parser.add_argument('-beta2', type=float, default=0.999, help='adam optimizer parameter [default: 0.999]')
    parser.add_argument('-epochs', type=int, default=1000, help='number of epochs for train [default: 1000]')
    parser.add_argument('-batch-size', type=int, default=8, help='batch size for training [default: 8]')
    parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
    parser.add_argument('-save-freq', type=int, default=1, help='save learned model for every "-save-freq" epoch')
    parser.add_argument('-cuda', action='store_true', default=False, help='enable the gpu')
    # data souce
    parser.add_argument('-random', action='store_true', default=False, help='use randomely created data to run program')
    parser.add_argument('-data-place', type=str, default='./data', help='prepared data path to run program')
    # model
    parser.add_argument('-multi-DRGAN', action='store_true', default=False, help='use multi image DR_GAN model')
    parser.add_argument('-images-perID', type=int, default=0, help='number of images per person to input to multi image DR_GAN')
    # option
    parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot(snapshot/{Single or Multiple}/{date}/{epoch}) [default: None]')
    parser.add_argument('-generate', action='store_true', default=None, help='Generate pose modified image from given image')
    parser.add_argument('-datatxt', type=str, default='./imagepaths', help='the txt containing the path of the training images')
    parser.add_argument('-cycle', action='store_true', default=False, help='if use cycle gan, only single can be used')
    parser.add_argument('-grey', action='store_true', default=False, help='if the image channel is one')
    parser.add_argument('-gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

    args = parser.parse_args()

    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)

    # update args and print
    if args.multi_DRGAN:
        args.save_dir = os.path.join(args.save_dir, 'Multi',datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    else:
        args.save_dir = os.path.join(args.save_dir, 'Single',datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    os.makedirs(args.save_dir)

    print("Parameters:")
    for attr, value in sorted(args.__dict__.items()):
        text ="\t{}={}\n".format(attr.upper(), value)
        print(text)
        with open('{}/Parameters.txt'.format(args.save_dir),'a') as f:
            f.write(text)



    # input data
    if args.random:
        images, id_labels, pose_labels, Nd, Np, Nz, channel_num = create_randomdata()
    else:
        print('n\Loading data from [%s]...' % args.data_place)
        images, id_labels, pose_labels, Nd, Np, Nz, channel_num = DataLoader(args.data_place)


    if args.grey:
        mydataset = mydata()
        Nd = mydataset.Nd
        Np = mydataset.Np
        Nz = 50
        channel_num = 3

    # model
    if args.snapshot is None:
        if not(args.multi_DRGAN):
            D = single_model.Discriminator256(Nd, Np, channel_num)
            G = single_model.Generator256(Np, Nz, channel_num)
        else:
            if args.images_perID==0:
                print("Please specify -images-perID of your data to input to multi_DRGAN")
                exit()
            else:
                D = multi_model.Discriminator(Nd, Np, channel_num)
                G = multi_model.Generator(Np, Nz, channel_num, args.images_perID)
    else:
        print('\nLoading model from [%s]...' % args.snapshot)
        try:

          #  D = torch.load('{}_D.pt'.format(args.snapshot))
            G = torch.load('{}_G.pt'.format(args.snapshot))
            #G_cycle = torch.load('{}_G_re.pt'.format(args.snapshot))
        except:
            print("Sorry, This snapshot doesn't exist.")
            exit()

    if not(args.generate):
        if not(args.multi_DRGAN):
            if not args.grey:
                train_single_DRGAN( Nd, Np, Nz, D, G, args, images=images, id_labels=id_labels, pose_labels=pose_labels)
            else:
                train_single_DRGAN( Nd, Np, Nz, D, G, args, dataset=mydataset)


        else:
            if args.batch_size % args.images_perID == 0:
                train_multiple_DRGAN(images, id_labels, pose_labels, Nd, Np, Nz, D, G, args)
            else:
                print("Please give valid combination of batch_size, images_perID")
                exit()
    else:
        # pose_code = [] # specify arbitrary pose code for every image
        #pose_code = np.random.uniform(-1,1, (images.shape[0], Np))

        pose_code = np.ones((images.shape[0],Np))
        pose_code_ = np.zeros((images.shape[0],Np))
        pose_code[range(images.shape[0]),1]=0
        pose_code_[range(images.shape[0]), pose_labels] = 1

        features = Generate_Image(images, pose_code, pose_code_, Nz, G, G, args)
