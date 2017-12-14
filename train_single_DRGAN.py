#!/usr/bin/env python
# encoding: utf-8

import os
import numpy as np
from scipy import misc
import pdb
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from util.one_hot import one_hot
from util.Is_D_strong import Is_D_strong
from util.log_learning import log_learning
from util.convert_image import convert_image
from util.DataAugmentation import FaceIdPoseDataset, Resize, RandomCrop
import itertools

def convert_greyimg(data):
    img = (data+1)/2.0*255
    return img.astype(np.uint8)


def train_single_DRGAN(Nd, Np, Nz, D_model, G_model, args, dataset=None,images=np.zeros((7000, 110, 110, 3)),id_labels=np.zeros(7000), pose_labels=np.zeros(7000)):
    if args.cuda:
        D_model.cuda()      

        G_model.cuda()

        # if args.cycle:
        #     G_cycle.cuda()

    D_model.train()
    G_model.train()
    # if args.cycle:
    #     G_cycle.train()

    lr_Adam    = args.lr
    beta1_Adam = args.beta1
    beta2_Adam = args.beta2

    image_size = images.shape[0]
    epoch_time = np.ceil(image_size / args.batch_size).astype(int)

    optimizer_D = optim.Adam(D_model.parameters(), lr = lr_Adam, betas=(beta1_Adam, beta2_Adam))
    optimizer_G = optim.Adam(G_model.parameters(), lr = lr_Adam, betas=(beta1_Adam, beta2_Adam))
    if args.cycle:
        optimizer_G_re = optim.Adam(itertools.chain(G_model.parameters(),G_model.parameters()), lr = lr_Adam, betas=(beta1_Adam, beta2_Adam))
    loss_criterion = nn.CrossEntropyLoss()
    loss_criterion_gan = nn.BCEWithLogitsLoss()
    loss_l2 = nn.MSELoss()
    if args.cycle:
        loss_criterion_cycle = nn.L1Loss()

    loss_log = []
    steps = 0
    # Load augmented data

    if args.grey:
        transformed_dataset = dataset
        print(len(dataset))
    else:
        transformed_dataset = FaceIdPoseDataset(images, id_labels, pose_labels,
                                                transform=transforms.Compose(
                                                    [Resize((256,256))]))
                                                    
    
   
    # transformed_dataset = mydataset(args, transform = transforms.Compose([Resize((110,110)), RandomCrop((96,96))]))
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=True)

    flag_D_strong  = False
    flag_D_cycle_strong = False
    for epoch in range(1,args.epochs+1):


        for i, batch_data in enumerate(dataloader):
            D_model.zero_grad()
            G_model.zero_grad()
            # if args.cycle:
            #
            #     G_cycle.zero_grad()

            if args.grey:
                batch_image = torch.FloatTensor(batch_data['img'].float())
                batch_id_label = batch_data['id']
                batch_pose_label = batch_data['pose']
            else:
                batch_image = torch.FloatTensor(batch_data[0].float())
                batch_id_label = batch_data[1]
                batch_pose_label = batch_data[2]
            
            minibatch_size = len(batch_image)

            batch_ones_label = torch.ones(minibatch_size)   # 真偽判別用のラベル
            batch_zeros_label = torch.zeros(minibatch_size)


            # ノイズと姿勢コードを生成
            fixed_noise = torch.FloatTensor(np.random.uniform(-1,1, (minibatch_size, Nz)))
            randnum = np.random.randint(Np, size=minibatch_size)
            tmp  = torch.LongTensor(randnum)

            pose_code = one_hot(tmp, Np)  # Condition 付に使用
            pose_code_label = torch.LongTensor(tmp) # CrossEntropy 誤差に使用

            if args.cycle:
                pose_cycle_code = one_hot(batch_pose_label,Np)
                pose_cycle_label= torch.LongTensor(batch_pose_label)



            if args.cuda:
                batch_image, batch_id_label, batch_pose_label, batch_ones_label, batch_zeros_label = \
                    batch_image.cuda(), batch_id_label.cuda(), batch_pose_label.cuda(), batch_ones_label.cuda(), batch_zeros_label.cuda()

                fixed_noise, pose_code, pose_code_label = \
                    fixed_noise.cuda(), pose_code.cuda(), pose_code_label.cuda()
                if args.cycle:
                    pose_cycle_code = pose_cycle_code.cuda()
                    pose_cycle_label = pose_cycle_label.cuda()



            batch_image, batch_id_label, batch_pose_label, batch_ones_label, batch_zeros_label = \
                Variable(batch_image), Variable(batch_id_label), Variable(batch_pose_label), Variable(batch_ones_label), Variable(batch_zeros_label)

            fixed_noise, pose_code, pose_code_label = \
                Variable(fixed_noise), Variable(pose_code), Variable(pose_code_label)

            if args.cycle:
                pose_cycle_code = Variable(pose_cycle_code)
                pose_cycle_label = Variable(pose_cycle_label)



            steps += 1

            # バッチ毎に交互に D と G の学習，　Dが90%以上の精度の場合は 1:4の比率で学習
            if not args.cycle:
                generated = G_model(batch_image, pose_code, fixed_noise)
                if flag_D_strong:
                    if i%5 == 0:
                        # Discriminator の学習
                        flag_D_strong = Learn_D(D_model, loss_criterion, loss_criterion_gan, optimizer_D, batch_image, generated, \
                                                batch_id_label, batch_pose_label, batch_ones_label, batch_zeros_label, epoch, steps, Nd, args)

                    else:
                        # Generatorの学習
                        Learn_G(D_model, loss_criterion, loss_criterion_gan, optimizer_G ,generated,\
                                batch_id_label, batch_ones_label, pose_code_label, epoch, steps, Nd, args)
                else:

                    if i%2==0:
                        # Discriminator の学習
                        flag_D_strong = Learn_D(D_model, loss_criterion, loss_criterion_gan, optimizer_D, batch_image, generated, \
                                                batch_id_label, batch_pose_label, batch_ones_label, batch_zeros_label, epoch, steps, Nd, args)

                    else:
                        # Generatorの学習
                        Learn_G(D_model, loss_criterion, loss_criterion_gan, optimizer_G ,generated, \
                                batch_id_label, batch_ones_label, pose_code_label, epoch, steps, Nd, args)
            else:
                generated = G_model(batch_image, pose_code, fixed_noise)
                generated_cycle = G_model(generated, pose_cycle_code, fixed_noise)
                if flag_D_strong:
                        if i % 5 == 0:
                             # Discriminator の学習
                             flag_D_strong = Learn_D(D_model, loss_criterion, loss_criterion_gan, optimizer_D, batch_image,generated, \
                                                batch_id_label, batch_pose_label, batch_ones_label, batch_zeros_label, epoch, steps, Nd, args)
                             flag_D_cycle_strong = Learn_D(D_model, loss_criterion, loss_criterion_gan, optimizer_D,
                                                      batch_image, generated_cycle, batch_id_label, batch_pose_label, batch_ones_label, \
                                                      batch_zeros_label, epoch, steps, Nd, args)

                        else:
                            # Generatorの学習
                             Learn_G_re(D_model, loss_criterion, loss_criterion_gan, loss_criterion_cycle,loss_l2, optimizer_G_re, generated, generated_cycle,\
                                        batch_image, batch_id_label, batch_ones_label, pose_code_label, pose_cycle_label, epoch, steps, Nd, args)
                    # els       e:
                    #     if i % 2 ==0 :
                    #         flag_D_strong = Learn_D(D_model, loss_criterion, loss_criterion_gan, optimizer_D, batch_image,
                    #                                 generated, \
                    #                                 batch_id_label, batch_pose_label, batch_ones_label, batch_zeros_label,
                    #                                 epoch, steps, Nd, args)
                    #         flag_D_cycle_strong = Learn_D(D_model, loss_criterion, loss_criterion_gan, optimizer_D,
                    #                                       batch_image, generated_cycle, \
                    #                                       batch_id_label, batch_pose_label, batch_ones_label,
                    #                                       batch_zeros_label, epoch, steps, Nd, args)
                    #     else:
                    #         # Generatorの学習
                    #         Learn_G_re(D_model, loss_criterion, loss_criterion_gan, loss_criterion_cycle, optimizer_G_re ,generated, generated_cycle, \
                    #                    batch_image, batch_id_label, batch_ones_label, pose_code_label, pose_cycle_label, epoch, steps, Nd, args)

                else:
                   if i % 2 == 0:
                       # Discriminator の学習
                       flag_D_strong = Learn_D(D_model, loss_criterion, loss_criterion_gan, optimizer_D, batch_image, generated, \
                                               batch_id_label, batch_pose_label, batch_ones_label, batch_zeros_label, epoch, steps, Nd, args)

                   else:
                       # Generatorの学習
                       Learn_G(D_model, loss_criterion, loss_criterion_gan, optimizer_G, generated, \
                               batch_id_label, batch_ones_label, pose_code_label, epoch, steps, Nd, args)


        if epoch%args.save_freq == 0:
            # 各エポックで学習したモデルを保存
            if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
            save_path_D = os.path.join(args.save_dir,'epoch{}_D.pt'.format(epoch))
            torch.save(D_model, save_path_D)
            save_path_G = os.path.join(args.save_dir,'epoch{}_G.pt'.format(epoch))
            torch.save(G_model, save_path_G)
            # save_path_G_re = os.path.join(args.save_dir, 'epoch{}_G_re.pt'.format(epoch))
            # torch.save(G_cycle, save_path_G_re)

            # 最後のエポックの学習前に生成した画像を１枚保存（学習の確認用）
            save_generated_image = convert_image(generated[0].cpu().data.numpy())
            save_path_image = os.path.join(args.save_dir, 'epoch{}_generatedimage.jpg'.format(epoch))

            misc.imsave(save_path_image, save_generated_image.astype(np.uint8))
            if args.cycle:

                save_cycle_image = convert_image(generated_cycle[0].cpu().data.numpy())
                save_path_cycle = os.path.join(args.save_dir, 'epoch{}_reconstructimage.jpg'.format(epoch))
                misc.imsave(save_path_cycle, save_cycle_image.astype(np.uint8))



def Learn_D(D_model, loss_criterion, loss_criterion_gan, optimizer_D, batch_image, generated, \
            batch_id_label, batch_pose_label, batch_ones_label, batch_zeros_label, epoch, steps, Nd, args, thresh = 0.9):

    real_output = D_model(batch_image)
    syn_output = D_model(generated.detach()) # .detach() をすることで Generatorまでの逆伝播計算省略

    # id,真偽, pose それぞれのロスを計算
    L_id    = loss_criterion(real_output[:, :Nd], batch_id_label)
    L_gan   = loss_criterion_gan(real_output[:, Nd], batch_ones_label) + loss_criterion_gan(syn_output[:, Nd], batch_zeros_label)
    L_pose  = loss_criterion(real_output[:, Nd+1:], batch_pose_label)

    d_loss = L_gan + L_id + L_pose

    d_loss.backward()
    optimizer_D.step()
    log_learning(epoch, steps, 'D', d_loss.data[0], args)

    # Discriminator の強さを判別
    flag_D_strong = Is_D_strong(real_output, syn_output, batch_id_label, batch_pose_label, Nd, thresh=thresh)

    return flag_D_strong



def Learn_G(D_model, loss_criterion, loss_criterion_gan, optimizer_G ,generated, \
            batch_id_label, batch_ones_label, pose_code_label, epoch, steps, Nd, args):

    syn_output=D_model(generated)

    # id についての出力と元画像のラベル, 真偽, poseについての出力と生成時に与えたposeコード の ロスを計算
    L_id    = loss_criterion(syn_output[:, :Nd], batch_id_label)
    L_gan   = loss_criterion_gan(syn_output[:, Nd], batch_ones_label)
    L_pose  = loss_criterion(syn_output[:, Nd+1:], pose_code_label)

    g_loss = L_gan + L_id + L_pose
    
    optimizer_G.zero_grad()
    g_loss.backward()
    optimizer_G.step()
    log_learning(epoch, steps, 'G', g_loss.data[0], args)

def Learn_G_re(D_model, loss_criterion, loss_criterion_gan, loss_cycle, loss_l2,  optimizer_G ,generated, generated_cycle, \
            batch_img, batch_id_label, batch_ones_label, pose_code_label, pose_code_cycle_label, epoch, steps, Nd, args):

    syn_output=D_model(generated)
    syn_output_cycle = D_model(generated_cycle)

    # id についての出力と元画像のラベル, 真偽, poseについての出力と生成時に与えたposeコード の ロスを計算
    L_id    = loss_criterion(syn_output[:, :Nd], batch_id_label)
    L_gan   = loss_criterion_gan(syn_output[:, Nd], batch_ones_label)
    L_pose  = loss_criterion(syn_output[:, Nd+1:], pose_code_label)

    L_id_cycle = loss_criterion(syn_output_cycle[:,:Nd], batch_id_label)
    L_gan_cycle = loss_criterion_gan(syn_output_cycle[:,Nd], batch_ones_label)
    L_pose_cycle = loss_criterion(syn_output_cycle[:, Nd+1:], pose_code_cycle_label)
    L_re    = loss_cycle(generated_cycle, batch_img)
    # L_tv = (loss_l2(generated_cycle[]))
    g_loss = L_gan + L_id + L_pose+0.2*L_re+0.8*(L_gan_cycle+L_id_cycle+L_pose_cycle)

    g_loss.backward()
    optimizer_G.step()
    log_learning(epoch, steps, 'G_re', g_loss.data[0], args)


