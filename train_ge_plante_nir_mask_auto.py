import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from unet_sa_v2 import UNet_SA_v2
from pgunet.FCN import FCN8s_1, VGGNet, FCN8s, FCN32s
from deeplabv3plus import Deeplab_v3plus
from HRNet import HighResolutionNet
from loss_0304 import Weight_Lossv5
from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from eval_net_ndvi_ndwi_nodeep import eval_ndwi_ndvi_nodeep1
from Datasets_GE_S21 import Dataset_ndwi_ndvi_Weight1
from Dataset_ge_plante_mask import Dataset_ge_plante_mask1
from Dataset_ge_plante_nir1 import Dataset_ge_plante_nir1
from losses import dice_coeff
from unet_sa_aspp_ndwi import UNet_SA_aspp_ndwi
from unet_sa_aspp_ndwi import UNet_SA_aspp_without_ndwi
from SegNet import SegNet
from segformer import SegFormer
from pvtv2 import pvt_v2_b0
from UNetFormer import UNetFormer
from swin_transformer_unet import SwinUNet
from model.vitcross_seg_modeling import VisionTransformer as ViT_seg
from model.vitcross_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from LANet import LANet
from DKFNet import DKFNet_DASPP_without_DCA, DKFNet_DCA_without_DASPP, DKFNet_DASPP_DCA, DKFNet_DCA_with_ASPP, DKFNet_DASPP_with_CA
from RedNet import RedNet
# from DenseASPP import DenseASPP
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from pgunet.NestedUNet import UNet
from pgunet.NestedUNet import NestedUNet

##训练unet网络，加上损失函数后的结果
dir_img = r"C:\Users\user\Desktop\experiment_ge_plante_fuse\google_dataset/"      #0~255
# C:\Users\user\Desktop\experiment_ge_plante_fuse\google_dataset
#C:\Users\user\Desktop\experiment_ge_plante_fuse\imbalance2\google
dir_plante = r"C:\Users\user\Desktop\experiment_ge_plante_fuse\plante_16bit_dataset/"   #0~1
# C:\Users\user\Desktop\experiment_ge_plante_fuse\plante_16bit_dataset
#C:\Users\user\Desktop\experiment_ge_plante_fuse\imbalance2\planet
dir_mask = r"C:\Users\user\Desktop\experiment_ge_plante_fuse\mask_dataset/"   #0~255
# C:\Users\user\Desktop\experiment_ge_plante_fuse\mask_dataset
#C:\Users\user\Desktop\experiment_ge_plante_fuse\imbalance2\mask
dir_weight = r"C:\Users\user\Desktop\experiment_ge_plante_fuse\mask_dataset/"
#C:\Users\user\Desktop\experiment_ge_plante_fuse\weight_new1
#C:\Users\user\Desktop\experiment_ge_plante_fuse\imbalance2\weight
dir_checkpoint0 = r"G:\ge_plante fuse\ckpt_imbalance1\ckpt_SwinUNet_ge_nir_4band_19985/"
# dir_checkpoint1 = r"G:\ge_plante fuse\ckpt_imbalance\ckpt_planet_6band_28665_1/"
# dir_checkpoint2 = r"G:\ge_plante fuse\ckpt_imbalance\ckpt_planet_6band_28665_2/"
# dir_checkpoint3 = r"G:\ge_plante fuse\ckpt_imbalance\ckpt_planet_6band_28665_3/"
# dir_checkpoint4 = r"G:\ge_plante fuse\ckpt_imbalance\ckpt_planet_6band_28665_4/"

#loss1: dif_value, a = 40, beta = 200; ndwi, a=1.5, beta=0.3

def weight_init(m):

    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        # nn.init.constant_(m.bias, 0)
    # 也可以判断是否为conv2d，使用相应的初始化方式
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
     # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
#get_args()是一个， 其中dest是可以直接在参数里面调用，例如，在模型中要初始化学习率，可以直接提供lr=args.lr(args=get_args()),这里的lr就是dest。
#而nargs='?'表示只有一个参数，nargs='*'表示有多个参数。其中带有--的为可选参数，-为该参数的短写形式，在这里前两个参数，例如'-e', '--epochs'，是
# 必须要有的，叫做name or flags；
# 如果在调用该参数时没有赋值就选用default值。
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=201,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=4,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=20,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()

if __name__ == '__main__':


    for i in range(0, 1):

        torch.manual_seed(i)
        # print("1",torch.manual_seed)
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        args = get_args()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device {device}')
        # net = UNet_SA_v2(n_channels=5, n_classes=1)
        # vgg_model = VGGNet(requires_grad=True, show_params=False)
        # net = FCN8s(pretrained_net=vgg_model, n_classes=1)
        # net = RedNet(n_classes=1)
        # net = UNet(n_channels=6, n_classes=1)
        # net = DKFNet_DASPP_DCA(n_channels=8, n_classes=1)
        # net = UNetFormer(n_classes=1)
        # config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        # config_vit.n_classes = 1
        # config_vit.n_skip = 3
        # config_vit.patches.grid = (int(512 / 16), int(512 / 16))
        # net = ViT_seg(config_vit, img_size=512, num_classes=1).cuda()
        # net.load_from(weights=np.load(config_vit.pretrained_path))
        net = SwinUNet(n_classes=1)
        # net = SegFormer(in_chans=4, n_classes=1, phi='b1')
        # net = pvt_v2_b0(in_chans=4, n_classes=1)
        # net = DKFNet_DASPP_without_DCA(n_channels=4, n_classes=1)
        # net = DKFNet_DCA_without_DASPP(n_channels=4, n_classes=1)
        # net = DKFNet_DCA_with_ASPP(n_channels=4, n_classes=1)
        # net = DKFNet_DASPP_with_CA(n_channels=4, n_classes=1)
        # net = LANet(in_channels=4, n_classes=1)
        # net = SegNet(n_channels=4, n_classes=1)
        # net = UNet_SA_aspp_ndwi(n_channels=4, n_classes=1)
        # net = Deeplab_v3plus(input_channels=4, n_classes=1)
        # net = NestedUNet(input_channels=4, n_classes=1)
        # net = UNet_SA_aspp_without_ndwi(n_channels=4, n_classes=1
        # net = HighResolutionNet(input_channels=4, n_classes=1)
        net.apply(weight_init)
        # if args.load:
        #     net.load_state_dict(
        #         torch.load(r"C:\Users\user\Desktop\plante_train\ckpt\ckpt_6band_unet_sa_v2_2078\CP_epoch399.pth",
        #                    map_location=device)
        #     )
        #     logging.info(f'Model loaded from {args.load}')
        net.to(device=device)

        epochs = args.epochs
        batch_size = args.batchsize
        lr = args.lr
        device = device
        img_scale = args.scale
        val_percent = args.val / 100
        save_cp = True

        dataset = Dataset_ge_plante_nir1(dir_img, dir_plante, dir_mask,  dir_weight)
        # print(len(dataset))
        n_val = int(len(dataset) * val_percent)  # len(dataset)返回的是ids的长度，即img的个数
        n_train = len(dataset) - n_val
        train, val = random_split(dataset, [n_train, n_val])
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

        # 当影像输入的帧数波动时，可以改变num_workers
        # 这里是创建一个SummaryWriter，对于每次试验，都会在runs文件夹里面生成一个experiment，comment是该experiment的名称，里面包含了LR，BS, SCALE的值
        # 之后可以利用writer.add_scalar向里面添加不同的元素
        writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
        global_step = 0

        logging.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {lr}
            Training size:   {n_train}
            Validation size: {n_val}
            Checkpoints:     {save_cp}
            Device:          {device.type}
            Images scaling:  {img_scale}
        ''')
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)  # weight_decay L2正则化的系数

        # # 常规的损失
        # criterion = Seg_BCELoss()
        criterion = nn.BCELoss()
        # #常规的损失

        for epoch in range(epochs):
            net.train()
            epoch_loss = 0
            # 进度条配置，total为总的迭代次数；desc为进度条的前缀；unit为每个迭代的单元
            with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
                for batch in train_loader:
                    imgs = batch['image']
                    true_masks = batch['mask']
                    nir = batch['nir']
                    weight = batch['weight']
                    # print(imgs.shape)
                    # print(true_masks.shape)

                    imgs = imgs.to(device=device, dtype=torch.float32)  # 将imgs和mask放到device上（GPU上）
                    # plante = plante.to(device=device, dtype=torch.float32)
                    mask_type = torch.float32 if net.n_classes == 1 else torch.long
                    true_masks = true_masks.to(device=device, dtype=mask_type)
                    nir = nir.to(device=device, dtype=torch.float32)
                    weight = weight.to(device=device, dtype=torch.float32)
                    # masks_pred = net(imgs, nir)
                    # masks_pred = net(imgs, nir)  ##for rednet
                    masks_pred = net(imgs)
                    # print(masks_pred.shape)    ##torch.Size([32, 7, 64, 64]) torch.Size([32, 1, 64, 64])
                    # print(true_masks)
                    if net.n_classes == 1:
                        masks_pred = masks_pred.squeeze(1)
                        true_masks = true_masks.squeeze(1)
                    else:
                        masks_pred = masks_pred
                        true_masks = true_masks
                    # print(masks_pred)
                    # print(masks_pred.shape, true_masks.shape)
                    loss = criterion(masks_pred, true_masks)
                    # criterion = Weight_Lossv5()  ##focalloss里面传入参数的顺序不能弄错
                    # loss = criterion(true_masks, masks_pred, weight)
                    # print(loss)
                    epoch_loss += loss.item()  # 一个元素张量可以用item得到元素值

                    loss = loss.cpu()
                    loss_ = str(loss.data.numpy())
                    with open('./loss_SwinUNet_ge_nir_4band_19985_seed{}.txt'.format(i), 'a', newline='') as f:  ##这里的newline=''相当于删掉txt里面的空行  DKFNet_DCA_DGLoss_without_DASPP
                        f.write(str(global_step))
                        f.write(' ')
                        f.write(loss_)
                        if global_step < 9999999:
                            f.write(' \r\n')
                    writer.add_scalar('Loss/train', loss.item(), global_step)  # 将标量添加到 summary,以便在summary中查看，

                    pbar.set_postfix(**{'loss (batch)': loss.item()})  ##输入一个字典，显示实验指标

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    pbar.update(imgs.shape[0])
                    global_step += 1  # 每过一个batch，global_step就+1
                    if global_step % (n_train // (1 * batch_size)) == 0:  # %是求余数， 这里是每一个batch计算一次val_score   ##这里的dataset需要替换为n_train
                        val_score = eval_ndwi_ndvi_nodeep1(net, val_loader, device, n_val)
                        print('val', epoch)  # 每一个batch，利用eval_net和val_loader来进行验证，从而得到每一个epoch的验证精度
                        if net.n_classes > 1:
                            logging.info('Validation cross entropy: {}'.format(val_score))
                            writer.add_scalar('Loss/test', val_score, epoch)

                        else:
                            logging.info('Validation Dice Coeff: {}'.format(val_score))
                            val_score_ = str(val_score)
                            with open('./valid_SwinUNet_ge_nir_4band_19985_seed{}.txt'.format(i), 'a', newline='') as f:
                                f.write(str(epoch))
                                f.write(' ')
                                f.write(val_score_)
                                if epoch < 9999999:
                                    f.write(' \r\n')
                            writer.add_scalar('Dice/test', val_score, epoch)  # 验证精度随epoch的变化情况
            if epoch % 1 == 0 or epoch == 199:
                if save_cp:
                    try:  # dir_checkpoint = r'C:\Users\zhoupu\Desktop\train_\checkpoints/'
                        os.mkdir(eval("dir_checkpoint" + str(i)))  # 这里是创建了一个目录，当dir_checkpoint不存在时，会在train_文件夹下面创建一个checkpoints文件夹
                        logging.info('Created checkpoint directory')
                    except OSError:
                        pass
                    torch.save(net.state_dict(),
                               eval("dir_checkpoint" + str(i)) + f'CP_epoch{epoch}.pth')
                    print('ckpt', epoch)
                    logging.info(f'Checkpoint {epoch} saved !')
                    print('done')

        writer.close()