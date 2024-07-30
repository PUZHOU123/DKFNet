import torch
import torch.nn.functional as F
from tqdm import tqdm

from dice_loss import dice_coeff


def eval_ndwi_ndvi_nodeep1(net, loader, device, n_val):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0
    # print("n_val是多少", n_val)
    with tqdm(total=n_val, desc='Validation round', unit='img', leave=False) as pbar:
        for batch in loader:
            imgs = batch['image']
            true_masks = batch['mask']
            nir = batch['nir']


            imgs = imgs.to(device=device, dtype=torch.float32)
            mask_type = torch.float32 if net.n_classes == 1 else torch.long
            true_masks = true_masks.to(device=device, dtype=mask_type)
            nir = nir.to(device=device, dtype=torch.float32)
            # mask_pred = net(imgs, nir)
            mask_pred = net(imgs)
            for true_mask, pred in zip(true_masks, mask_pred):
                pred = (pred > 0.5).float()  ##这里的0.5为threshold, 与预测时的threshold一样
                # print('1',pred.shape, true_mask.shape)  torch.Size([1, 160, 160]) torch.Size([1, 160, 160])
                if net.n_classes > 1:
                    tot += F.cross_entropy(pred.unsqueeze(dim=0), true_mask.unsqueeze(dim=0).squeeze(1)).item()
                else:
                    tot += dice_coeff(pred, true_mask.squeeze(dim=1)).item()
            pbar.update(imgs.shape[0])

    return tot / n_val
