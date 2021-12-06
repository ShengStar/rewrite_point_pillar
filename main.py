from train import bulid_net
import torch
from predict import predict
import numpy as np
if __name__ == '__main__':
    filename = '000116.bin'
    net = bulid_net()
    net.load_state_dict(torch.load("voxelnet-296960.pth"))
    net.eval()
    preds_dict , anchors_masks = net(filename)
    anchors_mask = torch.tensor(np.expand_dims(anchors_masks, 0))
    # box_preds = preds_dict["box_preds"]
    # cls_preds = preds_dict["cls_preds"]
    # dir_cls_preds = preds_dict["dir_cls_preds"]

    # box_preds = torch.tensor(np.load("batch_box_preds.npy").swapaxes(1,3))
    # cls_preds = torch.tensor(np.load("batch_cls_preds.npy").swapaxes(1,3))
    # dir_cls_preds = torch.tensor(np.load("batch_dir_preds.npy").swapaxes(1,3))
    # anchors_mask = torch.tensor(np.load("batch_anchors_mask.npy"))
    # print(anchors_mask.shape)
    # print(anchors_masks.shape)

    batch_anchor = torch.tensor(np.load("batch_anchors.npy"))

    box_preds = torch.tensor(preds_dict["box_preds"].swapaxes(1,3))
    cls_preds = torch.tensor(preds_dict["cls_preds"].swapaxes(1,3))
    dir_cls_preds = torch.tensor(preds_dict["dir_cls_preds"].swapaxes(1,3))
    # anchors_mask = torch.tensor(anchors_mask)
    forecast = predict(batch_anchor,box_preds,cls_preds,dir_cls_preds,anchors_mask)
    print(forecast)
