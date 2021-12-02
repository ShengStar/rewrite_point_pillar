from train import bulid_net
import torch
from predict import predict
import numpy as np
if __name__ == '__main__':
    filename = '000002.bin'
    net = bulid_net()
    net.load_state_dict(torch.load("voxelnet-296960.pth"))
    net.eval()
    preds_dict , anchors_mask = net(filename)
    box_preds = preds_dict["box_preds"]
    cls_preds = preds_dict["cls_preds"]
    dir_cls_preds = preds_dict["dir_cls_preds"]
    # box_preds = np.load("batch_box_preds.npy")
    # cls_preds = np.load("batch_cls_preds.npy")
    # dir_cls_preds = np.load("batch_dir_preds.npy")
    batch_anchor = np.load("batch_anchors.npy")
    print(box_preds.shape)
    print(cls_preds.shape)
    print(dir_cls_preds.shape)
    forecast = predict(batch_anchor,box_preds,cls_preds,dir_cls_preds,anchors_mask)
