from train import bulid_net
import torch
from predict import predict
import numpy as np

def get_start_result_anno():
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': [],
        'score': [],
    })
    return annotations

if __name__ == '__main__':
    filename = '/home8T/000116.bin'
    net = bulid_net()
    net.load_state_dict(torch.load("/home8T/voxelnet-296960.tckpt"))
    net.eval()
    preds_dict , anchors_masks = net(filename)
    anchors_mask = torch.tensor(np.expand_dims(anchors_masks, 0))
    # box_preds = preds_dict["box_preds"]
    # cls_preds = preds_dict["cls_preds"]
    # dir_cls_preds = preds_dict["dir_cls_preds"]
    box_preds = torch.tensor(np.load("/home8T/batch_box_preds.npy"))
    cls_preds = torch.tensor(np.load("/home8T/batch_cls_preds.npy"))
    dir_cls_preds = torch.tensor(np.load("/home8T/batch_dir_preds.npy"))
    anchors_mask = torch.tensor(np.load("/home8T/batch_anchors_mask.npy"))
    # print(anchors_mask.shape)
    # print(anchors_masks.shape)
    
    #anchors_mask = torch.tensor(anchors_mask)

    batch_anchor = torch.tensor(np.load("/home8T/batch_anchors.npy"))
    # box_preds = torch.tensor(preds_dict["box_preds"].swapaxes(1,3))
    # cls_preds = torch.tensor(preds_dict["cls_preds"].swapaxes(1,3))
    # dir_cls_preds = torch.tensor(preds_dict["dir_cls_preds"].swapaxes(1,3))

    preds_dict = predict(batch_anchor,box_preds,cls_preds,dir_cls_preds,anchors_mask)
    box_preds_lidar = preds_dict["box3d_lidar"].detach().cpu().numpy()
    label_preds = preds_dict["label_preds"].detach().cpu().numpy()
    scores = preds_dict["scores"].detach().cpu().numpy()
    box_preds = preds_dict["box3d_camera"]
    anno = get_start_result_anno()
    class_names = {
        0: 'Car',
        1: 'Pedestrian',
        2: 'Cyclist',
        3: 'Van',
        4: 'Person_sitting',
    }
    for box,box_lidar,score,label in zip(box_preds,box_preds_lidar,scores,label_preds):
        anno["name"].append(class_names[label])
        anno["bbox"].append(box)
        anno["score"].append(score)
        anno["dimensions"].append(box[3:6])
        anno["location"].append(box[:3])
    print(anno["score"])
