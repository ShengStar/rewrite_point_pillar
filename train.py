from net import backbone
from process import read_point,pointpillars,voxel_feature_extractor,Scatter
import torch
from torch import nn
import torchplus
from torchplus import metrics

class bulid_net(nn.Module):
    def __init__(self,):
        super().__init__()
        self.rpn_acc = metrics.Accuracy(
            dim=-1, encode_background_as_zeros=True)
        self.rpn_precision = metrics.Precision(dim=-1)
        self.rpn_recall = metrics.Recall(dim=-1)
        self.rpn_metrics = metrics.PrecisionRecall(
            dim=-1,
            thresholds=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95],
            use_sigmoid_score=True,
            encode_background_as_zeros=True)

        self.rpn_cls_loss = metrics.Scalar()
        self.rpn_loc_loss = metrics.Scalar()
        self.rpn_total_loss = metrics.Scalar()
        self.register_buffer("global_step", torch.LongTensor(1).zero_())
        self.voxel_feature_extractor = voxel_feature_extractor()
        self.rpn = backbone()
    def forward(self, filename):
        readpoint = read_point()
        x = readpoint(filename)  # 读取点云
        pointpillar = pointpillars()
        voxels, num_points_per_voxel, coors = pointpillar(x)  # 划分pillars
        #feature_expand = features_expand()
        #print(features_expand())
        features_9 = self.voxel_feature_extractor(voxels, num_points_per_voxel, coors)  # 特征拓展
        scatter = Scatter()
        x = scatter(features_9, coors, 1)
        #print(backbone())
        ret_dict =self.rpn(x)
        return ret_dict
