# SSD检测头
import torch
from core import box_torch_ops
def second_box_decode(box_encodings, anchors, encode_angle_to_vector=False, smooth_dim=False):
    """box decode for VoxelNet in lidar
    Args:
        boxes ([N, 7] Tensor): normal boxes: x, y, z, w, l, h, r
        anchors ([N, 7] Tensor): anchors
    """
    xa, ya, za, wa, la, ha, ra = torch.split(anchors, 1, dim=-1)
    if encode_angle_to_vector:
        xt, yt, zt, wt, lt, ht, rtx, rty = torch.split(
            box_encodings, 1, dim=-1)

    else:
        xt, yt, zt, wt, lt, ht, rt = torch.split(box_encodings, 1, dim=-1)

    # xt, yt, zt, wt, lt, ht, rt = torch.split(box_encodings, 1, dim=-1)
    za = za + ha / 2
    diagonal = torch.sqrt(la**2 + wa**2)
    xg = xt * diagonal + xa
    yg = yt * diagonal + ya
    zg = zt * ha + za
    if smooth_dim:
        lg = (lt + 1) * la
        wg = (wt + 1) * wa
        hg = (ht + 1) * ha
    else:

        lg = torch.exp(lt) * la
        wg = torch.exp(wt) * wa
        hg = torch.exp(ht) * ha
    if encode_angle_to_vector:
        rax = torch.cos(ra)
        ray = torch.sin(ra)
        rgx = rtx + rax
        rgy = rty + ray
        rg = torch.atan2(rgy, rgx)
    else:
        rg = rt + ra
    zg = zg - hg / 2
    return torch.cat([xg, yg, zg, wg, lg, hg, rg], dim=-1)

def predict(batch_anchor,box_preds,cls_preds,dir_cls_preds,anchors_mask):
    batch_size = 1
    nms_score_threshold = 0.05
    batch_anchors = batch_anchor
    #self._total_inference_count += batch_size
    batch_anchors_mask = anchors_mask.view(batch_size, -1)
    batch_box_preds = box_preds
    batch_cls_preds = cls_preds
    batch_box_preds = batch_box_preds.contiguous().view(1,-1,7)
    num_class_with_bg = 1
    batch_cls_preds = batch_cls_preds.contiguous().view(1,-1, 1)
    batch_box_preds = second_box_decode(batch_box_preds,batch_anchors)
    batch_dir_preds = dir_cls_preds
    batch_dir_preds = batch_dir_preds.contiguous().view(batch_size, -1, 2)
    predictions_dicts = []
    for box_preds, cls_preds, dir_preds,a_mask in zip(batch_box_preds,batch_cls_preds,batch_dir_preds,batch_anchors_mask):
        if a_mask is not None:
            box_preds = box_preds[a_mask]
            cls_preds = cls_preds[a_mask]
        if a_mask is not None:
            dir_preds = dir_preds[a_mask]
        dir_labels = torch.max(dir_preds, dim=-1)[1]
        total_scores = torch.sigmoid(cls_preds)
        print(total_scores)
        nms_func = box_torch_ops.nms
        selected_boxes = None
        selected_labels = None
        selected_scores = None
        selected_dir_labels = None
        if num_class_with_bg == 1:
            top_scores = total_scores.squeeze(-1)
            top_labels = torch.zeros(total_scores.shape[0],device=total_scores.device,dtype=torch.long)
        if nms_score_threshold >0.0:
            thresh = torch.tensor([nms_score_threshold],device=total_scores.device).type_as(total_scores)
            top_scores_keep = (top_scores >= thresh)
            top_scores = top_scores.masked_select(top_scores_keep)
        if top_scores.shape[0] != 0:
            if nms_score_threshold > 0.0:
                box_preds = box_preds[top_scores_keep]
                dir_labels = dir_labels[top_scores_keep]
                top_labels = top_labels[top_scores_keep]
            boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
            box_preds_corners = box_torch_ops.center_to_corner_box2d(boxes_for_nms[:, :2], boxes_for_nms[:, 2:4],boxes_for_nms[:, 4])
            print(box_preds_corners)
            boxes_for_nms = box_torch_ops.corner_to_standup_nd(box_preds_corners)
            # the nms in 3d detection just remove overlap boxes.
            selected = nms_func(
                        boxes_for_nms,
                        top_scores,
                        pre_max_size=self._nms_pre_max_size,
                        post_max_size=self._nms_post_max_size,
                        iou_threshold=self._nms_iou_threshold,
                    )
        else:
            selected = None
        if selected is not None:
            selected_boxes = box_preds[selected]
            selected_dir_labels = dir_labels[selected]
            selected_labels = top_labels[selected]
            selected_scores = top_scores[selected]
        # finally generate predictions.
        # 最终生成预测
        if selected_boxes is not None:
            box_preds = selected_boxes
            scores = selected_scores
            label_preds = selected_labels
            dir_labels = selected_dir_labels
            opp_labels = (box_preds[..., -1] > 0) ^ dir_labels.bool()
            box_preds[..., -1] += torch.where(
                        opp_labels,
                        torch.tensor(np.pi).type_as(box_preds),
                        torch.tensor(0.0).type_as(box_preds))
            final_box_preds = box_preds
            final_scores = scores
            final_labels = label_preds
            camera_box_origin = [0.5, 1.0, 0.5]
            box_corners = box_torch_ops.center_to_corner_box3d(
                    locs, dims, angles, camera_box_origin, axis=1)
            minxy = torch.min(box_corners_in_image, dim=1)[0]
            maxxy = torch.max(box_corners_in_image, dim=1)[0]
            box_2d_preds = torch.cat([minxy, maxxy], dim=1)
            predictions_dict = {
                    "bbox": box_2d_preds,
                    "box3d_lidar": final_box_preds,
                    "scores": final_scores,
                    "label_preds": label_preds,
                }
        else:
            predictions_dict = {
                    "bbox": None,
                    "box3d_lidar": None,
                    "scores": None,
                    "label_preds": None,
                }
        return predictions_dict

        










