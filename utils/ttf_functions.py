import torch 
from torch import nn 

def bbox_areas(bboxes, keep_axis=False):
    x_min, y_min, x_max, y_max = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    areas = (y_max - y_min + 1) * (x_max - x_min + 1)
    if keep_axis:
        return areas[:, None]
    return areas

def calc_region(bbox, ratio, featmap_size=None):
    """Calculate a proportional bbox region.

    The bbox center are fixed and the new h' and w' is h * ratio and w * ratio.

    Args:
        bbox (Tensor): Bboxes to calculate regions, shape (n, 4)
        ratio (float): Ratio of the output region.
        featmap_size (tuple): Feature map size used for clipping the boundary.

    Returns:
        tuple: x1, y1, x2, y2
    """
    x1 = torch.round((1 - ratio) * bbox[0] + ratio * bbox[2]).long()
    y1 = torch.round((1 - ratio) * bbox[1] + ratio * bbox[3]).long()
    x2 = torch.round(ratio * bbox[0] + (1 - ratio) * bbox[2]).long()
    y2 = torch.round(ratio * bbox[1] + (1 - ratio) * bbox[3]).long()
    if featmap_size is not None:
        x1 = x1.clamp(min=0, max=featmap_size[1] - 1)
        y1 = y1.clamp(min=0, max=featmap_size[0] - 1)
        x2 = x2.clamp(min=0, max=featmap_size[1] - 1)
        y2 = y2.clamp(min=0, max=featmap_size[0] - 1)
    return (x1, y1, x2, y2)
def simple_nms(heat, kernel=3, out_heat=None):
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    out_heat = heat if out_heat is None else out_heat
    return out_heat * keep
def ct_focal_loss(pred, gt, gamma=2.0):
    """
    Focal loss used in CornerNet & CenterNet. Note that the values in gt (label) are in [0, 1] since
    gaussian is used to reduce the punishment and we treat [0, 1) as neg example.

    Args:
        pred: tensor, any shape.
        gt: tensor, same as pred.
        gamma: gamma in focal loss.

    Returns:

    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)  # reduce punishment
    pos_loss = -torch.log(pred) * torch.pow(1 - pred, gamma) * pos_inds
    neg_loss = -torch.log(1 - pred) * torch.pow(pred, gamma) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        return neg_loss
    return (pos_loss + neg_loss) / num_pos
def calc_region(bbox, ratio, featmap_size=None):
    """Calculate a proportional bbox region.

    The bbox center are fixed and the new h' and w' is h * ratio and w * ratio.

    Args:
        bbox (Tensor): Bboxes to calculate regions, shape (n, 4)
        ratio (float): Ratio of the output region.
        featmap_size (tuple): Feature map size used for clipping the boundary.

    Returns:
        tuple: x1, y1, x2, y2
    """
    x1 = torch.round((1 - ratio) * bbox[0] + ratio * bbox[2]).long()
    y1 = torch.round((1 - ratio) * bbox[1] + ratio * bbox[3]).long()
    x2 = torch.round(ratio * bbox[0] + (1 - ratio) * bbox[2]).long()
    y2 = torch.round(ratio * bbox[1] + (1 - ratio) * bbox[3]).long()
    if featmap_size is not None:
        x1 = x1.clamp(min=0, max=featmap_size[1] - 1)
        y1 = y1.clamp(min=0, max=featmap_size[0] - 1)
        x2 = x2.clamp(min=0, max=featmap_size[1] - 1)
        y2 = y2.clamp(min=0, max=featmap_size[0] - 1)
    return (x1, y1, x2, y2)
def giou_loss(pred,
              target,
              weight,
              avg_factor=None):
    """GIoU loss.
    Computing the GIoU loss between a set of predicted bboxes and target bboxes.
    """
    pos_mask = weight > 0
    weight = weight[pos_mask].float()
    if avg_factor is None:
        avg_factor = torch.sum(pos_mask).float().item() + 1e-6
    bboxes1 = pred[pos_mask].view(-1, 4)
    bboxes2 = target[pos_mask].view(-1, 4)

    lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
    rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]
    wh = (rb - lt + 1).clamp(min=0)  # [rows, 2]
    enclose_x1y1 = torch.min(bboxes1[:, :2], bboxes2[:, :2])
    enclose_x2y2 = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1 + 1).clamp(min=0)

    overlap = wh[:, 0] * wh[:, 1]
    ap = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (bboxes1[:, 3] - bboxes1[:, 1] + 1)
    ag = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (bboxes2[:, 3] - bboxes2[:, 1] + 1)
    ious = overlap / (ap + ag - overlap)

    enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1]  # i.e. C in paper
    u = ap + ag - overlap
    gious = ious - (enclose_area - u) / enclose_area
    iou_distances = 1 - gious
    return torch.sum(iou_distances * weight)[None] / avg_factor

def _topk(scores, topk):
    batch, cat, height, width = scores.size()

    # both are (batch, 80, topk)
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), topk)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    # both are (batch, topk). select topk from 80*topk
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), topk)
    topk_clses = (topk_ind / topk).int()
    topk_ind = topk_ind.unsqueeze(2)
    topk_inds = topk_inds.view(batch, -1, 1).gather(1, topk_ind).view(batch, topk)
    topk_ys = topk_ys.view(batch, -1, 1).gather(1, topk_ind).view(batch, topk)
    topk_xs = topk_xs.view(batch, -1, 1).gather(1, topk_ind).view(batch, topk)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs
def ttf_decode(pred_heatmap,
               pred_wh,
               topk=200,
               down_ratio=4,
               output_h=128,
               output_w=128,
               score_thr=0.01,
               wh_agnostic=True,
               num_fg=1,
               rescale=False):
    batch, cat, height, width = pred_heatmap.size()
    pred_heatmap = pred_heatmap.detach().sigmoid_()
    wh = pred_wh.detach()

    # perform nms on heatmaps
    heat = simple_nms(pred_heatmap)  # used maxpool to filter the max score

    # topk = getattr(cfg, 'max_per_img', 100)
    topk = topk
    # (batch, topk)
    scores, inds, clses, ys, xs = _topk(heat, topk=topk)
    xs = xs.view(batch, topk, 1) * down_ratio
    ys = ys.view(batch, topk, 1) * down_ratio

    wh = wh.permute(0, 2, 3, 1).contiguous()
    wh = wh.view(wh.size(0), -1, wh.size(3))
    inds = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), wh.size(2))
    wh = wh.gather(1, inds)

    if not wh_agnostic:
        wh = wh.view(-1, topk, num_fg, 4)
        wh = torch.gather(wh, 2, clses[..., None, None].expand(
            clses.size(0), clses.size(1), 1, 4).long())

    wh = wh.view(batch, topk, 4)
    clses = clses.view(batch, topk, 1).float()
    scores = scores.view(batch, topk, 1)

    bboxes = torch.cat([xs - wh[..., [0]], ys - wh[..., [1]],
                        xs + wh[..., [2]], ys + wh[..., [3]]], dim=2)

    result_list = []
    # score_thr = getattr(cfg, 'score_thr', 0.01)
    score_thr=score_thr
    for batch_i in range(bboxes.shape[0]):
        scores_per_img = scores[batch_i]
        scores_keep = (scores_per_img > score_thr).squeeze(-1)

        scores_per_img = scores_per_img[scores_keep]
        bboxes_per_img = bboxes[batch_i][scores_keep]
        labels_per_img = clses[batch_i][scores_keep]
        # img_shape = img_metas[batch_i]['pad_shape']
        img_shape = [output_h*4,output_w*4]
        bboxes_per_img[:, 0::2] = bboxes_per_img[:, 0::2].clamp(min=0, max=img_shape[1] - 1)
        bboxes_per_img[:, 1::2] = bboxes_per_img[:, 1::2].clamp(min=0, max=img_shape[0] - 1)

        # if rescale:
        #     scale_factor = img_metas[batch_i]['scale_factor']
        #     bboxes_per_img /= bboxes_per_img.new_tensor(scale_factor)

        bboxes_per_img = torch.cat([bboxes_per_img, scores_per_img], dim=1)
        labels_per_img = labels_per_img.squeeze(-1)
        result_list.append((bboxes_per_img, labels_per_img))

    return result_list