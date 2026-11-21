# -*- coding: utf8 -*-
"""统一对比损失：same_loss（CVMCL）与 supcon_loss（SupCon）。

- same_loss  : CVMCL，按**查询图文件名**分组。组内（≥2 个不同物体）self-as-positive
               InfoNCE 把不同物体推开；同一物体的不同卫星视角（同 click）排除出负样本
               （不推开）；单物体收集起来跨图做 InfoNCE 推开。self 永远存在，损失不为空。
- supcon_loss: 物体级 SupCon（正样本 = 同场景 + 同 click）→ 拉近同一物体多视角，
               推开其它样本。
"""
import torch
import torch.nn.functional as F


def _object_key(qn, click_xy):
    """物体身份 = (查询图文件名, 四舍五入的点击点)。"""
    return (qn, round(float(click_xy[0]), 1), round(float(click_xy[1]), 1))


def info_nce_self_pos(x, obj_keys, tau=0.1):
    """
    self-as-positive InfoNCE：
    - 正样本 = 自己（i==i）；
    - 负样本 = 不同物体（不同 obj_key）；
    - 同一物体（同 obj_key）的其它视角被排除，既不拉近也不推开。
    x: (B, C, H, W)；obj_keys: (B,) 物体身份键序列。
    """
    B, C, H, W = x.shape
    x_flat = F.adaptive_avg_pool2d(x, output_size=(1, 1)).squeeze(-1).squeeze(-1)
    x_norm = F.normalize(x_flat, dim=1)
    sim = torch.matmul(x_norm, x_norm.T)  # (B, B)

    same_obj = torch.zeros(B, B, dtype=torch.bool, device=x.device)
    for i in range(B):
        for j in range(B):
            if obj_keys[i] == obj_keys[j]:
                same_obj[i, j] = True

    self_mask = torch.eye(B, dtype=torch.bool, device=x.device)
    neg_mask = ~self_mask & ~same_obj  # 不同物体

    losses = []
    for i in range(B):
        negs = neg_mask[i].nonzero().view(-1)
        if negs.numel() == 0:
            continue
        pos = sim[i, i].unsqueeze(0)
        neg = sim[i, negs]
        logits = torch.cat([pos, neg]) / tau
        label = torch.zeros(1, dtype=torch.long, device=x.device)
        losses.append(F.cross_entropy(logits.unsqueeze(0), label))
    if not losses:
        return x_flat.sum() * 0.0
    return torch.stack(losses).mean()


def same_loss(fused_features, query_names, click_xy, tau=0.1):
    """
    CVMCL（同图排斥）：
    - 按查询图文件名分组；
    - 组内有 ≥2 个不同物体（不同 click）：不同物体之间做 self-as-positive InfoNCE 推开，
      同一物体的不同卫星视角（同 click）被排除，不推开；
    - 单物体（一张图只有 1 个物体）收集起来跨图做 InfoNCE 推开（同物体视角同样排除）。
    因为 self 永远存在，每个 batch 损失都不为空（batch ≥ 2 时）。
    fused_features: (B, C, H, W)
    query_names: (B,) 查询图文件名序列
    click_xy: (B, 2) 点击点
    """
    groups = {}
    for i, qn in enumerate(query_names):
        groups.setdefault(qn, []).append(i)

    multi = []
    single = []
    for qn, g in groups.items():
        n_obj = len({(round(float(click_xy[i][0]), 1), round(float(click_xy[i][1]), 1)) for i in g})
        if n_obj >= 2:
            multi.append(g)
        else:
            single.extend(g)

    total = fused_features.sum() * 0.0
    for g in multi:
        keys = [_object_key(query_names[i], click_xy[i]) for i in g]
        total = total + info_nce_self_pos(fused_features[g], keys, tau=tau)
    if len(single) >= 2:
        keys = [_object_key(query_names[i], click_xy[i]) for i in single]
        total = total + info_nce_self_pos(fused_features[single], keys, tau=tau)
    return total


def supcon_loss(features, scene_id, click_xy, temperature=0.07):
    """
    Supervised contrastive loss on the fused features.
    Object identity = (scene id, click point): samples of the same
    object (same scene + same click, e.g. the same object seen in different
    satellite views) are positives; all other samples are negatives.
    features: (B, C, H, W) or (B, C)
    scene_id: (B,) int tensor, scene id (idx field in the annotation)
    click_xy: (B, 2) tensor, (x, y) click coordinate in query image
    """
    if features.dim() == 4:
        feats = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
    else:
        feats = features
    feats = F.normalize(feats, dim=-1)
    B = feats.shape[0]
    device = feats.device

    # build object identity labels by grouping (scene id, rounded click)
    key_to_label = {}
    labels = []
    for i in range(B):
        sid = int(scene_id[i])
        cx = round(float(click_xy[i][0]), 1)
        cy = round(float(click_xy[i][1]), 1)
        key = (sid, cx, cy)
        if key not in key_to_label:
            key_to_label[key] = len(key_to_label)
        labels.append(key_to_label[key])
    labels = torch.tensor(labels, device=device)

    # positive mask: same object and not self
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~torch.eye(B, dtype=torch.bool, device=device)

    sim_raw = torch.matmul(feats, feats.t()) / temperature
    sim = sim_raw.masked_fill(torch.eye(B, dtype=torch.bool, device=device), float('-inf'))
    sim_max = sim.max(dim=1, keepdim=True)[0].detach()
    exp_sim = torch.exp(sim - sim_max)
    denom_log = sim_max + torch.log(exp_sim.sum(dim=1, keepdim=True))

    num_pos = pos_mask.sum(dim=1).float()
    valid = num_pos > 0
    if not valid.any():
        return feats.sum() * 0.0

    pos_mean = (sim_raw * pos_mask.float()).sum(dim=1, keepdim=True) / num_pos.clamp(min=1).unsqueeze(1)
    loss_per = (denom_log - pos_mean).squeeze(1)
    return loss_per[valid].mean()


def compute_contrastive_loss(loss_type, fused_features, query_names, scene_id, click_xy):
    """根据 loss_type 选择对比损失；unknown 返回 0。"""
    if loss_type == 'same':
        return same_loss(fused_features, query_names, click_xy)
    elif loss_type == 'supcon':
        return supcon_loss(fused_features, scene_id, click_xy)
    else:
        return fused_features.sum() * 0.0
