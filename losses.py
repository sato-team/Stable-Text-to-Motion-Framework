from CLIP.clip import clip
from CLIP.clip import model
import torch

def topk_overlap_loss(gt, pred, K=2, metric='l1'):
    idx = torch.argsort(gt, descending=True)
    # print(idx)
    idx = idx[:K]
    pred_TopK_1 = pred.gather(-1,idx)
    gt_Topk_1 = gt.gather(-1,idx)

    idx_pred = torch.argsort(pred, descending=True)
    idx_pred = idx_pred[:K]
    try:
        gt_TopK_2 = gt.gather(-1, idx_pred)
    except Exception as e:
        print(e)
        print(gt.shape)
        print(idx_pred.shape)
    pred_TopK_2 = pred.gather(-1, idx_pred)

    gt_Topk_1_normed = torch.nn.functional.softmax(gt_Topk_1, dim=-1)
    pred_TopK_1_normed = torch.nn.functional.softmax(pred_TopK_1, dim=-1)
    gt_TopK_2_normed = torch.nn.functional.softmax(gt_TopK_2, dim=-1)
    pred_TopK_2_normed = torch.nn.functional.softmax(pred_TopK_2, dim=-1)

    def kl(a,b):
        return torch.nn.functional.kl_div(a.log(), b, reduction="batchmean")

    def jsd(a,b):
        loss = kl(a,b) + kl(b,a)
        loss /= 2
        return loss


    if metric == 'l1':
        loss = torch.abs((pred_TopK_1 - gt_Topk_1)) + torch.abs(gt_TopK_2 - pred_TopK_2)
        loss = loss/(2*K)
    elif metric == "l2":
        loss = torch.norm(pred_TopK_1 - gt_Topk_1, p=2) + torch.norm(gt_TopK_2 - pred_TopK_2, p=2)
        loss = loss/(2*K)
    elif metric == "kl-full":
        loss = kl(gt,pred)
    elif metric == "jsd-full":
        loss = jsd(gt,pred)
    elif metric == "kl-topk":
        loss = kl(gt_Topk_1_normed,pred_TopK_1_normed) + kl(gt_TopK_2_normed,pred_TopK_2_normed)
        loss /=2
    elif metric == "jsd-topk":
        loss = jsd(gt_Topk_1_normed, pred_TopK_1_normed) + jsd(gt_TopK_2_normed, pred_TopK_2_normed)
        loss /= 2
    return loss

def topk_overlap_loss_batch(gt,pred,K=2,metric='l1'):
    idx = torch.argsort(gt,dim=1,descending=True)
    # print(idx)
    idx = idx[:,:K]
    pred_TopK_1 = pred.gather(1,idx)
    gt_Topk_1 = gt.gather(1,idx)

    idx_pred = torch.argsort(pred,dim=1,descending=True)
    idx_pred = idx_pred[:,:K]
    try:
        gt_TopK_2 = gt.gather(1, idx_pred)
    except Exception as e:
        print(e)
        print(gt.shape)
        print(idx_pred.shape)
    pred_TopK_2 = pred.gather(1, idx_pred)

    gt_Topk_1_normed = torch.nn.functional.softmax(gt_Topk_1,dim=-1)
    pred_TopK_1_normed = torch.nn.functional.softmax(pred_TopK_1,dim=-1)
    gt_TopK_2_normed = torch.nn.functional.softmax(gt_TopK_2,dim=-1)
    pred_TopK_2_normed = torch.nn.functional.softmax(pred_TopK_2,dim=-1)

    def kl(a,b):
        return torch.nn.functional.kl_div(a.log(), b, reduction="batchmean")

    def jsd(a,b):
        loss = kl(a,b) + kl(b,a)
        loss /= 2
        return loss


    if metric == 'l1':
        loss = torch.abs((pred_TopK_1 - gt_Topk_1)) + torch.abs(gt_TopK_2 - pred_TopK_2)
        loss = loss/(2*K)
    elif metric == "l2":
        loss = torch.norm(pred_TopK_1 - gt_Topk_1, p=2) + torch.norm(gt_TopK_2 - pred_TopK_2, p=2)
        loss = loss/(2*K)
    elif metric == "kl-full":
        loss = kl(gt,pred)
    elif metric == "jsd-full":
        loss = jsd(gt,pred)
    elif metric == "kl-topk":
        loss = kl(gt_Topk_1_normed,pred_TopK_1_normed) + kl(gt_TopK_2_normed,pred_TopK_2_normed)
        loss /=2
    elif metric == "jsd-topk":
        loss = jsd(gt_Topk_1_normed, pred_TopK_1_normed) + jsd(gt_TopK_2_normed, pred_TopK_2_normed)
        loss /= 2
    return loss

