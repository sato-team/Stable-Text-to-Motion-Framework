import torch
import torch.nn as nn
import torch.nn.functional as F
class ReConsLoss(nn.Module):
    def __init__(self, recons_loss, nb_joints):
        super(ReConsLoss, self).__init__()
        
        if recons_loss == 'l1': 
            self.Loss = torch.nn.L1Loss()
        elif recons_loss == 'l2' : 
            self.Loss = torch.nn.MSELoss()
        elif recons_loss == 'l1_smooth' : 
            self.Loss = torch.nn.SmoothL1Loss()
        
        # 4 global motion associated to root
        # 12 local motion (3 local xyz, 3 vel xyz, 6 rot6d)
        # 3 global vel xyz
        # 4 foot contact
        self.nb_joints = nb_joints
        self.motion_dim = (nb_joints - 1) * 12 + 4 + 3 + 4
        
    def forward(self, motion_pred, motion_gt) : 
        loss = self.Loss(motion_pred[..., : self.motion_dim], motion_gt[..., :self.motion_dim])
        return loss
    
    def forward_vel(self, motion_pred, motion_gt) : 
        loss = self.Loss(motion_pred[..., 4 : (self.nb_joints - 1) * 3 + 4], motion_gt[..., 4 : (self.nb_joints - 1) * 3 + 4])
        return loss
    
def loss_robust(origin_prediction,perturbation_prediction,loss='l2'):
    if loss=='l1':
        return torch.nn.L1Loss(origin_prediction,perturbation_prediction)
    if loss=='l2':
        return torch.nn.MSELoss(origin_prediction,perturbation_prediction)
    if loss=='l1_smooth':
        return torch.nn.SmoothL1Loss(origin_prediction,perturbation_prediction)
    if loss=='jsd':
      return calculate_kl_divergence(origin_prediction,perturbation_prediction)


def calculate_kl_divergence(p, q):
    return torch.sum(p * (torch.log2(p) - torch.log2(q)))

def calculate_jsd_loss(p, q):
    # 将概率分布标准化为概率密度
    p_normalized = F.normalize(p, p=1, dim=-1)
    q_normalized = F.normalize(q, p=1, dim=-1)

    # 计算平均分布
    m = 0.5 * (p_normalized + q_normalized)

    # 计算两个分布与平均分布的 KL 散度
    kl_p = calculate_kl_divergence(p_normalized, m)
    kl_q = calculate_kl_divergence(q_normalized, m)

    # 计算 JSD
    jsd_loss = 0.5 * (kl_p + kl_q)

    return jsd_loss