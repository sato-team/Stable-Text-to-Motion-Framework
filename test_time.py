import os 
os.chdir('/root/autodl-tmp/SATO')
import torch
import numpy as np
import time

from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin
from torch.distributions import Categorical
import json
# import clip
from CLIP import clip
from attack import PGDAttacker
import options.option_transformer as option_trans
import models.vqvae as vqvae
import utils.utils_model as utils_model
import eval_trans_per as eval_trans
from dataset import dataset_TM_train
from dataset import dataset_TM_eval
from dataset import dataset_tokenize 
from metrics import batch_tvd,topk_overlap_loss,topK_overlap_true_loss
import models.t2m_trans as trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
from utils.word_vectorizer import WordVectorizer
from utils.losses import loss_robust
import wandb
import random
import time
from tqdm import tqdm
warnings.filterwarnings('ignore')
##### ---- Exp dirs ---- #####
args = option_trans.get_args_parser()
torch.manual_seed(args.seed)
device = 'cuda'
args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
args.vq_dir= os.path.join("./dataset/KIT-ML" if args.dataname == 'kit' else "./dataset/HumanML3D", f'{args.vq_name}')
os.makedirs(args.out_dir, exist_ok = True)
os.makedirs(args.vq_dir, exist_ok = True)
##wandb init
if args.wandb==True:
    wandb.init(config=args,
            project='T2M_adv-4090', group='_xhr0313', name='clip_train add loss attention adv-0.1')
##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

##### ---- preperation ---- #####
def target_model(embedd,encoder):
            logits = encoder(embedd)
            return torch.sigmoid(logits)
def target_model_sta(embedd, data, decoder,encoder):
    encoder(data,revise_embedding=embedd)
    decoder(data=data)
    return data.attn
# def crit(gt, pred):
#     return batch_tvd(gt,pred)
def crit(gt, pred, m_tokens_len, bs):
    loss_cls=0.0 
    for i in range(bs):
        # loss function     (26), (26, 513)
        loss_cls += loss_ce(pred[i][:m_tokens_len[i] + 1], gt[i][:m_tokens_len[i] + 1])/ bs
    return loss_cls

def crit_sta(gt, pred):
    return topk_overlap_loss(gt, pred,K=args.K).mean()
##### ---- Dataloader ---- #####
train_loader_token = dataset_tokenize.DATALoader(args.dataname, 1, unit_length=2**args.down_t)


w_vectorizer = WordVectorizer('./glove', 'our_vab')
val_loader = dataset_TM_eval.DATALoader(args.dataname, False, 32, w_vectorizer)

dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if args.dataname == 'kit' else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'

wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
##### ---- Network ---- #####
# clip_model, clip_preprocess = clip.load("/root/autodl-tmp/t2m/T2M-GPT-main/clip_weights/best_model_jsd_tvd.pt", device=torch.device('cuda'), jit=False)  # Must set jit=False for training
os.chdir('/root/autodl-tmp/SATO')
## load clip model and datasets
clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False)  # Must set jit=False for training
clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                       args.nb_code,
                       args.code_dim,
                       args.output_emb_width,
                       args.down_t,
                       args.stride_t,
                       args.width,
                       args.depth,
                       args.dilation_growth_rate)


trans_encoder = trans.Text2Motion_Transformer(num_vq=args.nb_code, 
                                embed_dim=1024, 
                                clip_dim=args.clip_dim, 
                                block_size=args.block_size, 
                                num_layers=9, 
                                n_head=16, 
                                drop_out_rate=args.drop_out_rate, 
                                fc_rate=args.ff_rate)


print ('loading checkpoint from {}'.format(args.resume_pth))
ckpt = torch.load(args.resume_pth, map_location='cpu')
net.load_state_dict(ckpt['net'], strict=True)
net.eval()
net.cuda()

print ('loading transformer checkpoint from {}'.format(args.resume_trans))
ckpt = torch.load(args.resume_trans, map_location='cpu')
trans_encoder.load_state_dict(ckpt['trans'], strict=True)
trans_encoder.eval()
trans_encoder.cuda()
print('loading chechpoint successfully!')
for batch in train_loader_token:
    pose, name = batch
    bs, seq = pose.shape[0], pose.shape[1]

    pose = pose.cuda().float() # bs, nb_joints, joints_dim, seq_len
    target = net.encode(pose)
    target = target.cpu().numpy()
    np.save(pjoin(args.vq_dir, name[0] +'.npy'), target)



train_loader = dataset_TM_train.DATALoader(args.dataname, 1, args.nb_code, args.vq_name, unit_length=2**args.down_t)
train_loader_iter = dataset_TM_train.cycle(train_loader)
mean = torch.from_numpy(np.load('./checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy')).cuda()
std = torch.from_numpy(np.load('./checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy')).cuda()
print('start')
start_time=time.time()
n=0
for batch in tqdm(train_loader_iter):
    clip_text, clip_text_perb, m_tokens, m_tokens_len = batch
    text = clip.tokenize(clip_text, truncate=True).cuda()
    feat_clip_text = clip_model.encode_text(text)[0].float()
    index_motion = trans_encoder.sample(feat_clip_text[0:1], False)
    pred_pose = net.forward_decoder(index_motion)
    n+=1
    
end_time=time.time()

print(start_time-end_time)