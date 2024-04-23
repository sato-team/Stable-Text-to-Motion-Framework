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
from CLIP.clip import clip
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
clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False) # Must set jit=False for training
clip_model_teacher, preprocess_teacher = clip.load("ViT-B/32", device=device, jit=False) # Must set jit=False for training
clip_model_teacher.eval()
for p in clip_model_teacher.parameters():
    p.requires_grad = False
# clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
clip_model.train()
# for p in clip_model.parameters():
#     p.requires_grad = False

net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                       args.nb_code,
                       args.code_dim,
                       args.output_emb_width,
                       args.down_t,
                       args.stride_t,
                       args.width,
                       args.depth,
                       args.dilation_growth_rate)


trans_encoder_origin = trans.Text2Motion_Transformer(num_vq=args.nb_code, 
                                embed_dim=args.embed_dim_gpt, 
                                clip_dim=args.clip_dim, 
                                block_size=args.block_size, 
                                num_layers=args.num_layers, 
                                n_head=args.n_head_gpt, 
                                drop_out_rate=args.drop_out_rate, 
                                fc_rate=args.ff_rate)

trans_encoder_pertubation =trans.Text2Motion_Transformer(num_vq=args.nb_code, 
                                embed_dim=args.embed_dim_gpt, 
                                clip_dim=args.clip_dim, 
                                block_size=args.block_size, 
                                num_layers=args.num_layers, 
                                n_head=args.n_head_gpt, 
                                drop_out_rate=args.drop_out_rate, 
                                fc_rate=args.ff_rate)

print ('loading checkpoint from {}'.format(args.resume_pth))
ckpt = torch.load(args.resume_pth, map_location='cpu')
net.load_state_dict(ckpt['net'], strict=True)
net.eval()
net.cuda()

if args.resume_trans is not None:
    print ('loading transformer checkpoint from {}'.format(args.resume_trans))
    ckpt = torch.load(args.resume_trans, map_location='cpu')
    trans_encoder_origin.load_state_dict(ckpt['trans'], strict=True)
    trans_encoder_origin.eval()
    trans_encoder_origin.cuda()
else:
    trans_encoder_origin.eval()
    trans_encoder_origin.cuda()
trans_encoder_pertubation.load_state_dict(ckpt['trans'], strict=True)
trans_encoder_pertubation.train()
trans_encoder_pertubation.cuda()
print('loading chechpoint successfully!')
##### ---- Optimizer & Scheduler ---- #####
# 获取 clip_model 的参数
clip_params = list(clip_model.parameters())

optimizer = utils_model.initial_optim(args.decay_option, args.lr, args.weight_decay, trans_encoder_pertubation, args.optimizer, eps=1e-3)

for param in clip_params:
    optimizer.add_param_group({'params': param})

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

##### ---- Optimization goals ---- #####
loss_ce = torch.nn.CrossEntropyLoss()
loss_rob=torch.nn.L1Loss()
sim_loss=torch.nn.L1Loss()
nb_iter, avg_loss_cls, avg_acc = 0, 0., 0.
right_num = 0
nb_sample_train = 0


##### ---- get code ---- #####
for batch in train_loader_token:
    pose, name = batch
    bs, seq = pose.shape[0], pose.shape[1]

    pose = pose.cuda().float() # bs, nb_joints, joints_dim, seq_len
    target = net.encode(pose)
    target = target.cpu().numpy()
    np.save(pjoin(args.vq_dir, name[0] +'.npy'), target)



train_loader = dataset_TM_train.DATALoader(args.dataname, args.batch_size, args.nb_code, args.vq_name, unit_length=2**args.down_t)
train_loader_iter = dataset_TM_train.cycle(train_loader)
# best_fid, best_fid_per, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = eval_trans.evaluation_transformer(args.out_dir, val_loader, net, trans_encoder_pertubation, logger, writer, 0, best_fid=1000, best_fid_perturbation=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, clip_model=clip_model, eval_wrapper=eval_wrapper)
best_fid=1000.0
best_fid_syn=1000.0
best_fid_per=1000.0
best_iter=0.0 
best_div=100.0
best_top1=0.0
best_top2=0.0
best_top3=0.0
best_matching=100.0
# start_time = time.time()
def topK_process(model, text):
    # Encode and normalize the search query using CLIP
    text_token = clip.tokenize(text, truncate=True).cuda()
    # tokens = text[5].split(' ')
    tokens_lens = [1+len(txt.split(' ')) for txt in text]

    text_encoded, weight = model.encode_text(text_token)

    # text_encoded /= text_encoded.norm(dim=-1, keepdim=True)
    # attention_weights = weight[-1][0][1+len(tokens)][:2+len(tokens)][1:][:-1]
    # attention_weights = weight[-1][range(len(weight[-1])), tokens_lens][:, :1+max(tokens_lens)][:, 1:][:, :-1]

    attention_weights_all = []
    for i in range(len(tokens_lens)):
        attention_weights = weight[-1][i][min(76, tokens_lens[i])][:1+min(75, max(tokens_lens))][1:][:-1]
        attention_weights_all.append(attention_weights)
    attention_weights = torch.stack(attention_weights_all, dim=0)

    return text_encoded.float(), attention_weights

# def topK_PGD_process(model, text, PGD, encoder):
#     # Encode and normalize the search query using CLIP
#     text_token = clip.tokenize(text, truncate=True).cuda()
#     # tokens = text[5].split(' ')
#     tokens_lens = [1+len(txt.split(' ')) for txt in text]
#     x = model.token_embedding(text_token).type(model.dtype)
    
#     x, attention_weights = PGD.perturb(device, m_tokens_len, bs, criterion=crit, x=x,encoder=encoder, a_indices=a_indices,y=target, tokens_lens=tokens_lens, model=model, text_token=text_token)

#     return x.float(), attention_weights

while nb_iter <= args.total_iter:
    # if nb_iter == 1000:
    #     print('time for 1000 iterations:', time.time()-start_time)
    
    batch = next(train_loader_iter)
    clip_text, clip_text_perb, m_tokens, m_tokens_len = batch  # origin data raw text

    m_tokens, m_tokens_len = m_tokens.cuda(), m_tokens_len.cuda()
    bs = m_tokens.shape[0]
    target = m_tokens    # (bs, 26)
    target = target.cuda()

    # text = clip.tokenize(clip_text, truncate=True).cuda()
    # text_perb = clip.tokenize(clip_text_perb, truncate=True).cuda()
    # feat_clip_text = clip_model.encode_text(text)[0].float()#text_tokens
    # new_embedd=clip_model.encode_text(text_perb)[0].float()

    if random.choice([True, False]):
        clip_text = clip_text  # 无truncate
    else:
        clip_text = clip_text_perb
        


    input_index = target[:,:-1]


    # ---------------------------attention loss-----------------------------------------------

    feat_clip_text, weight_ori = topK_process(clip_model, clip_text)
    feat_clip_text_teacher, weight_ori_teacher = topK_process(clip_model_teacher, clip_text)


    


    if args.pkeep == -1:
        proba = np.random.rand(1)[0]
        mask = torch.bernoulli(proba * torch.ones(input_index.shape,
                                                         device=input_index.device))
    else:
        mask = torch.bernoulli(args.pkeep * torch.ones(input_index.shape,
                                                         device=input_index.device))
    mask = mask.round().to(dtype=torch.int64)
    r_indices = torch.randint_like(input_index, args.nb_code)
    a_indices = mask*input_index+(1-mask)*r_indices
    
    if args.method=='PGD':
        
        X_PGDer = PGDAttacker(
        radius=args.x_pgd_radius, steps=args.x_pgd_step, step_size=args.x_pgd_step_size, random_start= \
        True, norm_type=args.x_pgd_norm_type, ascending=True
        )
        
        # we will perturb  e(x) to ensure robustness of models
        new_embedd = X_PGDer.perturb(device, m_tokens_len, bs, criterion=crit, x=feat_clip_text,encoder=trans_encoder_pertubation, a_indices=a_indices,y=target)
        new_embedd, weight_perb = topK_PGD_process(clip_model, clip_text, X_PGDer, encoder=trans_encoder_pertubation)
        loss_attention = topk_overlap_loss(weight_ori, weight_perb).sum()
    else:
        new_embedd, weight_perb = topK_process(clip_model, clip_text_perb)
        loss_attention = topk_overlap_loss(weight_ori, weight_perb).sum()
    
    cls_pred_sim_origin=trans_encoder_origin(a_indices,feat_clip_text_teacher)

    cls_pred_origin=trans_encoder_pertubation(a_indices, feat_clip_text)
    cls_pred_per = trans_encoder_pertubation(a_indices, new_embedd)

    cls_pred_origin = cls_pred_origin.contiguous()
    # cls_pred_sim_origin = cls_pred_sim_origin.contiguous()
    cls_pred_per = cls_pred_per.contiguous()
     

    sim_loss =0.0
    adv_loss =0.0
    loss_cls = 0.0
    for i in range(bs):
        # loss function     (26), (26, 513)
        loss_cls += loss_ce(cls_pred_per[i][:m_tokens_len[i] + 1], target[i][:m_tokens_len[i] + 1]) / bs
        adv_loss += loss_rob(cls_pred_origin[i][:m_tokens_len[i] + 1], cls_pred_per[i][:m_tokens_len[i] + 1]) / bs
        sim_loss+= loss_rob(cls_pred_origin[i][:m_tokens_len[i] + 1], cls_pred_sim_origin[i][:m_tokens_len[i] + 1]) / bs
        # Accuracy
        probs = torch.softmax(cls_pred_per[i][:m_tokens_len[i] + 1], dim=-1)

        if args.if_maxtest:
            _, cls_pred_index = torch.max(probs, dim=-1)

        else:
            dist = Categorical(probs)
            cls_pred_index = dist.sample()
        right_num += (cls_pred_index.flatten(0) == target[i][:m_tokens_len[i] + 1].flatten(0)).sum().item()




    loss_total=loss_cls+sim_loss*args.lambda_1 + adv_loss*args.lambda_2 + loss_attention*args.lambda_3
        # loss_total=loss_cls
    ## global loss
    if args.wandb==True:
        wandb.log({'loss_cls':loss_cls,'sim_loss':sim_loss,'adv_loss':adv_loss, 'attention': loss_attention})
    optimizer.zero_grad()
    loss_total.backward()
    optimizer.step()
    scheduler.step()

    avg_loss_cls = avg_loss_cls + loss_total
    nb_sample_train = nb_sample_train + (m_tokens_len + 1).sum().item()

    nb_iter += 1
    if nb_iter % args.print_iter ==  0 :
        avg_loss_cls = avg_loss_cls / args.print_iter
        avg_acc = right_num * 100 / nb_sample_train
        writer.add_scalar('./Loss/train', avg_loss_cls, nb_iter)
        writer.add_scalar('./ACC/train', avg_acc, nb_iter)
        msg = f"Train. Iter {nb_iter} : Loss. {avg_loss_cls:.5f}, ACC. {avg_acc:.4f}"
        logger.info(msg)
        if args.wandb==True:
            wandb.log({'Loss':avg_loss_cls,'ACC':avg_acc})
        avg_loss_cls = 0.
        right_num = 0
        nb_sample_train = 0
        
    if nb_iter % args.eval_iter ==  0:
        # if nb_iter < 3000:
        #     continue
        # clip_model.eval()
        # try:
        best_fid, best_fid_syn,best_fid_per,best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = eval_trans.evaluation_transformer(args.out_dir, val_loader, net, trans_encoder_pertubation, logger, writer, nb_iter, best_fid,best_fid_syn,best_fid_per, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, clip_model=clip_model, eval_wrapper=eval_wrapper)
        
        if args.wandb==True:
            wandb.log({'best_fid':best_fid,'best_fid_per':best_fid_per,'best_fid_syn':best_fid_syn,'best_iter':best_iter,'best_div':best_div,'best_top1':best_top1,'best_top2':best_top2,'best_top3':best_top3,'best_matching':best_matching})
        # except:
        #     print(nb_iter)
        # clip_model.train()
    if nb_iter == args.total_iter: 
        try:
            msg_final = f"Train. Iter {best_iter} : FID. {best_fid:.5f}, best_fid_syn. {best_fid_syn:.5f},FID_PER.{best_fid_per:.5f},Diversity. {best_div:.4f}, TOP1. {best_top1:.4f}, TOP2. {best_top2:.4f}, TOP3. {best_top3:.4f}"
            logger.info(msg_final)
            break
        except:
            print('total_iter can not eval') 
            break           
    