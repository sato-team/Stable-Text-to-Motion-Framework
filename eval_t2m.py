import os 
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import json
# import clip
from CLIP import clip

import options.option_transformer as option_trans
import models.vqvae as vqvae
import utils.utils_model as utils_model
import eval_trans_per as eval_trans
from dataset import dataset_TM_eval
import models.t2m_trans as trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
from tqdm import trange
warnings.filterwarnings('ignore')

##### ---- Exp dirs ---- #####
os.chdir('/root/autodl-tmp/SATO')
args = option_trans.get_args_parser()
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok = True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

from utils.word_vectorizer import WordVectorizer
w_vectorizer = WordVectorizer('./glove', 'our_vab')
val_loader = dataset_TM_eval.DATALoader(args.dataname, True, 32, w_vectorizer)

dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if args.dataname == 'kit' else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'

wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

##### ---- Network ---- #####

## load clip model and datasets

clip_model, clip_preprocess = clip.load(args.clip_path, device=torch.device('cuda'), jit=False)  # Must set jit=False for training
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
    trans_encoder.load_state_dict(ckpt['trans'], strict=True)
trans_encoder.train()
trans_encoder.cuda()
print('checkpoints loading successfully')

fid = []
fid_d=[]
div = []
top1 = []
top2 = []
top3 = []
matching = []
multi = []
repeat_time = 20
fid_p=[]
        
for i in range(repeat_time):
    print('repeat_time: ',i)
    best_fid,best_fid_p,best_fid_d, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_multi, writer, logger = eval_trans.evaluation_transformer_test(args.out_dir, val_loader, net, trans_encoder, logger, writer, 0, best_fid=1000,best_fid_p=1000,best_fid_dturbation=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, best_multi=0, clip_model=clip_model, eval_wrapper=eval_wrapper, draw=False, savegif=False, save=False, savenpy=(i==0))
    fid.append(best_fid)
    fid_p.append(best_fid_p)
    fid_d.append(best_fid_d)
    div.append(best_div)
    top1.append(best_top1)
    top2.append(best_top2)
    top3.append(best_top3)
    matching.append(best_matching)
    multi.append(best_multi)

    # print('fid: ', sum(fid)/(i+1))
    # print('fid_d',sum(fid_d)/(i+1))
    # print('div: ', sum(div)/(i+1))
    # print('top1: ', sum(top1)/(i+1))
    # print('top2: ', sum(top2)/(i+1))
    # print('top3: ', sum(top3)/(i+1))
    # print('matching: ', sum(matching)/(i+1))
    # print('multi: ', sum(multi)/(i+1))


print('final result:')
print('fid: ', sum(fid)/repeat_time)
print('fid_p',sum(fid_p)/repeat_time)
print('fid_d',sum(fid_d)/repeat_time)
print('div: ', sum(div)/repeat_time)
print('top1: ', sum(top1)/repeat_time)
print('top2: ', sum(top2)/repeat_time)
print('top3: ', sum(top3)/repeat_time)
print('matching: ', sum(matching)/repeat_time)
# print('multi: ', sum(multi)/repeat_time)

fid = np.array(fid)
fid_p=np.array(fid_p)
fid_d=np.array(fid_d)
div = np.array(div)
top1 = np.array(top1)
top2 = np.array(top2)
top3 = np.array(top3)
matching = np.array(matching)
# multi = np.array(multi)

msg_final = f"FID. {np.mean(fid):.3f}, {np.std(fid)*1.96/np.sqrt(repeat_time):.3f}, fid_p.{np.mean(fid_p):.3f}, {np.std(fid_p)*1.96/np.sqrt(repeat_time):.3f},fid_d.{np.mean(fid_d):.3f}, conf. {np.std(fid)*1.96/np.sqrt(repeat_time):.3f}, Diversity. {np.mean(div):.3f}, conf. {np.std(div)*1.96/np.sqrt(repeat_time):.3f}, TOP1. {np.mean(top1):.3f}, conf. {np.std(top1)*1.96/np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2)*1.96/np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3)*1.96/np.sqrt(repeat_time):.3f}, Matching. {np.mean(matching):.3f}, conf. {np.std(matching)*1.96/np.sqrt(repeat_time):.3f}, conf. {np.std(multi)*1.96/np.sqrt(repeat_time):.3f}"
logger.info(msg_final)
