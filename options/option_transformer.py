import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for Amass',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument('--method', type=str, default='adv', help='train or train with data_purturbation')
    ## dataloader
    parser.add_argument('--dataname', type=str, default='t2m', help='dataset directory')
    parser.add_argument('--batch-size', default=64, type=int, help='batch size')
    parser.add_argument('--fps', default=[20], nargs="+", type=int, help='frames per second')
    parser.add_argument('--seq-len', type=int, default=64, help='training motion length')
    
    ## optimization
    parser.add_argument('--total-iter', default=100000, type=int, help='number of total iterations to run')
    parser.add_argument('--warm-up-iter', default=10, type=int, help='number of total iterations for warmup')
    parser.add_argument('--lr', default=0.000001, type=float, help='max learning rate')
    # parser.add_argument('--lr', default=0.0001, type=float, help='max learning rate')
    parser.add_argument('--lr-scheduler', default=[150000], nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add_argument('--gamma', default=0.05, type=float, help="learning rate decay")
    
    parser.add_argument('--weight-decay', default=1e-6, type=float, help='weight decay') 
    parser.add_argument('--decay-option',default='all', type=str, choices=['all', 'noVQ'], help='disable weight decay on codebook')
    parser.add_argument('--optimizer',default='adamw', type=str, choices=['adam', 'adamw'], help='disable weight decay on codebook')
    
    ## vqvae arch
    parser.add_argument("--code-dim", type=int, default=512, help="embedding dimension")
    parser.add_argument("--nb-code", type=int, default=512, help="nb of embedding")
    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    parser.add_argument("--down-t", type=int, default=2, help="downsampling rate")
    parser.add_argument("--stride-t", type=int, default=2, help="stride size")
    parser.add_argument("--width", type=int, default=512, help="width of the network")
    parser.add_argument("--depth", type=int, default=3, help="depth of the network")
    parser.add_argument("--dilation-growth-rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument("--output-emb-width", type=int, default=512, help="output embedding width")
    parser.add_argument('--vq-act', type=str, default='relu', choices = ['relu', 'silu', 'gelu'], help='dataset directory')

    ## gpt arch
    parser.add_argument("--block-size", type=int, default=51, help="seq len")
    parser.add_argument("--embed-dim-gpt", type=int, default=1024, help="embedding dimension")
    parser.add_argument("--clip-dim", type=int, default=512, help="latent dimension in the clip feature")
    parser.add_argument("--num-layers", type=int, default=9, help="nb of transformer layers")
    parser.add_argument("--n-head-gpt", type=int, default=16, help="nb of heads")
    parser.add_argument("--ff-rate", type=int, default=4, help="feedforward size")
    parser.add_argument("--drop-out-rate", type=float, default=0.1, help="dropout ratio in the pos encoding")
    
    ## quantizer
    parser.add_argument("--quantizer", type=str, default='ema_reset', choices = ['ema', 'orig', 'ema_reset', 'reset'], help="eps for optimal transport")
    parser.add_argument('--quantbeta', type=float, default=1.0, help='dataset directory')

    ## resume  trans: /CV/xhr/xhr_project/Paper/text2Pose/t2m/T2M-GPT-main/output_human3d_9layer/debug/net_best_fid.pth
    # /root/autodl-tmp/SATO/pretrained/VQTransformer_corruption05/net_best_fid.pth
    parser.add_argument("--resume-pth", type=str, default='/root/autodl-tmp/SATO/pretrained/VQVAE/net_best_fid.pth', help='resume vq pth')
    # parser.add_argument("--resume-trans", type=str, default=None, help='resume gpt pth')
    parser.add_argument("--resume-trans", type=str, default='/root/autodl-tmp/SATO/output_human3d_9layer/debug/net_best_fid.pth', help='resume gpt pth')
    ## PGD
    parser.add_argument('--x_pgd_radius', type=float,default=0.005)
    parser.add_argument('--x_pgd_step', type=float,default=1)
    parser.add_argument('--x_pgd_step_size', type=float,default=0.01)
    parser.add_argument('--x_pgd_norm_type', type=str,default="l-infty")
    parser.add_argument('--lambda_1', type=float, default=0.5)
    parser.add_argument('--lambda_2', type=float, default=0.5)
    parser.add_argument('--lambda_3', type=float, default=0.5)
    parser.add_argument('--lambda_4', type=float, default=1e-2)
    parser.add_argument('--exp_name', type=str, default="debug")
    parser.add_argument('--K', type=int, default=4)
    parser.add_argument('--topk_prox_metric', type=str, choices=['l1', 'l2',"kl-full", 'jsd-full',"kl-topk", 'jsd-topk'], default='l1')


    ## output directory 
    parser.add_argument('--out-dir', type=str, default='/root/autodl-tmp/SATO/new', help='output directory')
    parser.add_argument('--exp-name', type=str, default='exp_debug', help='name of the experiment, will create a file inside out-dir')
    parser.add_argument('--vq-name', type=str, default='exp_debug', help='name of the generated dataset .npy, will create a file inside out-dir')
    ## other
    parser.add_argument('--print-iter', default=5000, type=int, help='print frequency')
    parser.add_argument('--eval-iter', default=100000, type=int, help='evaluation frequency')
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training. ')
    parser.add_argument("--if-maxtest", action='store_true', help="test in max")
    parser.add_argument('--pkeep', type=float, default=0.5, help='keep rate for gpt training')
    parser.add_argument('--wandb', default=False,action='store_true',)
    
    
    return parser.parse_args()