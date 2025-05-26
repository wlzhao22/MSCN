import os
import argparse
import torch

from torch.backends import cudnn
from utils.utils import *

from solver import Solver as ss


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    cudnn.benchmark = True
    # torch.manual_seed(2025)
    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)
    
    solver = ss(vars(config))

    if config.mode == 'train':
        solver.train(training_type='first_train')
    elif config.mode == 'test':
        solver.test()
    elif config.mode == 'memory_initial':
        solver.get_memory_initial_embedding(training_type='second_train')

    return solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=0)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--input_c', type=int, default=38)
    parser.add_argument('--output_c', type=int, default=38)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--temp_param',type=float, default=0.05)
    parser.add_argument('--lambd',type=float, default=0.01)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='SMD')
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test', 'memory_initial'])
    parser.add_argument('--data_path', type=str, default='./data/SMD/SMD/')
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--anomaly_ratio', type=float, default=0.0)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--n_memory', type=int, default=128, help='number of memory items')
    parser.add_argument('--momentum', type=float, default=0.9, help='memory changing speed')
    parser.add_argument('--num_workers', type=int, default=4*torch.cuda.device_count())
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--temperature', type=int, default=0.1)
    parser.add_argument('--memory_initial',type=str, default=False, help='whether it requires memory item embeddings. False: using random initialization, True: using customized intialization')
    parser.add_argument('--phase_type',type=str, default=None, help='whether it requires memory item embeddings. False: using random initialization, True: using customized intialization')
    parser.add_argument('--metrics',type=str,default='AF',help='how to measure the anomaly detection method')
    parser.add_argument('--backbone',type=str,default='ModernTCN',help='which backbone to use')
    parser.add_argument('--cache_window', type=int, default=14400)
    parser.add_argument('--score_window', type=int, default=30)
    
    ###ModernTCN
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--freq', type=str, default='m', help='time unit')

    parser.add_argument('--stem_ratio', type=int, default=6, help='stem ratio')
    parser.add_argument('--downsample_ratio', type=int, default=2, help='downsample_ratio')
    parser.add_argument('--ffn_ratio', type=int, default=2, help='ffn_ratio')
    parser.add_argument('--patch_size', type=int, default=16, help='the patch size')
    parser.add_argument('--patch_stride', type=int, default=8, help='the patch stride')

    parser.add_argument('--num_blocks', nargs='+',type=int, default=[1,1,1,1], help='num_blocks in each stage')
    parser.add_argument('--large_size', nargs='+',type=int, default=[31,29,27,13], help='big kernel size')
    parser.add_argument('--small_size', nargs='+',type=int, default=[5,5,5,5], help='small kernel size for structral reparam')
    parser.add_argument('--dims', nargs='+',type=int, default=[256,256,256,256], help='dmodels in each stage')
    parser.add_argument('--dw_dims', nargs='+',type=int, default=[256,256,256,256], help='dw dims in dw conv in each stage')

    parser.add_argument('--small_kernel_merged', type=str2bool, default=False, help='small_kernel has already merged or not')
    parser.add_argument('--call_structural_reparam', type=bool, default=False, help='structural_reparam after training')
    parser.add_argument('--use_multi_scale', type=str2bool, default=True, help='use_multi_scale fusion')

    #PatchTST
    # PatchTST
    parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    config = parser.parse_args()
    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    main(config)
