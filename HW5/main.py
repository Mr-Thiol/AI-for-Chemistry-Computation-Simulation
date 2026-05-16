import os
import argparse

from train import train
from sample import sample

from utils.config import Config


def main(args):    
    if args.mode == "train":
        # Load config
        config_path = args.config
        config = Config(config_path)
        config.global_setting.device = args.device
        config.global_setting.task_mode = 'train'
        config.global_setting.workdir = args.workdir
        config.global_setting.config_path = os.path.abspath(config_path)

        train(config) 
        
    elif args.mode == "sample":
        # Load config
        config_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.ckpt)))
        config_file = [x for x in os.listdir(config_dir) if '.yml' in x][0]
        config_path = os.path.join(config_dir,config_file)
        config = Config(config_path)
        
        config.global_setting.task_mode = 'sample'
        config.global_setting.task_name = args.sample_name
        config.global_setting.device = args.device
        config.sample.ckpt = args.ckpt
        
        if args.start_idx != None:
            config.sample.start_idx = args.start_idx
        if args.end_idx != None:
            config.sample.end_idx = args.end_idx
        
        sample(config)        
    
    else:
        raise ValueError(f"Mode {args.mode} not recognized.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 管理参数
    # settings for all
    parser.add_argument('--mode',        type=str, choices=['train','sample'], default='train', help='Running mode:train, sample')
    parser.add_argument('--device',      type=str, default='cuda:0')

    # settings for train
    parser.add_argument('--workdir',     type=str, default='./logs')
    parser.add_argument('--config',      type=str, default='./configs/qm9_default.yml')
    
    # settings for sample
    parser.add_argument('--sample_name', type=str, default='sample', help='The folder name for storing sampling results')
    parser.add_argument('--ckpt',        type=str, default=None,     help='The path of model ckpt for sampling task')
    parser.add_argument('--start_idx',   type=int, default=None,     help='The start index for sampling')
    parser.add_argument('--end_idx',     type=int, default=None,     help='The end index for sampling')

    args = parser.parse_args()

    main(args=args)

