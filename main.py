import argparse
import datetime
import os
import random
import torch
import numpy as np
from torch.backends import cudnn
from solver import Solver


def main(args):
    os.makedirs(args.src_model_path, exist_ok=True)

    solver = Solver(args)
    solver.train()
    solver.test()


def print_args(args):
    for k in dict(sorted(vars(args).items())).items():
        print(k)
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_classes', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--log_step', type=int, default=50)

    parser.add_argument('--dset', type=str, default='mnist', help=['mnist', 'fmnist'])
    parser.add_argument("--img_size", type=int, default=28, help="Img size")
    parser.add_argument("--patch_size", type=int, default=7, help="Patch Size")
    parser.add_argument("--n_channels", type=int, default=1, help="Number of channels")
    parser.add_argument('--data_path', type=str, default='/data/')
    parser.add_argument('--model_path', type=str, default='./model')
    parser.add_argument('--seed', type=int, default=100)

    parser.add_argument("--embed_dim", type=int, default=96, help="dimensionality of the latent space")
    parser.add_argument("--n_heads", type=int, default=8, help="number of heads to be used")
    parser.add_argument("--forward_mul", type=int, default=2, help="forward multiplier")
    parser.add_argument("--n_layers", type=int, default=4, help="number of encoder layers")
    parser.add_argument("--load_model", type=bool, default=False, help="Load saved model")

    args = parser.parse_args()
    print(args)
    
    args.src_model_path = os.path.join(args.model_path, args.dset)

    manual_seed = args.seed
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    np.random.seed(manual_seed)
    os.environ['PYTHONHASHSEED'] = str(manual_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(manual_seed)
        torch.cuda.manual_seed_all(manual_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    start_time = datetime.datetime.now()
    print("Started at " + str(start_time.strftime('%Y-%m-%d %H:%M:%S')))

    main(args)

    end_time = datetime.datetime.now()
    duration = end_time - start_time
    print("Ended at " + str(end_time.strftime('%Y-%m-%d %H:%M:%S')))
    print("Duration: " + str(duration))