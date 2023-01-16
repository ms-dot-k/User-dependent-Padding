import argparse
import random
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from src.models.visual_front import Visual_front_udp
from src.models.temporal_classifier import Temp_classifier
import os
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn import functional as F
from src.data.lrw import MultiDataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel as DP
import torch.utils.data.distributed
import torch.nn.parallel
import math
from matplotlib import pyplot as plt
import time
import glob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lrw', default="/mnt/hard/dataset/LRW")
    parser.add_argument('--model', default="Resnet18")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default='./data/checkpoints/Base/Baseline_85.847.ckpt')
    parser.add_argument("--checkpoint_udp", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=55)
    parser.add_argument("--total_step", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.00001)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--resnet", type=int, default=18)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--eval_step", type=int, default=30)

    parser.add_argument("--warmup", default=False)
    parser.add_argument("--augmentations", default=True)
    parser.add_argument("--mixup", default=False)

    parser.add_argument("--max_timesteps", type=int, default=29)
    parser.add_argument("--subject", type=str, default='FULL')

    parser.add_argument("--dataparallel", default=False)
    parser.add_argument("--distributed", default=False)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--gpu", type=str, default='0')
    args = parser.parse_args()
    return args


def train_net(args):
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.local_rank)
    torch.cuda.manual_seed_all(args.local_rank)
    random.seed(args.local_rank)
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['MASTER_PORT'] = '6666'

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')

    v_front = Visual_front_udp(in_channels=1)
    tcn = Temp_classifier()

    udps = {0: udp_gen(feat_size=112, pad=3, channel=1),
           1: udp_gen(feat_size=28, pad=1, channel=64),
           2: udp_gen(feat_size=28, pad=1, channel=64),
           3: udp_gen(feat_size=28, pad=1, channel=64),
           4: udp_gen(feat_size=28, pad=1, channel=64),
           5: udp_gen(feat_size=28, pad=1, channel=64),
           6: udp_gen(feat_size=28, pad=1, channel=128),
           7: udp_gen(feat_size=14, pad=1, channel=128),
           8: udp_gen(feat_size=14, pad=1, channel=128),
           9: udp_gen(feat_size=14, pad=1, channel=128),
           10: udp_gen(feat_size=14, pad=1, channel=256),
           11: udp_gen(feat_size=7, pad=1, channel=256),
           12: udp_gen(feat_size=7, pad=1, channel=256),
           13: udp_gen(feat_size=7, pad=1, channel=256),
           14: udp_gen(feat_size=7, pad=1, channel=512),
           15: udp_gen(feat_size=4, pad=1, channel=512),
           16: udp_gen(feat_size=4, pad=1, channel=512),
           }

    if args.checkpoint is not None:
        if args.local_rank == 0:
            print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage.cuda(args.local_rank))
        v_front.load_state_dict(checkpoint['v_front_state_dict'])
        tcn.load_state_dict(checkpoint['tcn_state_dict'])
        del checkpoint

    if args.checkpoint_udp is not None:
        if args.local_rank == 0:
            print(f"Loading checkpoint: {args.checkpoint_udp}")
        checkpoint = torch.load(args.checkpoint_udp, map_location=lambda storage, loc: storage.cuda(args.local_rank))
        udp = checkpoint['udp_state_dict']
        for i in range(17):
            udps[i] = udp[i].cpu().detach().clone()
        del checkpoint

    v_front.cuda()
    tcn.cuda()

    if args.distributed:
        v_front = DDP(v_front, device_ids=[args.local_rank], output_device=args.local_rank)
        tcn = DDP(tcn, device_ids=[args.local_rank], output_device=args.local_rank)
    elif args.dataparallel:
        v_front = DP(v_front)
        tcn = DP(tcn)

    _ = validate(v_front, tcn, udps)

def validate(v_front, tcn, udps, fast_validate=False, step=0, writer=None):
    with torch.no_grad():
        v_front.eval()
        tcn.eval()

        val_data = MultiDataset(
            lrw=args.lrw,
            mode='test',
            max_v_timesteps=args.max_timesteps,
            augmentations=False
        )

        dataloader = DataLoader(
            val_data,
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=args.workers,
            drop_last=False
        )

        criterion = nn.CrossEntropyLoss().cuda()
        batch_size = dataloader.batch_size
        if fast_validate:
            samples = min(2 * batch_size, int(len(dataloader.dataset)))
            max_batches = 2
        else:
            samples = int(len(dataloader.dataset))
            max_batches = int(len(dataloader))

        val_loss = []
        tot_cor, tot_v_cor, tot_a_cor, tot_num = 0, 0, 0, 0

        description = 'Check validation step' if fast_validate else 'Validation'
        if args.local_rank == 0:
            print(description)
        for i, batch in enumerate(dataloader):
            if args.local_rank == 0 and i % 10 == 0:
                if not fast_validate:
                    print("******** Validation : %d / %d ********" % ((i + 1) * batch_size, samples))
            v_in, target = batch

            v_feat = v_front(v_in.cuda(), udps)   #B,S,512
            pred = tcn(v_feat)

            loss = criterion(pred, target.long().cuda()).cpu().item()
            prediction = torch.argmax(pred.cpu(), dim=1).numpy()
            tot_cor += np.sum(prediction == target.long().numpy())

            tot_num += len(prediction)

            batch_size = pred.size(0)
            val_loss.append(loss)

            if i >= max_batches:
                break

        if args.local_rank == 0:
            if writer is not None:
                writer.add_scalar('Val/loss', np.mean(np.array(val_loss)), step)
                writer.add_scalar('Val/acc', tot_cor / tot_num, step)

        v_front.eval()
        tcn.eval()
        print('test_ACC:', tot_cor / tot_num, 'Cor:', tot_cor, 'tot_num:', tot_num)
        if fast_validate:
            return {}
        else:
            return np.mean(np.array(val_loss)), tot_cor / tot_num

def udp_gen(feat_size, pad, channel):
    return torch.zeros([(pad * feat_size * 4 + pad * pad * 4) * channel])

if __name__ == "__main__":
    args = parse_args()
    train_net(args)

