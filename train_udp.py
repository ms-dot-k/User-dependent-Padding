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
from src.data.lrw_udp import MultiDataset
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
    parser.add_argument("--checkpoint_dir", type=str, default='./data/checkpoints/LRWID_udp')
    parser.add_argument("--checkpoint", type=str, default='./data/checkpoints/Base/Baseline_85.847.ckpt')
    parser.add_argument("--checkpoint_udp", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=55)
    parser.add_argument("--total_step", type=int, default=300)  #200: 1min, 250: 3min, 300: 5min
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.00001)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--resnet", type=int, default=18)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--eval_step", type=int, default=20)

    parser.add_argument("--warmup", default=False)
    parser.add_argument("--augmentations", default=True)
    parser.add_argument("--mixup", default=False)

    parser.add_argument("--max_timesteps", type=int, default=29)
    parser.add_argument("--subject", type=int, default=0)
    parser.add_argument("--adapt_min", type=int, default=5)
    parser.add_argument("--fold", type=int, default=1)

    parser.add_argument("--layer_num", type=int, default=17, help='how many layers do we use. max 17')
    parser.add_argument("--location", type=str, default='front', help='from where udp will be applied. front, back')

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

    train_data = MultiDataset(
        lrw=args.lrw,
        mode='val',
        max_v_timesteps=args.max_timesteps,
        augmentations=args.augmentations,
        subject=args.subject,
        adapt_min=args.adapt_min,
        fold=args.fold
    )

    args.checkpoint_dir = args.checkpoint_dir + f'_s{args.subject}_{args.adapt_min}min_{args.layer_num}layer_min{args.lr}_{args.fold}'

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
        udp = checkpoint['udp_state_dict'].cpu()
        for i in range(args.layer_num):
            if args.location == 'front':
                udps[i] = udp[i].detach().clone()
            elif args.lcoation == 'back':
                udps[16-i] = udp[16-i].detach().clone()
        del checkpoint

    for k in range(args.layer_num):
        if args.location == 'front':
            udps[k].requires_grad = True
            params = [{'params': udps[k], 'name': f'{k}'} for k in range(args.layer_num)]
        elif args.location == 'back':
            udps[16-k].requires_grad = True
            params = [{'params': udps[16-k], 'name': f'{16-k}'} for k in range(args.layer_num)]
        else:
            assert 1 == 0, 'args.location should be one of [front, back]'

    f_optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    f_scheduler = None

    v_front.cuda()
    tcn.cuda()

    if args.distributed:
        v_front = DDP(v_front, device_ids=[args.local_rank], output_device=args.local_rank)
        tcn = DDP(tcn, device_ids=[args.local_rank], output_device=args.local_rank)
    elif args.dataparallel:
        v_front = DP(v_front)
        tcn = DP(tcn)

    validate(v_front, tcn, udps)
    train(v_front, tcn, udps, train_data, optimizer=f_optimizer, scheduler=f_scheduler, args=args)

def train(v_front, tcn, udps, train_data, optimizer, scheduler, args):
    best_val_acc = 0.0
    if args.local_rank == 0:
        writer = SummaryWriter(comment=os.path.split(args.checkpoint_dir)[-1])

    v_front.train()
    freeze_BN(v_front)
    tcn.train()
    freeze_BN(tcn)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None

    dataloader = DataLoader(
        train_data,
        shuffle=False if args.distributed else True,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler
    )

    tot_batch_num = len(dataloader)

    if args.mixup:
        criterion = nn.CrossEntropyLoss(reduction='none').cuda()
    else:
        criterion = nn.CrossEntropyLoss().cuda()

    samples = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    step = 0
    while step < args.total_step:
        loss_list = []
        acc_list = []
        if args.distributed:
            train_sampler.set_epoch(step)
        prev_time = time.time()
        for i, batch in enumerate(dataloader):
            step += 1
            if args.local_rank == 0 and i % 100 == 0:
                iter_time = (time.time() - prev_time) / 100
                prev_time = time.time()
                print("******** Training [%d / %d] : %d / %d, Iter Time : %.3f sec, Learning Rate of %s: %f ********" % (
                step, args.total_step, (i + 1) * batch_size, samples, iter_time, optimizer.param_groups[0]['name'], optimizer.param_groups[0]['lr']))
            v_in, target, _ = batch

            b_size = target.size(0)
            if args.mixup:
                weight = torch.from_numpy(np.random.beta(0.4, 0.4, size=[b_size])).float()
                new_ind = torch.randperm(b_size)
                v_in = weight.view(-1, 1, 1, 1, 1) * v_in + (1 - weight).view(-1, 1, 1, 1, 1) * v_in[new_ind]
                target_perm = target[new_ind]

            v_feat = v_front(v_in.cuda(), udps)   #B,S,512
            pred = tcn(v_feat)

            if args.mixup:
                weight = weight.cuda()
                loss = weight * criterion(pred, target.long().cuda()) + (1 - weight) * criterion(pred, target_perm.long().cuda())
                loss = loss.mean(0)
            else:
                loss = criterion(pred, target.long().cuda())

            if args.mixup:
                target = torch.max(target, target_perm)

            prediction = torch.argmax(pred, dim=1).cpu().numpy()

            acc = np.mean(prediction == target.long().numpy())

            loss_list.append(loss.cpu().item())
            acc_list.append(acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.distributed:
                dist.barrier()

            if args.local_rank == 0:
                if i % 100 == 0:
                    for j in range(2):
                        print('label: ', train_data.word_list[target.long().numpy()[j]])
                        print('prediction: ', train_data.word_list[prediction[j]])
                if writer is not None:
                    writer.add_scalar('Train/loss', loss.cpu().item(), step)
                    writer.add_scalar('Train/acc', acc, step)

                    for o in range(len(optimizer.param_groups)):
                        writer.add_scalar('lr/%s_lr' % optimizer.param_groups[o]['name'], optimizer.param_groups[o]['lr'], step)

            if step % args.eval_step == 0:
                if args.local_rank == 0:
                    logs = validate(v_front, tcn, udps, step=step, writer=writer)
                else:
                    logs = validate(v_front, tcn, udps, step=step, writer=None)

                if logs[1] > best_val_acc:
                    best_val_acc = logs[1]
                    if not os.path.exists(args.checkpoint_dir):
                        os.makedirs(args.checkpoint_dir)
                    if args.local_rank == 0:
                        bests = glob.glob(os.path.join(args.checkpoint_dir, 'Best_*.ckpt'))
                        for prev in bests:
                            os.remove(prev)
                        torch.save({'udp_state_dict': udps},
                                   os.path.join(args.checkpoint_dir, 'Best_%05d_acc_%.5f_cor_%d_num_%d.ckpt' % (
                                   step, logs[1], logs[2], logs[3])))

                if scheduler is not None:
                    scheduler.step()

    if scheduler is not None:
        scheduler.step()

    if args.local_rank == 0:
        print('Finishing training')


def validate(v_front, tcn, udps, fast_validate=False, step=0, writer=None):
    with torch.no_grad():
        v_front.eval()
        tcn.eval()

        val_data = MultiDataset(
            lrw=args.lrw,
            mode='test',
            max_v_timesteps=args.max_timesteps,
            augmentations=False,
            subject=args.subject
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
            v_in, target, _ = batch

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

        v_front.train()
        freeze_BN(v_front)
        tcn.train()
        freeze_BN(tcn)
        print('test_ACC:', tot_cor / tot_num)
        if fast_validate:
            return {}
        else:
            return np.mean(np.array(val_loss)), tot_cor / tot_num, tot_cor, tot_num

def udp_gen(feat_size, pad, channel):
    return torch.zeros([(pad * feat_size * 4 + pad * pad * 4) * channel])

def freeze_BN(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
        if isinstance(m, nn.BatchNorm3d):
            m.eval()
        if isinstance(m, nn.BatchNorm1d):
            m.eval()

if __name__ == "__main__":
    args = parse_args()
    train_net(args)

