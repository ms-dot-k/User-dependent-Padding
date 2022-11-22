import argparse
import random
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from src.models.visual_front import Visual_front
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
    parser.add_argument("--checkpoint_dir", type=str, default='./data/checkpoints/LRWID_Baseline')
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=55)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--weight_decay", type=float, default=0.00001)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--resnet", type=int, default=18)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--reduce_lr_at_1epoch", type=bool, default=False)
    parser.add_argument("--endurance", type=int, default=5)

    parser.add_argument("--warmup", default=False)
    parser.add_argument("--augmentations", default=True)
    parser.add_argument("--mixup", default=True)

    parser.add_argument("--max_timesteps", type=int, default=29)

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
        mode='train',
        max_v_timesteps=args.max_timesteps,
        augmentations=args.augmentations,
    )

    v_front = Visual_front(in_channels=1)
    tcn = Temp_classifier()

    params = [{'params': v_front.parameters(), 'name': 'v_front'},
              {'params': tcn.parameters(), 'name': 'tcn'}
              ]

    f_optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    if args.warmup:
        lambda1 = lambda epoch: math.pow(0.5, (epoch - 10 + 1) / float(15)) if epoch >= 10 else (epoch ** 3 / 1000)  # warmup 10, half 15
        f_scheduler = torch.optim.lr_scheduler.LambdaLR(f_optimizer, lr_lambda=lambda1)
        for i in range(args.start_epoch + 1):
            f_scheduler.step()
    else:
        f_scheduler = None

    if args.checkpoint is not None:
        if args.local_rank == 0:
            print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage.cuda(args.local_rank))
        v_front.load_state_dict(checkpoint['v_front_state_dict'])
        tcn.load_state_dict(checkpoint['tcn_state_dict'])
        del checkpoint

    v_front.cuda()
    tcn.cuda()

    if args.distributed:
        v_front = torch.nn.SyncBatchNorm.convert_sync_batchnorm(v_front)
        tcn = torch.nn.SyncBatchNorm.convert_sync_batchnorm(tcn)

    if args.distributed:
        v_front = DDP(v_front, device_ids=[args.local_rank], output_device=args.local_rank)
        tcn = DDP(tcn, device_ids=[args.local_rank], output_device=args.local_rank)
    elif args.dataparallel:
        v_front = DP(v_front)
        tcn = DP(tcn)

    train(v_front, tcn, train_data, args.epochs, optimizer=f_optimizer, scheduler=f_scheduler, args=args)

def train(v_front, tcn, train_data, epochs, optimizer, scheduler, args):
    best_val_acc = 0.0
    endurance = 0
    if args.local_rank == 0:
        writer = SummaryWriter(comment=os.path.split(args.checkpoint_dir)[-1])

    v_front.train()
    tcn.train()

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
    for epoch in range(args.start_epoch, epochs):
        loss_list = []
        acc_list = []
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if args.local_rank == 0:
            print(f"Epoch [{epoch}/{epochs}]")
        prev_time = time.time()
        for i, batch in enumerate(dataloader):
            step += 1
            if args.local_rank == 0 and i % 100 == 0:
                iter_time = (time.time() - prev_time) / 100
                prev_time = time.time()
                print("******** Training [%d / %d] : %d / %d, Iter Time : %.3f sec, Learning Rate of %s: %f ********" % (
                epoch, epochs, (i + 1) * batch_size, samples, iter_time, optimizer.param_groups[0]['name'], optimizer.param_groups[0]['lr']))
            v_in, target = batch

            b_size = target.size(0)
            if args.mixup:
                weight = torch.from_numpy(np.random.beta(0.4, 0.4, size=[b_size])).float()
                new_ind = torch.randperm(b_size)
                v_in = weight.view(-1, 1, 1, 1, 1) * v_in + (1 - weight).view(-1, 1, 1, 1, 1) * v_in[new_ind]
                target_perm = target[new_ind]

            v_feat = v_front(v_in.cuda())   #B,S,512
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

        if args.local_rank == 0:
            logs = validate(v_front, tcn, epoch=epoch, writer=writer)
        else:
            logs = validate(v_front, tcn, epoch=epoch, writer=None)

        if args.local_rank == 0:
            print('VAL_ACC: ', logs[1])
            print('Saving checkpoint: %d' % epoch)
            if args.distributed or args.dataparallel:
                v_state_dict = v_front.module.state_dict()
                tcn_state_dict = tcn.module.state_dict()
            else:
                v_state_dict = v_front.state_dict()
                tcn_state_dict = tcn.state_dict()
            if not os.path.exists(args.checkpoint_dir):
                os.makedirs(args.checkpoint_dir)
            torch.save({'v_front_state_dict': v_state_dict,
                        'tcn_state_dict': tcn_state_dict},
                       os.path.join(args.checkpoint_dir, 'Epoch_%04d_acc_%.5f.ckpt' % (epoch, logs[1])))

        if logs[1] > best_val_acc:
            endurance = 0
            best_val_acc = logs[1]
            if args.local_rank == 0:
                bests = glob.glob(os.path.join(args.checkpoint_dir, 'Best_*.ckpt'))
                for prev in bests:
                    os.remove(prev)
                torch.save({'v_front_state_dict': v_state_dict,
                            'tcn_state_dict': tcn_state_dict},
                           os.path.join(args.checkpoint_dir, 'Best_%04d_acc_%.5f.ckpt' % (epoch, logs[1])))
        else:
            endurance += 1

        if epoch == 1 and args.reduce_lr_at_1epoch:
            endurance = 0
            print('################# Reduce First LR #################')
            for g in optimizer.param_groups:
                g['lr'] = 0.01

        elif endurance >= args.endurance:
            endurance = 0
            print('################# Reduce LR #################')
            for g in optimizer.param_groups:
                current_lr = g['lr']
                g['lr'] = current_lr * 0.50

        if scheduler is not None:
            scheduler.step()

    if args.local_rank == 0:
        print('Finishing training')


def validate(v_front, tcn, fast_validate=False, epoch=0, writer=None):
    with torch.no_grad():
        v_front.eval()
        tcn.eval()

        val_data = MultiDataset(
            lrw=args.lrw,
            mode='test',
            max_v_timesteps=args.max_timesteps,
            augmentations=False,
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

            v_feat = v_front(v_in.cuda())   #B,S,512
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
                writer.add_scalar('Val/loss', np.mean(np.array(val_loss)), epoch)
                writer.add_scalar('Val/acc', tot_cor / tot_num, epoch)

        v_front.train()
        tcn.train()
        print('test_ACC:', tot_cor / tot_num)
        if fast_validate:
            return {}
        else:
            return np.mean(np.array(val_loss)), tot_cor / tot_num

if __name__ == "__main__":
    args = parse_args()
    train_net(args)

