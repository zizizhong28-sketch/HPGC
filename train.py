import os
import random
import argparse

import numpy as np
from glob import glob
from datetime import datetime

import torch
from pytorch3d.loss import chamfer_distance

from Utils.data import KITTI2019Dataset, StreamLoader, PointCloudDataset
import network_LWC
from tqdm import tqdm
import random
import warnings
warnings.filterwarnings("ignore")

seed = 11
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class StreamRecoder:
    def __init__(self):
        self.ls = []

    def refresh_stats(self, value):
        self.ls.append(value)
    
    def compute_mean(self, precision=5, reset=False):
        avg_value = round(np.array(self.ls).mean(), precision)
        if reset:
            self.ls = []
        return avg_value

parser = argparse.ArgumentParser(
    prog='train.py',
    description='Training.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--datatype', type=str, help='semantickitti or ford', default="semantickitti")
parser.add_argument('--gpu_id', type=int, help='gpu_id', default=0)
parser.add_argument('--model_save_folder', type=str, help='Directory where to save trained models.', default=f'./model/ckpt_kitti.pt')
parser.add_argument('--train_glob', type=str, help='Glob pattern to load point clouds.', default='../../tjc/datasets/KITTI2019Dataset/')
parser.add_argument('--K', type=int, help='$K$.', default=32)
parser.add_argument('--channel', type=int, help='Network channel.', default=64)
parser.add_argument('--bottleneck_channel', type=int, help='Bottleneck channel.', default=16)
parser.add_argument('--distri_num', type=int, help='the number of distribution (should be / by bottleneck_channel)', default=8)
parser.add_argument('--λ_R', type=float, help='Lambda for rate-distortion tradeoff.', default=1e-2)
parser.add_argument('--rate_loss_enable_step', type=int, help='Apply rate-distortion tradeoff at x steps.', default=5000)
parser.add_argument('--batch_size', type=int, help='Batch size (must be 1).', default=1)
parser.add_argument('--lr', type=float, help='Learning rate.', default=0.0005)
parser.add_argument('--lr_decay', type=float, help='Decays the learning rate to x times the original.', default=0.1)
parser.add_argument('--lr_decay_steps', type=int, help='Decays the learning rate at x step.', default=[70000, 120000])
parser.add_argument('--max_step', type=int, help='Train up to this number of steps.', default=140000)
args = parser.parse_args()

torch.cuda.set_device(args.gpu_id)

if args.datatype == "semantickitti":
    dilated_list = 4
    dataset = KITTI2019Dataset(data_root=args.train_glob, split="train")
elif args.datatype == "ford":
    dilated_list = [1, 1, 2]
    files = glob(args.train_glob)
    dataset = PointCloudDataset(files)
else:
    raise Exception("Unsupported datatype")

model = network_LWC.LWC(channel=args.channel, 
                          bottleneck_channel=args.bottleneck_channel,
                          distri_num=args.distri_num, 
                          dilated_list = dilated_list)

model = model.cuda().train()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

loader = StreamLoader(
    dataset = dataset,
    batch_size = args.batch_size,
    shuffle = True
)

global_step = 0
cd_recoder, bpp_recoder, loss_recoder = StreamRecoder(), StreamRecoder(), StreamRecoder()

progress_bar = tqdm(total=140000, ncols=150)

for _ in range(1, 9999):
    print(datetime.now())
    for batch_x in loader:
        
        batch_x = batch_x.cuda()

        if random.random() < 0.1 and global_step > 100000:
            K = random.randint(args.K + 1, 3600)
        else:
            K = args.K

        optimizer.zero_grad()
        rec_batch_x, bitrate = model(batch_x, K)
        # Get Loss
        chamfer_dist, _ = chamfer_distance(rec_batch_x, batch_x)
        loss = chamfer_dist
        if global_step > args.rate_loss_enable_step:
            loss = loss +  args.λ_R * bitrate
        loss.backward()
        global_step += 1
        progress_bar.refresh_stats(1)
  
        optimizer.step()

        cd_recoder.refresh_stats(chamfer_dist.item() * 100)
        bpp_recoder.refresh_stats(bitrate.item())
        loss_recoder.refresh_stats(loss.item())

        if global_step % 500 == 0:
            progress_bar.set_description(f'Step:{global_step} Ave CD: {cd_recoder.compute_mean(precision=3)} ' + \
                                         f'Bpp: {bpp_recoder.compute_mean(precision=3)} Loss: {loss_recoder.compute_mean(precision=3)}')

            # save model
            torch.save(model.state_dict(), args.model_save_folder)
        
        # Learning Rate Decay
        if global_step in args.lr_decay_steps:
            args.lr = args.lr * args.lr_decay
            for g in optimizer.param_groups:
                g['lr'] = args.lr
            print(f'Learning rate decay triggered at step {global_step}, LR is setting to{args.lr}.')

        if global_step > args.max_step:
            break
    
    if global_step > args.max_step:
        break
