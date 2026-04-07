"""
Uses HEM for anchor encoding and LWC for local prediction.
"""

import torch
import torchac
import numpy as np

import network_LWC
from Utils import operation
from Utils.data import load_bin_file, calc_bit_size

from glob import glob
from tqdm import tqdm
import os
import time
import random
from HEM.run import hem_encode

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

seed = 666
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

import warnings
warnings.filterwarnings("ignore")


class PerfTimer:
    def __init__(self):
        self.dict = {}
    
    def start_timer(self, label):
        torch.cuda.synchronize()
        self.dict[label] = time.time()
    
    def stop_timer(self, label):
        torch.cuda.synchronize()
        self.dict[label] = time.time() - self.dict[label]
    
    
    def compute_sum(self, precision=3, reset=False):
        t = 0
        for key in self.dict.keys():
            t += self.dict[key]
        t = round(t, precision)
        if reset:
            self.dict = {}
        return t
    
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

import argparse
parser = argparse.ArgumentParser(
    prog='encode.py',
    description='Compress point clouds using HPGC.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--input_globs', type=str, help='Glob pattern to load point clouds.', default='')
parser.add_argument('--compressed_path', type=str, default='./data/compressed/')
parser.add_argument('--datatype', type=str, help='Dataset type: semantickitti or ford', default="semantickitti")
parser.add_argument('--gpu_id', type=int, help='GPU device ID.', default=4)
parser.add_argument('--K', type=int, help='Number of points per bone.', default=32)
parser.add_argument('--distri_num', type=int, help='Number of distributions (must divide bottleneck_channel).', default=2)
parser.add_argument('--use_hem', action="store_true", help="Use hem (HEM) for skeleton, else GPCC")
parser.add_argument('--window_size', type=int, help='Window size for KNN.', default=16)
parser.add_argument('--octree_depth', type=int, help='Octree depth for HEM.', default=12)
args = parser.parse_args()

torch.cuda.set_device(args.gpu_id)

if args.datatype == "semantickitti":
    dilated_list = 4
    model_load_path = './model/ckpt_kitti.pt'
    gpcc_input_scale = 9.5
    pc_scale = 1
elif args.datatype == "ford":
    dilated_list = [1, 1, 2]
    model_load_path = './model/ckpt_ECC_ford_64.pt'
    gpcc_input_scale = 4
    pc_scale = 1000
else:
    raise Exception("Unsupported datatype")

#########################################
input_globs = args.input_globs
compressed_path = args.compressed_path
local_window_size = args.K
#########################################

if not os.path.exists(compressed_path):
    os.makedirs(compressed_path)

model = network_LWC.LWC(channel=64, 
                            bottleneck_channel=16, dilated_list = dilated_list)
model.load_state_dict(torch.load(model_load_path, map_location="cpu"))

model = model.cuda().eval()
files = np.array(glob(input_globs, recursive=True))

# HEM model checkpoint path
hem_ckpt_path = './HEM/outputs/kitti/best.ckpt'
# Base directory for HEM skeleton bitstream output (one sub-dir per file)
hem_output_base = os.path.join(compressed_path, 'hem')

print("#"*15 + f" local_window_size {local_window_size} " + "#"*15)
with torch.no_grad():
    time_recoder, bpp_recoder = StreamRecoder(), StreamRecoder()
    bone_bpp_recoder = StreamRecoder()
    tq = tqdm(files, ncols=150)
    for file_path in tq:
        timer = PerfTimer()
        pc = load_bin_file(file_path) / pc_scale
        batch_x = torch.tensor(pc).unsqueeze(0).cuda()
        N = batch_x.shape[1]

        filename_w_ext = os.path.split(file_path)[0].split('/')[-2]+'_'+os.path.split(file_path)[-1]
        compressed_head_path = os.path.join(compressed_path, filename_w_ext+'.h.bin')
        compressed_skin_path = os.path.join(compressed_path, filename_w_ext+'.s.bin')
        compressed_bone_path = os.path.join(compressed_path, filename_w_ext+'.b.bin')

        timer.start_timer("FPS+KNN")
        bones, local_windows = operation.NeighborSample(batch_x, local_window_size, no_anchor=True)
        timer.stop_timer("FPS+KNN")

        # ---- Skeleton (bone) encoding ----
        timer.start_timer("bone encode")
        # hem_encode returns: bone_stream_size, rec_pc, root_octant, min_, max_, db_center, db_extent
        hem_out_dir = os.path.join(hem_output_base, filename_w_ext)
        bone_stream_size, rec_pc, root_octant, min_, max_, db_center, db_extent = hem_encode(
            bones,
            args.octree_depth,
            hem_ckpt_path,
            type='kitti' if args.datatype == 'semantickitti' else args.datatype,
            output_dir=hem_out_dir,
            save_info=True
        )

        rec_bones = torch.tensor(rec_pc, dtype=torch.float32).cuda()
        torch.save(rec_bones, compressed_bone_path)
        timer.stop_timer("bone encode")

        timer.start_timer("shuffle_indices")
        cloest_idx = operation.shuffle_indices(bones, rec_bones)
        bones, local_windows = bones[cloest_idx], local_windows[cloest_idx]
        timer.stop_timer("shuffle_indices")
        
        timer.start_timer("align")
        aligned_windows = operation.AdaptiveAlign(local_windows, rec_bones)
        timer.stop_timer("align")

        timer.start_timer('Feature_Squeeze')
        knn_idx_list = operation.build_knn_indices(aligned_windows, args.window_size, dilated_list)
        feature = model.feature_squeeze(x = aligned_windows, knn_idx_list = knn_idx_list) # M, K, C
        max_pooled_feature = model.maxpool(feature.transpose(-1,-2)).squeeze(-1) # M, C
        
        timer.stop_timer('Feature_Squeeze')

        timer.start_timer('Entropy Moddule')
        knn_idx_list = operation.build_knn_indices(rec_bones.unsqueeze(0), 8, dilated_list)

        feature = model.entropy_Model(x = rec_bones.unsqueeze(0), knn_idx_list = knn_idx_list) # M, c * 2

        mu, sigma = model.parameterization(feature)
        
        mu, sigma = mu[0], sigma[0]

        timer.stop_timer('Entropy Module')

        timer.start_timer('Encoding')

        compact_fea = model.feature_sample(max_pooled_feature)

        quantized_compact_fea = torch.round(compact_fea)
        # Arithmetic encoding
        ########################
        distri_num = args.distri_num
        quantized_compact_fea = quantized_compact_fea.view(quantized_compact_fea.shape[0], distri_num, quantized_compact_fea.shape[1] // distri_num)
        min_v_value = []
        max_v_value = []
        bytestream = torch.tensor(0, dtype=torch.int, device=quantized_compact_fea.device)
        for i in range(distri_num):
            min_v_value.append(quantized_compact_fea[:,i].min().to(torch.int16))
            max_v_value.append(quantized_compact_fea[:,i].max().to(torch.int16))
            bytestream += torchac.encode_int16_normalized_cdf(
                operation.quantize_values(operation.cdf_range(mu[:,:,i]-min_v_value[-1], sigma[:,:,i], L=max_v_value[-1]-min_v_value[-1]+1), needs_normalization=True).cpu(), 
                (quantized_compact_fea[:,i]-min_v_value[-1]).cpu().to(torch.int16)
            )
        timer.stop_timer('Encoding')

        # ---- Write head file ----
        # Head file format (use_hem mode):
        #   local_window_size (uint16) x 1
        #   min_v_value       (int16)  x distri_num
        #   max_v_value       (int16)  x distri_num
        #   min_              (float64) -- skeleton distance quantization lower bound
        #   max_              (float64) -- skeleton distance quantization upper bound (= bin_num)
        #   db_center[0..2]   (float64) x 3
        #   db_extent         (float64)
        #   root_octant       (uint8)
        if args.use_hem:
            with open(compressed_head_path, 'wb') as fout:
                fout.write(np.array(local_window_size, dtype=np.uint16).tobytes())
                for mv in min_v_value:
                    fout.write(np.array(mv.item(), dtype=np.int16).tobytes())
                for mv in max_v_value:
                    fout.write(np.array(mv.item(), dtype=np.int16).tobytes())
                fout.write(np.array(min_, dtype=np.float64).tobytes())
                fout.write(np.array(max_, dtype=np.float64).tobytes())
                fout.write(np.array(db_center[0], dtype=np.float64).tobytes())
                fout.write(np.array(db_center[1], dtype=np.float64).tobytes())
                fout.write(np.array(db_center[2], dtype=np.float64).tobytes())
                fout.write(np.array(db_extent, dtype=np.float64).tobytes())
                fout.write(np.array(root_octant, dtype=np.uint8).tobytes())
        else:
            with open(compressed_head_path, 'wb') as fout:
                fout.write(np.array(local_window_size, dtype=np.uint16).tobytes())
                for mv in min_v_value:
                    fout.write(np.array(mv.item(), dtype=np.int16).tobytes())
                for mv in max_v_value:
                    fout.write(np.array(mv.item(), dtype=np.int16).tobytes())
                fout.write(np.array(gpcc_input_scale, dtype=np.float64).tobytes())

        with open(compressed_skin_path, 'wb') as fin:
            fin.write(bytestream)

        total_bits = bone_stream_size + calc_bit_size(compressed_skin_path) \
                    + calc_bit_size(compressed_head_path)
        bpp = total_bits / N
        enc_time = timer.compute_sum(precision=5)
        time_recoder.refresh_stats(enc_time)
        bpp_recoder.refresh_stats(bpp)
        bone_bpp_recoder.refresh_stats(bone_stream_size/N)
        tq.set_description(f"Bpp: {bpp_recoder.compute_mean(precision=3)}, Encode Time: {time_recoder.compute_mean(precision=3)}")
    print(f"Bpp: {bpp_recoder.compute_mean(precision=3)}, Encode Time: {time_recoder.compute_mean(precision=3)}")
