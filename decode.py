# -*- coding: utf-8 -*-

import argparse
import os
import time
import numpy as np
import torch
import torchac
from glob import glob

import network_LWC
from Utils import operation
from Utils.data import write_point_cloud
from HEM.run import hem_decode

import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    prog='decompress_bitstream.py',
    description='Decompress HPGC point clouds.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--compressed_path', type=str, default='./data/compressed/')
parser.add_argument('--output_path',     type=str, default='./data/decoded/')
parser.add_argument('--datatype', type=str, default='semantickitti',
                    choices=['semantickitti', 'ford'])
parser.add_argument('--gpu_id',    type=int, default=0)
parser.add_argument('--K',         type=int, default=32,  help='local window size')
parser.add_argument('--distri_num',type=int, default=2,   help='number of distributions')
parser.add_argument('--use_hem',   action='store_true',   help='use hem (HEM) mode for skeleton')
parser.add_argument('--hem_ckpt',  type=str,
                    default='./HEM/outputs/kitti/best.ckpt',
                    help='HEM model checkpoint path (required when --use_hem is apply_config)')
args = parser.parse_args()

torch.cuda.set_device(args.gpu_id)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
if args.datatype == 'semantickitti':
    dilated_list = 4
    model_load_path = './model/kitti/best.pt'
elif args.datatype == 'ford':
    dilated_list = [1, 1, 2]
    model_load_path = './model/ford/best.pt'
else:
    raise ValueError(f"Unknown datatype: {args.datatype}")

distri_num = args.distri_num
compressed_path = args.compressed_path
output_path = args.output_path
os.makedirs(output_path, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────

def parse_hem_header(head_path, distri_num):
    """
    Read head file written in use_hem mode.

    Binary layout (in order):
        local_window_size  uint16 × 1
        min_v_value        int16  × distri_num
        max_v_value        int16  × distri_num
        min_               float64 × 1   — skeleton distance quantization lower bound
        max_               float64 × 1   — skeleton distance quantization upper bound (= bin_num)
        db_center[0..2]    float64 × 3
        db_extent          float64 × 1
        root_octant        uint8  × 1
    """
    with open(head_path, 'rb') as f:
        K          = int(np.frombuffer(f.read(2), dtype=np.uint16)[0])
        min_v_list = [int(np.frombuffer(f.read(2), dtype=np.int16)[0]) for _ in range(distri_num)]
        max_v_list = [int(np.frombuffer(f.read(2), dtype=np.int16)[0]) for _ in range(distri_num)]
        min_       = float(np.frombuffer(f.read(8), dtype=np.float64)[0])
        max_       = float(np.frombuffer(f.read(8), dtype=np.float64)[0])
        db_center  = [float(np.frombuffer(f.read(8), dtype=np.float64)[0]) for _ in range(3)]
        db_extent  = float(np.frombuffer(f.read(8), dtype=np.float64)[0])
        root_octant= int(np.frombuffer(f.read(1), dtype=np.uint8)[0])
    return K, min_v_list, max_v_list, min_, max_, db_center, db_extent, root_octant


def parse_gpcc_header(head_path, distri_num):
    """
    Read head file written in non-use_hem (gpcc) mode.

    Binary layout:
        local_window_size  uint16 × 1
        min_v_value        int16  × distri_num
        max_v_value        int16  × distri_num
        gpcc_input_scale   float64 × 1
    """
    with open(head_path, 'rb') as f:
        K              = int(np.frombuffer(f.read(2), dtype=np.uint16)[0])
        min_v_list     = [int(np.frombuffer(f.read(2), dtype=np.int16)[0]) for _ in range(distri_num)]
        max_v_list     = [int(np.frombuffer(f.read(2), dtype=np.int16)[0]) for _ in range(distri_num)]
        gpcc_scale_val = float(np.frombuffer(f.read(8), dtype=np.float64)[0])
    return K, min_v_list, max_v_list, gpcc_scale_val


# ──────────────────────────────────────────────────────────────────────────────
# Load ECC (skin) model
# ──────────────────────────────────────────────────────────────────────────────
skin_model = network_LWC.LWC(
    channel=64, bottleneck_channel=16, dilated_list=dilated_list
)
skin_model.load_state_dict(torch.load(model_load_path, map_location='cpu'))
skin_model = skin_model.cuda().eval()

# ──────────────────────────────────────────────────────────────────────────────
# Find all head files
# ──────────────────────────────────────────────────────────────────────────────
head_files = sorted(glob(os.path.join(compressed_path, '*.h.bin')))
print(f"Found {len(head_files)} compressed file(s) in {compressed_path}")

with torch.no_grad():
    for head_path in head_files:
        stem = os.path.basename(head_path).replace('.h.bin', '')
        skin_path = os.path.join(compressed_path, stem + '.s.bin')
        bone_path = os.path.join(compressed_path, stem + '.b.bin')

        if not os.path.exists(skin_path):
            print(f"[SKIP] skin file not found: {skin_path}")
            continue

        t_start = time.time()
        print(f"\n=== Decoding: {stem} ===")

        # ── 1. Parse head ──
        if args.use_hem:
            K, min_v_list, max_v_list, min_, max_, db_center, db_extent, root_octant = \
                parse_hem_header(head_path, distri_num)
            print(f"  [Head] K={K}, min_v={min_v_list}, max_v={max_v_list}")
            print(f"         min_={min_:.4f}, max_={max_:.4f}, db_center={db_center}, "
                  f"db_extent={db_extent:.4f}, root_octant={root_octant}")
        else:
            K, min_v_list, max_v_list, gpcc_scale_val = \
                parse_gpcc_header(head_path, distri_num)
            print(f"  [Head] K={K}, min_v={min_v_list}, max_v={max_v_list}, "
                  f"gpcc_scale={gpcc_scale_val:.4f}")

        # ── 2. Reconstruct skeleton (bones) ──
        if args.use_hem:
            # Locate the corresponding HEM .bin file (under hem/<stem>/ sub-directory)
            hem_dir = os.path.join(compressed_path, 'hem', stem)
            hem_bins = sorted(glob(os.path.join(hem_dir, 'spher_*.bin')))
            if not hem_bins:
                print(f"  [WARN] No HEM .bin found in {hem_dir}, trying bone_path cache...")
                if os.path.exists(bone_path):
                    rec_bones = torch.load(bone_path).cuda()
                    print(f"  [Bone] Loaded from cache: {bone_path}, shape={rec_bones.shape}")
                else:
                    print(f"  [ERROR] Cannot find bone data, skipping.")
                    continue
            else:
                hem_bin = hem_bins[0]
                print(f"  [HEM] Decoding: {hem_bin}")
                rec_bones_np, bone_dec_time = hem_decode(
                    hem_bin, args.hem_ckpt,
                    type='kitti' if args.datatype == 'semantickitti' else args.datatype
                )
                rec_bones = torch.tensor(rec_bones_np, dtype=torch.float32).cuda()
                print(f"  [Bone] Reconstructed {rec_bones.shape[0]} bones in {bone_dec_time:.4f}s")
        else:
            # gpcc mode: load skeleton from cached .b.bin
            if not os.path.exists(bone_path):
                print(f"  [ERROR] bone file not found: {bone_path}, skipping.")
                continue
            rec_bones = torch.load(bone_path).cuda()
            print(f"  [Bone] Loaded from {bone_path}, shape={rec_bones.shape}")

        # ── 3. Decode skin features (arithmetic decoding → quantized features) ──
        M = rec_bones.shape[0]

        # Entropy model: compute mu/sigma from bones
        knn_idx_list = operation.build_knn_indices(rec_bones.unsqueeze(0), 8, dilated_list)
        feature = skin_model.entropy_Model(
            x=rec_bones.unsqueeze(0), knn_idx_list=knn_idx_list
        )  # (1, M, C)
        mu, sigma = skin_model.parameterization(feature)
        mu, sigma = mu[0], sigma[0]  # (M, distri_num, C//distri_num)

        # Arithmetic-decompress_bitstream skin bitstream
        with open(skin_path, 'rb') as fin:
            bytestream = fin.read()

        bottleneck_channel = skin_model.feature_sample.out_features  # 16

        quantized_compact_fea_parts = []
        offset = 0
        for i in range(distri_num):
            min_v = torch.tensor(min_v_list[i], dtype=torch.int16, device='cuda')
            max_v = torch.tensor(max_v_list[i], dtype=torch.int16, device='cuda')
            L = int(max_v.item()) - int(min_v.item()) + 1
            cdf = operation.quantize_values(
                operation.cdf_range(mu[:, :, i] - min_v, sigma[:, :, i], L=L),
                needs_normalization=True
            ).cpu()
            sym = torchac.decode_int16_normalized_cdf(
                cdf,
                bytestream[offset:]
            )
            sym = sym.cuda().to(torch.float32) + min_v.float()
            quantized_compact_fea_parts.append(sym)
            # Estimate byte offset for this distribution by re-encoding.
            # torchac.decode_int16_normalized_cdf consumes the full stream;
            # re-encoding gives the exact byte length consumed for this distribution.
            re_encoded = torchac.encode_int16_normalized_cdf(cdf, sym.cpu().to(torch.int16))
            offset += len(re_encoded)

        # Concatenate quantized_compact_fea: (M, distri_num, fea_per_distri) -> (M, bottleneck_channel)
        quantized_compact_fea = torch.stack(quantized_compact_fea_parts, dim=1)  # (M, distri_num, fea_per_distri)
        quantized_compact_fea = quantized_compact_fea.view(M, bottleneck_channel)  # (M, 16)

        # ── 4. Generate skin point cloud ──
        knn_idx_list = operation.build_knn_indices(rec_bones.unsqueeze(0), 8, dilated_list)
        skin_feature = skin_model.feature_stretch(
            quantized_compact_fea.unsqueeze(0),
            rec_bones.unsqueeze(0),
            knn_idx_list
        ).squeeze(0)  # (M, C)

        rec_windows = skin_model.point_generator(skin_feature, K)  # (M, K, 3)
        rec_windows = operation.InverseAlign(rec_windows, rec_bones)  # (M, K, 3)
        rec_pc = rec_windows.view(-1, 3)  # (M*K, 3)

        # ── 5. Merge + save ──
        # Optionally include bones: rec_full = torch.cat([rec_bones, rec_pc], dim=0)
        rec_full = rec_pc  # skin only, consistent with the HPGC paper

        output_file = os.path.join(output_path, stem + '_rec.ply')
        write_point_cloud(rec_full, output_file)

        t_total = time.time() - t_start
        print(f"  [Done] Decoded {rec_full.shape[0]} points → {output_file}")
        print(f"  [Time] {t_total:.4f}s")
