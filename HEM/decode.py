import time
import numpy as np
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import deque
import os
import sys

# Add project path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from hydra import initialize, compose
from data_preproc.Octree import DeOctree, dec2bin
from data_preproc.data_preprocess import spher2cart, cart2spher
from data_preproc import pt
from models import AgentSA
import numpyAc


def extract_params_from_filename(binfile):
    """Extract encoding parameters from the .bin filename.

    Filename format: spher_{level_num}_{bin_num}_{z_offset}.bin
    Example: spher_12_60_0.bin
    """
    basename = os.path.basename(binfile).replace('.bin', '')
    parts = basename.split('_')

    level_num = 12  # default
    bin_num = 60
    z_offset = 0

    if len(parts) >= 4:
        try:
            level_num = int(parts[1])
            bin_num = int(parts[2])
            z_offset = int(parts[3])
        except ValueError:
            pass

    return level_num, bin_num, z_offset


def cal_pos(parent_pos, i, cur_level, max_level):
    """Compute normalized child node position."""
    pos = torch.zeros_like(parent_pos)
    parent_pos_scaled = parent_pos * (2 ** max_level)
    parent_pos_scaled = torch.round(parent_pos_scaled).long()
    xyz = dec2bin(i, count=3)
    unit = 2 ** (max_level - cur_level + 1)
    for idx in range(3):
        pos[idx] = (xyz[idx] * unit + parent_pos_scaled[idx]) / (2 ** max_level)
    return pos


def decodeNode(pro, dec):
    """Decode one symbol using the given probability distribution."""
    root = dec.decode(np.expand_dims(pro, 0))
    return root


def decodeOct_ehem(binfile, model, context_size, level_k, pos_mm, bin_num, z_offset, level_num):
    """
    Decode an EHEM-format occupancy bitstream.

    Args:
        binfile:      Path to the .bin bitstream file.
        model:        AgentSA model instance.
        context_size: Context window size (default 8192).
        level_k:      K value for the context window (default 4).
        pos_mm:       Position normalization parameters [(min, max), ...].
        bin_num:      Number of radial quantization bins.
        z_offset:     Z-axis offset.
        level_num:    Octree depth.

    Returns:
        oct_seq (list): Decoded occupancy sequence.
        elapsed (float): Decoding time in seconds.
    """
    model.eval()
    max_level = level_num

    with torch.no_grad():
        elapsed = time.time()

        # Read oct_len from the companion file
        oct_len = None
        oct_len_file = binfile.replace('.bin', '_oct_len.txt')
        if os.path.exists(oct_len_file):
            with open(oct_len_file, 'r') as f:
                oct_len = int(f.read().strip())

        if oct_len is None:
            raise ValueError(f"Cannot find oct_len file: {oct_len_file}")

        # Initialize arithmetic decoder
        dec = numpyAc.arithmeticDeCoding(None, oct_len, 255, binfile)

        # Initialize input tensors
        # Shape: (context_size, level_k, 3)  —  [level, octant, occupancy]
        ipt = torch.zeros((context_size, level_k, 3)).long().cuda()
        ipt[:, :, 0] = 255  # padding value
        ipt[-1, -1, 1] = 1   # root node level = 1
        ipt[-1, -1, 2] = 1   # root node octant = 1
        ipt_pos = torch.zeros((context_size, level_k, 3)).cuda()

        # Get initial probability distribution (root node)
        output1, output2 = model(
            ipt[:1].unsqueeze(0), ipt_pos[:1].unsqueeze(0), None, enc=True
        )
        freqsinit = torch.softmax(output1[:, -1], 0).cpu().numpy()

        # Decode root node
        root = decodeNode(freqsinit, dec)
        node_id = 0

        ipt[-1, -1, 0] = root  # occupancy
        oct_seq = [root]

        # Initialize BFS queues: store (feature sequence, position sequence)
        nodeQ = deque()
        posQ = deque()
        nodeQ.append(ipt[-1, -(level_k - 1):].clone())
        posQ.append(ipt_pos[-1, -(level_k - 1):].clone())

        with tqdm(total=oct_len, desc="Decoding") as pbar:
            while nodeQ:
                ancients = nodeQ.popleft()
                ancient_pos = posQ.popleft()
                parent_pos = ancient_pos[-1]

                # Get binary representation of parent occupancy
                childOcu = dec2bin(ancients[-1, 0] + 1)
                childOcu.reverse()
                cur_level = ancients[-1][1] + 1

                # Stop if beyond max depth
                if cur_level > max_level:
                    continue

                # Iterate over 8 child nodes
                for i in range(8):
                    if not childOcu[i]:
                        continue

                    # Build current node feature
                    cur_feat = torch.vstack((
                        ancients,
                        torch.Tensor([[255, cur_level, i + 1]]).cuda()
                    ))

                    # Compute child node position
                    cur_pos = cal_pos(parent_pos, i, cur_level, max_level)
                    cur_pos = torch.vstack((ancient_pos.clone(), cur_pos))

                    # Slide context window
                    ipt[:-1] = ipt[1:].clone()
                    ipt[-1] = cur_feat
                    ipt_pos[:-1] = ipt_pos[1:].clone()
                    ipt_pos[-1] = cur_pos

                    # Forward pass to get probability distribution
                    output1, output2 = model(
                        ipt.unsqueeze(0), ipt_pos.unsqueeze(0), None, enc=True
                    )
                    probabilities = torch.softmax(output1[0, -1], 0).cpu().numpy()

                    # Decode occupancy
                    root = decodeNode(probabilities, dec)
                    node_id += 1
                    pbar.update(1)

                    oct_seq.append(root)

                    # Update queues
                    ipt[-1, -1, 0] = root
                    nodeQ.append(ipt[-1, 1:].clone())
                    posQ.append(ipt_pos[-1, 1:].clone())

                    if node_id >= oct_len:
                        pbar.close()
                        return oct_seq, time.time() - elapsed

                # No need to expand further at the last level
                if cur_level >= max_level:
                    continue

        return oct_seq, time.time() - elapsed


def reconstruct_pointcloud(oct_seq, pos_mm, bin_num, z_offset):
    """
    Reconstruct a point cloud from the decoded occupancy sequence.

    Args:
        oct_seq (list):  Decoded occupancy sequence.
        pos_mm (list):   Position normalization parameters.
        bin_num (int):   Number of radial quantization bins.
        z_offset (int):  Z-axis offset.

    Returns:
        points (ndarray, (N, 3)): Reconstructed point cloud in Cartesian coordinates.
    """
    oct_seq = np.array(oct_seq)

    # DeOctree expects occupancy values in the range [1, 255]
    points_spher = DeOctree(oct_seq + 1)

    # Convert spherical to Cartesian coordinates
    points_cart = spher2cart(points_spher)

    # Denormalize using pos_mm
    if pos_mm is not None and len(pos_mm) > 0:
        for i in range(min(3, len(pos_mm))):
            if isinstance(pos_mm[i][0], torch.Tensor):
                min_val = pos_mm[i][0].item()
                max_val = pos_mm[i][1].item()
            else:
                min_val, max_val = pos_mm[i]
            points_cart[:, i] = points_cart[:, i] * (max_val - min_val) + min_val

    return points_cart


def main(args):
    """Main decode function."""
    # Resolve config path
    if args.ckpt_path:
        root_path = args.ckpt_path.split("ckpt")[0]
        test_output_path = args.test_output_path if args.test_output_path else root_path + "test_output/"
        cfg_path = root_path
    else:
        raise ValueError("--ckpt_path is required")

    # Load Hydra config
    initialize(config_path=str(cfg_path), version_base='1.1')
    cfg = compose(config_name="config.yaml")

    # Load model
    model = AgentSA.load_from_checkpoint(
        args.ckpt_path,
        cfg=cfg
    ).cuda()
    model.eval()

    # Get model parameters
    context_size = cfg.model.context_size if hasattr(cfg.model, 'context_size') else 8192
    level_k = cfg.model.level_k if hasattr(cfg.model, 'level_k') else 4

    # Collect input files
    if args.bin_file:
        bin_files = [args.bin_file]
    elif args.test_files:
        bin_files = args.test_files
    else:
        raise ValueError("Please specify --bin_file or --test_files")

    for binfile in bin_files:
        binfile = Path(binfile)
        print(f"\nDecoding: {binfile}")

        # Extract parameters from filename
        level_num, bin_num, z_offset = extract_params_from_filename(str(binfile))
        print(f"Level num: {level_num}, Bin num: {bin_num}, Z offset: {z_offset}")

        # Load pos_mm companion file
        pos_mm_file = str(binfile).replace('.bin', '_pos_mm.txt')
        pos_mm = []
        if os.path.exists(pos_mm_file):
            with open(pos_mm_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        pos_mm.append((float(parts[0]), float(parts[1])))
        else:
            print(f"Warning: pos_mm file not found: {pos_mm_file}")
            # Use default values
            pos_mm = [(0, 1), (0, 1), (0, 1)]

        # Decode occupancy sequence
        oct_seq, elapsed = decodeOct_ehem(
            str(binfile), model, context_size, level_k,
            pos_mm, bin_num, z_offset, level_num
        )
        print(f"Decoding time: {elapsed:.4f}s")
        print(f"Oct sequence length: {len(oct_seq)}")

        # Reconstruct point cloud
        points = reconstruct_pointcloud(oct_seq, pos_mm, bin_num, z_offset)
        print(f"Reconstructed points: {len(points)}")

        # Save result
        output_ply = args.output_ply if args.output_ply else str(binfile).replace('.bin', '_decoded.ply')
        pt.write_ply_data(output_ply, points)
        print(f"Saved to: {output_ply}")

        # Optionally save oct_seq for verification
        oct_seq_file = str(binfile).replace('.bin', '_decoded_oct_seq.npy')
        np.save(oct_seq_file, np.array(oct_seq))
        print(f"Oct sequence saved to: {oct_seq_file}")


def get_args():
    parser = argparse.ArgumentParser(description='EHEM Decoder')
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--bin_file",
        type=str,
        default=None,
        help="Path to .bin file to decode"
    )
    parser.add_argument(
        "--test_files",
        nargs="*",
        default=None,
        help="Multiple .bin files to decode"
    )
    parser.add_argument(
        "--output_ply",
        type=str,
        default=None,
        help="Output .ply file path"
    )
    parser.add_argument(
        "--test_output_path",
        type=str,
        default=None,
        help="Test output directory path"
    )
    parser.add_argument(
        "--type",
        type=str,
        default='kitti',
        choices=['kitti', 'ford', 'nuscenes', 'obj'],
        help="Dataset type"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
