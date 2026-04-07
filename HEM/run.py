import os
import numpyAc
import numpy as np
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from models import AgentSA
import torch
import time
import math
import tempfile
import open3d as o3d
from pathlib import Path
from data_preproc.data_preprocess import glsproc_pc, spher2cart, cart2spher
from data_preproc.Octree import DeOctree, dec2bin
from tqdm import tqdm
from collections import deque


def hem_encode(ori_file, octree_depth, ckpt_path, type='kitti', output_dir='./temp/', save_info=True):
    """
    Encode a skeleton (bones) point cloud using HEM.
    Interface compatible with octformer_encode_backbone.

    Args:
        ori_file:     Skeleton point cloud Tensor, shape (M, 3) or (1, M, 3).
        octree_depth: Octree depth (lidar_level).
        ckpt_path:    Model checkpoint path.
        type:         Dataset type ('kitti', 'ford', 'nuscenes').
        output_dir:   Output directory for .bin and companion files.
        save_info:    Whether to save companion files needed for decoding.

    Returns (compatible with octformer_encode_backbone):
        bone_steam_size (float):   Encoded bit count.
        rec_pc          (ndarray, (M, 3)): Quantized reconstructed skeleton.
        root_octant     (int):     Root node octant index.
        min_            (float):   Radial quantization lower bound (= 0).
        max_            (float):   Radial quantization upper bound (= bin_num).
        db_center       (list):    Octree root node center [cx, cy, cz].
        db_extent       (float):   Octree root node half-extent.
    """
    # Preprocess: flatten input Tensor to (M, 3) numpy array
    if torch.is_tensor(ori_file):
        ori_file_np = ori_file.detach().cpu().numpy()
    else:
        ori_file_np = np.asarray(ori_file)
    if ori_file_np.ndim == 3:
        ori_file_np = ori_file_np[0]  # (1, M, 3) -> (M, 3)

    os.makedirs(output_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp:
        temp_path = tmp.name

    try:
        # Write temp file for glsproc_pc (simulates kitti .bin format: float32 x 3)
        # glsproc_pc's kitti branch reads 4 columns, so append a zero column
        dummy = np.zeros((ori_file_np.shape[0], 4), dtype=np.float32)
        dummy[:, :3] = ori_file_np.astype(np.float32)
        dummy.tofile(temp_path)

        quant_size = 80 if type == 'kitti' else 120 if type == 'nuscenes' else (2 ** 17)

        out_file, quantized_pc, pc, nodenum = glsproc_pc(
            temp_path,
            output_dir,
            'hem_enc_tmp',
            quant_size=quant_size,
            lidar_level=octree_depth,
            Layer_indexs=[],
            cylin=True,
            datatype=type,
            mode='test',
            save=True
        )

        # Read oct_seq
        oct_seq = np.load(out_file + '.npz', allow_pickle=True)['arr_0']
        if oct_seq.ndim == 2:
            oct_seq = oct_seq[:, np.newaxis, :]  # (N, 8) -> (N, 1, 8)

        # Extract root_octant / db_center / db_extent
        # oct_center and quant_size are fixed by glsproc_pc
        oct_center_val = quant_size / 2
        db_center  = [oct_center_val, oct_center_val, oct_center_val]
        db_extent  = float(oct_center_val)
        # root_octant: root node occupancy (first node, first column + 1)
        root_octant = int(oct_seq[0, -1, 0] + 1)  # stored as value-1, restore with +1

        # Radial quantization range
        min_ = 0.0
        max_ = float(nodenum)   # bin_num; aligns with octformer interface's max_ semantics

        # Build level-by-level data
        oct_seq_work = oct_seq.copy()
        oct_seq_work[:, :, 0] -= 1   # occupancy adjustment: 1-255 -> 0-254

        whole_ids = np.arange(len(oct_seq_work)).astype(np.int64)

        data, poss, extents, pos_mm, ids = [], [], [], [], []
        cur_level, cur_level_start = 1, 0
        lidar_level = octree_depth

        for i in range(len(oct_seq_work)):
            if oct_seq_work[i, -1, 1] > cur_level:
                level_data = oct_seq_work[cur_level_start:i, :, :3]
                level_data = np.concatenate(
                    (level_data[:, :, 1:], level_data[:, :, :1]), axis=2
                )
                data.append(level_data)

                cur_extent = oct_seq_work[cur_level_start:i, -1, 3:5]
                cur_pos    = oct_seq_work[cur_level_start:i, -1, 5:]
                pos_max, pos_min = cur_pos.max(), cur_pos.min()

                extents.append(
                    ((cur_extent - pos_min) / (pos_max - pos_min + 1e-9)).astype(np.float32)
                )
                poss.append(
                    ((cur_pos - pos_min) / (pos_max - pos_min + 1e-9)
                     ).astype(np.float32).transpose((1, 0))
                )
                pos_mm.append((pos_min, pos_max))
                ids.append(whole_ids[cur_level_start:i] - cur_level_start)

                cur_level_start = i
                cur_level = oct_seq_work[i, -1, 1]

        # Last level
        level_data = oct_seq_work[cur_level_start:, :, :3]
        level_data[:, :, 1] = np.clip(level_data[:, :, 1], None, lidar_level)
        level_data = np.concatenate(
            (level_data[:, :, 1:], level_data[:, :, :1]), axis=2
        )
        data.append(level_data)

        cur_extent = oct_seq_work[cur_level_start:, -1, 3:5]
        cur_pos    = oct_seq_work[cur_level_start:, -1, 5:]
        pos_max, pos_min = cur_pos.max(), cur_pos.min()
        extents.append(cur_extent.astype(np.float32))
        poss.append(
            ((cur_pos - pos_min) / (pos_max - pos_min + 1e-9)
             ).astype(np.float32).transpose((1, 0))
        )
        pos_mm.append((pos_min, pos_max))
        ids.append(whole_ids[cur_level_start:] - cur_level_start)

        # Convert to Tensors
        data_t    = [torch.tensor(d).cuda() for d in data]
        poss_t    = [torch.tensor(p).cuda() for p in poss]
        extents_t = [torch.tensor(e).cuda() for e in extents]
        ids_t     = [torch.tensor(i).cuda() for i in ids]

        pt_num  = len(quantized_pc)
        bin_num = nodenum
        z_offset = 0

        # Load model
        ckpt_dir = '/'.join(ckpt_path.replace('\\', '/').split('/')[:-1])
        cfg_path = ckpt_dir + '/'

        if not GlobalHydra().is_initialized():
            initialize(config_path=cfg_path, version_base='1.1')
        cfg = compose(config_name="config.yaml")

        model = AgentSA.load_from_checkpoint(ckpt_path, cfg=cfg).cuda()
        model.eval()

        context_size = 32768

        # Build oct_seq_codec for entropy coding
        # oct_seq_work already has -1 applied; extract the occupancy column
        oct_seq_codec = oct_seq_work[:, 0, 0].astype(np.int16)  # (total_nodes,)
        oct_len = len(oct_seq_codec)

        # Neural network inference
        elapsed   = 0.0
        proBit    = []
        coding_order = []
        coded_cnt = 0

        with torch.no_grad():
            interval = context_size
            for l, (level_data, level_extent, level_pos, level_ids) in enumerate(
                zip(data_t, extents_t, poss_t, ids_t)
            ):
                probabilities = torch.zeros(
                    (level_data.shape[0], model.cfg.model.token_num)
                ).cuda()

                for i in tqdm(
                    range(0, level_data.shape[0], interval),
                    desc=f"level {l}/{len(data_t)-1}"
                ):
                    ipt_data   = level_data[i: i + context_size].unsqueeze(0).long()
                    ipt_extent = level_extent[i: i + context_size].unsqueeze(0)
                    ipt_pos    = level_pos[:, i: i + context_size].unsqueeze(0)
                    node_id    = level_ids[i: i + context_size].unsqueeze(0)

                    t0 = time.time()
                    output1, output2 = model(ipt_data, ipt_extent, ipt_pos, enc=True)
                    elapsed += time.time() - t0

                    if level_data.shape[0] == 1:
                        probabilities[0] = torch.softmax(output1[:, -1], 1)
                        coding_order.append(
                            node_id[-1:].detach().cpu().numpy()
                        )
                        continue

                    p1 = torch.softmax(output1, 2)
                    p2 = torch.softmax(output2, 2)
                    probabilities[node_id[0, ::2]]  = p1[0]
                    probabilities[node_id[0, 1::2]] = p2[0]
                    coding_order.append(
                        node_id[0, ::2].detach().cpu().numpy() + coded_cnt
                    )
                    coding_order.append(
                        node_id[0, 1::2].detach().cpu().numpy() + coded_cnt
                    )

                coded_cnt += int(level_data.shape[0])
                proBit.append(probabilities.detach().cpu().numpy())

        proBit       = np.vstack(proBit)
        coding_order = np.concatenate(coding_order)

        # Entropy coding
        codec    = numpyAc.arithmeticCoding()
        out_name = f'spher_{len(data_t)}_{bin_num}_{z_offset}.bin'
        out_name = os.path.join(output_dir, out_name)
        os.makedirs(os.path.dirname(out_name), exist_ok=True) if os.path.dirname(out_name) else None

        _, real_rate = codec.encode(
            proBit[coding_order], oct_seq_codec[coding_order], out_name
        )

        # Save companion files needed for decoding
        if save_info:
            # Restore original occupancy (+1) for the saved oct_seq
            oct_seq_save = oct_seq_work.copy()
            oct_seq_save[:, :, 0] += 1
            np.save(out_name.replace('.bin', '_oct_seq.npy'), oct_seq_save)

            with open(out_name.replace('.bin', '_pos_mm.txt'), 'w') as f:
                for mn, mx in pos_mm:
                    f.write(f"{mn} {mx}\n")

            with open(out_name.replace('.bin', '_oct_len.txt'), 'w') as f:
                f.write(str(oct_len))

            with open(out_name.replace('.bin', '_level_num.txt'), 'w') as f:
                f.write(str(len(data_t)))

            with open(out_name.replace('.bin', '_bin_num.txt'), 'w') as f:
                f.write(str(bin_num))

            with open(out_name.replace('.bin', '_db_center.txt'), 'w') as f:
                f.write(' '.join(str(v) for v in db_center))

            with open(out_name.replace('.bin', '_db_extent.txt'), 'w') as f:
                f.write(str(db_extent))

            with open(out_name.replace('.bin', '_root_octant.txt'), 'w') as f:
                f.write(str(root_octant))

            with open(out_name.replace('.bin', '_min_max.txt'), 'w') as f:
                f.write(f"{min_} {max_}\n")

            print(f"\n[HEM Encoding Info Saved to {output_dir}]")

        print(f"\nbpp: {real_rate / pt_num:.6f}  |  time(s): {elapsed:.4f}"
              f"  |  pt_num: {pt_num}  |  oct_num: {oct_len}"
              f"  |  total_bits: {real_rate}")

        bone_steam_size = float(real_rate)
        rec_pc = quantized_pc  # ndarray (M, 3)

        return bone_steam_size, rec_pc, root_octant, min_, max_, db_center, db_extent

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        # Clean up temporary .npz files generated by glsproc_pc
        try:
            tmp_npz_candidates = [
                f for f in os.listdir(output_dir)
                if f.startswith('hem_enc_tmp_') and f.endswith('.npz')
            ]
            for fn in tmp_npz_candidates:
                os.remove(os.path.join(output_dir, fn))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# hem_decode: decoding counterpart of hem_encode
# ---------------------------------------------------------------------------

def _cal_pos_decode(parent_pos, i, cur_level, max_level):
    """Compute normalized child node coordinates (for context position features during decoding)."""
    pos = torch.zeros_like(parent_pos)
    parent_pos_scaled = parent_pos * (2 ** max_level)
    parent_pos_scaled = torch.round(parent_pos_scaled).long()
    xyz = dec2bin(i, count=3)
    unit = 2 ** (max_level - cur_level + 1)
    for idx in range(3):
        pos[idx] = (xyz[idx] * unit + parent_pos_scaled[idx]) / (2 ** max_level)
    return pos


def _decodeOct_ehem(binfile, model, context_size, level_k, oct_len, level_num):
    """Decode EHEM-format occupancy sequence (internal helper)."""
    model.eval()
    max_level = level_num

    with torch.no_grad():
        t0 = time.time()

        dec = numpyAc.arithmeticDeCoding(None, oct_len, 255, binfile)

        ipt     = torch.zeros((context_size, level_k, 3)).long().cuda()
        ipt[:, :, 0] = 255
        ipt[-1, -1, 1] = 1
        ipt[-1, -1, 2] = 1
        ipt_pos = torch.zeros((context_size, level_k, 3)).cuda()

        # Root node probability
        output1, _ = model(ipt[:1].unsqueeze(0), ipt_pos[:1].unsqueeze(0), None, enc=True)
        freqs_init  = torch.softmax(output1[:, -1], 0).cpu().numpy()

        root_ocu = dec.decode(np.expand_dims(freqs_init, 0))
        ipt[-1, -1, 0] = root_ocu
        oct_seq = [root_ocu]

        nodeQ = deque()
        posQ  = deque()
        nodeQ.append(ipt[-1, -(level_k - 1):].clone())
        posQ.append(ipt_pos[-1, -(level_k - 1):].clone())

        node_id = 0
        with tqdm(total=oct_len, desc="HEM Decoding") as pbar:
            pbar.update(1)
            while nodeQ:
                ancients    = nodeQ.popleft()
                ancient_pos = posQ.popleft()
                parent_pos  = ancient_pos[-1]

                child_ocu = dec2bin(int(ancients[-1, 0].item()) + 1)
                child_ocu.reverse()
                cur_level = int(ancients[-1, 1].item()) + 1

                if cur_level > max_level:
                    continue

                for i in range(8):
                    if not child_ocu[i]:
                        continue

                    cur_feat = torch.vstack((
                        ancients,
                        torch.tensor([[255, cur_level, i + 1]],
                                     dtype=torch.long, device='cuda')
                    ))
                    cur_pos  = _cal_pos_decode(parent_pos, i, cur_level, max_level)
                    cur_pos  = torch.vstack((ancient_pos.clone(), cur_pos))

                    ipt[:-1]     = ipt[1:].clone()
                    ipt[-1]      = cur_feat
                    ipt_pos[:-1] = ipt_pos[1:].clone()
                    ipt_pos[-1]  = cur_pos

                    output1, _ = model(
                        ipt.unsqueeze(0), ipt_pos.unsqueeze(0), None, enc=True
                    )
                    probs = torch.softmax(output1[0, -1], 0).cpu().numpy()

                    decoded = dec.decode(np.expand_dims(probs, 0))
                    node_id += 1
                    oct_seq.append(decoded)
                    pbar.update(1)

                    ipt[-1, -1, 0] = decoded
                    nodeQ.append(ipt[-1, 1:].clone())
                    posQ.append(ipt_pos[-1, 1:].clone())

                    if node_id >= oct_len - 1:
                        return oct_seq, time.time() - t0

        return oct_seq, time.time() - t0


def hem_decode(bin_file, ckpt_path, type='kitti'):
    """
    Decode a .bin file produced by hem_encode and reconstruct the skeleton point cloud.

    Args:
        bin_file:   Path to the encoded .bin file
                    (companion files such as _oct_seq.npy must be in the same directory).
        ckpt_path:  HEM model checkpoint path.
        type:       Dataset type (reserved for future use).

    Returns:
        rec_pc  (ndarray, (M, 3)): Reconstructed skeleton point cloud in Cartesian coordinates.
        elapsed (float):           Decoding time in seconds.
    """
    bin_file = str(bin_file)

    # Read companion files
    oct_len_file = bin_file.replace('.bin', '_oct_len.txt')
    if not os.path.exists(oct_len_file):
        raise FileNotFoundError(f"Cannot find oct_len file: {oct_len_file}")
    with open(oct_len_file) as f:
        oct_len = int(f.read().strip())

    level_num_file = bin_file.replace('.bin', '_level_num.txt')
    if os.path.exists(level_num_file):
        with open(level_num_file) as f:
            level_num = int(f.read().strip())
    else:
        parts = os.path.basename(bin_file).replace('.bin', '').split('_')
        level_num = int(parts[1]) if len(parts) >= 2 else 12

    bin_num_file = bin_file.replace('.bin', '_bin_num.txt')
    if os.path.exists(bin_num_file):
        with open(bin_num_file) as f:
            bin_num = int(f.read().strip())
    else:
        parts = os.path.basename(bin_file).replace('.bin', '').split('_')
        bin_num = int(parts[2]) if len(parts) >= 3 else 60

    pos_mm = []
    pos_mm_file = bin_file.replace('.bin', '_pos_mm.txt')
    if os.path.exists(pos_mm_file):
        with open(pos_mm_file) as f:
            for line in f:
                v = line.strip().split()
                if len(v) >= 2:
                    pos_mm.append((float(v[0]), float(v[1])))

    # Load model
    ckpt_dir = '/'.join(ckpt_path.replace('\\', '/').split('/')[:-1])
    cfg_path = ckpt_dir + '/'

    if not GlobalHydra().is_initialized():
        initialize(config_path=cfg_path, version_base='1.1')
    cfg = compose(config_name="config.yaml")

    model = AgentSA.load_from_checkpoint(ckpt_path, cfg=cfg).cuda()
    model.eval()

    context_size = getattr(getattr(cfg, 'model', None), 'context_size', 8192)
    level_k      = getattr(getattr(cfg, 'model', None), 'level_k', 4)

    # Decode occupancy sequence
    oct_seq, elapsed = _decodeOct_ehem(
        bin_file, model, context_size, level_k, oct_len, level_num
    )

    print(f"Decoded oct_seq length: {len(oct_seq)}  |  time: {elapsed:.4f}s")

    # Reconstruct point cloud
    oct_seq_np = np.array(oct_seq)
    # DeOctree expects occupancy values in the range [1, 255]
    points_spher = DeOctree(oct_seq_np + 1)

    # Dequantize: spherical -> Cartesian
    points_spher_denorm = points_spher * np.array(
        [1, 2 * math.pi / bin_num, math.pi / bin_num]
    )
    rec_pc = spher2cart(points_spher_denorm)

    return rec_pc, elapsed
