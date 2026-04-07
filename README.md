# HPGC

Official implementation of **HPGC**: Hierarchical Point Cloud Geometry Compression with Multi-Scale Geometric Context.

---

## Installation

Run the unified `install.sh` script (covers both HPGC and HEM dependencies):

```bash
bash install.sh
```

See [install.sh](install.sh) for details on what is installed.

---

## Data

### KITTI (SemanticKITTI)

Download the 80 GB velodyne laser data from the official [KITTI Odometry website](https://www.cvlibs.net/datasets/kitti/eval_odometry.php).

### Ford

The Ford dataset is distributed through the MPEG content server. You need MPEG membership to access it at [mpegfs.int-evry.fr](https://mpegfs.int-evry.fr/mpegcontent/ws-mpegcontent/MPEG-I).

---

## Evaluation

Make sure the metric tools are executable before running any evaluation:

```bash
chmod +x pc_error
chmod +x tmc3_v29
```

---

### Full HPGC Pipeline (Skeleton via HEM + Skin via LWC)

Use `--use_hem` to enable the HEM-based skeleton encoder. This is the full HPGC pipeline.

#### SemanticKITTI

**Encode:**
```bash
python encode.py \
    --input_globs   "./data/SemanticKITTI/*.bin" \
    --compressed_path ./data/SemanticKITTI/compressed/ \
    --datatype      semantickitti \
    --gpu_id        0 \
    --K             32 \
    --octree_depth  12 \
    --use_hem
```

**Decode:**
```bash
python decode.py \
    --compressed_path  ./data/SemanticKITTI/compressed/ \
    --output_path      ./data/SemanticKITTI/decoded/ \
    --datatype         semantickitti \
    --gpu_id           0 \
    --K                32 \
    --use_hem \
    --hem_ckpt         ./HEM/outputs/kitti/best.ckpt
```

**Evaluate:**
```bash
python eval_PSNR.py \
    --input_globs      "./data/SemanticKITTI/*.bin" \
    --decompressed_path ./data/SemanticKITTI/decoded/ \
    --datatype         semantickitti
```

#### Ford

**Encode:**
```bash
python encode.py \
    --input_globs   "./data/Ford/*.ply" \
    --compressed_path ./data/Ford/compressed/ \
    --datatype      ford \
    --gpu_id        0 \
    --K             64 \
    --octree_depth  12 \
    --use_hem
```

**Decode:**
```bash
python decode.py \
    --compressed_path  ./data/Ford/compressed/ \
    --output_path      ./data/Ford/decoded/ \
    --datatype         ford \
    --gpu_id           0 \
    --K                64 \
    --use_hem \
    --hem_ckpt         ./HEM/outputs/kitti/best.ckpt
```

**Evaluate:**
```bash
python eval_PSNR.py \
    --input_globs      "./data/Ford/*.ply" \
    --decompressed_path ./data/Ford/decoded/ \
    --datatype         ford
```

---

### LWC — Skin Module Only (without HEM skeleton)

Omit `--use_hem` to use the lightweight GPCC-based skeleton path. This evaluates only the **LWC** (Local Window Compression) skin module.

#### SemanticKITTI

**Encode:**
```bash
python encode.py \
    --input_globs   "./data/SemanticKITTI/*.bin" \
    --compressed_path ./data/SemanticKITTI/compressed_lwc/ \
    --datatype      semantickitti \
    --gpu_id        0 \
    --K             32
```

**Decode:**
```bash
python decode.py \
    --compressed_path  ./data/SemanticKITTI/compressed_lwc/ \
    --output_path      ./data/SemanticKITTI/decoded_lwc/ \
    --datatype         semantickitti \
    --gpu_id           0 \
    --K                32
```

**Evaluate:**
```bash
python eval_PSNR.py \
    --input_globs      "./data/SemanticKITTI/*.bin" \
    --decompressed_path ./data/SemanticKITTI/decoded_lwc/ \
    --datatype         semantickitti
```

#### Ford

**Encode:**
```bash
python encode.py \
    --input_globs   "./data/Ford/*.ply" \
    --compressed_path ./data/Ford/compressed_lwc/ \
    --datatype      ford \
    --gpu_id        0 \
    --K             64
```

**Decode:**
```bash
python decode.py \
    --compressed_path  ./data/Ford/compressed_lwc/ \
    --output_path      ./data/Ford/decoded_lwc/ \
    --datatype         ford \
    --gpu_id           0 \
    --K                64
```

**Evaluate:**
```bash
python eval_PSNR.py \
    --input_globs      "./data/Ford/*.ply" \
    --decompressed_path ./data/Ford/decoded_lwc/ \
    --datatype         ford
```

---

### HEM — Skeleton Module Only

Use the standalone HEM encoder/decoder to compress and reconstruct skeleton (bone) point clouds independently.

#### Encode (HEM standalone)

```bash
cd HEM
python encode.py \
    --ckpt_path  outputs/kitti/best.ckpt \
    --test_files "../data/SemanticKITTI/*.bin" \
    --type       kitti \
    --lidar_level 12 \
    --cylin
```

Output files are written to `outputs/kitti/test_output/` with the naming convention:
```
spher_{level_num}_{bin_num}_{z_offset}.bin
```

#### Decode (HEM standalone)

```bash
cd HEM
python decode.py \
    --ckpt_path  outputs/kitti/best.ckpt \
    --bin_file   outputs/kitti/test_output/spher_12_300000_0.bin \
    --output_ply outputs/kitti/test_output/rec.ply \
    --type       kitti
```

---

## Output File Structure

After encoding with `--use_hem`, the compressed directory contains:

```
compressed/
├── <name>.h.bin           # head file (K, min/max, db_center, db_extent, root_octant)
├── <name>.s.bin           # skin bitstream (torchac arithmetic coding)
├── <name>.b.bin           # cached reconstructed skeleton (torch tensor)
└── hem/
    └── <name>/
        ├── spher_{L}_{B}_{Z}.bin        # HEM bitstream
        ├── spher_..._oct_seq.npy        # octree occupancy sequence
        ├── spher_..._pos_mm.txt         # per-level position normalization
        ├── spher_..._oct_len.txt        # total octree node count
        ├── spher_..._level_num.txt      # number of octree levels
        ├── spher_..._bin_num.txt        # angular quantization bins
        ├── spher_..._db_center.txt      # octree root center
        ├── spher_..._db_extent.txt      # octree root extent
        └── spher_..._root_octant.txt    # root occupancy code
```

---

## Model Checkpoints

| Module | Path | Dataset |
|--------|------|---------|
| LWC (local) | `./model/lwc_best.pt` | SemanticKITTI / Ford | https://entuedu-my.sharepoint.com/:u:/g/personal/yuyang003_e_ntu_edu_sg/IQDSeztESvaWRaulhIpTFmmhAToH2R-we15wJERIghJA8tQ?e=LWNZll
| HEM (anchor) | `./HEM/outputs/hem_best.ckpt` | SemanticKITTI / Ford |  https://entuedu-my.sharepoint.com/:u:/g/personal/yuyang003_e_ntu_edu_sg/IQDAZBMlzah6TpYRy8KAqSmgAXSx6xYg3PVaXB-6p_chZmg?e=UyyAL1
|

---

## GNP - Global Normalized PSNR Metric

GNP (Global Normalized PSNR) is a full-reference point cloud quality metric used by HPGC for evaluating compression quality. It combines:

- **Density-adaptive global weighting**: Balances error contributions based on point distribution and local density
- **Regional keypoint analysis**: Focuses on perceptually important edge/corner regions

### Usage

```python
from gnp import gnp

# For kitti dataset 
gnp_value = gnp(origin_pc, recon_pc, peak_value=59.7)

# For Ford dataset 
gnp_value = gnp(origin_pc, recon_pc, peak_value=30000)
```

### Import Options

```python
# Import from main directory (recommended)
from gnp import gnp

# Import from HEM internal module
from HEM.metrics.utils import gnp

# Backward compatibility (old name still works)
from gnp import r_psnr  # equivalent to gnp()
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `origin_pcd` | array-like | - | Original point cloud (N x 3) |
| `recon_pcd` | array-like | - | Reconstructed point cloud (N x 3) |
| `peak_value` | float | 59.7 | Peak value for PSNR calculation |

### Returns

Returns a single GNP score (float) combining global and local quality assessments.
