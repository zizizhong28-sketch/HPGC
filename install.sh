#!/bin/bash
# ============================================================
#  HPGC — Unified Environment Setup
#  Covers: HPGC (LWC skin module) + HEM (skeleton module)
#
#  Requirements:
#    - Anaconda / Miniconda installed
#    - CUDA 11.7 compatible GPU
#
#  Usage:
#    bash install.sh
# ============================================================

set -e

ENV_NAME="hpgc"
PYTHON_VERSION="3.10"

echo "======================================================"
echo " Creating conda environment: ${ENV_NAME}"
echo "======================================================"
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
conda activate ${ENV_NAME}

# ── Core: PyTorch 2.0.1 + CUDA 11.7 ──────────────────────
echo "======================================================"
echo " Installing PyTorch 2.0.1 (CUDA 11.7)"
echo "======================================================"
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    pytorch-cuda=11.7 -c pytorch -c nvidia -y

# ── PyTorch3D ─────────────────────────────────────────────
echo "======================================================"
echo " Installing PyTorch3D"
echo "======================================================"
conda install pytorch3d=0.7.5 -c pytorch3d -y

# ── Pip packages: HPGC (LWC) ─────────────────────────────
echo "======================================================"
echo " Installing HPGC (LWC skin module) pip dependencies"
echo "======================================================"
pip install \
    torchac==0.9.3 \
    torch-scatter==2.1.2 \
    spconv-cu117==2.3.6 \
    cumm-cu117==0.4.11 \
    open3d==0.18.0 \
    plyfile==1.1 \
    numpy==1.26.4 \
    scipy \
    tqdm \
    matplotlib \
    h5py==3.12.1 \
    hdf5storage==0.1.19 \
    einops==0.8.0 \
    timm==1.0.13 \
    flash-attn==2.7.3 \
    ConfigArgParse==1.7 \
    pyntcloud \
    tensorboard==2.19.0 \
    trimesh \
    scikit-learn \
    scikit-image \
    geomloss \
    pybind11

# ── Pip packages: HEM (skeleton module) ──────────────────
echo "======================================================"
echo " Installing HEM (skeleton module) pip dependencies"
echo "======================================================"
pip install \
    lightning>=2.0.1 \
    hydra-core==1.3.2 \
    hydra-colorlog==1.2.0 \
    hydra-optuna-sweeper==1.2.0 \
    wandb \
    Ninja

# ── numpyAc (arithmetic coding, local install) ───────────
# numpyAc is not on PyPI; install from the local submodule if present.
echo "======================================================"
echo " Installing numpyAc"
echo "======================================================"
if [ -d "HEM/numpyAc" ]; then
    pip install -e HEM/numpyAc
elif [ -d "numpyAc" ]; then
    pip install -e numpyAc
else
    echo "[WARN] numpyAc source not found; please install it manually."
    echo "       Expected location: HEM/numpyAc  or  numpyAc/"
fi

# ── Build HEM C++ extension (fastutils) ──────────────────
echo "======================================================"
echo " Building HEM C++ extension (data_preproc/fastutils)"
echo "======================================================"
if [ -f "HEM/data_preproc/setup.py" ]; then
    pushd HEM/data_preproc
    python setup.py build_ext --inplace
    popd
else
    echo "[WARN] HEM/data_preproc/setup.py not found; skipping C++ extension build."
fi

echo ""
echo "======================================================"
echo " HPGC environment '${ENV_NAME}' is ready."
echo " Activate with:  conda activate ${ENV_NAME}"
echo "======================================================"
