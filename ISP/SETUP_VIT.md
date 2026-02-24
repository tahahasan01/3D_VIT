# ISP setup for Vit draping

## Done automatically
- `extra-data/shirt.npz`, `pants.npz`, `skirt.npz` were generated from the gt meshes.
- `extra-data/meshes/` already contains `tee-gt.obj`, `pants-gt.obj`, `skirt-gt.obj`.

## You must do manually

### 1. Download ISP checkpoints
- Go to [Google Drive](https://drive.google.com/file/d/1Zhr93ejWGobqDnJjE-P95ssNTDYSFNXS/view?usp=sharing) and download the checkpoints zip.
- Extract all `*.pth` files into this folder: `ISP/checkpoints/`
- Required files include: `shirt_sdf_f.pth`, `shirt_sdf_b.pth`, `shirt_rep.pth`, `shirt_atlas_f.pth`, `shirt_atlas_b.pth`, and similarly for pants/skirt, plus `drape_shirt.pth`, `drape_pants.pth`, `drape_skirt.pth`, `smpl_diffusion.pth`, `smpl_diffusion_TA.pth`, `layering.pth`.

### 2. SMPL models
- Register at http://smplify.is.tue.mpg.de/ and download SMPL 1.1.0.
- Place these files in `ISP/smpl_pytorch/`:
  - `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`
  - `basicModel_f_lbs_10_207_0_v1.0.0.pkl`

### 3. Python environment (for running ISP from Vit)
Vit runs `server/scripts/run_isp_drape.py` with `ISP_ROOT` set. That script needs:
- `torch` (with CUDA for GPU)
- **pytorch3d** (required; no Windows wheel on PyPI — use one of the following):
  - **Conda:** `conda create -n isp python=3.10 -y` then `conda activate isp` then `conda install pytorch torchvision pytorch3d -c pytorch -c nvidia -c fvcore` (or see [PyTorch3D install](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)).
  - **Build from source:** `pip install "git+https://github.com/facebookresearch/pytorch3d.git"` (needs Visual Studio Build Tools on Windows).
- `trimesh`, `scipy`, `numpy`

Set `ISP_PYTHON` in Vit's `server/.env` to that Python, e.g.:
`ISP_PYTHON=c:/Users/Syed Taha Hasan/Desktop/Vit/ISP/venv/Scripts/python.exe`  
or if using conda: `ISP_PYTHON=C:/ProgramData/anaconda3/envs/isp/python.exe`

## After setup
In Vit's `server/.env` set:
- `USE_ISP_DRAPING=1`
- `ISP_ROOT=c:/Users/Syed Taha Hasan/Desktop/Vit/ISP`

Restart the Vit backend and run a try-on; the garment should be draped on the body.
