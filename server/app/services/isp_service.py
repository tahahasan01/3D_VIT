"""ISP (Implicit Sewing Patterns) service — singleton for garment generation + draping.

Integrates the ISP model (Li et al., NeurIPS 2023) into the server for
generating realistic 3D garment meshes with proper sewing-pattern topology.

Loads all ISP neural networks once at startup. Provides:
  - generate_tpose_garment(garment_type, idx_G) → T-pose trimesh
  - drape_garment(garment_type, idx_G, pose, beta) → posed trimesh

Supports CPU-only inference by monkey-patching .cuda() when CUDA is unavailable.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path
from typing import Literal

import numpy as np
import trimesh

logger = logging.getLogger(__name__)

GarmentKind = Literal["tee", "pants", "skirt"]

# ISP root directory (sibling of server/)
_ISP_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "ISP"
_CHECKPOINT_DIR = _ISP_ROOT / "checkpoints"
_EXTRA_DATA_DIR = _ISP_ROOT / "extra-data"
_SMPL_DIR = _ISP_ROOT / "smpl_pytorch"

# Minimum checkpoints needed for a tee-shirt drape
_REQUIRED_CKPTS_TEE = [
    "shirt_sdf_f.pth", "shirt_sdf_b.pth", "shirt_rep.pth",
    "shirt_atlas_f.pth", "shirt_atlas_b.pth",
    "drape_shirt.pth", "smpl_diffusion.pth",
]

_REQUIRED_CKPTS_PANTS = [
    "pants_sdf_f.pth", "pants_sdf_b.pth", "pants_rep.pth",
    "pants_atlas_f.pth", "pants_atlas_b.pth",
    "drape_pants.pth", "smpl_diffusion_TA.pth",
]

_REQUIRED_CKPTS_SKIRT = [
    "skirt_sdf_f.pth", "skirt_sdf_b.pth", "skirt_rep.pth",
    "skirt_atlas_f.pth", "skirt_atlas_b.pth",
    "drape_skirt.pth", "smpl_diffusion.pth",
]


def _check_checkpoints_available(kind: GarmentKind | None = None) -> list[str]:
    """Return list of missing checkpoint files."""
    needed = set(_REQUIRED_CKPTS_TEE)
    if kind == "pants":
        needed = set(_REQUIRED_CKPTS_PANTS)
    elif kind == "skirt":
        needed = set(_REQUIRED_CKPTS_SKIRT)
    elif kind is None:
        # All
        needed = set(_REQUIRED_CKPTS_TEE + _REQUIRED_CKPTS_PANTS + _REQUIRED_CKPTS_SKIRT)
    missing = [f for f in needed if not (_CHECKPOINT_DIR / f).is_file()]
    return sorted(missing)


def is_isp_available(kind: GarmentKind = "tee") -> bool:
    """Check if ISP is ready to use for the given garment type."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return len(_check_checkpoints_available(kind)) == 0


# ---------------------------------------------------------------------------
# CPU compatibility — monkey-patch .cuda() when no GPU
# ---------------------------------------------------------------------------

_cuda_patched = False


def _ensure_cuda_compat():
    """If CUDA unavailable, patch torch so .cuda() is a no-op (returns self)."""
    global _cuda_patched
    if _cuda_patched:
        return
    import torch

    if torch.cuda.is_available():
        _cuda_patched = True
        return

    logger.info("CUDA not available — patching torch .cuda() for CPU-only ISP inference")

    _orig_tensor_cuda = torch.Tensor.cuda

    def _tensor_cuda_noop(self, *args, **kwargs):
        return self

    torch.Tensor.cuda = _tensor_cuda_noop

    _orig_module_cuda = torch.nn.Module.cuda

    def _module_cuda_noop(self, *args, **kwargs):
        return self.to("cpu")

    torch.nn.Module.cuda = _module_cuda_noop

    # Patch torch.load to always use map_location='cpu'
    _orig_load = torch.load

    def _cpu_load(*args, **kwargs):
        kwargs.setdefault("map_location", "cpu")
        return _orig_load(*args, **kwargs)

    torch.load = _cpu_load

    _cuda_patched = True


# ---------------------------------------------------------------------------
# ISP Service (singleton, thread-safe)
# ---------------------------------------------------------------------------

class ISPService:
    """Singleton that wraps all ISP model loading, reconstruction, and draping."""

    _instance: ISPService | None = None
    _lock = threading.Lock()

    def __new__(cls) -> ISPService:
        with cls._lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._initialized = False
                cls._instance = inst
            return cls._instance

    # ── Lazy initialization ────────────────────────────────────────────

    def _ensure_initialized(self):
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            self._do_initialize()
            self._initialized = True

    def _do_initialize(self):
        import torch

        _ensure_cuda_compat()

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("ISP service initializing on device=%s", self._device)

        # Add ISP to sys.path so its internal imports work
        isp_str = str(_ISP_ROOT)
        if isp_str not in sys.path:
            sys.path.insert(0, isp_str)

        # Need to chdir temporarily for relative path references inside ISP
        old_cwd = os.getcwd()
        os.chdir(str(_ISP_ROOT))

        try:
            self._load_modules()
            self._garment_models: dict = {}
            self._draping_models: dict = {}
            self._smpl_data: dict = {}
            self._uv_cache: dict = {}

            # Load SMPL server (for draping)
            self._load_smpl()

            # Pre-create UV meshes used in reconstruction + draping
            self._uv_cache[180] = self._create_uv_data(180)
            self._uv_cache[200] = self._create_uv_data(200)

        finally:
            os.chdir(old_cwd)

    def _load_modules(self):
        """Import ISP modules (already on sys.path)."""
        from networks import SDF, drape, unet
        from utils.ISP import reconstruct_batch
        from utils import mesh_reader
        from utils.draping import (
            generate_fix_mask,
            generate_fix_mask_bottom,
            barycentric_faces,
            transform_pose,
            draping,
        )
        from utils.skinning import infer_smpl
        from smpl_pytorch.body_models import SMPL

        # Store as instance attributes for later use
        self._SDF = SDF
        self._drape = drape
        self._unet = unet
        self._reconstruct_batch = reconstruct_batch
        self._mesh_reader = mesh_reader
        self._generate_fix_mask = generate_fix_mask
        self._generate_fix_mask_bottom = generate_fix_mask_bottom
        self._barycentric_faces = barycentric_faces
        self._transform_pose = transform_pose
        self._draping_fn = draping
        self._infer_smpl = infer_smpl
        self._SMPL = SMPL

    def _create_uv_data(self, res: int):
        """Create UV mesh grid at given resolution."""
        import torch

        uv_vertices, uv_faces = self._mesh_reader.create_uv_mesh(res, res, debug=False)
        mesh_uv = trimesh.Trimesh(uv_vertices, uv_faces, process=False, validate=False)
        edges = torch.LongTensor(mesh_uv.edges).to(self._device)
        uv_vertices_t = torch.FloatTensor(uv_vertices).to(self._device)
        return {
            "vertices": uv_vertices,
            "faces": uv_faces,
            "mesh": mesh_uv,
            "edges": edges,
            "vertices_t": uv_vertices_t,
        }

    # ── SMPL loading ──────────────────────────────────────────────────

    def _load_smpl(self):
        """Load SMPL model and compute rest-pose transforms for pants."""
        import torch

        smpl_server = self._SMPL(
            model_path=str(_SMPL_DIR),
            gender="f",
            use_hands=False,
            use_feet_keypoints=False,
            dtype=torch.float32,
        ).to(self._device)

        smpl_faces = smpl_server.faces

        # Compute rest-pose (A-pose) transforms for pants skinning
        pose = torch.zeros(1, 72, device=self._device)
        beta = torch.zeros(1, 10, device=self._device)
        pose = pose.reshape(24, 3)
        pose[1, 2] = 0.35
        pose[2, 2] = -0.35
        pose = pose.reshape(-1).unsqueeze(0)

        with torch.no_grad():
            w, tfs, _, pose_offsets, _, _ = self._infer_smpl(pose, beta, smpl_server)
            Rot_rest = torch.einsum("nk,kij->nij", w.squeeze(), tfs.squeeze())
            pose_offsets_rest = pose_offsets.squeeze()

        self._smpl_data = {
            "server": smpl_server,
            "faces": smpl_faces,
            "Rot_rest": Rot_rest,
            "pose_offsets_rest": pose_offsets_rest,
        }

        # Compute T-pose body bounds for coordinate alignment
        with torch.no_grad():
            tpose = torch.zeros(1, 72, device=self._device)
            tbeta = torch.zeros(1, 10, device=self._device)
            _, _, verts_tpose, _, _, _ = self._infer_smpl(tpose, tbeta, smpl_server)
            v_np = verts_tpose[0].cpu().numpy()
            self._smpl_tpose_y_min = float(v_np[:, 1].min())
            self._smpl_tpose_y_max = float(v_np[:, 1].max())
            logger.info(
                "SMPL T-pose Y bounds: [%.3f, %.3f]",
                self._smpl_tpose_y_min, self._smpl_tpose_y_max,
            )

    # ── Garment model loading (lazy per type) ─────────────────────────

    def _ensure_garment_models(self, kind: GarmentKind):
        """Load ISP reconstruction models for a garment type (lazy)."""
        if kind in self._garment_models:
            return
        import torch

        missing = _check_checkpoints_available(kind)
        if missing:
            raise FileNotFoundError(
                f"ISP checkpoints missing for '{kind}': {missing}. "
                f"Download from https://drive.google.com/file/d/1Zhr93ejWGobqDnJjE-P95ssNTDYSFNXS/view "
                f"and extract *.pth into {_CHECKPOINT_DIR}"
            )

        SDF = self._SDF
        rep_size = 32
        ckpt = str(_CHECKPOINT_DIR)
        data = str(_EXTRA_DATA_DIR)

        # Architecture depends on garment type
        configs = {
            "tee": {
                "stat_file": "shirt.npz",
                "sdf_f_out": 1 + 11, "sdf_b_out": 1 + 10,
                "rep_samples": 400,
                "prefix": "shirt",
            },
            "pants": {
                "stat_file": "pants.npz",
                "sdf_f_out": 1 + 7, "sdf_b_out": 1 + 7,
                "rep_samples": 200,
                "prefix": "pants",
            },
            "skirt": {
                "stat_file": "skirt.npz",
                "sdf_f_out": 1 + 4, "sdf_b_out": 1 + 4,
                "rep_samples": 300,
                "prefix": "skirt",
            },
        }
        cfg = configs[kind]
        stat = np.load(os.path.join(data, cfg["stat_file"]))

        model_sdf_f = SDF.SDF2branch_deepSDF(
            d_in=2 + rep_size, d_out=cfg["sdf_f_out"],
            dims=[256] * 6, skip_in=[3],
        ).to(self._device)
        model_sdf_b = SDF.SDF2branch_deepSDF(
            d_in=2 + rep_size, d_out=cfg["sdf_b_out"],
            dims=[256] * 6, skip_in=[3],
        ).to(self._device)
        model_rep = SDF.learnt_representations(
            rep_size=rep_size, samples=cfg["rep_samples"],
        ).to(self._device)
        model_atlas_f = SDF.SDF(
            d_in=2 + rep_size, d_out=3,
            dims=[256] * 6, skip_in=[3],
        ).to(self._device)
        model_atlas_b = SDF.SDF(
            d_in=2 + rep_size, d_out=3,
            dims=[256] * 6, skip_in=[3],
        ).to(self._device)

        prefix = cfg["prefix"]
        model_sdf_f.load_state_dict(torch.load(os.path.join(ckpt, f"{prefix}_sdf_f.pth"), map_location=self._device))
        model_sdf_b.load_state_dict(torch.load(os.path.join(ckpt, f"{prefix}_sdf_b.pth"), map_location=self._device))
        model_rep.load_state_dict(torch.load(os.path.join(ckpt, f"{prefix}_rep.pth"), map_location=self._device))
        model_atlas_f.load_state_dict(torch.load(os.path.join(ckpt, f"{prefix}_atlas_f.pth"), map_location=self._device))
        model_atlas_b.load_state_dict(torch.load(os.path.join(ckpt, f"{prefix}_atlas_b.pth"), map_location=self._device))

        model_sdf_f.eval()
        model_sdf_b.eval()
        model_rep.eval()
        model_atlas_f.eval()
        model_atlas_b.eval()

        self._garment_models[kind] = {
            "sdf_f": model_sdf_f,
            "sdf_b": model_sdf_b,
            "rep": model_rep,
            "atlas_f": model_atlas_f,
            "atlas_b": model_atlas_b,
            "y_center": float(stat["y_center"]),
            "diag_max": float(stat["diag_max"]),
            "latent_codes": model_rep.weights.detach(),
        }

    def _ensure_draping_models(self, kind: GarmentKind):
        """Load ISP draping models for a garment type (lazy)."""
        if kind in self._draping_models:
            return
        import torch

        drape_mod = self._drape
        ckpt = str(_CHECKPOINT_DIR)

        if kind == "tee":
            model_draping = drape_mod.Pred_decoder_uv_linear(
                d_in=82 + 32, d_hidden=512, depth=8, skip_layer=[4], tanh=False,
            ).to(self._device)
            model_draping.load_state_dict(
                torch.load(os.path.join(ckpt, "drape_shirt.pth"), map_location=self._device)
            )

            model_diffusion = drape_mod.skip_connection(
                d_in=3, width=512, depth=8, d_out=6890, skip_layer=[],
            ).to(self._device)
            model_diffusion.load_state_dict(
                torch.load(os.path.join(ckpt, "smpl_diffusion.pth"), map_location=self._device)
            )

            model_draping.eval()
            model_diffusion.eval()

            self._draping_models[kind] = {
                "draping": model_draping,
                "diffusion": model_diffusion,
                "is_pants": False,
            }

        elif kind == "pants":
            model_draping = drape_mod.Pred_decoder_uv_linear(
                d_in=82 + 32, d_hidden=512, depth=8, skip_layer=[4], tanh=False,
            ).to(self._device)
            model_draping.load_state_dict(
                torch.load(os.path.join(ckpt, "drape_pants.pth"), map_location=self._device)
            )

            model_diffusion = drape_mod.skip_connection(
                d_in=3, width=512, depth=8, d_out=6890, skip_layer=[],
            ).to(self._device)
            model_diffusion.load_state_dict(
                torch.load(os.path.join(ckpt, "smpl_diffusion_TA.pth"), map_location=self._device)
            )

            model_draping.eval()
            model_diffusion.eval()

            self._draping_models[kind] = {
                "draping": model_draping,
                "diffusion": model_diffusion,
                "is_pants": True,
            }

        elif kind == "skirt":
            model_draping = drape_mod.Pred_decoder_uv_linear(
                d_in=82 + 32, d_hidden=512, depth=8, skip_layer=[4], tanh=False,
            ).to(self._device)
            model_draping.load_state_dict(
                torch.load(os.path.join(ckpt, "drape_skirt.pth"), map_location=self._device)
            )

            model_diffusion = drape_mod.skip_connection(
                d_in=3, width=512, depth=8, d_out=6890, skip_layer=[],
            ).to(self._device)
            model_diffusion.load_state_dict(
                torch.load(os.path.join(ckpt, "smpl_diffusion.pth"), map_location=self._device)
            )

            model_draping.eval()
            model_diffusion.eval()

            self._draping_models[kind] = {
                "draping": model_draping,
                "diffusion": model_diffusion,
                "is_pants": False,
            }

    # ── Public API ────────────────────────────────────────────────────

    def get_num_garments(self, kind: GarmentKind) -> int:
        """Return how many garment variants exist for a type."""
        self._ensure_initialized()
        self._ensure_garment_models(kind)
        return len(self._garment_models[kind]["latent_codes"])

    # ── Best T-shirt selection ────────────────────────────────────────

    # Pre-computed top-15 **closed-front** tee indices sorted by
    # Y-coverage * front-coverage score.  idx=248 has best torso coverage
    # (Y=0.534, 670 front verts) followed by 112, 40, 252.
    # Computed via offline scan of all 400 codebook entries.
    _BEST_TEE_INDICES: list[int] = [
        248, 112, 40, 332, 252, 44, 164, 92, 176, 32,
        216, 152, 12, 296, 264,
    ]

    _best_tee_idx_cache: int | None = None

    def find_best_tee_idx(self) -> int:
        """Return the best codebook index for a full T-shirt.

        Uses pre-computed rankings; validates at runtime that the top
        candidate actually produces a reasonable mesh.  Falls back through
        the ranking if not.
        """
        if self._best_tee_idx_cache is not None:
            return self._best_tee_idx_cache

        self._ensure_initialized()
        self._ensure_garment_models("tee")

        # Try pre-ranked indices, pick first that succeeds with good extents
        for idx in self._BEST_TEE_INDICES:
            try:
                mesh = self.generate_tpose_garment("tee", idx_G=idx, resolution=180)
                verts = np.asarray(mesh.vertices)
                y_ext = verts[:, 1].max() - verts[:, 1].min()
                x_ext = verts[:, 0].max() - verts[:, 0].min()
                # Verify front-centre coverage (reject open-front)
                cx = verts[:, 0].mean()
                cz = verts[:, 2].mean()
                mid_y = (verts[:, 1].min() + verts[:, 1].max()) / 2
                front_ctr = (
                    (np.abs(verts[:, 0] - cx) < 0.02)
                    & (verts[:, 2] > cz)
                    & (np.abs(verts[:, 1] - mid_y) < 0.12)
                ).sum()
                if y_ext > 0.30 and x_ext > 0.40 and front_ctr > 3:
                    self._best_tee_idx_cache = idx
                    logger.info(
                        "Selected tee idx_G=%d (Y=%.3f X=%.3f front=%d verts=%d)",
                        idx, y_ext, x_ext, front_ctr, len(mesh.vertices),
                    )
                    return idx
            except Exception:
                continue

        # Fallback: idx 252 is verified closed-front
        self._best_tee_idx_cache = 252
        return 252

    # ── Best Pants selection ───────────────────────────────────────────

    # Pre-computed top-10 pants indices sorted by Y extent (longest legs).
    # Computed via offline scan of all 200 codebook entries.
    _BEST_PANTS_INDICES: list[int] = [180, 135, 10, 130, 190, 45, 160, 175, 65, 55]

    _best_pants_idx_cache: int | None = None

    def find_best_pants_idx(self) -> int:
        """Return the best codebook index for full-length trousers.

        Uses pre-computed rankings; validates at runtime that the top
        candidate actually produces a reasonable mesh.  Falls back through
        the ranking if not.
        """
        if self._best_pants_idx_cache is not None:
            return self._best_pants_idx_cache

        self._ensure_initialized()
        self._ensure_garment_models("pants")

        for idx in self._BEST_PANTS_INDICES:
            try:
                mesh = self.generate_tpose_garment("pants", idx_G=idx, resolution=180)
                y_ext = mesh.bounds[1][1] - mesh.bounds[0][1]
                if y_ext > 0.40:  # at least 40 cm in server coords
                    self._best_pants_idx_cache = idx
                    logger.info(
                        "Selected pants idx_G=%d (Y=%.3f verts=%d)",
                        idx, y_ext, len(mesh.vertices),
                    )
                    return idx
            except Exception:
                continue

        self._best_pants_idx_cache = 180
        return 180

    # ------------------------------------------------------------------
    # Coordinate alignment: ISP SMPL → server body system
    # ------------------------------------------------------------------
    # ISP SMPL body: origin at pelvis, Y-up, feet at Y ≈ −1.12, head ≈ +0.54
    # Server body:  feet at Y = 0, head at Y ≈ 1.75, Y-up
    # We shift so feet touch Y = 0 then scale to server height (1.75 m).
    _SERVER_BODY_HEIGHT = 1.75  # default male body in body_generator

    def _align_to_server_body(
        self,
        mesh: trimesh.Trimesh,
        reference_body: trimesh.Trimesh | None = None,
    ) -> trimesh.Trimesh:
        """Translate + scale an ISP mesh to the server body coordinate system.

        If *reference_body* is given its Y-bounds are used as the "real"
        body extent; otherwise we use the cached SMPL T-pose bounds.
        """
        if reference_body is not None:
            y_min = reference_body.bounds[0][1]
            y_max = reference_body.bounds[1][1]
        else:
            # Use the SMPL T-pose body bounds we loaded at init time
            y_min = self._smpl_tpose_y_min
            y_max = self._smpl_tpose_y_max

        isp_height = y_max - y_min
        if isp_height < 0.01:
            isp_height = 1.0  # safety

        scale = self._SERVER_BODY_HEIGHT / isp_height

        # Shift so that the bottom of the ISP body maps to Y = 0
        mesh.vertices[:, 1] -= y_min
        # Scale to server body height
        mesh.vertices *= scale
        return mesh

    def generate_tpose_garment(
        self,
        kind: GarmentKind,
        idx_G: int = 0,
        resolution: int = 180,
    ) -> trimesh.Trimesh:
        """Generate a T-pose garment mesh using ISP (no draping).

        Parameters
        ----------
        kind : "tee" | "pants" | "skirt"
        idx_G : index into the learned garment codebook
        resolution : UV grid resolution (default 180)

        Returns
        -------
        trimesh.Trimesh in SMPL coordinate space (Y-up, metres).
        """
        self._ensure_initialized()
        self._ensure_garment_models(kind)

        models = self._garment_models[kind]
        uv = self._uv_cache.get(resolution)
        if uv is None:
            uv = self._create_uv_data(resolution)
            self._uv_cache[resolution] = uv

        latent_code = models["latent_codes"][idx_G]

        mesh_sewing, mesh_atlas_f, mesh_atlas_b, mesh_pattern_f, mesh_pattern_b = \
            self._reconstruct_batch(
                models["sdf_f"], models["sdf_b"],
                models["atlas_f"], models["atlas_b"],
                latent_code, uv["vertices_t"], uv["faces"], uv["edges"],
                which=kind, resolution=resolution,
            )

        # Scale to SMPL coordinates
        mesh_sewing.vertices = mesh_sewing.vertices * models["diag_max"] / 2
        mesh_sewing.vertices[:, 1] += models["y_center"]

        # Transform from ISP SMPL coords (origin at pelvis) to server
        # body coords (feet at Y=0, head at ~1.75)
        mesh_sewing = self._align_to_server_body(mesh_sewing)

        return mesh_sewing

    def drape_garment(
        self,
        kind: GarmentKind,
        idx_G: int = 0,
        pose: np.ndarray | None = None,
        beta: np.ndarray | None = None,
        resolution: int = 180,
        smooth: bool = True,
    ) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
        """Generate and drape a garment on a posed SMPL body.

        Parameters
        ----------
        kind : "tee" | "pants" | "skirt"
        idx_G : garment codebook index
        pose : (72,) axis-angle rotations. None = T-pose.
        beta : (10,) SMPL shape params. None = mean shape.
        resolution : UV grid resolution for ISP reconstruction
        smooth : apply Taubin smoothing on the result

        Returns
        -------
        (garment_mesh, body_mesh) — both in world coordinates.
        """
        import torch

        self._ensure_initialized()
        self._ensure_garment_models(kind)
        self._ensure_draping_models(kind)

        device = self._device
        models = self._garment_models[kind]
        drape_m = self._draping_models[kind]

        # ── Pose & shape tensors ─────────────────────────────────
        if pose is None:
            pose_t = torch.zeros(72, device=device)
        else:
            pose_t = torch.FloatTensor(pose).to(device)

        if beta is None:
            beta_t = torch.zeros(10, device=device)
        else:
            beta_t = torch.FloatTensor(beta).to(device)

        # Transform pose (zero root Y-tilt)
        pose_2d, rotate_original, rotate_zero_inv = self._transform_pose(
            pose_t.unsqueeze(0)
        )
        pose_t = pose_2d.squeeze()

        # ── SMPL forward pass ────────────────────────────────────
        smpl_server = self._smpl_data["server"]
        with torch.no_grad():
            w_smpl, tfs, verts_body, pose_offsets, shape_offsets, root_J = \
                self._infer_smpl(pose_t.unsqueeze(0), beta_t.unsqueeze(0), smpl_server)
            packed_input_smpl = [w_smpl, tfs, pose_offsets, shape_offsets]
            root_J_np = root_J.squeeze(0).cpu().numpy()

        # ── ISP reconstruct ──────────────────────────────────────
        uv_180 = self._uv_cache[180]
        uv_200 = self._uv_cache[200]

        latent_code = models["latent_codes"][idx_G]

        mesh_sewing, mesh_atlas_f, mesh_atlas_b, mesh_pattern_f, mesh_pattern_b = \
            self._reconstruct_batch(
                models["sdf_f"], models["sdf_b"],
                models["atlas_f"], models["atlas_b"],
                latent_code, uv_180["vertices_t"], uv_180["faces"], uv_180["edges"],
                which=kind, resolution=180,
            )

        # Scale to SMPL coordinates
        diag_max = models["diag_max"]
        y_center = models["y_center"]
        mesh_sewing.vertices = mesh_sewing.vertices * diag_max / 2
        mesh_sewing.vertices[:, 1] += y_center
        mesh_atlas_f.vertices = mesh_atlas_f.vertices * diag_max / 2
        mesh_atlas_b.vertices = mesh_atlas_b.vertices * diag_max / 2
        mesh_atlas_f.vertices[:, 1] += y_center
        mesh_atlas_b.vertices[:, 1] += y_center

        # ── Prepare draping inputs ───────────────────────────────
        mesh_uv_200 = uv_200["mesh"]
        uv_faces_cuda = torch.LongTensor(uv_200["faces"]).to(device)

        if kind in ("tee",):
            fix_mask = self._generate_fix_mask(mesh_pattern_f, mesh_pattern_b, mesh_uv_200)
        else:
            fix_mask = self._generate_fix_mask_bottom(mesh_pattern_f, mesh_pattern_b, mesh_uv_200)

        fix_mask = torch.FloatTensor(fix_mask).to(device)

        barycentric_uv_f, closest_face_idx_uv_f = self._barycentric_faces(
            mesh_pattern_f, mesh_uv_200, return_tensor=True,
        )
        barycentric_uv_b, closest_face_idx_uv_b = self._barycentric_faces(
            mesh_pattern_b, mesh_uv_200, return_tensor=True,
        )

        vertices_T_f = torch.FloatTensor(mesh_atlas_f.vertices).to(device)
        vertices_T_b = torch.FloatTensor(mesh_atlas_b.vertices).to(device)

        packed_input = [
            vertices_T_f, vertices_T_b,
            mesh_atlas_f.faces, mesh_atlas_b.faces,
            barycentric_uv_f, barycentric_uv_b,
            closest_face_idx_uv_f, closest_face_idx_uv_b,
            fix_mask, latent_code,
        ]

        # ── Drape ────────────────────────────────────────────────
        is_pants = drape_m["is_pants"]
        Rot_rest = self._smpl_data["Rot_rest"] if is_pants else None
        po_rest = self._smpl_data["pose_offsets_rest"] if is_pants else None

        garment_skinning_f, garment_skinning_b, garment_skinning, garment_faces = \
            self._draping_fn(
                packed_input, pose_t, beta_t,
                drape_m["diffusion"], drape_m["draping"],
                uv_faces_cuda, packed_input_smpl,
                is_pants=is_pants,
                Rot_rest=Rot_rest,
                pose_offsets_rest=po_rest,
            )

        # ── Build result meshes ──────────────────────────────────
        final_verts = garment_skinning.squeeze(0).cpu().numpy()
        sewing_faces = mesh_sewing.faces

        garment_mesh = trimesh.Trimesh(
            final_verts, sewing_faces, process=False, validate=False,
        )

        # Rotate back to original orientation
        garment_mesh.vertices = np.einsum(
            "ij,nj->ni",
            rotate_original[0],
            np.einsum(
                "ij,nj->ni",
                rotate_zero_inv[0],
                garment_mesh.vertices - root_J_np,
            ),
        ) + root_J_np

        if smooth:
            trimesh.smoothing.filter_taubin(garment_mesh, lamb=0.5)

        # Body mesh
        body_verts = verts_body[0].detach().cpu().numpy()
        body_mesh = trimesh.Trimesh(body_verts, self._smpl_data["faces"])
        body_mesh.vertices = np.einsum(
            "ij,nj->ni",
            rotate_original[0],
            np.einsum(
                "ij,nj->ni",
                rotate_zero_inv[0],
                body_mesh.vertices - root_J_np,
            ),
        ) + root_J_np

        # Transform from ISP SMPL coords (origin at pelvis) to server
        # body coords (feet at Y=0, head at ~1.75)
        garment_mesh = self._align_to_server_body(garment_mesh, body_mesh)
        body_mesh = self._align_to_server_body(body_mesh)

        return garment_mesh, body_mesh


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_service: ISPService | None = None
_service_lock = threading.Lock()


def get_isp_service() -> ISPService:
    """Get or create the singleton ISP service (lazy initialization)."""
    global _service
    if _service is not None and _service._initialized:
        return _service
    with _service_lock:
        if _service is None:
            _service = ISPService()
        _service._ensure_initialized()
        return _service
