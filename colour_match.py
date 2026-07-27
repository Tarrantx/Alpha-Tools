"""
Batch colour matching — force a set of textures to share one reference image's
colour statistics, to kill frame-to-frame colour flicker in animations.

Two methods (see COLOURMATCH_METHODS):
  * Reinhard (mean/std) — shift/scale each channel so its mean and standard
    deviation match the reference. Gentle; removes overall cast/exposure drift
    while keeping each frame's own detail. (Same transfer ESRGAN Tools uses for
    per-tile colour normalisation.)
  * Histogram match — remap each channel so its whole tonal DISTRIBUTION matches
    the reference (quantile mapping). Stronger; better when palettes differ more.

Two colourspaces (COLOURMATCH_SPACES): "LAB" (perceptual, decorrelated — cleaner
matching of cast/brightness) or "RGB" (direct). LAB is computed in float here
(accurate, unlike PIL's lossy 8-bit LAB), so matching a frame to itself is a
near-identity no-op.

Statistics are always taken from OPAQUE pixels only — transparent/black fill
would otherwise skew every frame's "average colour". Only RGB is modified; the
alpha channel is passed through untouched.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

COLOURMATCH_METHODS = ["Reinhard (mean/std)", "Histogram match"]
COLOURMATCH_SPACES = ["LAB", "RGB"]

_NQ = 256                       # quantile samples stored per channel
_ALPHA_FLOOR = 8                # alpha > this = opaque enough to sample stats from

# sRGB <-> linear <-> XYZ(D65) <-> LAB, all in float (D65 white point).
_M_RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                       [0.2126729, 0.7151522, 0.0721750],
                       [0.0193339, 0.1191920, 0.9503041]], dtype=np.float64)
_M_XYZ2RGB = np.linalg.inv(_M_RGB2XYZ)
_WHITE = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """HxWx3 float sRGB (0-255) -> HxWx3 float LAB (L 0-100, a/b ~ -128..127)."""
    r = rgb.astype(np.float64) / 255.0
    lin = np.where(r <= 0.04045, r / 12.92, ((r + 0.055) / 1.055) ** 2.4)
    xyz = lin @ _M_RGB2XYZ.T / _WHITE
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def _lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """HxWx3 float LAB -> HxWx3 float sRGB (0-255, unclamped)."""
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    f = np.stack([fx, fy, fz], axis=-1)
    xyz = np.where(f ** 3 > 0.008856, f ** 3, (f - 16.0 / 116.0) / 7.787) * _WHITE
    lin = xyz @ _M_XYZ2RGB.T
    srgb = np.where(lin <= 0.0031308, 12.92 * lin,
                    1.055 * np.maximum(lin, 0.0) ** (1 / 2.4) - 0.055)
    return srgb * 255.0


def _to_space(rgb_uint8: np.ndarray, space: str) -> np.ndarray:
    if str(space).upper() == "LAB":
        return _srgb_to_lab(rgb_uint8.astype(np.float64))
    return rgb_uint8.astype(np.float64)


def _from_space(arr: np.ndarray, space: str) -> np.ndarray:
    if str(space).upper() == "LAB":
        arr = _lab_to_srgb(arr)
    return np.clip(arr, 0, 255).astype(np.uint8)


def _opaque_mask(rgba: Image.Image) -> np.ndarray:
    a = np.asarray(rgba.convert("RGBA").getchannel("A"), dtype=np.uint8)
    mask = a > _ALPHA_FLOOR
    # Fall back to the whole image if there's essentially nothing opaque.
    if mask.sum() < max(16, a.size // 1000):
        mask = np.ones(a.shape, dtype=bool)
    return mask


def compute_reference_stats(ref_rgba: Image.Image, space: str = "LAB") -> dict:
    """Per-channel statistics of the reference's opaque pixels, in `space`:
    means, stds (for Reinhard) and quantile values (for histogram match)."""
    rgb = np.asarray(ref_rgba.convert("RGB"), dtype=np.uint8)
    mask = _opaque_mask(ref_rgba)
    sp = _to_space(rgb, space)
    px = sp[mask]                                   # N x 3
    q = np.linspace(0.0, 1.0, _NQ)
    means, stds, quantiles = [], [], []
    for c in range(3):
        vals = px[:, c]
        means.append(float(vals.mean()))
        stds.append(float(vals.std()))
        quantiles.append(np.quantile(vals, q).astype(np.float64).tolist())
    return {"space": str(space), "means": means, "stds": stds,
            "quantiles": quantiles}


def apply_colour_match(rgba: Image.Image, ref: dict, method: str,
                       strength: float) -> Image.Image:
    """Return `rgba` with its RGB matched to the reference stats (alpha kept).
    `strength` 0..1 blends the correction with the original."""
    if not ref or strength <= 0.0:
        return rgba
    if rgba.mode != "RGBA":
        return rgba
    space = ref.get("space", "LAB")
    rgb = np.asarray(rgba.convert("RGB"), dtype=np.uint8)
    mask = _opaque_mask(rgba)
    sp = _to_space(rgb, space)
    out = sp.copy()

    if str(method).lower().startswith("hist"):
        q = np.linspace(0.0, 1.0, _NQ)
        for c in range(3):
            src_q = np.quantile(sp[..., c][mask], q)      # source quantile values
            ref_q = np.asarray(ref["quantiles"][c], dtype=np.float64)
            # Monotonic quantile map: a value at source-rank r -> ref value at r.
            src_q = np.maximum.accumulate(src_q)          # guard against ties
            out[..., c] = np.interp(sp[..., c], src_q, ref_q)
    else:                                                 # Reinhard mean/std
        rmeans, rstds = ref["means"], ref["stds"]
        for c in range(3):
            vals = sp[..., c][mask]
            sm, ss = float(vals.mean()), float(vals.std())
            if ss < 1e-6:
                continue
            out[..., c] = (sp[..., c] - sm) / ss * float(rstds[c]) + float(rmeans[c])

    if strength < 1.0:
        out = out * float(strength) + sp * (1.0 - float(strength))

    matched = _from_space(out, space)
    res = Image.fromarray(matched, mode="RGB").convert("RGBA")
    res.putalpha(rgba.getchannel("A"))
    return res


def ref_signature(ref: dict) -> tuple:
    """Small hashable signature of a reference-stats dict, for cache keys."""
    if not ref:
        return ()
    return (ref.get("space"), tuple(ref.get("means", ())),
            tuple(ref.get("stds", ())))
