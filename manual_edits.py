"""Manual per-pixel touch-up edit layer for Alpha Tools.

A SOURCE-resolution overlay the user paints in the preview to fix pixels that no
setting can. Two independent channels, applied at different points in the
pipeline (see alpha_pipeline.process_pil):

  * RGB edits   (rgb + rgb_mask)   applied EARLY  — before tone, so tone /
    perceptual / FX / quantise reprocess them and keep them colour-consistent.
  * alpha edits (alpha + alpha_mask) applied LATE — after snap / clamp / shape
    (so those can't destroy the painted values) but before compose, so the rest
    of post-processing still applies.

Brush size and coordinates are in SOURCE pixels, so edits are zoom / pan /
resolution invariant and land identically on the full-res save and in batch.
The pipeline stages resample the layer (nearest) to whatever resolution they run
at: the full-res save (1:1, exact), the ~1024px live preview (downscaled), or a
full-res ROI crop (a sliced region). Batch uses `fit="place"` — fixed x,y, no
scaling, clipped to the target — per the same-size-images assumption.

Undo is per-stroke: each stroke (press→drag→release) snapshots the prior state
of every square it touches and can be reverted by replaying those snapshots in
reverse. The sidecar (`*.edits.npz`) stores only the flattened layer, not the
undo history.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

MAX_UNDO = 24                       # stroke history depth
CHANNELS = ("alpha", "rgb")
MODES = ("paint", "erase")


class EditLayer:
    """Full-resolution manual edit layer for one source image."""

    def __init__(self, width: int, height: int):
        self.w = int(width)
        self.h = int(height)
        self.rgb = np.zeros((self.h, self.w, 3), np.uint8)
        self.rgb_mask = np.zeros((self.h, self.w), bool)
        self.alpha = np.zeros((self.h, self.w), np.uint8)
        self.alpha_mask = np.zeros((self.h, self.w), bool)
        self._version = 0                        # bumped on every committed change
        self._undo: list = []                    # each: list of square snapshots
        self._pending: Optional[list] = None     # snapshots for the active stroke

    # -- state ---------------------------------------------------------------
    def version(self) -> int:
        return self._version

    def has_rgb(self) -> bool:
        return bool(self.rgb_mask.any())

    def has_alpha(self) -> bool:
        return bool(self.alpha_mask.any())

    def is_empty(self) -> bool:
        return not self.has_rgb() and not self.has_alpha()

    def can_undo(self) -> bool:
        return bool(self._undo)

    # -- painting ------------------------------------------------------------
    def begin_stroke(self):
        """Start a stroke; snapshots accumulate until end_stroke()."""
        self._pending = []

    def apply_dab(self, cx: float, cy: float, size: int, channel: str,
                  mode: str = "paint", alpha_value: int = 0,
                  rgb_value=(0, 0, 0)):
        """Paint (or erase) a square of `size` SOURCE px centred on (cx, cy).
        No-op if the square falls entirely outside the image."""
        if channel not in CHANNELS or mode not in MODES:
            raise ValueError(f"bad channel/mode: {channel}/{mode}")
        s = max(1, int(size))
        half = s // 2
        x0 = max(0, int(round(cx)) - half)
        y0 = max(0, int(round(cy)) - half)
        x1 = min(self.w, x0 + s)
        y1 = min(self.h, y0 + s)
        if x1 <= x0 or y1 <= y0:
            return
        standalone = self._pending is None       # allow a dab without begin/end
        if standalone:
            self._pending = []
        self._snapshot(x0, y0, x1, y1)
        if channel == "alpha":
            if mode == "erase":
                self.alpha_mask[y0:y1, x0:x1] = False
            else:
                self.alpha[y0:y1, x0:x1] = int(alpha_value) & 0xFF
                self.alpha_mask[y0:y1, x0:x1] = True
        else:                                     # rgb
            if mode == "erase":
                self.rgb_mask[y0:y1, x0:x1] = False
            else:
                self.rgb[y0:y1, x0:x1] = np.asarray(rgb_value, np.uint8)[:3]
                self.rgb_mask[y0:y1, x0:x1] = True
        # Bump per dab (not per stroke) so the live preview cache invalidates
        # mid-drag; undo granularity stays per-stroke (begin/end_stroke).
        self._version += 1
        if standalone:
            self.end_stroke()

    def end_stroke(self):
        """Commit the active stroke to the undo history (version bumps per dab)."""
        if not self._pending:
            self._pending = None
            return
        self._undo.append(self._pending)
        if len(self._undo) > MAX_UNDO:
            self._undo.pop(0)
        self._pending = None

    def _snapshot(self, x0, y0, x1, y1):
        self._pending.append((
            x0, y0, x1, y1,
            self.rgb[y0:y1, x0:x1].copy(),
            self.rgb_mask[y0:y1, x0:x1].copy(),
            self.alpha[y0:y1, x0:x1].copy(),
            self.alpha_mask[y0:y1, x0:x1].copy(),
        ))

    def undo(self) -> bool:
        """Revert the most recent stroke. Returns False if nothing to undo."""
        if not self._undo:
            return False
        # Replay the stroke's square snapshots in REVERSE so overlapping dabs
        # restore correctly (the earliest snapshot holds the true prior value).
        for x0, y0, x1, y1, rgb, rgbm, a, am in reversed(self._undo.pop()):
            self.rgb[y0:y1, x0:x1] = rgb
            self.rgb_mask[y0:y1, x0:x1] = rgbm
            self.alpha[y0:y1, x0:x1] = a
            self.alpha_mask[y0:y1, x0:x1] = am
        self._version += 1
        return True

    def clear(self):
        """Delete all manual edits (undoable as a single step)."""
        if self.is_empty():
            return
        self.begin_stroke()
        self._snapshot(0, 0, self.w, self.h)
        self.rgb_mask[:] = False
        self.alpha_mask[:] = False
        self.end_stroke()
        self._version += 1

    # -- rendering into the pipeline ----------------------------------------
    def _render(self, channel: str, region, out_w: int, out_h: int, fit: str):
        """Return (values, mask) for `channel` over source `region`, sized to
        (out_w, out_h). `fit`: 'scale' resamples (nearest); 'place' pastes at
        (0,0) clipped (no scaling). region=None means the whole layer."""
        if channel == "rgb":
            vals_src, mask_src = self.rgb, self.rgb_mask
        else:
            vals_src, mask_src = self.alpha, self.alpha_mask

        if region is None:
            x0, y0, x1, y1 = 0, 0, self.w, self.h
        else:
            x0 = max(0, min(self.w, int(region[0])))
            y0 = max(0, min(self.h, int(region[1])))
            x1 = max(0, min(self.w, int(region[2])))
            y1 = max(0, min(self.h, int(region[3])))
        if x1 <= x0 or y1 <= y0:
            return None, None
        vsub = vals_src[y0:y1, x0:x1]
        msub = mask_src[y0:y1, x0:x1]

        if fit == "place":
            hh = min(out_h, vsub.shape[0])
            ww = min(out_w, vsub.shape[1])
            if channel == "rgb":
                vals = np.zeros((out_h, out_w, 3), np.uint8)
            else:
                vals = np.zeros((out_h, out_w), np.uint8)
            mask = np.zeros((out_h, out_w), bool)
            vals[:hh, :ww] = vsub[:hh, :ww]
            mask[:hh, :ww] = msub[:hh, :ww]
            return vals, mask

        # fit == "scale": exact when the region already matches the target.
        if (x1 - x0) == out_w and (y1 - y0) == out_h:
            return vsub.copy(), msub.copy()
        size = (out_w, out_h)
        N = Image.Resampling.NEAREST
        if channel == "rgb":
            vals = np.asarray(Image.fromarray(vsub, "RGB").resize(size, N))
        else:
            vals = np.asarray(Image.fromarray(vsub, "L").resize(size, N))
        mask = np.asarray(
            Image.fromarray((msub.astype(np.uint8) * 255), "L").resize(size, N)
        ) > 127
        return vals, mask

    # -- persistence (sidecar) ----------------------------------------------
    def save(self, path):
        np.savez_compressed(
            str(path), w=self.w, h=self.h, rgb=self.rgb, rgb_mask=self.rgb_mask,
            alpha=self.alpha, alpha_mask=self.alpha_mask)

    @classmethod
    def load(cls, path) -> "EditLayer":
        with np.load(str(path)) as z:
            layer = cls(int(z["w"]), int(z["h"]))
            layer.rgb = z["rgb"]
            layer.rgb_mask = z["rgb_mask"]
            layer.alpha = z["alpha"]
            layer.alpha_mask = z["alpha_mask"]
        return layer

    def flatten(self) -> "EditLayer":
        """A copy holding just the pixel data (no undo history) — used for batch,
        where the layer is pickled to each worker process; dropping the undo
        stack keeps the payload small and independent of later interactive edits."""
        c = EditLayer(self.w, self.h)
        c.rgb = self.rgb.copy()
        c.rgb_mask = self.rgb_mask.copy()
        c.alpha = self.alpha.copy()
        c.alpha_mask = self.alpha_mask.copy()
        c._version = self._version
        return c

    def apply_granulate_dab(self, cx, cy, size: int, base_rgb, amount: int):
        """Paint a square of per-pixel colour JITTER around the underlying
        source colour — breaks flat upscale banding into subtle speckle. Writes
        the RGB channel. `base_rgb` = source RGB (HxWx3 uint8, layer-sized);
        `amount` = max per-channel jitter (0 = paint the source colour flat)."""
        if base_rgb is None:
            return
        amount = max(0, int(amount))
        s = max(1, int(size))
        half = s // 2
        x0 = max(0, int(round(cx)) - half)
        y0 = max(0, int(round(cy)) - half)
        x1 = min(self.w, x0 + s)
        y1 = min(self.h, y0 + s)
        if x1 <= x0 or y1 <= y0:
            return
        standalone = self._pending is None
        if standalone:
            self._pending = []
        self._snapshot(x0, y0, x1, y1)
        sub = base_rgb[y0:y1, x0:x1, :3].astype(np.int16)
        if amount > 0:
            noise = np.random.randint(-amount, amount + 1, size=sub.shape,
                                      dtype=np.int16)
            sub = np.clip(sub + noise, 0, 255)
        self.rgb[y0:y1, x0:x1] = sub.astype(np.uint8)
        self.rgb_mask[y0:y1, x0:x1] = True
        self._version += 1
        if standalone:
            self.end_stroke()

    def placed_into(self, w: int, h: int) -> "EditLayer":
        """A copy of this layer sized (w, h): identical when the size matches,
        otherwise pasted at (0,0) and clipped (fixed x,y). Used when loading a
        saved edit layer onto a different image."""
        if (self.w, self.h) == (int(w), int(h)):
            return self.flatten()
        c = EditLayer(int(w), int(h))
        hh, ww = min(c.h, self.h), min(c.w, self.w)
        c.rgb[:hh, :ww] = self.rgb[:hh, :ww]
        c.rgb_mask[:hh, :ww] = self.rgb_mask[:hh, :ww]
        c.alpha[:hh, :ww] = self.alpha[:hh, :ww]
        c.alpha_mask[:hh, :ww] = self.alpha_mask[:hh, :ww]
        c._version = self._version
        return c

    @staticmethod
    def sidecar_path(image_path) -> Path:
        p = Path(image_path)
        return p.with_suffix(p.suffix + ".edits.npz")


# -- pipeline entry points (called by alpha_pipeline stages) -----------------
def apply_rgb(rgb_pil: Image.Image, layer: Optional[EditLayer],
              region=None, fit: str = "scale") -> Image.Image:
    """Return `rgb_pil` (RGB) with the layer's RGB edits composited in."""
    if layer is None or not layer.has_rgb():
        return rgb_pil
    W, H = rgb_pil.size
    vals, mask = layer._render("rgb", region, W, H, fit)
    if mask is None or not mask.any():
        return rgb_pil
    arr = np.array(rgb_pil.convert("RGB"), np.uint8)
    arr[mask] = vals[mask]
    return Image.fromarray(arr, "RGB")


def apply_alpha(alpha_pil: Image.Image, layer: Optional[EditLayer],
                region=None, fit: str = "scale") -> Image.Image:
    """Return `alpha_pil` (L) with the layer's alpha edits composited in."""
    if layer is None or not layer.has_alpha():
        return alpha_pil
    W, H = alpha_pil.size
    vals, mask = layer._render("alpha", region, W, H, fit)
    if mask is None or not mask.any():
        return alpha_pil
    arr = np.array(alpha_pil.convert("L"), np.uint8)
    arr[mask] = vals[mask]
    return Image.fromarray(arr, "L")


class EditController:
    """Turns paint events (in SOURCE px) into EditLayer operations. GUI-free:
    the preview widget reports press/drag/release; the UI panel sets the tool,
    brush size and values. Interpolates dabs along a drag so fast strokes leave
    no gaps. `on_change` fires after any layer mutation (to refresh the preview).
    """
    TOOLS = ("pencil", "eraser", "picker", "granulate")

    def __init__(self, layer: EditLayer, on_change=None):
        self.layer = layer
        self.tool = "pencil"
        self.channel = "alpha"                 # set from the view mode
        self.brush = 4                         # source px
        self.alpha_value = 255
        self.rgb_value = (255, 0, 0)
        self.granulate_amount = 12             # max per-channel jitter (granulate)
        self.source_rgb = None                 # HxWx3 uint8 base for granulate
        self.on_change = on_change
        self._last = None
        self._stroking = False
        self._gran_base = None                 # frozen effective RGB for a stroke

    def _effective_rgb(self):
        """Source RGB with the current manual RGB edits composited in — the base
        the Granulate tool jitters around, so it BUILDS ON pencil edits instead
        of erasing them back to the source colour."""
        if self.source_rgb is None:
            return None
        m = self.layer.rgb_mask
        if not m.any():
            return self.source_rgb             # no edits yet: source as-is
        return np.where(m[..., None], self.layer.rgb, self.source_rgb)

    def _dab(self, sx, sy):
        if self.tool == "granulate":
            self.layer.apply_granulate_dab(sx, sy, self.brush, self._gran_base,
                                           self.granulate_amount)
            return
        mode = "erase" if self.tool == "eraser" else "paint"
        self.layer.apply_dab(sx, sy, self.brush, self.channel, mode,
                             alpha_value=self.alpha_value, rgb_value=self.rgb_value)

    def begin(self, sx, sy):
        """Begin a paint/erase stroke. No-op for the picker (handled upstream)."""
        if self.tool == "picker":
            return
        if self.tool == "granulate":
            # Freeze the base once per stroke (source + existing edits) so a
            # stroke can't compound its own jitter and pencil edits survive.
            self._gran_base = self._effective_rgb()
        self.layer.begin_stroke()
        self._stroking = True
        self._last = (sx, sy)
        self._dab(sx, sy)
        self._changed()

    def to(self, sx, sy):
        if not self._stroking:
            return
        x0, y0 = self._last
        dist = max(abs(sx - x0), abs(sy - y0))
        steps = max(1, int(dist / max(1, self.brush / 2.0)))
        for i in range(1, steps + 1):
            self._dab(x0 + (sx - x0) * i / steps, y0 + (sy - y0) * i / steps)
        self._last = (sx, sy)
        self._changed()

    def end(self):
        if not self._stroking:
            return
        self.layer.end_stroke()
        self._gran_base = None
        self._stroking = False
        self._last = None
        self._changed()

    def undo(self):
        if self.layer.undo():
            self._changed()

    def clear(self):
        self.layer.clear()
        self._changed()

    def _changed(self):
        if self.on_change:
            self.on_change()
