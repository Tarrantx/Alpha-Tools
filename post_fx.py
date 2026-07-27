"""
ESRGAN Tools — post-processing effects via ImageMagick.

These run AFTER the AI upscale (and, in atlas mode, after the upscale of each
padded tile, BEFORE the padding is cropped off — so convolution kernels in
the effects get symmetric mirror context and don't introduce new seams).

magick.exe is discovered at runtime: bundled next to the frozen exe, on PATH,
or borrowed from a sibling app's installation (Alpha-Tools / Vtracer-Tools).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# Display labels (also the values stored on disk in params)
EFFECT_NONE        = "None"
EFFECT_UNSHARP     = "Unsharp Mask (sharpen)"
EFFECT_BILATERAL   = "Bilateral Smooth (clean)"
EFFECT_SATURATION  = "Saturation Boost (vibrance)"
EFFECT_EMBOSS      = "Emboss (metallic etch)"
EFFECT_CHROME        = "Chrome (liquid / shiny)"
EFFECT_COLOUR_TINT   = "Colour Tint (multiply blend)"
EFFECT_FILM_GRAIN    = "Film Grain (photographic noise)"
EFFECT_FROSTED       = "Frosted Glass (diffuse blur)"
EFFECT_LOMO          = "Lomography (retro colour cast)"
EFFECT_PAINT         = "Paint (oil paint)"
EFFECT_GRANULATE     = "Granulate (seam / tone blend)"
EFFECT_SCREEN        = "Screen (CRT scanlines)"
EFFECT_TURBULENCE    = "Turbulence (chaotic warp)"
EFFECT_NORMAL_MAP    = "Normal Map (tangent-space)"
EFFECT_BEVEL         = "Bevel (inner edge relief)"
EFFECT_DAVEHILL      = "Dave Hill (hyper-real photo)"
EFFECT_TEXTURIZE     = "Texturize (fabric / canvas)"
EFFECT_WATERCOLOR    = "Watercolor (painted)"
EFFECT_SCREENFX      = "Screen Effects (mesh / displace)"

# Order in the dropdown: cleanup/sharpening first (most-used universal),
# then stylistic, then retro/texture effects, then output-changing last.
EFFECTS_LIST = [
    EFFECT_NONE,
    EFFECT_UNSHARP,
    EFFECT_BILATERAL,
    EFFECT_SATURATION,
    EFFECT_EMBOSS,
    EFFECT_CHROME,
    EFFECT_COLOUR_TINT,
    EFFECT_FILM_GRAIN,
    EFFECT_FROSTED,
    EFFECT_LOMO,
    EFFECT_PAINT,
    EFFECT_GRANULATE,
    EFFECT_SCREEN,
    EFFECT_TURBULENCE,
    EFFECT_NORMAL_MAP,
    EFFECT_BEVEL,
    EFFECT_DAVEHILL,
    EFFECT_TEXTURIZE,
    EFFECT_WATERCOLOR,
    EFFECT_SCREENFX,
]


# Display label → tile dimension in pixels. 0 = no tiling (apply globally).
# Same set as the Tune-tab visual grid for consistency, so users can pair
# the visual grid with a matching tile size for effects.
TILE_SIZE_OPTIONS = {
    "Off":       0,
    "4×4":       4,
    "8×8":       8,
    "16×16":     16,
    "32×32":     32,
    "64×64":     64,
    "128×128":   128,
    "256×256":   256,
    "512×512":   512,
}


def effect_strips_alpha(effect: str) -> bool:
    """True for effects whose output is a different kind of image (normal map)
    where reattaching the source alpha would be meaningless."""
    return effect == EFFECT_NORMAL_MAP


def _bundled_dir() -> Optional[Path]:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return None


def find_magick() -> Optional[str]:
    """Return absolute path to magick.exe, or None if not findable."""
    # 1. Bundled next to the frozen exe (preferred for shipped MSI)
    bundled = _bundled_dir()
    if bundled is not None:
        for sub in ("", "imagemagick", "vendor/imagemagick"):
            p = bundled / sub / "magick.exe"
            if p.exists():
                return str(p)
    # 2. On PATH
    on_path = shutil.which("magick")
    if on_path:
        return on_path
    # 3. Sibling apps that already bundle it, or common installs
    candidates = [
        r"C:\Scripts\Alpha-Tools\imagemagick\magick.exe",
        r"C:\Scripts\Vtracer-Tools\imagemagick\magick.exe",
        r"C:\Program Files\ImageMagick\magick.exe",
        r"C:\Program Files (x86)\ImageMagick\magick.exe",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def _effect_args(effect: str, strength: float, **extras) -> Optional[list[str]]:
    """Return ImageMagick args (between input and output paths) for an effect.
    strength is a 0-2 multiplier (1.0 = nominal). Returns None for unknown,
    and for effects implemented in Python rather than IM (see apply_effect).
    extras carries per-effect parameters (e.g. sketch_edge, sketch_contrast)."""
    s = max(0.0, min(strength, 2.0))

    if effect == EFFECT_UNSHARP:
        # `-unsharp radius x sigma + amount + threshold`
        # Radius/sigma fixed at a sensible 1×1; amount scales with strength,
        # threshold suppresses over-sharpening of smooth areas.
        amount = max(0.2, s * 1.0)
        return ["-unsharp", f"1x1+{amount:.2f}+0.05"]

    if effect == EFFECT_BILATERAL:
        # `-selective-blur radius x sigma + threshold%` — universally
        # supported edge-preserving smoother (works on IM 6 and all 7).
        # Only blurs pixels whose values are within threshold% of their
        # neighbours, so sharp edges stay sharp. (Note: was originally
        # `-bilateral-blur` which requires IM 7.0.7-22+, not in all builds.)
        sigma = max(1.0, s * 3.0)
        threshold = max(5, int(round(s * 15)))   # 5%..30%
        return ["-selective-blur", f"0x{sigma:.1f}+{threshold}%"]

    if effect == EFFECT_SATURATION:
        # `-modulate brightness, saturation, hue` — all percentages.
        # Strength 0 → ~110% (a touch), 1 → 200% (double), 2 → 300% (triple).
        sat_pct = max(110.0, 100.0 + s * 100.0)
        return ["-modulate", f"100,{sat_pct:.0f},100"]

    if effect == EFFECT_EMBOSS:
        return None  # handled as a multi-pass Python helper in apply_effect

    if effect == EFFECT_CHROME:
        return None  # handled as a multi-pass Python helper in apply_effect

    if effect == EFFECT_FILM_GRAIN:
        # Flood a clone to neutral gray, add Gaussian noise, blur slightly,
        # then Softlight-composite over the source.  Softlight from near-gray
        # lightens highlights and darkens shadows without shifting overall
        # tonality — the classic film-grain transfer curve.
        attenuation = max(0.1, s * 0.6)
        return [
            "(", "+clone",
            "-fill", "gray50", "-colorize", "100",
            "-attenuate", f"{attenuation:.2f}",
            "+noise", "Gaussian",
            "-blur", "0x0.5",
            ")",
            "-compose", "Softlight", "-composite",
        ]

    if effect == EFFECT_FROSTED:
        # Fred Weinhaus frosted-glass approach: pre-blur → spread → post-blur.
        # Pre-blur (applied first) controls texture detail visible through the
        # glass — lower = more detail, higher = more diffuse.
        # Spread displaces pixels randomly within the given radius (the main
        # "thickness" of the frosted effect).
        # Post-blur is a light cleanup pass that softens the harsh jagged edges
        # that -spread introduces at region boundaries.
        pre_blur  = max(0.0, float(extras.get("frost_pre_blur",  5.0)))
        spread    = max(0,   int(extras.get("frost_spread",      5)))
        post_blur = max(0.0, float(extras.get("frost_post_blur", 0.5)))
        args: list[str] = []
        if pre_blur > 0:
            args += ["-blur", f"0x{pre_blur:.1f}"]
        args += ["-spread", str(spread)]
        if post_blur > 0:
            args += ["-blur", f"0x{post_blur:.1f}"]
        return args

    if effect == EFFECT_LOMO:
        # Three lomo types via selective channel reduction (matching Fred Weinhaus's
        # channel-swap approach) + saturation boost + corner vignette:
        #   turquoise — reduce Red   → boosts cyan/teal complement
        #   purple    — reduce Green → boosts magenta/purple complement
        #   lime      — reduce Blue  → boosts yellow/lime complement
        lomo_type = str(extras.get("lomo_type", "turquoise")).lower()
        if lomo_type == "purple":
            channel_args = ["-channel", "Green", "-evaluate", "multiply", "0.65", "+channel"]
        elif lomo_type == "lime":
            channel_args = ["-channel", "Blue",  "-evaluate", "multiply", "0.55", "+channel"]
        else:  # turquoise (default)
            channel_args = ["-channel", "Red",   "-evaluate", "multiply", "0.65", "+channel"]
        return [
            *channel_args,
            "-modulate", "100,160,100",
            # Vignette removed: it operates on the full RGB canvas, so it
            # darkens pixels that are transparent in the source image.  When
            # the alpha channel is reattached those pixels appear as solid
            # black, creating a heavy dark oval around the sprite content.
        ]

    if effect == EFFECT_PAINT:
        # Fred Weinhaus painteffect pipeline:
        #   gamma → selective-blur (edge-preserving) → oil-paint → edge clone
        # The edge clone: sharpen → morphology edgein → multiply by edgegain
        # → grey → negate → Multiply-composite back as dark outlines.
        gamma     = max(0.1, float(extras.get("paint_gamma",  1.0)))
        blur      = max(0.0, float(extras.get("paint_blur",   2.0)))
        paint_r   = max(0,   int(extras.get("paint_radius",   2)))
        edge_gain = max(0.0, float(extras.get("paint_edge",   8.0)))

        args: list[str] = []
        if abs(gamma - 1.0) > 0.001:
            args += ["-gamma", f"{gamma:.2f}"]
        if blur > 0:
            args += ["-selective-blur", f"0x{blur:.1f}+20%"]
        if paint_r > 0:
            args += ["-paint", str(paint_r)]
        if edge_gain > 0:
            args += [
                "(", "+clone",
                "-sharpen", "0x2",
                "-morphology", "edgein", "diamond:1",
                "-evaluate", "multiply", f"{edge_gain:.1f}",
                "-colorspace", "gray",
                "-negate",
                ")",
                "-compose", "Multiply", "-composite",
            ]
        return args or None

    if effect == EFFECT_GRANULATE:
        # IM -sketch used as a granulation / seam-blending tool.
        # The stroke layer is composited over the original, then blended back
        # against the original at `gran_blend` opacity so intensity is fully
        # controllable.  gran_blend=0 → original unchanged; 1 → full effect.
        sigma      = max(1, int(extras.get("gran_sigma",   5)))
        angle      = int(extras.get("gran_angle",  45))
        blend      = max(0.0, min(1.0, float(extras.get("gran_blend",  0.3))))
        compose_op = str(extras.get("gran_compose", "Overlay"))
        # compose:args = % of original (source) in the final blend
        orig_pct   = int((1.0 - blend) * 100)
        # +clone at the start of () clones the parent-stack image (in.png),
        # applies the sketch stroke to produce the granulation layer, then
        # composites it over the original.  mpr:orig is then reloaded for the
        # Blend step so the final output is lerped back toward the untouched
        # original.  This matches the pattern used by Mottle, Film Grain, etc.
        return [
            "-write", "mpr:orig",
            "(", "+clone", "-sketch", f"0x{sigma}+{angle}", ")",
            "-compose", compose_op, "-composite",
            "mpr:orig",
            "-compose", "Blend",
            "-define", f"compose:args={orig_pct}",
            "-composite",
        ]

    if effect == EFFECT_TURBULENCE:
        # Fine-grained random displacement (small blur → high-frequency chaos)
        # vs Mottle's coarse blobs.  More distance, less smoothing = jagged
        # ripple rather than gentle swell.
        distance = max(2, int(s * 14))
        gran = max(0.3, s * 1.5)
        return [
            "(", "+clone",
            "+noise", "Random",
            "-blur", f"0x{gran:.1f}",
            "-contrast-stretch", "0%",
            ")",
            "-compose", "Displace",
            "-define", f"compose:args={distance}x{distance}",
            "-composite",
        ]

    if effect == EFFECT_SCREEN:
        return None  # handled in pure Python (see _screen_scanlines)

    return None


def _screen_scanlines(img: "Image.Image", strength: float) -> "Image.Image":
    """CRT scanline effect: darken every Nth row. Pure-numpy replacement for
    the old ImageMagick `-fx` per-pixel expression, which was O(W*H) and
    single-threaded — seconds per large texture and a subprocess per call.
    A vectorised row-slice multiply does the same job orders of magnitude
    faster with no magick.exe dependency.

    Higher strength → denser (smaller spacing) and darker scanlines.
    """
    import numpy as np

    s = max(0.0, min(strength, 2.0))
    spacing = max(2, int(8 - s * 3))             # s=0→8 rows, s=2→2 rows
    light = max(0.4, 1.0 - (0.15 + s * 0.25))    # s=0→0.85, s=1→0.60

    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    arr[::spacing] *= light
    return Image.fromarray(
        np.clip(arr, 0, 255).astype(np.uint8), mode="RGB"
    )


def _apply_then_crosshatch(
    img: "Image.Image",
    magick: str,
    base_args: list[str],
    length: int = 7,
    balance: int = 0,
    sharp: int = 10,
    mix: int = 100,
) -> "Image.Image":
    """Apply base-effect args then overlay a crosshatch diagonal-blur texture.

    Pipeline (Fred Weinhaus crosshatch):
      1. Apply base_args to the image (e.g. unsharp mask or bilateral smooth).
      2. Clone the result and apply a +45° diagonal motion blur.
      3. Clone again and apply a −45° diagonal motion blur.
      4. Blend the two diagonal layers using balance (−100→all+45, 0→equal,
         +100→all−45).
      5. Apply a secondary unsharp pass at strength `sharp`.
      6. Mix the crosshatch result back against the original at `mix`%
         (mix=100→full effect, mix=0→original unchanged).

    base_args: IM argument list for the pre-crosshatch base effect — pass
               the output of _effect_args() for EFFECT_UNSHARP or
               EFFECT_BILATERAL.
    length:    diagonal motion-blur length in pixels (default 7).
    balance:   −100..100 bias toward the +45 or −45 diagonal (default 0).
    sharp:     secondary unsharp strength after blending (default 10, 0=off).
    mix:       0–100 blend of crosshatch result vs original (default 100).
    """
    creationflags = 0x08000000 if sys.platform == "win32" else 0

    # Blend weight for the -45° layer as the Blend *source*.
    # b1=(balance+100)/2 → 0 at balance=-100 (all +45), 100 at balance=+100 (all -45).
    b1 = int((balance + 100) / 2)
    sharp_args: list[str] = ["-unsharp", f"0x1+{sharp}+0"] if sharp > 0 else []
    blend_mix = max(0, min(100, mix))

    with tempfile.TemporaryDirectory(prefix="crosshatch_") as td:
        td_path  = Path(td)
        in_path  = td_path / "in.png"
        out_path = td_path / "out.png"
        img.save(in_path, format="PNG")

        cmd = [
            magick,
            str(in_path),
            "-write", "mpr:orig",      # save original for final mix
            *base_args,                # apply base effect (unsharp / bilateral)
            "-write", "mpr:base",      # save base result
            "-delete", "0",            # clear stack
            # +45° diagonal blur of base
            "(", "mpr:base", "-motion-blur", f"0x{length}+45",  ")",
            # −45° diagonal blur of base
            "(", "mpr:base", "-motion-blur", f"0x{length}+135", ")",
            # Blend diagonals: dst=plus45, src=minus45, args=b1%
            # → minus45*(b1/100) + plus45*(1−b1/100)
            "-compose", "Blend",
            "-define", f"compose:args={b1}",
            "-composite",
            # Secondary sharpen on the blended crosshatch
            *sharp_args,
            # Mix crosshatch result (src) with original (dst) at mix%
            "(", "mpr:orig", ")",
            "+swap",
            "-compose", "Blend",
            "-define", f"compose:args={blend_mix}",
            "-composite",
            "-alpha", "off",
            str(out_path),
        ]

        r = subprocess.run(cmd, capture_output=True, text=True,
                           creationflags=creationflags)
        if r.returncode != 0:
            raise RuntimeError(
                f"ImageMagick crosshatch failed (exit {r.returncode}): "
                f"{r.stderr.strip() or r.stdout.strip()}"
            )
        with Image.open(out_path) as opened:
            return opened.copy()


def _davehill_fx(
    img: "Image.Image",
    magick: str,
    brightness: float = 1.0,
    contrast: float = 0.0,
    gain: int = 40,
) -> "Image.Image":
    """Dave Hill photographic effect (Fred Weinhaus davehilleffect).

    Produces the hyper-real, over-sharpened, high-contrast look associated
    with photographer Dave Hill:

      1. Brightness multiplier (-evaluate multiply) and optional sigmoidal
         contrast adjustment on the base image.
      2. Difference of Gaussians (DoG:0,0,4) high-pass filter cloned from
         the enhanced base; composited over it with VividLight.
      3. Second DoG (DoG:0,0,6.9) high-pass on the VividLight result;
         composited with HardLight.
      4. Blend the fully-processed result back against the original at the
         gain percentage (gain=0 → original unchanged, gain=100 → full effect).

    brightness: multiply factor, 1.0 = no change, >1 brightens.
    contrast:   sigmoidal strength, 0 = off; positive = more contrast,
                negative = less contrast (range -10 to 10).
    gain:       0-100 — how much of the processed result replaces the original.
    """
    creationflags = 0x08000000 if sys.platform == "win32" else 0

    with tempfile.TemporaryDirectory(prefix="davehill_") as td:
        td_path  = Path(td)
        in_path  = td_path / "in.png"
        out_path = td_path / "out.png"
        img.save(in_path, format="PNG")

        brightness_args: list[str] = []
        if abs(brightness - 1.0) > 0.001:
            brightness_args = ["-evaluate", "multiply", f"{brightness:.3f}"]

        contrast_args: list[str] = []
        if abs(contrast) > 0.01:
            if contrast >= 0:
                contrast_args = ["-sigmoidal-contrast", f"{contrast:.1f}x50%"]
            else:
                contrast_args = ["+sigmoidal-contrast", f"{abs(contrast):.1f}x50%"]

        # compose:args for the final Blend: percentage of the *source* (original).
        # At gain=40: args=60 → result = 60% orig + 40% processed = 40% effect. ✓
        blend_orig_pct = 100 - max(0, min(100, int(gain)))

        cmd = [
            magick, str(in_path),
            "-write", "mpr:orig",
            # Step 1: tonal adjustments
            *brightness_args,
            *contrast_args,
            # Step 2: DoG high-pass → VividLight composite
            "(", "+clone",
                "-morphology", "Convolve", "DoG:0,0,4",
                "-auto-level",
            ")",
            "-compose", "VividLight", "-composite",
            # Step 3: second DoG high-pass → HardLight composite
            "(", "+clone",
                "-morphology", "Convolve", "DoG:0,0,6.9",
                "-auto-level",
            ")",
            "-compose", "HardLight", "-composite",
            # Step 4: blend processed result back against original at gain%
            "mpr:orig",
            "-compose", "Blend",
            "-define", f"compose:args={blend_orig_pct}",
            "-composite",
            "-alpha", "off",
            str(out_path),
        ]

        r = subprocess.run(cmd, capture_output=True, text=True,
                           creationflags=creationflags)
        if r.returncode != 0:
            raise RuntimeError(
                f"ImageMagick davehilleffect failed (exit {r.returncode}): "
                f"{r.stderr.strip() or r.stdout.strip()}"
            )
        with Image.open(out_path) as opened:
            return opened.copy()


def _watercolor_fx(
    img: "Image.Image",
    magick: str,
    smoothing: int = 0,
    edge: float = 5.0,
    mixing: int = 33,
    contrast: float = 0.0,
) -> "Image.Image":
    """Watercolor effect (Fred Weinhaus watercolor), a faithful port of his
    single convert pipeline:

      1. Optional sigmoidal contrast (centred at 50%) then mean-shift
         edge-preserving smoothing on the base image.
      2. Sobel edge map (all directions, Lighten-composed), negated and
         raised to the `edge` power to control edge prominence.
      3. Luminize-composite the edge map onto the smoothed base (dark edge
         lines over flat colour regions).
      4. Blend that edged version back over the smoothed base at `mixing`%.

    smoothing: mean-shift window (sxs+10%); 0 = none. Odd values avoid a
               half-pixel shift.
    edge:      power applied to the negated Sobel edges; >=1, higher = stronger.
    mixing:    0-100 — percentage of edged content in the final blend.
    contrast:  sigmoidal contrast factor; 0 = none.
    """
    creationflags = 0x08000000 if sys.platform == "win32" else 0

    with tempfile.TemporaryDirectory(prefix="watercolor_") as td:
        td_path  = Path(td)
        in_path  = td_path / "in.png"
        out_path = td_path / "out.png"
        img.save(in_path, format="PNG")

        pre: list[str] = []
        if abs(float(contrast)) > 0.001:
            pre += ["-sigmoidal-contrast", f"{float(contrast):.2f}x50%"]
        s = max(0, int(smoothing))
        if s > 0:
            pre += ["-mean-shift", f"{s}x{s}+10%"]

        e = max(1.0, float(edge))
        m = max(0, min(100, int(mixing)))

        cmd = [
            magick, str(in_path),
            *pre,
            # Sobel edge map → negate → pow(edge)
            "(", "-clone", "0",
                "-define", "convolve:scale=!",
                "-define", "morphology:compose=Lighten",
                "-morphology", "Convolve", "Sobel:>",
                "-negate", "-evaluate", "pow", f"{e:.3f}",
            ")",
            # luminize the edges onto the smoothed base
            "(", "-clone", "0", "-clone", "1",
                "-compose", "luminize", "-composite",
            ")",
            # drop the raw edge map, then blend edged-base over base at mixing%
            "-delete", "1",
            "-define", f"compose:args={m}",
            "-compose", "blend", "-composite",
            "-alpha", "off",
            str(out_path),
        ]

        r = subprocess.run(cmd, capture_output=True, text=True,
                           creationflags=creationflags)
        if r.returncode != 0:
            raise RuntimeError(
                f"ImageMagick watercolor failed (exit {r.returncode}): "
                f"{r.stderr.strip() or r.stdout.strip()}"
            )
        with Image.open(out_path) as opened:
            return opened.copy()


# Cached per-IM-build Euclidean distance scale (distance is stored as px*scale).
_BEVEL_DIST_SCALE: "Optional[float]" = None


def _bevel_distance_scale(magick: str, td: Path) -> float:
    """Measure IM's Euclidean-distance scale once: a vertical edge, sample the
    distance 10 px in. Cached for the process."""
    global _BEVEL_DIST_SCALE
    if _BEVEL_DIST_SCALE is not None:
        return _BEVEL_DIST_SCALE
    import numpy as np
    creationflags = 0x08000000 if sys.platform == "win32" else 0
    half = td / "scale_m.png"
    a = np.zeros((40, 160), np.uint8); a[:, :80] = 255
    Image.fromarray(a, "L").save(half)
    d = td / "scale_d.png"
    subprocess.run([magick, str(half), "-morphology", "Distance:-1",
                    "Euclidean", "-depth", "16", str(d)],
                   capture_output=True, creationflags=creationflags)
    v = np.asarray(Image.open(d))
    _BEVEL_DIST_SCALE = max(1e-6, float(v[20, 70]) / 10.0)
    return _BEVEL_DIST_SCALE


def _bevel_dist_px(magick: str, mask_bool, td: Path, name: str,
                   scale: float):
    """Euclidean distance in PIXELS from the False region into the True region
    of mask_bool (16-bit distance transform ÷ measured scale)."""
    import numpy as np
    creationflags = 0x08000000 if sys.platform == "win32" else 0
    m = td / f"{name}_m.png"
    Image.fromarray((mask_bool.astype(np.uint8) * 255), "L").save(m)
    d = td / f"{name}_d.png"
    subprocess.run([magick, str(m), "-morphology", "Distance:-1", "Euclidean",
                    "-depth", "16", str(d)],
                   capture_output=True, creationflags=creationflags)
    return np.asarray(Image.open(d)).astype(np.float32) / scale


def _bevel_fx(
    img: "Image.Image",
    magick: str,
    alpha: "Optional[Image.Image]" = None,
    width: int = 8,
    azimuth: int = 135,
    elevation: int = 30,
    depth: int = 100,
    form: str = "inner",
    innermode: str = "hard",
    outermode: str = "raised",
    soften: int = 0,
    taper: int = 0,
    bcolor: str = "#808080",
    colormode: str = "dark",
    ring_alpha: str = "opaque",
) -> "Image.Image":
    """Shaded edge bevel built from the alpha SHAPE (Fred Weinhaus bevel scope).

    A Euclidean distance transform of the alpha gives the distance from the
    silhouette edge; a per-form height field is built from it, `-shade`d with
    the light azimuth/elevation, and hardlight-composited over the RGB.

      form=inner   bevel inside the edge; shape unchanged.
                   innermode hard = linear ramp, smooth = rounded (smoothstep).
      form=outer   bevel in a `width`-px ring OUTSIDE the edge; the alpha is
                   EXPANDED to include the ring, which is filled with bcolor.
                   outermode raised/lowered flips the relief direction.
      form=emboss  ridge straddling the edge (inner + outer); alpha expanded.

    width      bevel band width in px      depth   relief contrast (≈Fred -d)
    soften     px blur of the height field taper   shapes the outer ramp (0-3)
    Returns RGBA (RGB beveled + output alpha). No-op when there's no usable
    alpha edge (returns the input as RGBA)."""
    import numpy as np
    from PIL import ImageFilter, ImageColor
    creationflags = 0x08000000 if sys.platform == "win32" else 0

    rgb = img.convert("RGB")
    if alpha is None:
        out = rgb.convert("RGBA")
        out.putalpha(Image.new("L", rgb.size, 255))
        return out
    a = np.asarray(alpha.convert("L"))
    # The silhouette is every non-transparent pixel, so semi-transparent
    # CONTENT keeps its alpha (and isn't mistaken for new ring material).
    mask = a > 0
    if not mask.any() or mask.all():       # no edge to bevel → no-op
        out = rgb.convert("RGBA")
        out.putalpha(alpha.convert("L"))
        return out

    def _smoothstep(x):
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    def _bleed_outward(values, filled, iters):
        """Propagate `values` from the `filled` region into surrounding empty
        pixels, one ring per iteration — each empty pixel inherits the value of
        the nearest filled pixel (same flood used for RGB edge-dilation)."""
        h0, w0 = values.shape
        vals = values.astype(np.float32).copy()
        fill = filled.copy()
        for _ in range(int(iters)):
            if fill.all():
                break
            pf = np.pad(fill, 1, mode="constant", constant_values=False)
            pv = np.pad(vals, 1, mode="edge")
            new_fill = fill.copy()
            new_vals = vals.copy()
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nf = pf[1 + dy:1 + dy + h0, 1 + dx:1 + dx + w0]
                nv = pv[1 + dy:1 + dy + h0, 1 + dx:1 + dx + w0]
                take = (~new_fill) & nf
                new_vals[take] = nv[take]
                new_fill |= take
            fill, vals = new_fill, new_vals
        return vals

    with tempfile.TemporaryDirectory(prefix="bevel_") as t:
        td = Path(t)
        scale = _bevel_distance_scale(magick, td)
        w = float(max(1, width))
        din = _bevel_dist_px(magick, mask, td, "in", scale)     # inward
        dout = _bevel_dist_px(magick, ~mask, td, "out", scale)  # outward
        ring = (~mask) & (dout <= w)                             # new transparent ring

        # Alpha for the new outer ring (outer/emboss only):
        #   opaque  — solid 255 border (default).
        #   inherit — each ring pixel takes the nearest silhouette-edge alpha.
        #   fade    — inherited edge alpha tapered to 0 at the outer rim.
        def _ring_alpha_values():
            ra = str(ring_alpha).strip().lower()
            if "inherit" in ra or "fade" in ra:
                bled = _bleed_outward(a, mask, int(round(w)))
                if "fade" in ra:
                    bled = bled * np.clip(1.0 - dout / w, 0.0, 1.0)
                return np.clip(bled, 0, 255).astype(np.uint8)
            return np.full(a.shape, 255, np.uint8)

        # Output alpha preserves the ORIGINAL alpha values (gray / white /
        # mixed) — the binary `mask` is only the shape geometry for the
        # distance transform, never the output alpha.
        out_alpha = a.copy()
        if form == "inner":
            h = np.clip(din / w, 0.0, 1.0)
            if innermode == "smooth":
                h = _smoothstep(h)
            H = np.where(mask, h, 0.0)
        elif form == "outer":
            h = np.clip(1.0 - dout / w, 0.0, 1.0)
            if taper > 0:
                h = h ** (1.0 + 0.5 * taper)
            if outermode == "lowered":
                h = np.where(ring, 1.0 - h, 0.0)
            H = np.where(mask, 1.0, np.where(ring, h, 0.0))
            out_alpha = np.where(ring, _ring_alpha_values(), a).astype(np.uint8)
        else:  # emboss — ridge straddling the edge
            inner_h = np.clip(1.0 - din / w, 0.0, 1.0)
            outer_h = np.clip(1.0 - dout / w, 0.0, 1.0)
            if taper > 0:
                outer_h = outer_h ** (1.0 + 0.5 * taper)
            H = np.where(mask, inner_h, np.where(ring, outer_h, 0.0))
            out_alpha = np.where(ring, _ring_alpha_values(), a).astype(np.uint8)

        if soften > 0:
            H = np.asarray(
                Image.fromarray((np.clip(H, 0, 1) * 255).astype(np.uint8), "L")
                .filter(ImageFilter.GaussianBlur(float(soften))),
                np.float32) / 255.0

        hpath = td / "H.png"
        spath = td / "shaded.png"
        bpath = td / "base.png"
        opath = td / "out.png"
        Image.fromarray((np.clip(H, 0, 1) * 255).astype(np.uint8), "L").save(hpath)
        # depth → extra contrast on the shaded relief (depth 100 ≈ nominal).
        dc = max(-100, min(100, (int(depth) - 100) // 2 + 25))
        subprocess.run([magick, str(hpath), "-shade", f"{azimuth}x{elevation}",
                        "-auto-level", "-brightness-contrast", f"0,{dc}",
                        str(spath)], capture_output=True, creationflags=creationflags)

        # RGB to composite over; outer/emboss ring is new area → fill bcolor.
        base = np.asarray(rgb).copy()
        if form in ("outer", "emboss"):
            try:
                base[ring] = ImageColor.getrgb(bcolor)
            except ValueError:
                base[ring] = (128, 128, 128)
        Image.fromarray(base, "RGB").save(bpath)
        # 'light' colormode lifts the relief toward highlights; 'dark' keeps the
        # neutral hardlight relief (shadows read as recessed).
        compose = "screen" if colormode == "light" else "hardlight"
        r = subprocess.run([magick, str(bpath), str(spath), "-compose", compose,
                            "-composite", "-alpha", "off", str(opath)],
                           capture_output=True, text=True, creationflags=creationflags)
        if r.returncode != 0:
            raise RuntimeError(
                f"ImageMagick bevel failed (exit {r.returncode}): "
                f"{r.stderr.strip() or r.stdout.strip()}"
            )
        out = Image.open(opath).convert("RGBA")
        out.putalpha(Image.fromarray(out_alpha, "L"))
        return out


def effect_needs_magick(effect: str) -> bool:
    """True for IM-implemented effects (not 'None' and not Python-implemented)."""
    if effect in (None, "", EFFECT_NONE):
        return False
    if effect in (EFFECT_NORMAL_MAP, EFFECT_COLOUR_TINT, EFFECT_SCREEN):
        return False  # implemented in Python — no ImageMagick needed
    if effect == EFFECT_BEVEL:
        return True
    if effect == EFFECT_DAVEHILL:
        return True
    return True


def _normal_map(img: "Image.Image", strength: float) -> "Image.Image":
    """Generate a tangent-space normal map from the image's luminance.

    Uses central-difference gradients (fast, no scipy) and packs them as RGB:
        R = 0.5 + 0.5·(-dx / |N|)        (left bright → right dark)
        G = 0.5 + 0.5·(-dy / |N|)        (DirectX convention; flip for OpenGL)
        B = 0.5 + 0.5·( 1   / |N|)       (always points "out" of surface)

    strength scales the in-plane gradient magnitude — 1.0 nominal, 0.5
    subtle, 2.0 exaggerated relief.

    Atlas-safe: the central-difference at the edge uses replicate padding
    via numpy slicing, and when called from _upscale_single_tile the input
    is already mirror-padded, so seams remain clean after the crop.
    """
    import numpy as np

    s = max(0.1, min(strength, 4.0))
    gray = np.array(img.convert("L"), dtype=np.float32) / 255.0

    # Horizontal gradient (central difference, replicate at edges)
    dx = np.zeros_like(gray)
    dx[:, 1:-1] = (gray[:, 2:] - gray[:, :-2]) * 0.5
    dx[:, 0] = dx[:, 1]
    dx[:, -1] = dx[:, -2]

    # Vertical gradient
    dy = np.zeros_like(gray)
    dy[1:-1, :] = (gray[2:, :] - gray[:-2, :]) * 0.5
    dy[0, :] = dy[1, :]
    dy[-1, :] = dy[-2, :]

    dx *= s
    dy *= s

    z = np.ones_like(gray)
    length = np.sqrt(dx * dx + dy * dy + z * z)
    nx = -dx / length
    ny = -dy / length
    nz = z / length

    r = np.clip((nx * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    g = np.clip((ny * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    b = np.clip((nz * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)

    return Image.fromarray(np.stack([r, g, b], axis=2), mode="RGB")


# --- Texturize (Fred Weinhaus texturize port) -----------------------------

def _tx_box_blur_axis(a: "np.ndarray", r: int, axis: int) -> "np.ndarray":
    """Edge-padded 1-D box mean (window 2r+1) along one axis, via cumsum —
    O(1) per pixel regardless of radius."""
    if r <= 0:
        return a
    a = np.moveaxis(a, axis, 0)
    n = a.shape[0]
    padded = np.pad(a, [(r, r)] + [(0, 0)] * (a.ndim - 1), mode="edge")
    cs = np.cumsum(padded, axis=0, dtype=np.float64)
    cs = np.concatenate([np.zeros((1,) + cs.shape[1:], dtype=cs.dtype), cs], axis=0)
    out = (cs[2 * r + 1: 2 * r + 1 + n] - cs[0:n]) / float(2 * r + 1)
    return np.moveaxis(out, 0, axis).astype(np.float32)


def _tx_gaussian1d(a: "np.ndarray", sigma: float, axis: int) -> "np.ndarray":
    """1-D Gaussian blur along one axis, approximated by 3 box passes
    (no scipy dependency). sigma<=0 returns the input unchanged."""
    if sigma <= 0:
        return a.astype(np.float32)
    # Box width for 3 passes that approximates a Gaussian of this sigma.
    r = max(1, int(round(sigma * 1.13)))
    out = a.astype(np.float32)
    for _ in range(3):
        out = _tx_box_blur_axis(out, r, axis)
    return out


def _tx_stretch(a: "np.ndarray") -> "np.ndarray":
    """Contrast-stretch a float array to full 0..1 dynamic range."""
    lo = float(a.min())
    hi = float(a.max())
    if hi - lo < 1e-6:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def _tx_shade(height: "np.ndarray", azimuth: float, elevation: float) -> "np.ndarray":
    """ImageMagick-style `-shade azimuthxelevation`: treat the array as a
    height map, light it from the given direction, return the cosine shading
    (0..1). Gives the 'bump' format its 3-D relief."""
    az = np.deg2rad(azimuth)
    el = np.deg2rad(elevation)
    lx = np.cos(el) * np.cos(az)
    ly = np.cos(el) * np.sin(az)
    lz = np.sin(el)
    gx = np.zeros_like(height)
    gy = np.zeros_like(height)
    gx[:, 1:-1] = (height[:, 2:] - height[:, :-2]) * 0.5
    gy[1:-1, :] = (height[2:, :] - height[:-2, :]) * 0.5
    # Surface normal (-gx, -gy, 1) normalised, dotted with the light vector.
    nlen = np.sqrt(gx * gx + gy * gy + 1.0)
    shade = (-gx * lx - gy * ly + lz) / nlen
    return np.clip(shade, 0.0, 1.0)


def _texturize_fx(
    img: "Image.Image",
    threshold: float = 2.0,
    blur: float = 9.0,
    widen: float = 0.0,
    spread: float = 0.0,
    gnoise: float = 5.0,
    fmt: str = "bump",
    mix: float = 25.0,
    contrast: float = 0.0,
    seed: int = 1,
    azimuth: int = 135,
    elevation: int = 45,
) -> "Image.Image":
    """Port of Fred Weinhaus's `texturize` script: synthesise a woven /
    canvas texture from seeded noise and blend it over the image.

    Pipeline: seeded random noise → keep the brightest `threshold`% as dots →
    optional 2-D widen blur → two orthogonal 1-D Gaussian blurs multiplied
    (the cross-hatch weave) → contrast-stretch → additive grain → optional
    spread jitter → optional `bump` shading for relief → contrast → blend
    over the source at `mix`%.

    Operates on RGB; the chain reattaches the source alpha afterwards.
    """
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w = rgb.shape[:2]
    rng = np.random.default_rng(int(seed))

    # 1. Seeded noise → keep the brightest `threshold`% as white dots.
    noise = rng.random((h, w), dtype=np.float32)
    t = max(0.0, min(100.0, float(threshold)))
    if t <= 0.0:
        dots = np.zeros((h, w), dtype=np.float32)
    else:
        cutoff = float(np.percentile(noise, 100.0 - t))
        dots = (noise >= cutoff).astype(np.float32)

    # 2. Optional 2-D widen of the dots before streaking.
    if widen > 0:
        dots = _tx_gaussian1d(dots, float(widen), axis=0)
        dots = _tx_gaussian1d(dots, float(widen), axis=1)

    # 3. Two orthogonal 1-D blurs multiplied → the woven cross-hatch.
    horiz = _tx_gaussian1d(dots, float(blur), axis=1)
    vert = _tx_gaussian1d(dots, float(blur), axis=0)
    tex = _tx_stretch(horiz * vert)

    # 4. Additive Gaussian grain.
    if gnoise > 0:
        tex = tex + rng.normal(0.0, float(gnoise) / 100.0, size=tex.shape).astype(np.float32)
        tex = np.clip(tex, 0.0, 1.0)

    # 5. Optional spread (random per-pixel jitter).
    s = int(round(spread))
    if s > 0:
        dy = rng.integers(-s, s + 1, size=tex.shape)
        dx = rng.integers(-s, s + 1, size=tex.shape)
        yy = np.clip(np.arange(h)[:, None] + dy, 0, h - 1)
        xx = np.clip(np.arange(w)[None, :] + dx, 0, w - 1)
        tex = tex[yy, xx]

    # 6. 'bump' relief shading (plain skips this).
    if str(fmt).lower() == "bump":
        tex = _tx_shade(tex, float(azimuth), float(elevation))

    # 7. Contrast around mid-grey before blending.
    c = max(0.0, float(contrast))
    if c > 0:
        tex = np.clip((tex - 0.5) * (1.0 + c * 0.15) + 0.5, 0.0, 1.0)

    # 8. Linear blend of the (grey) texture over the source at mix%.
    m = max(0.0, min(100.0, float(mix))) / 100.0
    tex_rgb = (tex[:, :, None] * 255.0).astype(np.float32)
    out = rgb * (1.0 - m) + tex_rgb * m
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


# --- Screen Effects (Fred Weinhaus screeneffects port) ---------------------

def _screen_maps(w: int, h: int, spacing: float, rotate: bool):
    """Two sinusoidal screen maps (mx varies along x, my along y) plus their
    averaged+stretched merge, each HxW in 0..1. Mirrors Fred's screeneffects:
    frequency = image_dim/spacing, averaged across the two axes, evaluated as
    0.5 + 0.5*sin(2*pi*freq*u). `rotate` turns the screen 45 degrees by
    rotating the evaluation coordinates about the centre."""
    spacing = max(0.5, float(spacing))
    avef = 0.5 * (w / spacing + h / spacing)        # Fred's averaged frequency
    X, Y = np.meshgrid(np.arange(w, dtype=np.float32),
                       np.arange(h, dtype=np.float32))
    ux = X / max(1, w - 1)
    uy = Y / max(1, h - 1)
    if rotate:
        ca = sa = np.float32(np.sqrt(0.5))          # cos/sin 45 degrees
        rx = (ux - 0.5) * ca - (uy - 0.5) * sa + 0.5
        ry = (ux - 0.5) * sa + (uy - 0.5) * ca + 0.5
        ux, uy = rx, ry
    tp = 2.0 * np.pi
    mx = 0.5 + 0.5 * np.sin(tp * avef * ux)
    my = 0.5 + 0.5 * np.sin(tp * avef * uy)
    avg = (mx + my) * 0.5
    lo, hi = float(avg.min()), float(avg.max())     # -contrast-stretch 0
    if hi - lo > 1e-6:
        avg = (avg - lo) / (hi - lo)
    return mx, my, avg, X, Y


def _screen_effects_fx(
    img: "Image.Image",
    type_: str = "simple",
    spacing: float = 6.0,
    rotate: bool = False,
    color_hex: str = "#000000",
    amount: float = 6.0,
    method: int = 1,
) -> "Image.Image":
    """Screen effects (Fred Weinhaus screeneffects).

    Builds a sinusoidal screen pattern at `spacing`-pixel pitch (optionally
    rotated 45 degrees) and either:
      simple   — colourises the pattern (black point -> `color_hex`, white ->
                 white) and multiply-composites it over the image: a tinted
                 screen-door / mesh overlay.
      displace — uses the pattern as a displacement map to warp the image,
                 like viewing it through textured glass. `amount` = max pixel
                 shift; `method` chooses which maps drive x/y:
                   1 = merged map for both, 2 = x-map->x / y-map->y,
                   3 = swapped (y-map->x / x-map->y).
    Operates on RGB; the chain reattaches the source alpha.
    """
    from PIL import ImageColor
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w = rgb.shape[:2]
    mx, my, avg, X, Y = _screen_maps(w, h, spacing, rotate)

    if str(type_).lower().startswith("d"):          # displace
        m = int(method)
        if m == 2:
            dmx, dmy = mx, my
        elif m == 3:
            dmx, dmy = my, mx
        else:
            dmx = dmy = avg
        amt = max(0.0, float(amount))
        sx = np.clip(X + (dmx - 0.5) * 2.0 * amt, 0, w - 1)
        sy = np.clip(Y + (dmy - 0.5) * 2.0 * amt, 0, h - 1)
        x0 = np.floor(sx).astype(np.int32); x1 = np.clip(x0 + 1, 0, w - 1)
        y0 = np.floor(sy).astype(np.int32); y1 = np.clip(y0 + 1, 0, h - 1)
        wx = (sx - x0)[..., None]; wy = (sy - y0)[..., None]
        out = (rgb[y0, x0] * (1 - wx) * (1 - wy) +
               rgb[y0, x1] * wx * (1 - wy) +
               rgb[y1, x0] * (1 - wx) * wy +
               rgb[y1, x1] * wx * wy)
    else:                                           # simple
        bc = np.array(ImageColor.getrgb(color_hex)[:3], dtype=np.float32)
        patt = bc[None, None, :] + (255.0 - bc[None, None, :]) * avg[..., None]
        out = rgb * patt / 255.0

    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def _colour_tint(
    img: "Image.Image",
    opacity: float,
    color_hex: str,
) -> "Image.Image":
    """Multiply-blend a solid colour fill over the RGB image.

    opacity  — 0.0 = original unchanged, 1.0 = full multiply blend.
    color_hex — CSS hex string, e.g. '#0080FF'.

    Multiply formula: result = img × color, then lerp with original
    by opacity so the user can dial in a subtle tint or a heavy cast.
    Alpha is not touched — callers strip and restore it.
    """
    import numpy as np

    opacity = max(0.0, min(1.0, opacity))
    h = color_hex.lstrip("#")
    try:
        cr = int(h[0:2], 16) / 255.0
        cg = int(h[2:4], 16) / 255.0
        cb = int(h[4:6], 16) / 255.0
    except (ValueError, IndexError):
        cr, cg, cb = 0.0, 0.502, 1.0   # fallback: medium blue

    arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0

    # Multiply each channel by the corresponding colour component
    multiplied = np.stack([
        arr[:, :, 0] * cr,
        arr[:, :, 1] * cg,
        arr[:, :, 2] * cb,
    ], axis=2)

    # Lerp: original × (1 − opacity) + multiplied × opacity
    result = arr * (1.0 - opacity) + multiplied * opacity
    return Image.fromarray(
        np.clip(result * 255.0, 0, 255).astype(np.uint8), mode="RGB"
    )


# ── DDS export ────────────────────────────────────────────────────────────────

# Maps the UI format label to the ImageMagick dds:compression value.
DDS_FORMATS: "dict[str, str]" = {
    "DDS DXT1": "dxt1",   # BC1 — RGB, no/1-bit alpha; smallest file size
    "DDS DXT5": "dxt5",   # BC3 — RGBA, full 8-bit alpha
}


def save_dds(
    img: "Image.Image",
    path: "Path",
    compression: str = "dxt5",
) -> None:
    """Save *img* to *path* as a DDS file using ImageMagick.

    compression: 'dxt1' (BC1 — RGB, no / 1-bit alpha)  or
                 'dxt5' (BC3 — RGBA, full 8-bit alpha).

    A single mip level (level 0) is written — use a dedicated DDS tool
    (e.g. texconv) if you need a full mip chain.

    Raises RuntimeError if magick.exe is not found or the IM call fails."""
    magick = find_magick()
    if magick is None:
        raise RuntimeError(
            "ImageMagick (magick.exe) not found. Install ImageMagick and add "
            "it to PATH, or place magick.exe alongside the app."
        )

    creationflags = 0x08000000 if sys.platform == "win32" else 0

    with tempfile.TemporaryDirectory(prefix="dds_save_") as td:
        in_path = Path(td) / "in.png"
        # compress_level=1: this PNG is a throwaway handed straight to an IM
        # subprocess, so minimise zlib effort rather than file size.
        img.save(in_path, format="PNG", compress_level=1)
        cmd = [magick, str(in_path),
               "-define", f"dds:compression={compression}",
               str(path)]
        res = subprocess.run(cmd, capture_output=True, text=True,
                             creationflags=creationflags)
        if res.returncode != 0:
            err = res.stderr.strip() or res.stdout.strip()
            raise RuntimeError(
                f"ImageMagick DDS save failed (exit {res.returncode}): {err}"
            )


def apply_effect_chain(
    img: Image.Image,
    chain: "list[tuple[str, float]]",
    tile_size: int = 0,
    alpha: "Optional[Image.Image]" = None,
) -> Image.Image:
    """Apply an ordered sequence of (effect_name, strength) pairs to a PIL
    image. 'None' entries are skipped. Useful for piping e.g.
        Bilateral Smooth → Unsharp Mask → Saturation Boost
    as a single composite step.

    If `tile_size > 0`, the image is sliced into tile_size×tile_size blocks
    and the chain is applied to each tile independently before re-stitching.
    Use this for tile-based assets (sprite sheets, atlases) where effects
    should respect tile boundaries instead of bleeding across them. Each
    tile is processed in isolation — a convolution kernel near a tile edge
    sees only that tile's pixels, not the neighbours.

    If a normal-map (or any other alpha-stripping) effect appears in the
    chain, the result is RGB from that point on — callers should check
    effect_strips_alpha on each entry if they need to suppress alpha
    reattachment downstream."""
    if not chain:
        return img
    if tile_size <= 0:
        return _apply_chain_inplace(img, chain, alpha)
    # Tiled mode can't sensibly bevel per-tile (it would bevel each tile's
    # edge), so the alpha-aware path is skipped there — the bevel no-ops.
    return _apply_chain_tiled(img, chain, tile_size)


def _apply_chain_inplace(
    img: Image.Image,
    chain: "list",
    alpha: "Optional[Image.Image]" = None,
) -> Image.Image:
    """Walk the chain and accumulate effects on one image. `alpha` (the shape)
    is threaded to alpha-aware effects (Bevel); an effect that returns RGBA
    updates the carried alpha for later steps. Returns RGBA carrying the final
    alpha when one is in play (the bevel may have expanded it), else RGB.
    Chain items may be (effect, strength) or (effect, strength, extras_dict)."""
    result = img.convert("RGB") if img.mode == "RGBA" else img
    cur_alpha = alpha
    for item in chain:
        effect, strength = item[0], item[1]
        extras = item[2] if len(item) > 2 else {}
        if not effect or effect == EFFECT_NONE:
            continue
        out = apply_effect(result, effect, strength, alpha=cur_alpha, **extras)
        if effect_strips_alpha(effect):
            cur_alpha = None
            result = out.convert("RGB") if out.mode != "RGB" else out
        elif out.mode == "RGBA":
            cur_alpha = out.getchannel("A")
            result = out.convert("RGB")
        else:
            result = out
    if cur_alpha is not None:
        result = result.convert("RGBA")
        result.putalpha(cur_alpha)
    return result


def _apply_chain_tiled(
    img: Image.Image,
    chain: "list[tuple[str, float]]",
    tile_size: int,
) -> Image.Image:
    """Slice into tile_size×tile_size blocks, run the chain on each tile in
    isolation, stitch results back together. Tile origin is (0, 0); edge
    tiles are clipped if the image doesn't divide evenly."""
    w, h = img.size
    # Run the first tile up front to discover the output mode (e.g. the
    # chain might strip alpha via a Normal Map step — the canvas needs
    # to be RGB rather than RGBA).
    first_w = min(tile_size, w)
    first_h = min(tile_size, h)
    first_out = _apply_chain_inplace(img.crop((0, 0, first_w, first_h)), chain)
    canvas = Image.new(first_out.mode, (w, h))
    canvas.paste(first_out, (0, 0))

    for y0 in range(0, h, tile_size):
        for x0 in range(0, w, tile_size):
            if y0 == 0 and x0 == 0:
                continue
            x1 = min(x0 + tile_size, w)
            y1 = min(y0 + tile_size, h)
            tile = img.crop((x0, y0, x1, y1))
            canvas.paste(_apply_chain_inplace(tile, chain), (x0, y0))
    return canvas


def chain_strips_alpha(chain: "list") -> bool:
    """True if any effect in the chain strips alpha (e.g. Normal Map)."""
    if not chain:
        return False
    return any(effect_strips_alpha(item[0]) for item in chain)


def _chrome_fx(
    img: "Image.Image",
    magick: str,
    alpha: "Optional[Image.Image]" = None,
    smooth: float = 3.0,
    cycles: int = 2,
    intensity: int = 100,
    azimuth: int = 135,
    elevation: int = 45,
    blend: float = 0.6,
    compose: str = "Screen",
    source: str = "image",
) -> "Image.Image":
    """Fred Weinhaus chrome effect — sinusoidal CLUT + shade, with colour retention.

    Pass 1: Build a 1×100 sinusoidal colour lookup table (CLUT).
    Pass 2: Grayscale → blur → shade → auto-level → CLUT → contrast adjust →
            composite chrome map over original colour via compose operator →
            blend result back against original at `blend` opacity.

    source='image'  — chrome is derived from the image luminance (default).
    source='alpha'  — Fred's -A extract: chrome is derived from the alpha
                      SHAPE instead, so the silhouette itself looks chromed.
                      Falls back to 'image' when no alpha is supplied.
    compose='None' outputs the raw greyscale chrome map without colour blending.
    """
    creationflags = 0x08000000 if sys.platform == "win32" else 0
    use_extract = (source == "alpha" and alpha is not None)

    with tempfile.TemporaryDirectory(prefix="chrome_") as td:
        td_path  = Path(td)
        in_path  = td_path / "in.png"
        alpha_path = td_path / "alpha.png"
        lut_path = td_path / "lut.png"
        out_path = td_path / "out.png"

        # compress_level=1: this PNG is a throwaway handed straight to an IM
        # subprocess, so minimise zlib effort rather than file size.
        img.save(in_path, format="PNG", compress_level=1)
        if use_extract:
            alpha.convert("L").save(alpha_path, format="PNG", compress_level=1)
        # Grayscale source for the chrome chain: the alpha shape (extract) or
        # the image itself.
        gray_src = str(alpha_path) if use_extract else str(in_path)

        # Pass 1: build sinusoidal CLUT
        # -function sinusoid frequency,phase,amplitude,bias
        # → bias + amplitude * sin(2π * frequency * u + phase_degrees)
        # phase -90° shifts the wave so u=0 maps to the trough (dark start).
        r = subprocess.run(
            [magick, "-size", "1x100", "gradient:", "-rotate", "90",
             "-function", "sinusoid", f"{cycles},-90,0.5,0.5",
             str(lut_path)],
            capture_output=True, text=True, creationflags=creationflags,
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"ImageMagick chrome CLUT failed (exit {r.returncode}): "
                f"{r.stderr.strip() or r.stdout.strip()}"
            )

        # intensity=100 (default) → no extra contrast; scales ±50 around that
        contrast_delta = max(-100, min(100, (intensity - 100) // 2))
        do_compose = compose and compose.lower() != "none"

        if do_compose:
            # Pass 2a: shade + CLUT + composite over colour original + blend back
            # Stack trace (key steps):
            #   [in] → write mpr:orig → shade/CLUT/contrast/sRGB → [chrome]
            #   → write mpr:chrome → delete 0 → []
            #   → mpr:orig, mpr:chrome → [orig, chrome]
            #   → compose(dst=orig, src=chrome) → [composited]
            #   → mpr:orig → [composited, orig]
            #   → Blend at orig_pct% orig → [blended]
            orig_pct = int((1.0 - blend) * 100)
            cmd2 = [
                magick, str(in_path),
                "-write", "mpr:orig", "-delete", "0",
                gray_src,                         # image gray, or the alpha shape
                "-colorspace", "gray",
                "-blur", f"0x{smooth:.1f}",
                "-shade", f"{azimuth}x{elevation}",
                "-auto-level",
                str(lut_path), "-clut",
                "-brightness-contrast", f"0,{contrast_delta}",
                "-colorspace", "sRGB",
                "-write", "mpr:chrome",
                "-delete", "0",
                "mpr:orig",
                "mpr:chrome",
                "-compose", compose, "-composite",
                "mpr:orig",
                "-compose", "Blend",
                "-define", f"compose:args={orig_pct}",
                "-composite",
                "-alpha", "off",
                str(out_path),
            ]
        else:
            # Pass 2b: raw chrome map only, no colour composite
            cmd2 = [
                magick, gray_src,
                "-colorspace", "gray",
                "-blur", f"0x{smooth:.1f}",
                "-shade", f"{azimuth}x{elevation}",
                "-auto-level",
                str(lut_path), "-clut",
                "-brightness-contrast", f"0,{contrast_delta}",
                "-alpha", "off",
                str(out_path),
            ]

        r = subprocess.run(
            cmd2, capture_output=True, text=True, creationflags=creationflags,
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"ImageMagick chrome processing failed (exit {r.returncode}): "
                f"{r.stderr.strip() or r.stdout.strip()}"
            )

        with Image.open(out_path) as opened:
            return opened.copy()


def _emboss_m1(
    img: "Image.Image",
    magick: str,
    azimuth: int = 135,
    elevation: int = 45,
    depth: int = 1,
    compose: str = "Overlay",
) -> "Image.Image":
    """Fred Weinhaus emboss Method 1 — Variation In Compose.

    Pass 1: convert to grayscale, apply -shade {azimuth}x{elevation}.
    Pass 2: polynomial stretch driven by the shaded image's mean luminance
            (aa=depth, bb=mean*(1-depth)), then composite with the original
            via the chosen compose operator.

    compose='None' outputs the stretched grayscale emboss layer only.
    """
    creationflags = 0x08000000 if sys.platform == "win32" else 0

    with tempfile.TemporaryDirectory(prefix="emboss_") as td:
        td_path     = Path(td)
        in_path     = td_path / "in.png"
        shaded_path = td_path / "shaded.png"
        out_path    = td_path / "out.png"

        # compress_level=1: this PNG is a throwaway handed straight to an IM
        # subprocess, so minimise zlib effort rather than file size.
        img.save(in_path, format="PNG", compress_level=1)

        # Pass 1: shade the grayscale version
        r = subprocess.run(
            [magick, str(in_path),
             "-colorspace", "gray",
             "-shade", f"{azimuth}x{elevation}",
             str(shaded_path)],
            capture_output=True, text=True, creationflags=creationflags,
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"ImageMagick emboss shade failed (exit {r.returncode}): "
                f"{r.stderr.strip() or r.stdout.strip()}"
            )

        # Extract mean luminance of the shaded result to parameterise the stretch
        r = subprocess.run(
            [magick, str(shaded_path), "-format", "%[fx:mean]", "info:"],
            capture_output=True, text=True, creationflags=creationflags,
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"ImageMagick emboss mean extraction failed: "
                f"{r.stderr.strip() or r.stdout.strip()}"
            )
        try:
            mean = float(r.stdout.strip())
        except ValueError:
            mean = 0.5

        aa = float(depth)
        bb = mean * (1.0 - depth)

        # Pass 2: polynomial stretch + composite over the colour original.
        # IMPORTANT: in_path (colour) must be the destination so IM composites
        # the grayscale emboss layer as the source onto the colour canvas —
        # reversing this makes IM use a grayscale canvas and strips all colour.
        if compose and compose.lower() != "none":
            cmd2 = [
                magick, str(in_path),
                "(",
                str(shaded_path),
                "-function", "polynomial", f"{aa},{bb:.6f}",
                "-colorspace", "sRGB",
                ")",
                "-compose", compose, "-composite",
                "-alpha", "off",
                str(out_path),
            ]
        else:
            # No composite — just output the stretched grayscale emboss layer
            cmd2 = [
                magick, str(shaded_path),
                "-function", "polynomial", f"{aa},{bb:.6f}",
                "-alpha", "off",
                str(out_path),
            ]

        r = subprocess.run(
            cmd2, capture_output=True, text=True, creationflags=creationflags,
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"ImageMagick emboss composite failed (exit {r.returncode}): "
                f"{r.stderr.strip() or r.stdout.strip()}"
            )

        with Image.open(out_path) as opened:
            return opened.copy()


def _emboss_m2(
    img: "Image.Image",
    magick: str,
    azimuth: int = 135,
    depth: int = 1,
    intensity: int = 0,
    compose: str = "None",
) -> "Image.Image":
    """Fred Weinhaus emboss Method 2 — Photoshop approach.

    Offsets a grayscale copy by `depth` px in the light azimuth direction and
    takes the signed difference centred on mid-gray (orig − shifted + 0.5),
    giving a directional edge relief. `intensity` (−50..50) adjusts contrast.
    Uses azimuth + depth + intensity (no elevation), per Fred.

    compose='None' outputs the raw grayscale emboss layer; otherwise the layer
    is blended over the colour original with the chosen compose operator.
    """
    import math
    creationflags = 0x08000000 if sys.platform == "win32" else 0
    d = max(1, int(depth))
    dx = int(round(d * math.cos(math.radians(azimuth))))
    dy = int(round(-d * math.sin(math.radians(azimuth))))   # image y axis points down
    contrast = max(-100, min(100, int(intensity) * 2))
    roll = f"{dx:+d}{dy:+d}"

    with tempfile.TemporaryDirectory(prefix="emboss2_") as td:
        in_path = Path(td) / "in.png"
        out_path = Path(td) / "out.png"
        img.save(in_path, format="PNG", compress_level=1)

        if compose and compose.lower() != "none":
            cmd = [
                magick, str(in_path),
                "(", "+clone", "-colorspace", "gray",
                    "(", "+clone", "-roll", roll, ")",
                    "-compose", "Mathematics",
                    "-define", "compose:args=0,-1,1,0.5", "-composite",
                    "-brightness-contrast", f"0,{contrast}",
                    "-colorspace", "sRGB",
                ")",
                "-compose", compose, "-composite",
                "-alpha", "off", str(out_path),
            ]
        else:
            cmd = [
                magick, str(in_path), "-colorspace", "gray",
                "(", "+clone", "-roll", roll, ")",
                "-compose", "Mathematics",
                "-define", "compose:args=0,-1,1,0.5", "-composite",
                "-brightness-contrast", f"0,{contrast}",
                "-alpha", "off", str(out_path),
            ]

        r = subprocess.run(cmd, capture_output=True, text=True,
                           creationflags=creationflags)
        if r.returncode != 0:
            raise RuntimeError(
                f"ImageMagick emboss (method 2) failed (exit {r.returncode}): "
                f"{r.stderr.strip() or r.stdout.strip()}"
            )
        with Image.open(out_path) as opened:
            return opened.copy()


def apply_effect(
    img: Image.Image,
    effect: str,
    strength: float = 1.0,
    alpha: "Optional[Image.Image]" = None,
    **extras,
) -> Image.Image:
    """Apply a single named effect to a PIL image. Most effects operate on RGB
    only; the Bevel effect uses `alpha` (the shape) and returns RGBA (with a
    possibly-expanded alpha). Returns a
    new PIL Image. Raises RuntimeError if magick.exe is needed but missing.

    ``extras`` carries optional per-effect parameters such as ``color``
    (hex string) and ``opacity`` (0-1 float) for EFFECT_COLOUR_TINT."""
    if effect in (None, "", EFFECT_NONE):
        return img

    # Python-implemented effects (no magick.exe needed)
    if effect == EFFECT_NORMAL_MAP:
        return _normal_map(img, strength)
    if effect == EFFECT_COLOUR_TINT:
        return _colour_tint(
            img,
            opacity=float(extras.get("opacity", 1.0)),
            color_hex=str(extras.get("color", "#0080FF")),
        )
    if effect == EFFECT_SCREEN:
        return _screen_scanlines(img, strength)
    if effect == EFFECT_TEXTURIZE:
        return _texturize_fx(
            img,
            threshold = float(extras.get("texturize_threshold",  2.0)),
            blur      = float(extras.get("texturize_blur",        9.0)),
            widen     = float(extras.get("texturize_widen",       0.0)),
            spread    = float(extras.get("texturize_spread",      0.0)),
            gnoise    = float(extras.get("texturize_gnoise",      5.0)),
            fmt       = str(extras.get("texturize_format",     "bump")),
            mix       = float(extras.get("texturize_mix",        25.0)),
            contrast  = float(extras.get("texturize_contrast",    0.0)),
            seed      = int(extras.get("texturize_seed",            1)),
            azimuth   = int(extras.get("texturize_azimuth",       135)),
            elevation = int(extras.get("texturize_elevation",      45)),
        )
    if effect == EFFECT_SCREENFX:
        return _screen_effects_fx(
            img,
            type_     = str(extras.get("screenfx_type",     "simple")),
            spacing   = float(extras.get("screenfx_spacing",     6.0)),
            rotate    = bool(extras.get("screenfx_rotate",     False)),
            color_hex = str(extras.get("screenfx_color",   "#000000")),
            amount    = float(extras.get("screenfx_amount",      6.0)),
            method    = int(extras.get("screenfx_method",          1)),
        )
    # Multi-pass IM effects (need magick but run their own subprocess calls)
    if effect == EFFECT_CHROME:
        magick = find_magick()
        if magick is None:
            raise RuntimeError(
                "ImageMagick (magick.exe) not found. Install ImageMagick and add "
                "it to PATH, or place magick.exe alongside the app."
            )
        return _chrome_fx(
            img, magick, alpha,
            smooth    = float(extras.get("chrome_smooth",     3.0)),
            cycles    = int(extras.get("chrome_cycles",         2)),
            intensity = int(extras.get("chrome_intensity",    100)),
            azimuth   = int(extras.get("chrome_azimuth",      135)),
            elevation = int(extras.get("chrome_elevation",     45)),
            blend     = float(extras.get("chrome_blend",       0.6)),
            compose   = str(extras.get("chrome_compose",  "Screen")),
            source    = str(extras.get("chrome_source",   "image")),
        )
    if effect == EFFECT_EMBOSS:
        magick = find_magick()
        if magick is None:
            raise RuntimeError(
                "ImageMagick (magick.exe) not found. Install ImageMagick and add "
                "it to PATH, or place magick.exe alongside the app."
            )
        if int(extras.get("emboss_method", 1)) == 2:
            return _emboss_m2(
                img, magick,
                azimuth   = int(extras.get("emboss_azimuth",   135)),
                depth     = int(extras.get("emboss_depth",       1)),
                intensity = int(extras.get("emboss_intensity",   0)),
                compose   = str(extras.get("emboss_compose", "Overlay")),
            )
        return _emboss_m1(
            img, magick,
            azimuth   = int(extras.get("emboss_azimuth",   135)),
            elevation = int(extras.get("emboss_elevation",  45)),
            depth     = int(extras.get("emboss_depth",       1)),
            compose   = str(extras.get("emboss_compose", "Overlay")),
        )
    if effect == EFFECT_DAVEHILL:
        magick = find_magick()
        if magick is None:
            raise RuntimeError(
                "ImageMagick (magick.exe) not found. Install ImageMagick and add "
                "it to PATH, or place magick.exe alongside the app."
            )
        return _davehill_fx(
            img, magick,
            brightness = float(extras.get("davehill_brightness", 1.0)),
            contrast   = float(extras.get("davehill_contrast",   0.0)),
            gain       = int(extras.get("davehill_gain",          40)),
        )
    if effect == EFFECT_WATERCOLOR:
        magick = find_magick()
        if magick is None:
            raise RuntimeError(
                "ImageMagick (magick.exe) not found. Install ImageMagick and add "
                "it to PATH, or place magick.exe alongside the app."
            )
        return _watercolor_fx(
            img, magick,
            smoothing = int(extras.get("watercolor_smoothing",   0)),
            edge      = float(extras.get("watercolor_edge",      5.0)),
            mixing    = int(extras.get("watercolor_mixing",      33)),
            contrast  = float(extras.get("watercolor_contrast",  0.0)),
        )
    if effect == EFFECT_BEVEL:
        magick = find_magick()
        if magick is None:
            raise RuntimeError(
                "ImageMagick (magick.exe) not found. Install ImageMagick and add "
                "it to PATH, or place magick.exe alongside the app."
            )
        return _bevel_fx(
            img, magick, alpha,
            width     = int(extras.get("bevel_width",        8)),
            azimuth   = int(extras.get("bevel_azimuth",    135)),
            elevation = int(extras.get("bevel_elevation",   30)),
            depth     = int(extras.get("bevel_depth",      100)),
            form      = str(extras.get("bevel_form",   "inner")),
            innermode = str(extras.get("bevel_innermode", "hard")),
            outermode = str(extras.get("bevel_outermode", "raised")),
            soften    = int(extras.get("bevel_soften",       0)),
            taper     = int(extras.get("bevel_taper",        0)),
            bcolor    = str(extras.get("bevel_bcolor",  "#808080")),
            colormode = str(extras.get("bevel_colormode", "dark")),
            ring_alpha= str(extras.get("bevel_ring_alpha", "opaque")),
        )
    # Crosshatch overlay on unsharp / bilateral — intercept before the
    # simple single-pass path so we can wrap base_args in the multi-pass
    # crosshatch pipeline.
    if effect in (EFFECT_UNSHARP, EFFECT_BILATERAL) and extras.get("crosshatch"):
        magick = find_magick()
        if magick is None:
            raise RuntimeError(
                "ImageMagick (magick.exe) not found. Install ImageMagick and add "
                "it to PATH, or place magick.exe alongside the app."
            )
        base_args = _effect_args(effect, strength, **extras) or []
        return _apply_then_crosshatch(
            img, magick, base_args,
            length  = int(extras.get("crosshatch_length",   7)),
            balance = int(extras.get("crosshatch_balance",  0)),
            sharp   = int(extras.get("crosshatch_sharp",   10)),
            mix     = int(extras.get("crosshatch_mix",    100)),
        )
    # IM-implemented effects
    args = _effect_args(effect, strength, **extras)
    if args is None:
        return img

    magick = find_magick()
    if magick is None:
        raise RuntimeError(
            "ImageMagick (magick.exe) not found. Install ImageMagick and add "
            "it to PATH, or place magick.exe alongside the app."
        )

    creationflags = 0x08000000 if sys.platform == "win32" else 0  # no console window

    with tempfile.TemporaryDirectory(prefix="esrgan_fx_") as td:
        in_path = Path(td) / "in.png"
        out_path = Path(td) / "out.png"
        # compress_level=1: this PNG is a throwaway handed straight to an IM
        # subprocess, so minimise zlib effort rather than file size.
        img.save(in_path, format="PNG", compress_level=1)
        # -alpha off before the output path guarantees IM never writes an alpha
        # channel into the temp PNG, preventing unexpected RGBA leakage between
        # chained effects even when a compose op introduces one internally.
        cmd = [magick, str(in_path), *args, "-alpha", "off", str(out_path)]
        result = subprocess.run(cmd, capture_output=True, text=True,
                                creationflags=creationflags)
        if result.returncode != 0:
            err = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(
                f"ImageMagick failed (exit {result.returncode}): {err}"
            )
        with Image.open(out_path) as opened:
            return opened.copy()
