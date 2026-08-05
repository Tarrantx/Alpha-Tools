"""
Alpha-channel processing pipeline for upscaled game textures.

Pure functions — no GUI state. The GUI calls `process_image()` for batch work
and the individual stage functions (e.g. `level_alpha`, `n_level_snap`) when
building a live-preview chain.

Engine choice: Pillow + numpy for the per-pixel math (fast, in-process), with
ImageMagick reserved for headroom and vtracer.exe shelled out for the optional
pre-stage. ImageMagick stays bundled but is not on the hot path today.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# Optional: libimagequant for higher-quality palette quantization. This is
# the same algorithm pngquant uses; it preserves rare-but-saturated colours
# (small highlights, accents) much better than PIL's default median-cut,
# which is critical for character art where every distinct hue matters.
try:
    import imagequant as _imagequant
    HAS_IMAGEQUANT = True
except ImportError:
    _imagequant = None
    HAS_IMAGEQUANT = False


# Suppress the console-window flash when launching helpers from the GUI build.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def quantize_image(img: Image.Image, colors: int) -> Image.Image:
    """Reduce the palette to N colours, preferring libimagequant when
    available (better preservation of rare-but-saturated colours) and
    falling back to PIL's median-cut otherwise.

    Returns an RGB image so downstream code can treat the result uniformly.
    Dithering is disabled in both paths — vtracer wants solid regions to
    trace, not stippled approximations of intermediate colours.
    """
    if colors <= 0:
        return img.convert("RGB")
    colors = max(2, min(256, int(colors)))
    if HAS_IMAGEQUANT:
        # imagequant operates on RGBA; convert in and back to RGB out
        rgba = img.convert("RGBA")
        try:
            paletted = _imagequant.quantize_pil_image(
                rgba, dithering_level=0.0, max_colors=colors,
            )
            return paletted.convert("RGB")
        except RuntimeError:
            # libimagequant couldn't meet min_quality — shouldn't happen with
            # the defaults but fall through to median-cut just in case.
            pass
    rgb = img.convert("RGB")
    return rgb.quantize(
        colors=colors, dither=Image.Dither.NONE,
    ).convert("RGB")


def quantize_rgba(img: Image.Image, colors: int,
                  dither_level: float = 0.0) -> Image.Image:
    """Reduce to <= `colors` palette entries while RETAINING the alpha channel
    (libimagequant — the pngquant engine — which quantises RGBA and keeps a
    per-entry alpha). Returns a paletted 'P' image with an RGBA palette;
    convert('RGBA') restores per-pixel alpha, and saving as PNG yields a true
    8-bit (or 4-bit) PNG with transparency.

    Used to mimic the PS1 palette constraint (256 colours = 8-bit CLUT,
    16 = 4-bit) on the final output. Falls back to PIL's FASTOCTREE quantiser,
    which also preserves alpha, when libimagequant is unavailable.

    dither_level (0.0-1.0) applies error-diffusion dithering — 0 = flat colour
    bands (default), higher = more dithering to hide banding at small palettes.
    """
    colors = max(2, min(256, int(colors)))
    d = max(0.0, min(1.0, float(dither_level)))
    rgba = img.convert("RGBA")
    if HAS_IMAGEQUANT:
        try:
            return _imagequant.quantize_pil_image(
                rgba, dithering_level=d, max_colors=colors,
            )
        except RuntimeError:
            pass
    return rgba.quantize(
        colors=colors, method=Image.Quantize.FASTOCTREE,
        dither=Image.Dither.FLOYDSTEINBERG if d > 0 else Image.Dither.NONE,
    )


# Small cache so the live preview doesn't re-read + re-quantise the reference
# palette image on every unrelated slider move. Keyed on path + mtime.
_PALETTE_CACHE: dict = {}


def _load_palette_image(path: str) -> "Optional[Image.Image]":
    """Load a reference palette image as a PIL 'P'-mode palette source (its own
    distinct colours), cached by path+mtime. None if it can't be read."""
    try:
        p = Path(path)
        key = (str(p), p.stat().st_mtime)
    except OSError:
        return None
    hit = _PALETTE_CACHE.get(key)
    if hit is not None:
        return hit
    try:
        ref = Image.open(path).convert("RGB")
    except Exception:
        return None
    cols = ref.getcolors(maxcolors=256)
    ncol = len(cols) if cols else 256
    pal = ref.quantize(colors=max(1, min(256, ncol)))
    _PALETTE_CACHE.clear()          # only ever cache the current palette
    _PALETTE_CACHE[key] = pal
    return pal


def quantize_to_palette(img: Image.Image, palette_src: Image.Image,
                        dither_level: float = 0.0) -> Image.Image:
    """Remap `img`'s RGB onto the exact colours of `palette_src` (a P-mode
    palette image), preserving the original alpha channel. dither_level > 0
    enables Floyd-Steinberg error diffusion. Returns RGBA."""
    rgba = img.convert("RGBA")
    alpha = rgba.getchannel("A")
    dither = Image.Dither.FLOYDSTEINBERG if float(dither_level) > 0 else Image.Dither.NONE
    q = rgba.convert("RGB").quantize(palette=palette_src, dither=dither)
    out = q.convert("RGB").convert("RGBA")
    out.putalpha(alpha)
    return out


# --- Tool discovery --------------------------------------------------------

def _bundled_dir() -> Optional[Path]:
    """Directory next to the frozen .exe (cx_Freeze install) or None when
    running from source."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return None


def _configure_magick_env() -> None:
    """Point ImageMagick at the bundled codec modules and config files.

    The portable magick.exe locates its codec DLLs (modules/coders/IM_MOD_RL_*.dll)
    and its XML configs via Windows registry keys that only exist if
    ImageMagick was installed through its own installer. On a fresh machine
    that just received our MSI those keys are absent and magick.exe fails with:

        RegistryKeyLookupFailed `CoderModulesPath`
        no decode delegate for this image format `SVG` / `PNG`

    Setting these env vars makes magick.exe skip the registry lookup and use
    the bundled files. setdefault leaves any user-set values untouched.

    Only takes effect when the bundled vendor/imagemagick/ folder is present
    (the frozen MSI build). In dev mode the function bails out and any
    system ImageMagick on PATH is used as before via its own registry keys.
    """
    bundled_root = _bundled_dir()
    if bundled_root is None:
        return
    bundled = bundled_root / "vendor" / "imagemagick"
    if not bundled.is_dir():
        return
    os.environ.setdefault("MAGICK_HOME", str(bundled))
    os.environ.setdefault("MAGICK_CONFIGURE_PATH", str(bundled))
    # Codec DLLs live in modules/coders/, not directly in modules/.
    coders_dir = bundled / "modules" / "coders"
    if coders_dir.is_dir():
        os.environ.setdefault("MAGICK_CODER_MODULE_PATH", str(coders_dir))
    filters_dir = bundled / "modules" / "filters"
    if filters_dir.is_dir():
        os.environ.setdefault("MAGICK_CODER_FILTER_PATH", str(filters_dir))


_configure_magick_env()


def find_magick() -> Optional[str]:
    # 1. Bundled next to the frozen app
    bundled = _bundled_dir()
    if bundled is not None:
        candidate = bundled / "vendor" / "imagemagick" / "magick.exe"
        if candidate.exists():
            return str(candidate)
    # 2. On PATH
    on_path = shutil.which("magick")
    if on_path:
        return on_path
    # 3. Common install locations
    for p in [
        r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe",
        r"C:\Program Files\ImageMagick\magick.exe",
        r"C:\Scripts\Dependencies\Imagemagick\magick.exe",
    ]:
        if os.path.exists(p):
            return p
    return None


def find_vtracer() -> Optional[str]:
    bundled = _bundled_dir()
    if bundled is not None:
        candidate = bundled / "vtracer.exe"
        if candidate.exists():
            return str(candidate)
    on_path = shutil.which("vtracer")
    if on_path:
        return on_path
    for p in [
        r"C:\Scripts\Vtracer-Tools\vtracer.exe",
        r"C:\Scripts\Alpha-Tools\vtracer.exe",
    ]:
        if os.path.exists(p):
            return p
    return None


# --- Parameter defaults ----------------------------------------------------

DEFAULT_PARAMS = {
    # Vtracer pre-stage
    "vtracer_enabled":            False,
    "vtracer_ignore_alpha":       False,  # True = trace raw RGB, don't mask transparent areas
    "vtracer_pre_quantize":       16,    # 0 disables pre-quantize
    "vtracer_mode":               "pixel",    # pixel|polygon|spline (curve fitting)
    "vtracer_hierarchical":       "stacked",  # stacked|cutout
    "vtracer_color_precision":    6,     # vtracer --color_precision
    "vtracer_filter_speckle":     4,     # vtracer --filter_speckle
    "vtracer_gradient_step":      16,    # vtracer --gradient_step
    "vtracer_corner_threshold":   60,    # spline-only, degrees
    "vtracer_segment_length":     4.0,   # spline-only, pixels
    "vtracer_splice_threshold":   45,    # spline-only, degrees

    # Alpha source
    "alpha_source":               "fix",   # "fix" | "create" | "fuzzythresh" | "rangethresh"

    # Fuzzythresh-path synthesis (Fred Weinhaus weighted iterative threshold),
    # anchored to the original alpha.
    "fuzzythresh_target":         "white",     # "white" | "gray"
    "fuzzythresh_radius":         0,           # Fred -r: blurs the weight map (0 = off)
    "fuzzythresh_invert":         False,       # True = treat dark region as opaque (light bg)
    "fuzzythresh_seed_mode":      "grow",      # "shrink" | "grow" — how the protected seed is built
    "fuzzythresh_seed_grow":      0,           # % of region (brightest-first) kept as protected seed
    "fuzzythresh_seed_shrink":    0,           # px to erode region for the protected seed (shrink mode)
    "fuzzythresh_expand_limit":   0,           # px to dilate the region for outward fringe growth
    "fuzzythresh_alpha_floor":    16,          # original alpha <= this is treated as transparent

    # Rangethresh-path synthesis (Fred Weinhaus colour-range key), anchored to
    # the original alpha. Pixel passes when all 3 channels are within range.
    "rangethresh_target":         "white",     # "white" | "gray"
    "rangethresh_colorspace":     "sRGB",      # "sRGB" | "HSV" | "HSL"
    "rangethresh_unit":           "8bit",      # "8bit" (0-255) | "percent" (0-100)
    "rangethresh_r_low":          0,           # channel 1 (R / H) low
    "rangethresh_r_high":         255,         # channel 1 high
    "rangethresh_g_low":          0,           # channel 2 (G / S) low
    "rangethresh_g_high":         255,
    "rangethresh_b_low":          0,           # channel 3 (B / V or L) low
    "rangethresh_b_high":         255,
    "rangethresh_seed_mode":      "grow",      # "shrink" | "grow"; grow@0% = no seed → range decides all
    "rangethresh_seed_grow":      0,
    "rangethresh_seed_shrink":    0,
    "rangethresh_expand_limit":   0,
    "rangethresh_alpha_floor":    16,          # original alpha <= this is treated as transparent

    # Fix-path levelling (percent of full alpha range)
    "fix_level_low":              30.0,
    "fix_level_high":             70.0,

    # Create-path synthesis (luminance-anchored)
    "create_target":              "gray",      # "gray" | "white"
    "create_alpha_floor":         16,          # original alpha <= this is treated as transparent
    "create_seed_mode":           "shrink",    # "shrink" | "grow" — how the protected seed is built
    "create_seed_shrink":         2,           # px to erode original alpha for seed (shrink mode)
    "create_seed_grow":           0,           # % of region (brightest-first) kept as seed (grow mode)
    "create_expand_limit":        4,           # px to dilate original alpha for growth boundary
    "create_lum_threshold":       10.0,        # min luminance % to include in fringe (0-100)
    "create_lum_mode":            "perceptual",  # "perceptual" | "max-channel"
    "create_lum_invert":          False,         # True = dark pixels pass (invert threshold direction)

    # Incongruence edge refine — colour-aware halo removal. Walks inward from
    # the transparency border (up to incong_reach px) eating boundary pixels
    # that are INCONGRUENT with the local interior content (upscaler fringe),
    # and stopping at a content match, a sharp colour step (genuine outline),
    # a thin feature (no interior reference nearby), or the reach cap.
    # Runs between alpha synthesis and blur/despeckle.
    "incong_enabled":             False,
    "incong_reach":               4,      # max px eaten inward from the border
    "incong_content_thresh":      35.0,   # boundary-vs-interior RGB distance below this = content, stop
    "incong_line_thresh":         90.0,   # outward RGB step above this = feature line (outline), stop
    "incong_smooth":              True,   # majority-smooth the eaten mask (de-jag)
    "incong_alpha_floor":         16,     # alpha > this = subject silhouette; keep below a grey subject's alpha (PS1 127/143)

    # N-level snap (preserves quantised plateaus like 0/128/255)
    "snap_enabled":               False,
    "snap_levels":                3,
    "snap_low_threshold":         25,
    "snap_high_threshold":        230,
    "snap_mid_tolerance":         40,

    # Edge refinement
    "blur_radius":                0.0,
    "despeckle":                  0,

    # Alpha shape — shrink / grow the alpha mask (post-snap, post-binary).
    # Use negative values via the GUI's bipolar slider; the pipeline reads
    # alpha_erosion and alpha_dilation independently.
    "alpha_erosion":              0,    # pixels to shrink the alpha mask
    "alpha_dilation":             0,    # pixels to grow the alpha mask

    # Final alpha clamp
    "clamp_mode":                 "none",   # "none" | "binary" | "tri-clamp" | "ps1"
    "binary_threshold":           128,
    "tri_clamp_lo":               64,
    "tri_clamp_hi":               192,

    # PS1 region-aware tri-state snap (clamp_mode == "ps1").
    # Quantises to exactly {0, ps1_semi_value, 255}; flat semi-transparent
    # surfaces keep the semi value, cutout-edge gradients go binary.
    # Emulator semi value: DuckStation = 143, Beetle PSX HW = 127.
    "ps1_semi_value":             143,
    "ps1_semi_tolerance":         40,   # +/- band counted as "near semi"
    "ps1_region_min":             2,    # opening radius (px) to drop false-semi edges

    # RGB processing
    "edge_dilation":              0,
    "edge_dilation_threshold":    16,    # alpha > this is a valid colour source
    "premultiply":                False,

    # Tone adjustment (PIL.ImageEnhance, 1.0 = no change)
    "saturation":                 1.0,
    "contrast":                   1.0,
    "brightness":                 1.0,
    # Per-channel colour level (Fred Weinhaus levelcolors). -100..100 each;
    # >0 brightens that channel, <0 darkens it, 0 = no change.
    "level_r":                    0,
    "level_g":                    0,
    "level_b":                    0,

    # Stylise
    "pixelate_factor":            1.0,   # 1 = off; N = ~NxN blocks. Fractional (1.5) = milder
    "soften_edges":               0,     # px Gaussian blur of RGB along the alpha outline; 0 = off

    # Output colour depth — "Full colour" | "256 (8-bit)" | "16 (4-bit)".
    # Quantises the final image to a palette (alpha retained). Final stage.
    "quantize_mode":              "Full colour",
    "quantize_dither":            0.0,     # 0-1 error-diffusion dither (blocks when pixelating)
    "quantize_use_palette":       False,   # match a reference palette image instead of a colour count
    "quantize_palette_path":      "",      # path to the reference palette image

    # Batch colour match to a reference image (kills animation colour flicker).
    # Set + driven by the Batch tab; inert in single-image Tune (no reference).
    # colourmatch_ref_stats (the precomputed reference statistics) is injected
    # into the params by the batch runner, not stored here.
    "colourmatch_enabled":        False,
    "colourmatch_method":         "Reinhard (mean/std)",  # | "Histogram match"
    "colourmatch_strength":       1.0,     # 0 = off, 1 = full match
    "colourmatch_colorspace":     "LAB",   # "LAB" | "RGB"

    # Post-processing effects chain (up to 5 slots, applied in order).
    # Each slot is independently None or one of post_fx.EFFECTS_LIST.
    # Slots with effect="None" are skipped.
    "fx_effect_1":                "None",
    "fx_strength_1":              1.0,
    "fx_color_1":                 "#0080FF",
    "fx_opacity_1":               1.0,
    "fx_emboss_azimuth_1":        135,
    "fx_emboss_elevation_1":      45,
    "fx_emboss_depth_1":          1,
    "fx_emboss_compose_1":        "Overlay",
    "fx_emboss_method_1":       1,
    "fx_emboss_intensity_1":    0,
    "fx_gran_sigma_1":            5,
    "fx_gran_angle_1":            45,
    "fx_gran_blend_1":            0.3,
    "fx_gran_compose_1":          "Overlay",
    "fx_chrome_smooth_1":         3.0,
    "fx_chrome_cycles_1":         2,
    "fx_chrome_intensity_1":      100,
    "fx_chrome_azimuth_1":        135,
    "fx_chrome_elevation_1":      45,
    "fx_chrome_blend_1":          0.6,
    "fx_chrome_compose_1":        "Screen",
    "fx_chrome_source_1":        "image",
    "fx_frost_spread_1":          5,
    "fx_frost_pre_blur_1":        5.0,
    "fx_frost_post_blur_1":       0.5,
    "fx_lomo_type_1":             "turquoise",
    "fx_paint_gamma_1":           1.0,
    "fx_paint_blur_1":            2.0,
    "fx_paint_radius_1":          2,
    "fx_paint_edge_1":            8.0,
    "fx_bevel_width_1":           5,
    "fx_bevel_azimuth_1":         135,
    "fx_bevel_elevation_1":       30,
    "fx_bevel_depth_1":           100,
    "fx_bevel_bcolor_1":          "#808080",
    "fx_bevel_colormode_1":       "dark",

    "fx_bevel_form_1":            "inner",
    "fx_bevel_innermode_1":       "hard",
    "fx_bevel_outermode_1":       "raised",
    "fx_bevel_soften_1":          0,
    "fx_bevel_taper_1":           0,
    "fx_bevel_ring_alpha_1":      "Opaque",
    "fx_davehill_brightness_1":   1.0,
    "fx_davehill_contrast_1":     0.0,
    "fx_davehill_gain_1":         40,
    "fx_crosshatch_1":            False,
    "fx_crosshatch_length_1":     7,
    "fx_crosshatch_balance_1":    0,
    "fx_crosshatch_sharp_1":      10,
    "fx_crosshatch_mix_1":        100,
    "fx_texturize_threshold_1":   2.0,
    "fx_texturize_blur_1":        9.0,
    "fx_texturize_widen_1":       0.0,
    "fx_texturize_spread_1":      0.0,
    "fx_texturize_gnoise_1":      5.0,
    "fx_texturize_format_1":      "bump",
    "fx_texturize_mix_1":         25.0,
    "fx_texturize_contrast_1":    0.0,
    "fx_texturize_seed_1":        1,
    "fx_texturize_azimuth_1":     135,
    "fx_texturize_elevation_1":   45,
    "fx_watercolor_smoothing_1":  0,
    "fx_watercolor_edge_1":       5.0,
    "fx_watercolor_mixing_1":     33,
    "fx_watercolor_contrast_1":   0.0,
    "fx_screenfx_type_1":         "simple",
    "fx_screenfx_spacing_1":      6.0,
    "fx_screenfx_rotate_1":       False,
    "fx_screenfx_color_1":        "#000000",
    "fx_screenfx_amount_1":       6.0,
    "fx_screenfx_method_1":       1,
    "fx_effect_2":                "None",
    "fx_strength_2":              1.0,
    "fx_color_2":                 "#0080FF",
    "fx_opacity_2":               1.0,
    "fx_emboss_azimuth_2":        135,
    "fx_emboss_elevation_2":      45,
    "fx_emboss_depth_2":          1,
    "fx_emboss_compose_2":        "Overlay",
    "fx_emboss_method_2":       1,
    "fx_emboss_intensity_2":    0,
    "fx_gran_sigma_2":            5,
    "fx_gran_angle_2":            45,
    "fx_gran_blend_2":            0.3,
    "fx_gran_compose_2":          "Overlay",
    "fx_chrome_smooth_2":         3.0,
    "fx_chrome_cycles_2":         2,
    "fx_chrome_intensity_2":      100,
    "fx_chrome_azimuth_2":        135,
    "fx_chrome_elevation_2":      45,
    "fx_chrome_blend_2":          0.6,
    "fx_chrome_compose_2":        "Screen",
    "fx_chrome_source_2":        "image",
    "fx_frost_spread_2":          5,
    "fx_frost_pre_blur_2":        5.0,
    "fx_frost_post_blur_2":       0.5,
    "fx_lomo_type_2":             "turquoise",
    "fx_paint_gamma_2":           1.0,
    "fx_paint_blur_2":            2.0,
    "fx_paint_radius_2":          2,
    "fx_paint_edge_2":            8.0,
    "fx_bevel_width_2":           5,
    "fx_bevel_azimuth_2":         135,
    "fx_bevel_elevation_2":       30,
    "fx_bevel_depth_2":           100,
    "fx_bevel_bcolor_2":          "#808080",
    "fx_bevel_colormode_2":       "dark",

    "fx_bevel_form_2":            "inner",
    "fx_bevel_innermode_2":       "hard",
    "fx_bevel_outermode_2":       "raised",
    "fx_bevel_soften_2":          0,
    "fx_bevel_taper_2":           0,
    "fx_bevel_ring_alpha_2":      "Opaque",
    "fx_davehill_brightness_2":   1.0,
    "fx_davehill_contrast_2":     0.0,
    "fx_davehill_gain_2":         40,
    "fx_crosshatch_2":            False,
    "fx_crosshatch_length_2":     7,
    "fx_crosshatch_balance_2":    0,
    "fx_crosshatch_sharp_2":      10,
    "fx_crosshatch_mix_2":        100,
    "fx_texturize_threshold_2":   2.0,
    "fx_texturize_blur_2":        9.0,
    "fx_texturize_widen_2":       0.0,
    "fx_texturize_spread_2":      0.0,
    "fx_texturize_gnoise_2":      5.0,
    "fx_texturize_format_2":      "bump",
    "fx_texturize_mix_2":         25.0,
    "fx_texturize_contrast_2":    0.0,
    "fx_texturize_seed_2":        1,
    "fx_texturize_azimuth_2":     135,
    "fx_texturize_elevation_2":   45,
    "fx_watercolor_smoothing_2":  0,
    "fx_watercolor_edge_2":       5.0,
    "fx_watercolor_mixing_2":     33,
    "fx_watercolor_contrast_2":   0.0,
    "fx_screenfx_type_2":         "simple",
    "fx_screenfx_spacing_2":      6.0,
    "fx_screenfx_rotate_2":       False,
    "fx_screenfx_color_2":        "#000000",
    "fx_screenfx_amount_2":       6.0,
    "fx_screenfx_method_2":       1,
    "fx_effect_3":                "None",
    "fx_strength_3":              1.0,
    "fx_color_3":                 "#0080FF",
    "fx_opacity_3":               1.0,
    "fx_emboss_azimuth_3":        135,
    "fx_emboss_elevation_3":      45,
    "fx_emboss_depth_3":          1,
    "fx_emboss_compose_3":        "Overlay",
    "fx_emboss_method_3":       1,
    "fx_emboss_intensity_3":    0,
    "fx_gran_sigma_3":            5,
    "fx_gran_angle_3":            45,
    "fx_gran_blend_3":            0.3,
    "fx_gran_compose_3":          "Overlay",
    "fx_chrome_smooth_3":         3.0,
    "fx_chrome_cycles_3":         2,
    "fx_chrome_intensity_3":      100,
    "fx_chrome_azimuth_3":        135,
    "fx_chrome_elevation_3":      45,
    "fx_chrome_blend_3":          0.6,
    "fx_chrome_compose_3":        "Screen",
    "fx_chrome_source_3":        "image",
    "fx_frost_spread_3":          5,
    "fx_frost_pre_blur_3":        5.0,
    "fx_frost_post_blur_3":       0.5,
    "fx_lomo_type_3":             "turquoise",
    "fx_paint_gamma_3":           1.0,
    "fx_paint_blur_3":            2.0,
    "fx_paint_radius_3":          2,
    "fx_paint_edge_3":            8.0,
    "fx_bevel_width_3":           5,
    "fx_bevel_azimuth_3":         135,
    "fx_bevel_elevation_3":       30,
    "fx_bevel_depth_3":           100,
    "fx_bevel_bcolor_3":          "#808080",
    "fx_bevel_colormode_3":       "dark",

    "fx_bevel_form_3":            "inner",
    "fx_bevel_innermode_3":       "hard",
    "fx_bevel_outermode_3":       "raised",
    "fx_bevel_soften_3":          0,
    "fx_bevel_taper_3":           0,
    "fx_bevel_ring_alpha_3":      "Opaque",
    "fx_davehill_brightness_3":   1.0,
    "fx_davehill_contrast_3":     0.0,
    "fx_davehill_gain_3":         40,
    "fx_crosshatch_3":            False,
    "fx_crosshatch_length_3":     7,
    "fx_crosshatch_balance_3":    0,
    "fx_crosshatch_sharp_3":      10,
    "fx_crosshatch_mix_3":        100,
    "fx_texturize_threshold_3":   2.0,
    "fx_texturize_blur_3":        9.0,
    "fx_texturize_widen_3":       0.0,
    "fx_texturize_spread_3":      0.0,
    "fx_texturize_gnoise_3":      5.0,
    "fx_texturize_format_3":      "bump",
    "fx_texturize_mix_3":         25.0,
    "fx_texturize_contrast_3":    0.0,
    "fx_texturize_seed_3":        1,
    "fx_texturize_azimuth_3":     135,
    "fx_texturize_elevation_3":   45,
    "fx_watercolor_smoothing_3":  0,
    "fx_watercolor_edge_3":       5.0,
    "fx_watercolor_mixing_3":     33,
    "fx_watercolor_contrast_3":   0.0,
    "fx_screenfx_type_3":         "simple",
    "fx_screenfx_spacing_3":      6.0,
    "fx_screenfx_rotate_3":       False,
    "fx_screenfx_color_3":        "#000000",
    "fx_screenfx_amount_3":       6.0,
    "fx_screenfx_method_3":       1,
    "fx_effect_4":                "None",
    "fx_strength_4":              1.0,
    "fx_color_4":                 "#0080FF",
    "fx_opacity_4":               1.0,
    "fx_emboss_azimuth_4":        135,
    "fx_emboss_elevation_4":      45,
    "fx_emboss_depth_4":          1,
    "fx_emboss_compose_4":        "Overlay",
    "fx_emboss_method_4":       1,
    "fx_emboss_intensity_4":    0,
    "fx_gran_sigma_4":            5,
    "fx_gran_angle_4":            45,
    "fx_gran_blend_4":            0.3,
    "fx_gran_compose_4":          "Overlay",
    "fx_chrome_smooth_4":         3.0,
    "fx_chrome_cycles_4":         2,
    "fx_chrome_intensity_4":      100,
    "fx_chrome_azimuth_4":        135,
    "fx_chrome_elevation_4":      45,
    "fx_chrome_blend_4":          0.6,
    "fx_chrome_compose_4":        "Screen",
    "fx_chrome_source_4":        "image",
    "fx_frost_spread_4":          5,
    "fx_frost_pre_blur_4":        5.0,
    "fx_frost_post_blur_4":       0.5,
    "fx_lomo_type_4":             "turquoise",
    "fx_paint_gamma_4":           1.0,
    "fx_paint_blur_4":            2.0,
    "fx_paint_radius_4":          2,
    "fx_paint_edge_4":            8.0,
    "fx_bevel_width_4":           5,
    "fx_bevel_azimuth_4":         135,
    "fx_bevel_elevation_4":       30,
    "fx_bevel_depth_4":           100,
    "fx_bevel_bcolor_4":          "#808080",
    "fx_bevel_colormode_4":       "dark",

    "fx_bevel_form_4":            "inner",
    "fx_bevel_innermode_4":       "hard",
    "fx_bevel_outermode_4":       "raised",
    "fx_bevel_soften_4":          0,
    "fx_bevel_taper_4":           0,
    "fx_bevel_ring_alpha_4":      "Opaque",
    "fx_davehill_brightness_4":   1.0,
    "fx_davehill_contrast_4":     0.0,
    "fx_davehill_gain_4":         40,
    "fx_crosshatch_4":            False,
    "fx_crosshatch_length_4":     7,
    "fx_crosshatch_balance_4":    0,
    "fx_crosshatch_sharp_4":      10,
    "fx_crosshatch_mix_4":        100,
    "fx_texturize_threshold_4":   2.0,
    "fx_texturize_blur_4":        9.0,
    "fx_texturize_widen_4":       0.0,
    "fx_texturize_spread_4":      0.0,
    "fx_texturize_gnoise_4":      5.0,
    "fx_texturize_format_4":      "bump",
    "fx_texturize_mix_4":         25.0,
    "fx_texturize_contrast_4":    0.0,
    "fx_texturize_seed_4":        1,
    "fx_texturize_azimuth_4":     135,
    "fx_texturize_elevation_4":   45,
    "fx_watercolor_smoothing_4":  0,
    "fx_watercolor_edge_4":       5.0,
    "fx_watercolor_mixing_4":     33,
    "fx_watercolor_contrast_4":   0.0,
    "fx_screenfx_type_4":         "simple",
    "fx_screenfx_spacing_4":      6.0,
    "fx_screenfx_rotate_4":       False,
    "fx_screenfx_color_4":        "#000000",
    "fx_screenfx_amount_4":       6.0,
    "fx_screenfx_method_4":       1,
    "fx_effect_5":                "None",
    "fx_strength_5":              1.0,
    "fx_color_5":                 "#0080FF",
    "fx_opacity_5":               1.0,
    "fx_emboss_azimuth_5":        135,
    "fx_emboss_elevation_5":      45,
    "fx_emboss_depth_5":          1,
    "fx_emboss_compose_5":        "Overlay",
    "fx_emboss_method_5":       1,
    "fx_emboss_intensity_5":    0,
    "fx_gran_sigma_5":            5,
    "fx_gran_angle_5":            45,
    "fx_gran_blend_5":            0.3,
    "fx_gran_compose_5":          "Overlay",
    "fx_chrome_smooth_5":         3.0,
    "fx_chrome_cycles_5":         2,
    "fx_chrome_intensity_5":      100,
    "fx_chrome_azimuth_5":        135,
    "fx_chrome_elevation_5":      45,
    "fx_chrome_blend_5":          0.6,
    "fx_chrome_compose_5":        "Screen",
    "fx_chrome_source_5":        "image",
    "fx_frost_spread_5":          5,
    "fx_frost_pre_blur_5":        5.0,
    "fx_frost_post_blur_5":       0.5,
    "fx_lomo_type_5":             "turquoise",
    "fx_paint_gamma_5":           1.0,
    "fx_paint_blur_5":            2.0,
    "fx_paint_radius_5":          2,
    "fx_paint_edge_5":            8.0,
    "fx_bevel_width_5":           5,
    "fx_bevel_azimuth_5":         135,
    "fx_bevel_elevation_5":       30,
    "fx_bevel_depth_5":           100,
    "fx_bevel_bcolor_5":          "#808080",
    "fx_bevel_colormode_5":       "dark",

    "fx_bevel_form_5":            "inner",
    "fx_bevel_innermode_5":       "hard",
    "fx_bevel_outermode_5":       "raised",
    "fx_bevel_soften_5":          0,
    "fx_bevel_taper_5":           0,
    "fx_bevel_ring_alpha_5":      "Opaque",
    "fx_davehill_brightness_5":   1.0,
    "fx_davehill_contrast_5":     0.0,
    "fx_davehill_gain_5":         40,
    "fx_crosshatch_5":            False,
    "fx_crosshatch_length_5":     7,
    "fx_crosshatch_balance_5":    0,
    "fx_crosshatch_sharp_5":      10,
    "fx_crosshatch_mix_5":        100,
    "fx_texturize_threshold_5":   2.0,
    "fx_texturize_blur_5":        9.0,
    "fx_texturize_widen_5":       0.0,
    "fx_texturize_spread_5":      0.0,
    "fx_texturize_gnoise_5":      5.0,
    "fx_texturize_format_5":      "bump",
    "fx_texturize_mix_5":         25.0,
    "fx_texturize_contrast_5":    0.0,
    "fx_texturize_seed_5":        1,
    "fx_texturize_azimuth_5":     135,
    "fx_texturize_elevation_5":   45,
    "fx_watercolor_smoothing_5":  0,
    "fx_watercolor_edge_5":       5.0,
    "fx_watercolor_mixing_5":     33,
    "fx_watercolor_contrast_5":   0.0,
    "fx_screenfx_type_5":         "simple",
    "fx_screenfx_spacing_5":      6.0,
    "fx_screenfx_rotate_5":       False,
    "fx_screenfx_color_5":        "#000000",
    "fx_screenfx_amount_5":       6.0,
    "fx_screenfx_method_5":       1,
    # Tile size for the post-processing chain. "Off" applies effects to the
    # whole image; "16×16" / "32×32" / ... applies the chain independently
    # to each NxN tile and stitches back. Useful for sprite-sheet / atlas
    # textures where effects should respect tile boundaries.
    "fx_tile_size":               "Off",

    # DDS export compression (only used when output_path has a .dds suffix).
    # 'dxt1' = BC1 (RGB, no/1-bit alpha); 'dxt5' = BC3 (RGBA, full alpha).
    "dds_compression":            "dxt5",
}


def _build_fx_chain(params: dict) -> list:
    """Pull (effect, strength, extras) tuples for non-None slots.

    ``extras`` carries per-effect parameters so that apply_effect can retrieve
    them without widening the public API. Non-relevant keys are ignored by each
    effect handler.
    """
    chain = []
    for i in (1, 2, 3, 4, 5):
        eff = params.get(f"fx_effect_{i}", "None")
        if eff and eff != "None":
            extras = {
                "color":            str(params.get(f"fx_color_{i}",            "#0080FF")),
                "opacity":          float(params.get(f"fx_opacity_{i}",          1.0)),
                "emboss_azimuth":   int(params.get(f"fx_emboss_azimuth_{i}",   135)),
                "emboss_elevation": int(params.get(f"fx_emboss_elevation_{i}",  45)),
                "emboss_depth":     int(params.get(f"fx_emboss_depth_{i}",       1)),
                "emboss_compose":   str(params.get(f"fx_emboss_compose_{i}", "Overlay")),
                "emboss_method":    int(params.get(f"fx_emboss_method_{i}",      1)),
                "emboss_intensity": int(params.get(f"fx_emboss_intensity_{i}",   0)),
                "gran_sigma":       int(params.get(f"fx_gran_sigma_{i}",         5)),
                "gran_angle":       int(params.get(f"fx_gran_angle_{i}",        45)),
                "gran_blend":       float(params.get(f"fx_gran_blend_{i}",     0.3)),
                "gran_compose":     str(params.get(f"fx_gran_compose_{i}",  "Overlay")),
                "chrome_smooth":    float(params.get(f"fx_chrome_smooth_{i}",   3.0)),
                "chrome_cycles":    int(params.get(f"fx_chrome_cycles_{i}",       2)),
                "chrome_intensity": int(params.get(f"fx_chrome_intensity_{i}",  100)),
                "chrome_azimuth":   int(params.get(f"fx_chrome_azimuth_{i}",    135)),
                "chrome_elevation": int(params.get(f"fx_chrome_elevation_{i}",   45)),
                "chrome_blend":     float(params.get(f"fx_chrome_blend_{i}",    0.6)),
                "chrome_compose":   str(params.get(f"fx_chrome_compose_{i}", "Screen")),
                "chrome_source":    str(params.get(f"fx_chrome_source_{i}", "image")),
                "frost_spread":     int(params.get(f"fx_frost_spread_{i}",          5)),
                "frost_pre_blur":   float(params.get(f"fx_frost_pre_blur_{i}",    5.0)),
                "frost_post_blur":  float(params.get(f"fx_frost_post_blur_{i}",   0.5)),
                "lomo_type":        str(params.get(f"fx_lomo_type_{i}", "turquoise")),
                "paint_gamma":      float(params.get(f"fx_paint_gamma_{i}",  1.0)),
                "paint_blur":       float(params.get(f"fx_paint_blur_{i}",   2.0)),
                "paint_radius":     int(params.get(f"fx_paint_radius_{i}",     2)),
                "paint_edge":       float(params.get(f"fx_paint_edge_{i}",   8.0)),
                "bevel_width":           int(params.get(f"fx_bevel_width_{i}",       8)),
                "bevel_azimuth":         int(params.get(f"fx_bevel_azimuth_{i}",   135)),
                "bevel_elevation":       int(params.get(f"fx_bevel_elevation_{i}",  30)),
                "bevel_depth":           int(params.get(f"fx_bevel_depth_{i}",     100)),
                "bevel_form":            str(params.get(f"fx_bevel_form_{i}",   "inner")),
                "bevel_innermode":       str(params.get(f"fx_bevel_innermode_{i}", "hard")),
                "bevel_outermode":       str(params.get(f"fx_bevel_outermode_{i}", "raised")),
                "bevel_soften":          int(params.get(f"fx_bevel_soften_{i}",       0)),
                "bevel_taper":           int(params.get(f"fx_bevel_taper_{i}",        0)),
                "bevel_bcolor":          str(params.get(f"fx_bevel_bcolor_{i}", "#808080")),
                "bevel_colormode":       str(params.get(f"fx_bevel_colormode_{i}", "dark")),
                "bevel_ring_alpha":      str(params.get(f"fx_bevel_ring_alpha_{i}", "Opaque")),
                "davehill_brightness":   float(params.get(f"fx_davehill_brightness_{i}", 1.0)),
                "davehill_contrast":     float(params.get(f"fx_davehill_contrast_{i}",   0.0)),
                "davehill_gain":         int(params.get(f"fx_davehill_gain_{i}",           40)),
                "crosshatch":            bool(params.get(f"fx_crosshatch_{i}",          False)),
                "crosshatch_length":     int(params.get(f"fx_crosshatch_length_{i}",        7)),
                "crosshatch_balance":    int(params.get(f"fx_crosshatch_balance_{i}",       0)),
                "crosshatch_sharp":      int(params.get(f"fx_crosshatch_sharp_{i}",        10)),
                "crosshatch_mix":        int(params.get(f"fx_crosshatch_mix_{i}",         100)),
                "texturize_threshold":   float(params.get(f"fx_texturize_threshold_{i}",  2.0)),
                "texturize_blur":        float(params.get(f"fx_texturize_blur_{i}",        9.0)),
                "texturize_widen":       float(params.get(f"fx_texturize_widen_{i}",       0.0)),
                "texturize_spread":      float(params.get(f"fx_texturize_spread_{i}",      0.0)),
                "texturize_gnoise":      float(params.get(f"fx_texturize_gnoise_{i}",      5.0)),
                "texturize_format":      str(params.get(f"fx_texturize_format_{i}",     "bump")),
                "texturize_mix":         float(params.get(f"fx_texturize_mix_{i}",        25.0)),
                "texturize_contrast":    float(params.get(f"fx_texturize_contrast_{i}",    0.0)),
                "texturize_seed":        int(params.get(f"fx_texturize_seed_{i}",            1)),
                "texturize_azimuth":     int(params.get(f"fx_texturize_azimuth_{i}",       135)),
                "texturize_elevation":   int(params.get(f"fx_texturize_elevation_{i}",      45)),
                "watercolor_smoothing":  int(params.get(f"fx_watercolor_smoothing_{i}",      0)),
                "watercolor_edge":       float(params.get(f"fx_watercolor_edge_{i}",        5.0)),
                "watercolor_mixing":     int(params.get(f"fx_watercolor_mixing_{i}",        33)),
                "watercolor_contrast":   float(params.get(f"fx_watercolor_contrast_{i}",    0.0)),
                "screenfx_type":         str(params.get(f"fx_screenfx_type_{i}",     "simple")),
                "screenfx_spacing":      float(params.get(f"fx_screenfx_spacing_{i}",     6.0)),
                "screenfx_rotate":       bool(params.get(f"fx_screenfx_rotate_{i}",     False)),
                "screenfx_color":        str(params.get(f"fx_screenfx_color_{i}",   "#000000")),
                "screenfx_amount":       float(params.get(f"fx_screenfx_amount_{i}",      6.0)),
                "screenfx_method":       int(params.get(f"fx_screenfx_method_{i}",          1)),
            }
            chain.append((eff, float(params.get(f"fx_strength_{i}", 1.0)), extras))
    return chain


# --- Image helpers ---------------------------------------------------------

def load_rgba(path) -> Image.Image:
    """Load any supported image as RGBA. Adds a fully-opaque alpha if the
    source had none, so downstream code can assume an alpha channel exists."""
    img = Image.open(path)
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    return img


def split_rgba(rgba: Image.Image) -> tuple[Image.Image, Image.Image]:
    return rgba.convert("RGB"), rgba.getchannel("A")


# --- Auto-classification (for batch routing) ------------------------------

# Preset stems used by auto-pick (must match files under presets/).
PRESET_DEFAULT          = "01 Default (pass-through)"
PRESET_BINARY_CUTOUT    = "08 Upscaled - binary cutout"
PRESET_PS1_DUCKSTATION  = "11 PS1 - DuckStation STP (0-143-255)"
PRESET_PS1_BEETLE       = "12 PS1 - Beetle PSX HW (0-127-255)"


def classify_for_preset(path) -> tuple[str, str]:
    """Inspect a texture and recommend a preset for batch auto-pick.

    Returns (preset_stem, reason). The preset stem matches a shipped preset
    file under presets/ (without the .json extension). The reason is a short
    human-readable string for the batch log.

    DuckStation encodes the texture mode in the dump filename, which is a more
    reliable signal than the (post-upscale, smeared) alpha histogram:
      * texpage-STP4 / texpage-STP8  → uses the semi-transparency bit
                                       → PS1 DuckStation preset (semi = 143)
      * texpage-P4 / texpage-P8      → paletted, no semi → binary cutout

    Beetle PSX HW dumps are hash-named with no mode hint, so they fall back to
    the alpha histogram:
      * No alpha / all transparent / fully opaque → 01 Default (pass-through)
      * Anything with meaningful alpha            → PS1 Beetle preset (semi=127)
        (the region-aware snap yields a clean binary result when there are no
        flat semi surfaces, so this is safe for binary Beetle textures too).
    """
    name = Path(path).name
    # 1. DuckStation filename hint — authoritative when present.
    if "-STP4-" in name or "-STP8-" in name:
        return (PRESET_PS1_DUCKSTATION, "DuckStation STP texture — semi 143")
    if "-P4-" in name or "-P8-" in name:
        return (PRESET_BINARY_CUTOUT, "DuckStation paletted — no semi (binary)")

    # 2. Histogram fallback (Beetle / unknown sources).
    img = Image.open(path)
    if "A" not in img.getbands():
        return (PRESET_DEFAULT, "no alpha channel")

    a = np.asarray(img.convert("RGBA").getchannel("A"))
    if a.max() == 0:
        return (PRESET_DEFAULT, "all transparent")
    if a.min() == 255 and a.max() == 255:
        return (PRESET_DEFAULT, "fully opaque")

    # Has meaningful alpha — sniff which plateaus are present so the log is
    # useful for spot-checking.
    total = a.size
    plateaus = []
    if (a <= 25).sum() / total > 0.01:
        plateaus.append("0")
    if ((a >= 88) & (a <= 168)).sum() / total > 0.01:
        plateaus.append("semi")
    if (a >= 230).sum() / total > 0.01:
        plateaus.append("255")
    if not plateaus:
        # Alpha values exist but no clear plateau — it's smeared/graded
        plateaus.append("graded")
    reason = "Beetle/unknown, plateaus: " + "/".join(plateaus)
    return (PRESET_PS1_BEETLE, reason)


def compose_rgba(rgb: Image.Image, alpha: Image.Image) -> Image.Image:
    rgba = rgb.convert("RGBA")
    rgba.putalpha(alpha)
    return rgba


# --- Stage 1: Vtracer pre-stage -------------------------------------------

def vtracer_pre_stage(src_path: Path, tmp_dir: Path, params: dict) -> Path:
    """Optional palette-quantize + vtracer color trace + rasterize.

    Useful for textures with originally-simple palettes that the upscaler
    polluted with hallucinated gradients/noise. Output is a clean PNG at
    the same dimensions as the source.
    """
    vtracer = find_vtracer()
    if vtracer is None:
        raise RuntimeError("vtracer.exe not found (looked in bundled dir, "
                           "PATH, and C:\\Scripts\\Vtracer-Tools).")
    magick = find_magick()
    if magick is None:
        raise RuntimeError("magick.exe not found (needed for SVG rasterize).")

    src = Path(src_path)
    orig = Image.open(src)
    orig_w, orig_h = orig.size
    orig_mode = "RGBA" if "A" in orig.getbands() else "RGB"

    # 0. Vtracer has no concept of alpha — it traces every RGB region it
    # sees, so the originally-transparent pixels (whose RGB is often garbage
    # or whatever the upscaler bled in) become a polygon and, with
    # --hierarchical stacked, the SVG's background fill. To stop that
    # leaking through (especially in Create-alpha mode where the alpha is
    # later resynthesised from this RGB), replace transparent pixels with
    # white before tracing.
    #
    # CRITICAL: use a strict binary mask (alpha == 0), not PIL's paste with
    # an alpha mask. paste(mask=alpha) does alpha-blending which would mix
    # white into anti-aliased edges (alpha 1..254) and wash out every
    # saturated colour at the silhouette. Only truly-transparent pixels
    # should be replaced; edge pixels keep their original RGB.
    #
    # When `vtracer_ignore_alpha` is set the user has opted out of letting the
    # alpha influence the trace at all: the RGB is traced exactly as-is (no
    # white-fill). The original alpha is still reattached unchanged in step 4.
    ignore_alpha = bool(params.get("vtracer_ignore_alpha", False))
    if orig_mode == "RGBA" and not ignore_alpha:
        rgba_arr = np.asarray(orig.convert("RGBA"))
        rgb_arr  = rgba_arr[:, :, :3].copy()
        a_arr    = rgba_arr[:, :, 3]
        rgb_arr[a_arr == 0] = (255, 255, 255)
        prep = Image.fromarray(rgb_arr, mode="RGB")
    else:
        prep = orig.convert("RGB")

    # 1. Pre-quantize the palette (throws away upscaler noise so vtracer
    #    doesn't trace it as detail). Uses libimagequant when available
    #    so rare-but-saturated colours survive.
    if params["vtracer_pre_quantize"] > 0:
        quant = quantize_image(prep, params["vtracer_pre_quantize"])
        vtracer_input = tmp_dir / "quantized.png"
        quant.save(vtracer_input, compress_level=1)
    else:
        vtracer_input = tmp_dir / "prepped.png"
        prep.save(vtracer_input, compress_level=1)

    # 2. Vtracer color trace → SVG. We always pass the spline-specific flags
    # too — vtracer silently ignores them when mode != spline.
    svg_path = tmp_dir / "traced.svg"
    cmd = [
        vtracer,
        "--input", str(vtracer_input),
        "--output", str(svg_path),
        "--colormode", "color",
        "--mode",              str(params["vtracer_mode"]),
        "--hierarchical",      str(params["vtracer_hierarchical"]),
        "--color_precision",   str(params["vtracer_color_precision"]),
        "--filter_speckle",    str(params["vtracer_filter_speckle"]),
        "--gradient_step",     str(params["vtracer_gradient_step"]),
        "--corner_threshold",  str(params["vtracer_corner_threshold"]),
        "--segment_length",    str(params["vtracer_segment_length"]),
        "--splice_threshold",  str(params["vtracer_splice_threshold"]),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True,
                         creationflags=_NO_WINDOW)
    if res.returncode != 0 or not svg_path.exists():
        raise RuntimeError(
            "vtracer failed:\n"
            f"  exit code: {res.returncode}\n"
            f"  stdout: {res.stdout.strip()[:500]}\n"
            f"  stderr: {res.stderr.strip()[:500]}"
        )

    # 3. Rasterize SVG → PNG at the original dimensions. We re-merge the
    #    original alpha afterwards so vtracer's RGB cleanup doesn't disturb
    #    the alpha channel (which we process separately downstream).
    rasterized = tmp_dir / "rasterized.png"
    cmd = [
        magick,
        "-background", "none",
        "-density", "200",
        str(svg_path),
        "-resize", f"{orig_w}x{orig_h}!",
        str(rasterized),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True,
                         creationflags=_NO_WINDOW)
    if res.returncode != 0 or not rasterized.exists():
        # The classic failure here was 'rsvg-convert delegate not found'
        # (delegates.xml shipped a delegate that shelled out to an external
        # rsvg-convert.exe that user machines didn't have). Build-time patch
        # to delegates.xml should prevent that — see setup.py.
        raise RuntimeError(
            "ImageMagick failed to rasterise the SVG produced by vtracer:\n"
            f"  exit code: {res.returncode}\n"
            f"  stdout: {res.stdout.strip()[:500]}\n"
            f"  stderr: {res.stderr.strip()[:500]}\n"
            f"  magick path: {magick}\n"
            "If the error mentions 'rsvg-convert' the bundled delegates.xml "
            "patch did not apply — rebuild the MSI."
        )

    # 4. Recombine the cleaned RGB with the original alpha
    cleaned_rgb = Image.open(rasterized).convert("RGB")
    if orig_mode == "RGBA":
        recombined = cleaned_rgb.convert("RGBA")
        recombined.putalpha(orig.getchannel("A"))
    else:
        recombined = cleaned_rgb
    out_path = tmp_dir / "vtracer_out.png"
    recombined.save(out_path, compress_level=1)
    return out_path


# --- Stage 2: Alpha level (Fix path) --------------------------------------

def level_alpha(alpha: Image.Image, low_pct: float, high_pct: float) -> Image.Image:
    """Stretch input range [low_pct, high_pct] to full [0, 255]. Mirrors
    ImageMagick's `-level low%,high%`."""
    low  = max(0.0, min(255.0, low_pct  * 2.55))
    high = max(0.0, min(255.0, high_pct * 2.55))
    if high <= low + 1e-6:
        # Degenerate: hard threshold at the midpoint of the (collapsed) range
        midpoint = (low + high) / 2
        lut = [0 if i < midpoint else 255 for i in range(256)]
    else:
        scale = 255.0 / (high - low)
        lut = [max(0, min(255, int(round((i - low) * scale)))) for i in range(256)]
    return alpha.point(lut)


# --- Stage 2 (alt): Synthesize alpha from RGB (Create path) ---------------

def create_alpha_from_existing(alpha: Image.Image,
                               target_fill: str,
                               fuzz_pct: float) -> Image.Image:
    """Aggressively rebuild the alpha channel from the existing alpha values.
    Mirrors the original Alphatools `3.CreateAlpha_*.bat` chain, which
    operates on the extracted alpha (NOT on RGB luminance — earlier versions
    of this app got that wrong).

    Chain:
        -level 1%,100%     (mild black-point clamp)
        -level 0%,10%      (stretch shadows: anything > 10% clips to white)
        -level 10%,100%    (re-stretch)
        -fuzz F% -fill X -opaque white    (snap near-white to target_fill)
        -fuzz F% -fill 0  -opaque black   (snap near-black to 0)  *bidirectional*

    Net effect with the default fuzz=50%: input alpha=0 stays 0; any non-zero
    input alpha gets pushed to the target plateau. Lower fuzz values leave a
    band of mid-greys around the midpoint (preserves anti-aliased transitions
    when you want a tighter outline). The bidirectional fuzz catches the
    'outside the texture' greys (near-black) that the original mono-directional
    fuzz left behind.
    """
    a = np.asarray(alpha.convert("L"), dtype=np.float32)

    # -level 1%,100% (mild black-point clamp)
    a = np.clip((a - 2.55) * (255.0 / (255.0 - 2.55)), 0, 255)

    # -level 0%,10% (stretch shadows: anything > 10% clips to white)
    a = np.clip(a * (255.0 / 25.5), 0, 255)

    # -level 10%,100% (re-stretch)
    a = np.clip((a - 25.5) * (255.0 / (255.0 - 25.5)), 0, 255)

    # Bidirectional fuzz — snap both extremes to their plateaus. Cap at 50%
    # so the two bands meet at the midpoint instead of overlapping (above 50%
    # would otherwise wipe everything since both masks would catch the same
    # values and we'd evaluate them sequentially on the modified buffer).
    effective_fuzz = min(50.0, float(fuzz_pct))
    fuzz_value = 255.0 * (effective_fuzz / 100.0)
    fill = {"gray": 128, "white": 255, "black": 0}.get(target_fill, 128)
    # Evaluate both masks on the ORIGINAL `a` so the order of `np.where`
    # calls can't cross-contaminate them.
    mask_white = a >= (255.0 - fuzz_value)
    mask_black = a <= fuzz_value
    a = np.where(mask_black, 0,    a)
    a = np.where(mask_white, fill, a)   # near-white wins ties (midpoint case)

    return Image.fromarray(a.astype(np.uint8), mode="L")


def _build_seed(region_mask: np.ndarray, lum: np.ndarray, mode: str,
                shrink_px: int, grow_pct: float) -> np.ndarray:
    """Protected, always-opaque seed within `region_mask`. Shared by the create
    and fuzzythresh paths so either seeding method is available to both.

    mode='shrink': the region eroded inward by `shrink_px` px — pulls the
                   protected core back from the silhouette boundary
                   (shrink_px=0 keeps the whole region as seed).
    mode='grow':   the brightest `grow_pct`% of the region by luminance — grows
                   the core outward from the luminance peak (0% ≈ nothing,
                   100% = the whole region).
    """
    if str(mode) == "grow":
        p = max(0.0, min(100.0, float(grow_pct)))
        if p <= 0.0 or not region_mask.any():
            return np.zeros_like(region_mask)
        if p >= 100.0:
            return region_mask.copy()
        cutoff = float(np.percentile(lum[region_mask], 100.0 - p))
        return region_mask & (lum >= cutoff)
    # shrink — spatial erosion of the region
    s = int(shrink_px)
    if s <= 0:
        return region_mask.copy()
    region_img = Image.fromarray(region_mask.astype(np.uint8) * 255, mode="L")
    return np.asarray(erode_alpha(region_img, s), dtype=np.uint8) > 128


def create_alpha_from_luminance(
    rgb: Image.Image,
    original_alpha: Image.Image,
    seed_shrink: int,
    expand_limit: int,
    lum_threshold: float,
    target_fill: str,
    lum_mode: str,
    alpha_floor: int = 0,
    lum_invert: bool = False,
    seed_mode: str = "shrink",
    seed_grow: float = 0.0,
) -> Image.Image:
    """Synthesise a new alpha from RGB luminance, anchored to the original alpha.

    Strategy:
      1. Threshold original alpha at > alpha_floor to get a binary interior mask.
         Pixels at or below alpha_floor are treated as transparent — this excludes
         upscaler-contaminated pixels whose original alpha=0 was nudged to 1–15.
      2. Build the protected interior seed via seed_mode (see _build_seed):
         'shrink' erodes the region by seed_shrink px; 'grow' keeps the
         brightest seed_grow% of the region by luminance. Seed pixels are kept
         unconditionally (never evaluated against luminance).
      3. Dilate the original mask by expand_limit px → outer growth boundary.
      4. Fringe = growth boundary minus seed.  In the fringe only, pixels whose
         luminance >= lum_threshold are included.
      5. Output: fill_value where (seed OR (fringe AND lum_pass)), else 0.
    """
    rgb_arr = np.asarray(rgb.convert("RGB"), dtype=np.float32)
    if lum_mode == "max-channel":
        lum = np.max(rgb_arr, axis=2)
    else:  # perceptual
        lum = (0.299 * rgb_arr[:, :, 0] +
               0.587 * rgb_arr[:, :, 1] +
               0.114 * rgb_arr[:, :, 2])

    orig_arr = np.asarray(original_alpha.convert("L"), dtype=np.uint8)
    # Pixels with alpha > alpha_floor are inside the original boundary.
    # alpha_floor > 0 discards upscaler-contaminated pixels whose original
    # alpha=0 was nudged slightly above zero during AI upscaling.
    region_mask = orig_arr > int(alpha_floor)
    binary_img = Image.fromarray(region_mask.astype(np.uint8) * 255, mode="L")

    seed_mask  = _build_seed(region_mask, lum, seed_mode, seed_shrink, seed_grow)
    limit_img  = dilate_alpha(binary_img, expand_limit) if expand_limit > 0 else binary_img

    limit_mask = np.asarray(limit_img, dtype=np.uint8) > 128
    fringe_mask = limit_mask & ~seed_mask

    threshold_val = float(lum_threshold) * 255.0 / 100.0
    lum_pass = (lum <= threshold_val) if lum_invert else (lum >= threshold_val)
    final_mask  = seed_mask | (fringe_mask & lum_pass)

    fill_value = 128 if target_fill == "gray" else 255
    result = np.zeros(orig_arr.shape, dtype=np.uint8)
    result[final_mask] = fill_value
    return Image.fromarray(result, mode="L")


def _box_blur_axis(a: np.ndarray, r: int, axis: int) -> np.ndarray:
    """Edge-padded 1-D box mean (window 2r+1) along one axis, via cumsum —
    O(1) per pixel regardless of radius."""
    a = np.moveaxis(a, axis, 0)
    n = a.shape[0]
    padded = np.pad(a, [(r, r)] + [(0, 0)] * (a.ndim - 1), mode="edge")
    cs = np.cumsum(padded, axis=0, dtype=np.float64)
    cs = np.concatenate([np.zeros((1,) + cs.shape[1:], dtype=cs.dtype), cs], axis=0)
    out = (cs[2 * r + 1: 2 * r + 1 + n] - cs[0:n]) / float(2 * r + 1)
    return np.moveaxis(out, 0, axis).astype(np.float32)


def _gaussian_blur_f(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur for a float32 array (3 separable box blurs ≈ Gaussian).
    PIL's GaussianBlur rejects 32-bit-float ('F') images, so we do it in numpy;
    this also avoids the precision loss of routing a huge-dynamic-range weight
    map through an 8-bit image."""
    if sigma <= 0:
        return arr.astype(np.float32)
    r = max(1, int(round(sigma)))
    a = arr.astype(np.float32)
    for _ in range(3):
        a = _box_blur_axis(a, r, 0)
        a = _box_blur_axis(a, r, 1)
    return a


def _fuzzythresh_value(gray: np.ndarray, radius: float,
                       max_iter: int = 100) -> float:
    """Fred Weinhaus fuzzythresh auto-threshold.

    Iterates T = (mu_fg + mu_bg) / 2 until T stops changing, where each class
    mean is a *weighted* mean: the weight of a pixel is the normalised inverse
    square distance of its graylevel to that class's mean, so pixels near the
    class centre dominate and outliers are suppressed. `radius` (Fred's -r)
    Gaussian-blurs the weight map so neighbouring pixels correlate spatially,
    which removes noise from the result. radius=0 → no spatial correlation.
    """
    g = gray.astype(np.float32)
    eps = 1e-6
    T = (float(g.min()) + float(g.max())) / 2.0

    def weighted_mean(mask: np.ndarray) -> float:
        mu0 = float(g[mask].mean())
        w = 1.0 / ((g - mu0) ** 2 + eps)        # inverse-square distance (whole image)
        if radius > 0:
            w = _gaussian_blur_f(w, float(radius))
        wm = w[mask]
        denom = float(wm.sum())
        return float((wm * g[mask]).sum() / denom) if denom > eps else mu0

    for _ in range(max_iter):
        fg = g > T
        if not fg.any() or fg.all():
            break
        new_T = (weighted_mean(fg) + weighted_mean(~fg)) / 2.0
        if abs(new_T - T) < 0.5:
            T = new_T
            break
        T = new_T
    return T


def create_alpha_fuzzythresh(
    rgb: Image.Image,
    original_alpha: Optional[Image.Image] = None,
    target_fill: str = "white",
    radius: float = 0,
    invert: bool = False,
    seed_grow: float = 0.0,
    expand_limit: int = 0,
    seed_mode: str = "grow",
    seed_shrink: int = 0,
    alpha_floor: int = 0,
) -> Image.Image:
    """Automatic alpha synthesis via Fred Weinhaus's fuzzythresh auto-threshold,
    anchored to the original alpha.

    The luminance threshold is found by the weighted iterative method
    (`_fuzzythresh_value`; `radius` adds spatial correlation). Pixels above it
    are opaque (below, if `invert`). The result is anchored to the source alpha:

      * region       = original alpha > alpha_floor (the source silhouette;
                       the whole image when the source is fully opaque / has no
                       alpha). alpha_floor > 0 discards upscaler-contaminated
                       pixels whose original alpha=0 was nudged to 1-15.
      * expand_limit = px to dilate the region → outer growth boundary, letting
                       the threshold recover a fringe just outside the silhouette.
      * seed_mode    = how the protected always-opaque seed is built (see
                       _build_seed): 'grow' keeps the brightest seed_grow% of the
                       region by luminance; 'shrink' erodes the region inward by
                       seed_shrink px. Seed pixels bypass the threshold test.

    Output: target_fill where seed OR (inside the boundary AND passes threshold).
    """
    gray_arr = np.asarray(rgb.convert("L"), dtype=np.float32)
    T = _fuzzythresh_value(gray_arr, float(radius))
    fuzz_pass = (gray_arr <= T) if invert else (gray_arr > T)
    fill_value = 128 if target_fill == "gray" else 255

    # Region = source silhouette (whole image when there's no real alpha).
    if original_alpha is not None:
        region = np.asarray(original_alpha.convert("L"), dtype=np.uint8) > int(alpha_floor)
    else:
        region = np.ones(gray_arr.shape, dtype=bool)
    if not region.any():
        return Image.fromarray(np.zeros(gray_arr.shape, dtype=np.uint8), mode="L")

    # Outer growth boundary = region dilated by expand_limit px.
    if expand_limit > 0:
        region_img = Image.fromarray(region.astype(np.uint8) * 255, mode="L")
        boundary = np.asarray(dilate_alpha(region_img, int(expand_limit)),
                              dtype=np.uint8) > 128
    else:
        boundary = region

    seed = _build_seed(region, gray_arr, seed_mode, seed_shrink, seed_grow)
    final_mask = seed | (boundary & fuzz_pass)
    result = np.zeros(gray_arr.shape, dtype=np.uint8)
    result[final_mask] = fill_value
    return Image.fromarray(result, mode="L")


# Colourspaces for the rangethresh path. HSV/HSL select by hue/saturation far
# better than RGB; kept numpy-only so the alpha path stays ImageMagick-free.
RANGETHRESH_COLORSPACES = ["sRGB", "HSV", "HSL"]


def _to_colorspace_arr(rgb: Image.Image, colorspace: str) -> np.ndarray:
    """RGB image → HxWx3 uint8 array in the requested colourspace (each channel
    scaled 0-255)."""
    cs = str(colorspace).upper()
    if cs == "HSV":
        return np.asarray(rgb.convert("HSV"), dtype=np.uint8)
    if cs == "HSL":
        arr = np.asarray(rgb.convert("RGB"), dtype=np.float32) / 255.0
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        mx = arr.max(axis=2); mn = arr.min(axis=2)
        diff = mx - mn
        L = (mx + mn) / 2.0
        S = np.where(diff == 0, 0.0,
                     diff / (1.0 - np.abs(2.0 * L - 1.0) + 1e-12))
        H = np.zeros_like(mx)
        nz = diff != 0
        rmax = nz & (mx == r)
        gmax = nz & (mx == g) & ~rmax
        bmax = nz & (mx == b) & ~rmax & ~gmax
        H[rmax] = ((g - b)[rmax] / diff[rmax]) % 6.0
        H[gmax] = ((b - r)[gmax] / diff[gmax]) + 2.0
        H[bmax] = ((r - g)[bmax] / diff[bmax]) + 4.0
        H = (H / 6.0) % 1.0
        out = np.stack([H, np.clip(S, 0, 1), np.clip(L, 0, 1)], axis=2) * 255.0
        return np.clip(out, 0, 255).astype(np.uint8)
    return np.asarray(rgb.convert("RGB"), dtype=np.uint8)   # sRGB (default)


def create_alpha_rangethresh(
    rgb: Image.Image,
    original_alpha: Optional[Image.Image] = None,
    r_range: "tuple[int, int]" = (0, 255),
    g_range: "tuple[int, int]" = (0, 255),
    b_range: "tuple[int, int]" = (0, 255),
    unit: str = "8bit",
    colorspace: str = "sRGB",
    target_fill: str = "white",
    seed_mode: str = "grow",
    seed_shrink: int = 0,
    seed_grow: float = 0.0,
    expand_limit: int = 0,
    alpha_floor: int = 0,
) -> Image.Image:
    """Colour-range alpha synthesis (Fred Weinhaus rangethresh), anchored to the
    original alpha.

    A pixel passes (becomes opaque) when ALL THREE channels of the chosen
    colourspace fall within their [low, high] ranges. Anchoring mirrors the
    fuzzythresh path: a protected seed (seed_mode) plus an expand_limit growth
    boundary; the range test only decides the non-seed pixels inside it.

    r/g/b_range — (low, high) for channels 1/2/3 of the colourspace
        (R,G,B for sRGB; H,S,V for HSV; H,S,L for HSL).
    unit — '8bit' (ranges 0-255) or 'percent' (0-100, scaled to 0-255).
    alpha_floor — original alpha <= this is treated as transparent (discards
        upscaler-contaminated near-zero alpha).
    """
    cs = _to_colorspace_arr(rgb, colorspace)                 # HxWx3 uint8
    lum = np.asarray(rgb.convert("L"), dtype=np.float32)      # grow-mode seed
    sc = 2.55 if str(unit).lower() == "percent" else 1.0

    def _chan_pass(idx, rng):
        v = cs[..., idx].astype(np.float32)
        return (v >= float(rng[0]) * sc) & (v <= float(rng[1]) * sc)

    range_pass = (_chan_pass(0, r_range)
                  & _chan_pass(1, g_range)
                  & _chan_pass(2, b_range))
    fill_value = 128 if target_fill == "gray" else 255

    if original_alpha is not None:
        region = np.asarray(original_alpha.convert("L"), dtype=np.uint8) > int(alpha_floor)
    else:
        region = np.ones(range_pass.shape, dtype=bool)
    if not region.any():
        return Image.fromarray(np.zeros(range_pass.shape, dtype=np.uint8), mode="L")

    if expand_limit > 0:
        region_img = Image.fromarray(region.astype(np.uint8) * 255, mode="L")
        boundary = np.asarray(dilate_alpha(region_img, int(expand_limit)),
                              dtype=np.uint8) > 128
    else:
        boundary = region

    seed = _build_seed(region, lum, seed_mode, seed_shrink, seed_grow)
    final_mask = seed | (boundary & range_pass)
    result = np.zeros(range_pass.shape, dtype=np.uint8)
    result[final_mask] = fill_value
    return Image.fromarray(result, mode="L")


# --- Stage 3: N-level alpha snap ------------------------------------------

def n_level_snap(alpha: Image.Image,
                 n_levels: int,
                 low_thresh: int,
                 high_thresh: int,
                 mid_tolerance: int) -> Image.Image:
    """Snap alpha values back to quantised plateaus, preserving the edge
    gradients between plateaus.

    For default n_levels=3 the plateaus are [0, 128, 255]:
      * input <= low_thresh         -> 0
      * |input - 128| <= mid_tol    -> 128
      * input >= high_thresh        -> 255
      * everything else             -> unchanged  (the edge transition)

    For n_levels=N>3, plateaus are evenly spaced between 0 and 255, and the
    intermediate ones each use a ±mid_tolerance band.
    """
    if n_levels < 2:
        return alpha

    plateaus = np.linspace(0, 255, n_levels).round().astype(np.int32)
    inputs = np.arange(256, dtype=np.int32)
    lut = inputs.astype(np.uint8).copy()  # identity by default

    # Apply the intermediate plateaus first, then the transparent (0) and
    # opaque (255) clamps last so the clamps always win where bands overlap.
    # Otherwise a wide mid_tolerance reaching down into the low_thresh band
    # (reachable from the GUI: low_thresh up to 127, mid_tolerance up to 64)
    # would re-stamp fully-transparent pixels as semi-transparent (128),
    # contradicting the documented "input <= low_thresh -> 0" rule.
    for plateau in plateaus:
        if plateau == 0 or plateau == 255:
            continue
        mask = np.abs(inputs - int(plateau)) <= mid_tolerance
        lut[mask] = int(plateau)

    # Low/high clamps take precedence over any intermediate band.
    lut[inputs <= low_thresh] = 0
    lut[inputs >= high_thresh] = 255

    arr = np.asarray(alpha, dtype=np.uint8)
    return Image.fromarray(lut[arr], mode="L")


# --- Stage 4: Edge refinement ---------------------------------------------

def gaussian_blur_alpha(alpha: Image.Image, radius: float) -> Image.Image:
    if radius <= 0:
        return alpha
    return alpha.filter(ImageFilter.GaussianBlur(radius=radius))


def despeckle_alpha(alpha: Image.Image, strength: int) -> Image.Image:
    """Median-filter the alpha to remove orphan noise pixels. `strength` is
    the number of 3x3 median passes."""
    if strength <= 0:
        return alpha
    out = alpha
    for _ in range(strength):
        out = out.filter(ImageFilter.MedianFilter(size=3))
    return out


def binary_threshold(alpha: Image.Image, threshold: int) -> Image.Image:
    """Force alpha to pure 0 or 255 at the given threshold."""
    lut = [0 if i < threshold else 255 for i in range(256)]
    return alpha.point(lut)


def tri_clamp_alpha(
    alpha: Image.Image,
    lo_thresh: int = 64,
    hi_thresh: int = 192,
    mid: int = 128,
) -> Image.Image:
    """Snap all alpha values to exactly 0, `mid`, or 255.

    Values below lo_thresh → 0 (transparent).
    Values above hi_thresh → 255 (opaque).
    Everything in between    → mid (semi-transparent; default 128).

    Useful for PS1-style textures where the hardware supports three
    distinct alpha states: fully transparent, semi-transparent, and opaque.
    The semi level is configurable because emulators dump it differently
    (DuckStation 143, Beetle PSX HW 127).
    """
    lut = [
        0 if i < lo_thresh else (255 if i > hi_thresh else int(mid))
        for i in range(256)
    ]
    return alpha.point(lut)


def ps1_alpha_snap(
    alpha: Image.Image,
    semi_value: int = 143,
    semi_tolerance: int = 40,
    region_min: int = 2,
    binary_thresh: int = 128,
) -> Image.Image:
    """Quantise alpha to exactly {0, semi_value, 255} for PS1 HD-replacement
    textures, keeping flat semi-transparent *surfaces* at semi_value while
    forcing cutout-edge gradients to binary 0/255.

    PS1 emulators treat any non-opaque alpha as the per-primitive
    semi-transparency (STP) flag — DuckStation explicitly maps any alpha < 255
    to "blend this texel". So the upscaler's anti-aliased edges (partial alpha)
    would otherwise become unintended blended halos on re-injection. We isolate
    genuinely-flat semi regions from edge gradients with a morphological
    *opening* of the near-semi mask: thin bands where a 0->255 edge merely
    passes through the semi value are eroded away, while thick flat semi blobs
    survive. Everything not in a flat semi region is hard-thresholded to 0/255,
    guaranteeing no partial alpha leaks into the output.

    semi_value      target for semi pixels (DuckStation 143, Beetle 127)
    semi_tolerance  +/- band around semi_value counted as "near semi"
    region_min      opening radius in px; larger removes wider false-semi edges
    binary_thresh   non-semi pixels >= this -> 255, else 0
    """
    arr = np.asarray(alpha.convert("L"))
    sv = int(semi_value)

    # Candidate semi pixels, then morphological opening to drop thin edge bands.
    near = np.abs(arr.astype(np.int16) - sv) <= int(semi_tolerance)
    near_u8 = near.astype(np.uint8) * 255
    r = int(region_min)
    if r > 0:
        opened = _morph_np(near_u8, r, np.minimum)   # erode
        opened = _morph_np(opened,  r, np.maximum)   # dilate -> opening
    else:
        opened = near_u8
    semi_region = opened > 128

    out = np.where(arr >= int(binary_thresh), 255, 0).astype(np.uint8)
    out[semi_region] = np.uint8(sv)
    return Image.fromarray(out, mode="L")


# --- Stage 4b: Alpha shape (erode / dilate the alpha mask) ----------------

def _morph_1d(x: np.ndarray, r: int, op) -> np.ndarray:
    """Min/max filter over a (2r+1)-wide window along the last axis, with
    edge replication at the borders. `op` is np.minimum (erode) or np.maximum
    (dilate)."""
    n = x.shape[-1]
    xp = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(r, r)], mode="edge")
    out = xp[..., 0:n].copy()
    for k in range(1, 2 * r + 1):
        out = op(out, xp[..., k:k + n])
    return out


def _morph_np(arr: np.ndarray, r: int, op) -> np.ndarray:
    """Separable square min/max filter on a 2-D numpy array (Chebyshev radius
    r). One 1-D pass per axis; `op` is np.minimum (erode) or np.maximum
    (dilate)."""
    if r <= 0:
        return arr
    out = _morph_1d(arr, r, op)                       # horizontal pass
    out = _morph_1d(np.moveaxis(out, 0, -1), r, op)   # vertical pass
    return np.moveaxis(out, -1, 0)


def _separable_morph(alpha: Image.Image, iterations: int, op) -> Image.Image:
    """Separable square morphology. N iterations of a 3x3 PIL Min/MaxFilter
    are mathematically a single erosion/dilation by a (2N+1)x(2N+1) square
    structuring element (Chebyshev radius N). A square SE is separable, so we
    run a 1D pass along each axis instead of N full 3x3 filter passes —
    O(radius) vectorised numpy ops rather than O(radius) C-level rank-filter
    passes. This is byte-for-byte identical to the old iterated PIL filters
    (verified against random inputs) but ~7x faster, and the speedup grows
    with the radius (sliders reach 100 px).
    """
    r = int(iterations)
    if r <= 0:
        return alpha
    out = _morph_np(np.asarray(alpha.convert("L")), r, op)
    return Image.fromarray(out, mode="L")


def erode_alpha(alpha: Image.Image, iterations: int) -> Image.Image:
    """Shrink the alpha mask by `iterations` pixels around its boundary —
    contracts bright (opaque) regions by one pixel per iteration (3x3 square
    SE, equivalent to the previous iterated MinFilter).

    Use to 'pull the outline back' when the alpha covers more than the
    intended silhouette.
    """
    return _separable_morph(alpha, iterations, np.minimum)


def dilate_alpha(alpha: Image.Image, iterations: int) -> Image.Image:
    """Grow the alpha mask by `iterations` pixels around its boundary. The
    inverse of erosion. Use to 'fill out the outline' when the alpha is
    tighter than the intended silhouette.
    """
    return _separable_morph(alpha, iterations, np.maximum)


# --- Stage 5: RGB edge dilation (alpha bleed) -----------------------------

def dilate_rgb(rgb: Image.Image,
               alpha: Image.Image,
               iterations: int,
               source_threshold: int = 16) -> Image.Image:
    """Bleed opaque RGB outward into transparent regions, one pixel per
    iteration. Prevents the dark/coloured halos that appear when an upscaled
    texture is bilinear-filtered or mip-mapped in-engine.

    `source_threshold` (0-255) is the alpha value above which a pixel is
    treated as a trustworthy colour source. Pixels at or below it are
    considered transparent and get their colour bled in. Default 16 ignores
    upscaler smear in nominally-transparent regions.
    """
    if iterations <= 0:
        return rgb

    rgb_arr = np.asarray(rgb.convert("RGB"), dtype=np.uint8).copy()
    mask = np.asarray(alpha.convert("L"), dtype=np.uint8) > source_threshold

    h, w = mask.shape
    for _ in range(iterations):
        # Nothing left to bleed into — the mask covers every pixel. Further
        # iterations can't change anything, so stop early instead of paying
        # for more full-array pads/copies (cheap guard, big win when the
        # transparent region is small or the bleed has already converged).
        if mask.all():
            break

        # Pad: mask edges are False (so we never bleed FROM outside);
        # RGB edges are copies of the inside (irrelevant — never sampled).
        padded_mask = np.pad(mask, 1, mode="constant", constant_values=False)
        padded_rgb  = np.pad(rgb_arr, ((1, 1), (1, 1), (0, 0)), mode="edge")

        new_mask = mask.copy()
        new_rgb  = rgb_arr.copy()

        # Each transparent pixel inherits colour from the first opaque
        # 4-neighbour we find (up, down, left, right). Order is arbitrary but
        # deterministic, which is fine — bleed is iterative anyway.
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbour_mask = padded_mask[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
            neighbour_rgb  = padded_rgb[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
            fill_here = (~new_mask) & neighbour_mask
            new_rgb[fill_here] = neighbour_rgb[fill_here]
            new_mask |= fill_here

        # No pixel was filled this pass — the bleed front has stalled (e.g.
        # an isolated transparent region with no opaque neighbours). Stop.
        if not np.any(new_mask & ~mask):
            rgb_arr = new_rgb
            break

        rgb_arr = new_rgb
        mask = new_mask

    return Image.fromarray(rgb_arr, mode="RGB")


# --- Stage 6a: Tone adjustment (saturation / contrast / brightness) -------

def adjust_intensity(rgb: Image.Image,
                     saturation: float,
                     contrast: float,
                     brightness: float) -> Image.Image:
    """Restore vibrance / tonal range to an upscaled / cleaned texture.
    All factors centred on 1.0 (no change). Order: saturation, contrast,
    brightness — applied independently via PIL.ImageEnhance."""
    out = rgb
    if abs(saturation - 1.0) > 1e-3:
        out = ImageEnhance.Color(out).enhance(float(saturation))
    if abs(contrast - 1.0) > 1e-3:
        out = ImageEnhance.Contrast(out).enhance(float(contrast))
    if abs(brightness - 1.0) > 1e-3:
        out = ImageEnhance.Brightness(out).enhance(float(brightness))
    return out


def level_colors_rgb(rgb: Image.Image, level_r: float, level_g: float,
                     level_b: float) -> Image.Image:
    """Per-channel colour level adjustment (Fred Weinhaus levelcolors, RGB
    only — alpha handled elsewhere). Each value is in [-100, 100]:

      > 0  brightens the channel  — pulls the white point down to (1 - t),
           so out = in / (1 - t)  (more of that colour).
      < 0  darkens the channel    — raises the black point to |t|,
           so out = (in - |t|) / (1 - |t|)  (less of that colour).
      = 0  no change.

    This is ImageMagick's per-channel `-level lo,hi` with lo/hi derived from
    the slider, exactly as levelcolors applies -level channel-by-channel.
    """
    vals = (float(level_r), float(level_g), float(level_b))
    if all(abs(v) < 1e-6 for v in vals):
        return rgb
    arr = np.asarray(rgb.convert("RGB"), dtype=np.float32) / 255.0
    out = arr.copy()
    for c, v in enumerate(vals):
        t = max(-100.0, min(100.0, v)) / 100.0
        if abs(t) < 1e-6:
            continue
        if t > 0:                                   # brighten (white point down)
            hi = max(1.0 - t, 1e-6)
            out[..., c] = np.clip(arr[..., c] / hi, 0.0, 1.0)
        else:                                       # darken (black point up)
            lo = -t
            out[..., c] = np.clip((arr[..., c] - lo) / max(1.0 - lo, 1e-6),
                                  0.0, 1.0)
    return Image.fromarray((out * 255.0 + 0.5).astype(np.uint8), mode="RGB")


# --- Stage 6b: Pixelate ----------------------------------------------------

def pixelate(rgba: Image.Image, factor: float,
             block: "Optional[tuple]" = None) -> Image.Image:
    """Re-pixelate an upscaled texture: downsample by `factor` then upsample
    back to original size, both with nearest-neighbour. factor<=1 is a no-op.

    `factor` may be fractional (e.g. 1.5) for a milder effect between integer
    steps: the intermediate size is rounded, so blocks come out a mix of
    floor/ceil widths rather than a uniform NxN grid — a gentler pixelation
    than the next whole factor.

    `block` = (block_w, block_h) overrides the block SIZE (the resize ratio)
    instead of deriving it from this image's own dimensions. The ROI full-res
    preview passes the WHOLE image's block size so a crop pixelates on the same
    grid as the full image (otherwise the crop's own width gives a slightly
    different ratio and the blocks drift out of alignment).

    Operates on the composed RGBA so RGB and alpha are pixelated together
    — otherwise you get pixelated colour with smooth alpha edges (or vice
    versa), which looks wrong.
    """
    if factor <= 1:
        return rgba
    w, h = rgba.size
    if block is not None:
        sw = max(1, int(round(w / block[0])))
        sh = max(1, int(round(h / block[1])))
    else:
        sw = max(1, int(round(w / factor)))
        sh = max(1, int(round(h / factor)))
    small = rgba.resize((sw, sh), resample=Image.Resampling.NEAREST)
    return small.resize((w, h), resample=Image.Resampling.NEAREST)


def soften_edges(rgb: Image.Image, alpha: Image.Image, radius: int) -> Image.Image:
    """Gaussian-soften the RGB in a band along the alpha outline only; the
    alpha channel and the texture interior are left untouched. radius=0 is a
    no-op.

    The outline is isolated from the alpha silhouette: a band of ~`radius` px
    just inside the boundary is where the softening is applied, feathered so
    the blend ramps smoothly and the interior stays crisp. The blur itself is
    alpha-weighted (normalised convolution) so transparent-area fill colours
    can't bleed into the visible edge and create a dark halo.

    radius sets both the blur strength and the width of the softened band.
    """
    r = int(radius)
    if r <= 0:
        return rgb
    rgb_arr = np.asarray(rgb.convert("RGB"), dtype=np.float32)
    a = np.asarray(alpha.convert("L"), dtype=np.uint8)
    sil = a > 0
    if not sil.any() or sil.all():
        # No outline to soften (fully transparent or fully opaque).
        return rgb

    sigma = float(r)
    w = sil.astype(np.float32)
    # Alpha-weighted (normalised) blur: only in-silhouette colour contributes.
    num = _gaussian_blur_f(rgb_arr * w[:, :, None], sigma)
    den = _gaussian_blur_f(w, sigma)[:, :, None]
    blurred = num / np.maximum(den, 1e-6)

    # Edge band = a linear ramp from 1 at the boundary to 0 at `r` px inside,
    # built by accumulating single-pixel erosions. A pixel `d` px in from the
    # edge survives `d` erosions, giving depth=min(d,r); band = 1 - depth/r.
    # This bounds the softened region to exactly `r` px (no interior bleed) and
    # ramps smoothly. (A blurred ring would overshoot far past the band.)
    cur_img = Image.fromarray(sil.astype(np.uint8) * 255, mode="L")
    depth = np.zeros(sil.shape, dtype=np.float32)
    for _ in range(r):
        cur_img = erode_alpha(cur_img, 1)
        depth += (np.asarray(cur_img, dtype=np.uint8) > 128).astype(np.float32)
    band = np.clip(1.0 - depth / float(r), 0.0, 1.0) * w
    band = band[:, :, None]

    out = rgb_arr * (1.0 - band) + blurred * band
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


# --- Stage 7: Premultiply --------------------------------------------------

def premultiply_rgb(rgb: Image.Image, alpha: Image.Image) -> Image.Image:
    rgb_arr = np.asarray(rgb.convert("RGB"), dtype=np.float32)
    a = np.asarray(alpha.convert("L"), dtype=np.float32) / 255.0
    pre = rgb_arr * a[:, :, np.newaxis]
    return Image.fromarray(np.clip(pre, 0, 255).astype(np.uint8), mode="RGB")


# --- Full pipeline ---------------------------------------------------------

@contextmanager
def _maybe_tempdir(supplied: Optional[Path]):
    if supplied is not None:
        yield Path(supplied)
    else:
        with tempfile.TemporaryDirectory(prefix="alphatools_") as t:
            yield Path(t)


def _clear_existing_output(path: Path) -> None:
    """Remove an existing output file so a subsequent save is a guaranteed
    clean overwrite. Handles the Windows read-only case (unlink raises
    PermissionError; clearing the read-only bit then retrying succeeds). A
    genuinely locked file (open in another app) still can't be removed — the
    caller's save will then surface a clear error rather than silently keeping
    the stale file."""
    if not path.exists():
        return
    try:
        path.unlink()
    except PermissionError:
        import stat
        os.chmod(path, stat.S_IWRITE)
        path.unlink()


# --- Incongruence edge refine (colour-aware halo removal) -------------------
#
# AI upscaling bleeds the (often black) fill under transparent regions into the
# opaque edge, leaving a dark fringe that luminance thresholds can't separate
# from genuinely dark content. This stage walks inward from the transparency
# border and reclassifies fringe pixels by COLOUR INCONGRUENCE instead:
# a boundary pixel is eaten while it differs from the local interior content,
# and survives when it matches that content, when crossing it would step over a
# sharp colour line (a genuine outline), or when no interior reference exists
# nearby (thin feature). Where fringe and content are the same colour (e.g. a
# shadowed edge) nothing is eaten — and nothing needs to be, since the fringe
# is invisible there.

def _dilate_bool(m: np.ndarray) -> np.ndarray:
    """4-neighbour dilation of a boolean mask."""
    g = m.copy()
    g[1:, :] |= m[:-1, :]; g[:-1, :] |= m[1:, :]
    g[:, 1:] |= m[:, :-1]; g[:, :-1] |= m[:, 1:]
    return g


def _box_blur_f(a: np.ndarray, r: int) -> np.ndarray:
    """Mean of a single-channel array over a (2r+1) box, via an integral image.
    Used to smooth the opacity mask so its gradient gives a stable inward
    normal across the whole peel band."""
    H, W = a.shape
    ii = np.zeros((H + 1, W + 1), np.float64)
    ii[1:, 1:] = a.cumsum(0).cumsum(1)
    ys, xs = np.arange(H), np.arange(W)
    y0, y1 = np.clip(ys - r, 0, H), np.clip(ys + r + 1, 0, H)
    x0, x1 = np.clip(xs - r, 0, W), np.clip(xs + r + 1, 0, W)
    tot = ii[np.ix_(y1, x1)] - ii[np.ix_(y0, x1)] - ii[np.ix_(y1, x0)] + ii[np.ix_(y0, x0)]
    cnt = ((y1 - y0)[:, None] * (x1 - x0)[None, :]).astype(np.float64)
    return tot / cnt


def incongruence_refine_alpha(
    rgb_img: Image.Image,
    alpha_img: Image.Image,
    reach: int = 4,
    content_thresh: float = 35.0,
    line_thresh: float = 90.0,
    smooth: bool = True,
    alpha_floor: int = 16,
) -> Image.Image:
    """Colour-aware fringe removal: peel the alpha boundary inward up to
    `reach` px, eating pixels incongruent with the local interior content.

      reach          — max pixels eaten inward from the transparency border
      content_thresh — boundary pixel within this Euclidean RGB distance of
                       the local interior mean = content reached, stop
      line_thresh    — step from the just-eaten outward pixel above this =
                       sharp feature line (genuine outline), stop
      smooth         — majority-filter the eaten mask (2 passes) so per-pixel
                       decisions don't leave a jagged silhouette; only applied
                       within the modified band, never adds opacity where the
                       original alpha was transparent
      alpha_floor    — pixels with alpha > this form the subject silhouette to
                       peel from. The transparency border is the OUTER edge of
                       that silhouette. Must sit BELOW a semi-transparent grey
                       subject's alpha (e.g. PS1 127/143), or that subject is
                       split into an opaque/transparent speckle and the peel
                       eats its interior. Default 16 (also drops upscaler
                       contamination); set near 128 only for a strictly
                       opaque-vs-transparent source.

    Eaten pixels get alpha 0; everything else keeps its original alpha value
    (semi-transparency preserved).
    """
    rgb = np.asarray(rgb_img.convert("RGB"), dtype=np.float32)
    a = np.asarray(alpha_img.convert("L"), dtype=np.uint8)
    opaque = a > int(alpha_floor)
    reach = int(reach)
    if reach <= 0 or not opaque.any() or opaque.all():
        return alpha_img

    # Interior colour reference, sampled DIRECTIONALLY (inward along the local
    # normal) rather than over a flat window. A window bleeds sideways across
    # colour boundaries — e.g. a pure-green stripe a few px away contaminates a
    # dark tile's reference and makes valid content look like halo. Walking
    # inward follows the geometry, so the reference is the content actually
    # behind each boundary pixel. Pixels whose inward ray never lands on opaque
    # content (thin features) have no reference and are frozen.
    H, W = opaque.shape
    soft = _box_blur_f(opaque.astype(np.float64), max(2, reach))
    gy, gx = np.gradient(soft)                         # points toward interior
    mag = np.sqrt(gy * gy + gx * gx) + 1e-6
    ny, nx = gy / mag, gx / mag
    Y, X = np.mgrid[0:H, 0:W]
    ref_acc = np.zeros((H, W, 3), np.float64)
    ref_cnt = np.zeros((H, W), np.float64)
    for t in (reach, reach + 1, reach + 2):            # sample just past the band
        sy = np.clip(np.round(Y + ny * t).astype(np.int32), 0, H - 1)
        sx = np.clip(np.round(X + nx * t).astype(np.int32), 0, W - 1)
        so = opaque[sy, sx]
        sc = rgb[sy, sx]
        ref_acc[so] += sc[so]
        ref_cnt[so] += 1.0
    has_ref = ref_cnt > 0
    ref = np.zeros((H, W, 3), np.float32)
    ref[has_ref] = (ref_acc[has_ref] / ref_cnt[has_ref][..., None]).astype(np.float32)
    d_interior = np.sqrt(((rgb - ref) ** 2).sum(-1))

    # Fill colour (what bleeds in under the transparency — usually near-black).
    # A halo intrudes TOWARD this colour, so a pixel is only halo if it is
    # closer to the fill than its interior reference is. This stops valid
    # content brighter/farther-from-fill than a contaminated reference from
    # being eaten where there is no actual halo (clean or thin edges).
    border_trans = (~opaque) & _dilate_bool(opaque)
    if border_trans.any():
        fill = rgb[border_trans].mean(0)
    else:
        fill = np.zeros(3, np.float32)
    FILL_MARGIN = 10.0
    d_px_fill = np.sqrt(((rgb - fill) ** 2).sum(-1))
    d_ref_fill = np.sqrt(((ref - fill) ** 2).sum(-1))
    toward_fill = d_px_fill + FILL_MARGIN < d_ref_fill

    trans = ~opaque
    frozen = np.zeros_like(opaque)
    eaten = np.zeros_like(opaque)
    eaten_col = np.zeros_like(rgb)
    for _layer in range(reach):
        cand = opaque & ~trans & ~frozen & _dilate_bool(trans)
        if not cand.any():
            break
        # Sharp-step (outline) guard: max colour step to any eaten 4-neighbour.
        step = np.zeros(opaque.shape, dtype=np.float32)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nb_eaten = np.zeros_like(eaten)
            nb_col = np.zeros_like(eaten_col)
            ys = slice(max(dy, 0), opaque.shape[0] + min(dy, 0))
            yd = slice(max(-dy, 0), opaque.shape[0] + min(-dy, 0))
            xs = slice(max(dx, 0), opaque.shape[1] + min(dx, 0))
            xd = slice(max(-dx, 0), opaque.shape[1] + min(-dx, 0))
            nb_eaten[yd, xd] = eaten[ys, xs]
            nb_col[yd, xd] = eaten_col[ys, xs]
            d = np.sqrt(((rgb - nb_col) ** 2).sum(-1))
            step = np.where(nb_eaten, np.maximum(step, d), step)

        freeze_now = cand & (
            ~has_ref                                    # thin feature
            | (d_interior <= float(content_thresh))     # content reached
            | (step > float(line_thresh))               # genuine outline
            | ~toward_fill                               # not moving toward the
                                                        # fill colour → not halo
        )
        eat_now = cand & ~freeze_now
        frozen |= freeze_now
        trans |= eat_now
        eaten |= eat_now
        eaten_col[eat_now] = rgb[eat_now]

    if not eaten.any():
        return alpha_img

    final_opaque = ~trans
    if smooth:
        # Majority-smooth the post-eat opacity, applied only within the
        # modified band; never adds opacity where original alpha was 0.
        sm = final_opaque.copy()
        for _ in range(2):
            s = np.zeros(sm.shape, dtype=np.int32)
            mp = np.pad(sm.astype(np.int32), 1, mode="edge")
            for dy in (0, 1, 2):
                for dx in (0, 1, 2):
                    s += mp[dy:dy + sm.shape[0], dx:dx + sm.shape[1]]
            sm = s >= 5
        band = _dilate_bool(eaten)
        final_opaque = np.where(band, sm & opaque, final_opaque)

    out = np.where(final_opaque, a, 0).astype(np.uint8)
    return Image.fromarray(out, mode="L")


# --- Pipeline stages (shared by process_pil and the preview stage cache) ---
#
# Each stage is a pure function of its inputs + params (no in-place mutation of
# its arguments). process_pil runs them top-to-bottom; PreviewPipeline runs the
# exact same functions but memoises each one, so an unchanged early stage isn't
# recomputed when only a later parameter moved. Keeping the stage bodies in one
# place guarantees the cached preview and the batch/file path can never drift.

def _stage_load(img: Image.Image, params: dict, tmp_dir: Path):
    """Optional vtracer pre-stage, then split into base RGB + original alpha."""
    if params["vtracer_enabled"]:
        vt_src = tmp_dir / "vtracer_src.png"
        img.save(vt_src, format="PNG", compress_level=1)
        img = load_rgba(vtracer_pre_stage(vt_src, tmp_dir, params))
    return split_rgba(img)


def _stage_build_alpha(rgb_base: Image.Image, original_alpha: Image.Image,
                       params: dict) -> Image.Image:
    if params["alpha_source"] == "fix":
        return level_alpha(
            original_alpha, params["fix_level_low"], params["fix_level_high"],
        )
    elif params["alpha_source"] == "fuzzythresh":
        return create_alpha_fuzzythresh(
            rgb_base,
            original_alpha,
            target_fill  = str(params["fuzzythresh_target"]),
            radius       = float(params["fuzzythresh_radius"]),
            invert       = bool(params["fuzzythresh_invert"]),
            seed_mode    = str(params["fuzzythresh_seed_mode"]),
            seed_grow    = float(params["fuzzythresh_seed_grow"]),
            seed_shrink  = int(params["fuzzythresh_seed_shrink"]),
            expand_limit = int(params["fuzzythresh_expand_limit"]),
            alpha_floor  = int(params["fuzzythresh_alpha_floor"]),
        )
    elif params["alpha_source"] == "rangethresh":
        return create_alpha_rangethresh(
            rgb_base,
            original_alpha,
            r_range      = (int(params["rangethresh_r_low"]), int(params["rangethresh_r_high"])),
            g_range      = (int(params["rangethresh_g_low"]), int(params["rangethresh_g_high"])),
            b_range      = (int(params["rangethresh_b_low"]), int(params["rangethresh_b_high"])),
            unit         = str(params["rangethresh_unit"]),
            colorspace   = str(params["rangethresh_colorspace"]),
            target_fill  = str(params["rangethresh_target"]),
            seed_mode    = str(params["rangethresh_seed_mode"]),
            seed_shrink  = int(params["rangethresh_seed_shrink"]),
            seed_grow    = float(params["rangethresh_seed_grow"]),
            expand_limit = int(params["rangethresh_expand_limit"]),
            alpha_floor  = int(params["rangethresh_alpha_floor"]),
        )
    return create_alpha_from_luminance(
        rgb_base,
        original_alpha,
        int(params["create_seed_shrink"]),
        int(params["create_expand_limit"]),
        float(params["create_lum_threshold"]),
        str(params["create_target"]),
        str(params["create_lum_mode"]),
        alpha_floor=int(params["create_alpha_floor"]),
        lum_invert=bool(params["create_lum_invert"]),
        seed_mode=str(params["create_seed_mode"]),
        seed_grow=float(params["create_seed_grow"]),
    )


def _stage_incong_refine(rgb_base: Image.Image, work_alpha: Image.Image,
                         params: dict) -> Image.Image:
    # Colour-aware halo removal. Runs on the crisp post-synthesis alpha and
    # BEFORE blur/despeckle: it needs a sharp boundary to walk from, and the
    # later stages then soften/shape the corrected edge. The first refine
    # stage that reads RGB — the others operate on the alpha alone.
    if params.get("incong_enabled", False) and int(params.get("incong_reach", 0)) > 0:
        work_alpha = incongruence_refine_alpha(
            rgb_base, work_alpha,
            reach=int(params["incong_reach"]),
            content_thresh=float(params["incong_content_thresh"]),
            line_thresh=float(params["incong_line_thresh"]),
            smooth=bool(params["incong_smooth"]),
            alpha_floor=int(params.get("incong_alpha_floor", 16)),
        )
    return work_alpha


def _stage_edge_refine(work_alpha: Image.Image, params: dict) -> Image.Image:
    # Blur + despeckle run BEFORE snap so the gaussian smooths the raw upscaled
    # alpha first; snap then quantises the already-clean edge into crisp
    # plateaus rather than re-softening values snap just committed.
    if params["blur_radius"] > 0:
        work_alpha = gaussian_blur_alpha(work_alpha, params["blur_radius"])
    if params["despeckle"] > 0:
        work_alpha = despeckle_alpha(work_alpha, params["despeckle"])
    return work_alpha


def _stage_snap(work_alpha: Image.Image, params: dict) -> Image.Image:
    if params["snap_enabled"]:
        work_alpha = n_level_snap(
            work_alpha,
            params["snap_levels"],
            params["snap_low_threshold"],
            params["snap_high_threshold"],
            params["snap_mid_tolerance"],
        )
    return work_alpha


def _stage_clamp(work_alpha: Image.Image, params: dict) -> Image.Image:
    clamp_mode = str(params.get("clamp_mode", "none"))
    if clamp_mode == "binary":
        work_alpha = binary_threshold(
            work_alpha, int(params.get("binary_threshold", 128)),
        )
    elif clamp_mode == "tri-clamp":
        work_alpha = tri_clamp_alpha(
            work_alpha,
            int(params.get("tri_clamp_lo",  64)),
            int(params.get("tri_clamp_hi", 192)),
        )
    elif clamp_mode == "ps1":
        work_alpha = ps1_alpha_snap(
            work_alpha,
            semi_value=int(params.get("ps1_semi_value", 143)),
            semi_tolerance=int(params.get("ps1_semi_tolerance", 40)),
            region_min=int(params.get("ps1_region_min", 2)),
            binary_thresh=int(params.get("binary_threshold", 128)),
        )
    return work_alpha


def _stage_alpha_shape(work_alpha: Image.Image, params: dict) -> Image.Image:
    if params["alpha_erosion"] > 0:
        work_alpha = erode_alpha(work_alpha, params["alpha_erosion"])
    if params["alpha_dilation"] > 0:
        work_alpha = dilate_alpha(work_alpha, params["alpha_dilation"])
    return work_alpha


def _stage_rgb_dilate(rgb_base: Image.Image, work_alpha: Image.Image,
                      params: dict) -> Image.Image:
    # Dilates against the *cleaned* alpha so the "valid colour" region matches
    # the final cutout.
    if params["edge_dilation"] > 0:
        return dilate_rgb(
            rgb_base, work_alpha,
            params["edge_dilation"],
            source_threshold=params["edge_dilation_threshold"],
        )
    return rgb_base


def _stage_tone(rgb: Image.Image, params: dict) -> Image.Image:
    out = adjust_intensity(
        rgb, params["saturation"], params["contrast"], params["brightness"],
    )
    return level_colors_rgb(
        out, params.get("level_r", 0), params.get("level_g", 0),
        params.get("level_b", 0),
    )


def _stage_premultiply(rgb: Image.Image, work_alpha: Image.Image,
                       params: dict) -> Image.Image:
    if params["premultiply"]:
        return premultiply_rgb(rgb, work_alpha)
    return rgb


def _stage_compose(rgb: Image.Image, work_alpha: Image.Image) -> Image.Image:
    return compose_rgba(rgb, work_alpha)


def _stage_pixelate(out: Image.Image, params: dict) -> Image.Image:
    # Last spatial step — affects RGB + alpha together so edges stay aligned.
    if float(params["pixelate_factor"]) > 1:
        block = (params.get("_roi_ctx") or {}).get("pixelate_block")
        return pixelate(out, float(params["pixelate_factor"]), block=block)
    return out


def _stage_soften_edges(out: Image.Image, params: dict) -> Image.Image:
    """Gaussian-soften the RGB along the alpha outline (RGB only; alpha kept)."""
    r = int(params.get("soften_edges", 0))
    if r <= 0 or out.mode != "RGBA":
        return out
    alpha = out.getchannel("A")
    softened = soften_edges(out.convert("RGB"), alpha, r)
    res = softened.convert("RGBA")
    res.putalpha(alpha)
    return res


def _stage_fx(out: Image.Image, params: dict) -> Image.Image:
    """Post-processing effects chain (ImageMagick + Python). Runs on RGB only;
    the cleaned alpha is preserved and reattached unless an effect strips it
    (e.g. Normal Map)."""
    fx_chain = _build_fx_chain(params)
    if not fx_chain:
        return out
    try:
        import post_fx
    except ImportError:
        return out
    if out.mode == "RGBA":
        rgb_only = out.convert("RGB")
        alpha_only = out.split()[3]
    else:
        rgb_only = out.convert("RGB")
        alpha_only = None
    tile_label = params.get("fx_tile_size", "Off")
    tile_px = post_fx.TILE_SIZE_OPTIONS.get(tile_label, 0)
    # Pass the cleaned alpha to the chain so the Bevel effect can shape itself
    # from the silhouette (and expand it for outer/emboss bevels).
    processed = post_fx.apply_effect_chain(
        rgb_only, fx_chain, tile_size=tile_px, alpha=alpha_only)
    if post_fx.chain_strips_alpha(fx_chain):
        return processed.convert("RGB") if processed.mode != "RGB" else processed
    if processed.mode == "RGBA":
        # The chain carried the alpha through (a bevel may have expanded it).
        return processed
    if alpha_only is None:
        return processed
    if processed.size != alpha_only.size:
        alpha_only = alpha_only.resize(processed.size, Image.Resampling.LANCZOS)
    out = processed.convert("RGBA")
    out.putalpha(alpha_only)
    return out


# Output colour-depth quantisation (final stage). Mimics fixed-palette CLUTs
# (PS1 uses 256 = 8-bit and 16 = 4-bit). Alpha is retained (quantize_rgba).
# Ordered (label, max-colours) — 0 = no reduction. The UI builds its dropdown
# from QUANTIZE_LABELS so the choices and the mapping can't drift apart.
QUANTIZE_MODES = [
    ("Full colour", 0),
    ("256 (8-bit)", 256),
    ("128 (7-bit)", 128),
    ("64 (6-bit)",  64),
    ("32 (5-bit)",  32),
    ("16 (4-bit)",  16),
    ("8 (3-bit)",   8),
    ("4 (2-bit)",   4),
    ("2 (1-bit)",   2),
]
QUANTIZE_LABELS = [label for label, _ in QUANTIZE_MODES]
_QUANTIZE_COLORS = {label: n for label, n in QUANTIZE_MODES if n > 0}


def _stage_colourmatch(out: Image.Image, params: dict) -> Image.Image:
    """Match this image's colours to a reference's statistics (RGB only; alpha
    kept). Batch-only: inert unless the batch runner has enabled it and injected
    `colourmatch_ref_stats`."""
    if not params.get("colourmatch_enabled", False) or out.mode != "RGBA":
        return out
    ref = params.get("colourmatch_ref_stats")
    if not ref:
        return out
    import colour_match
    return colour_match.apply_colour_match(
        out, ref,
        method=str(params.get("colourmatch_method", "Reinhard (mean/std)")),
        strength=float(params.get("colourmatch_strength", 1.0)),
    )


def _quantize_at_pixelart_res(out: Image.Image, factor: float, quant_fn):
    """Run `quant_fn` at the pixel-art block resolution (downscale by `factor`,
    quantise, nearest-upscale back), so dithering lands as chunky pixel-art
    blocks instead of full-res speckle. NEAREST resize preserves 'P' mode, so a
    paletted (indexed) result stays indexed."""
    w, h = out.size
    sw = max(1, int(round(w / factor)))
    sh = max(1, int(round(h / factor)))
    small = out.resize((sw, sh), Image.Resampling.NEAREST)
    q = quant_fn(small)
    return q.resize((w, h), Image.Resampling.NEAREST)


def _stage_quantize(out: Image.Image, params: dict) -> Image.Image:
    dither = float(params.get("quantize_dither", 0.0))
    factor = float(params.get("pixelate_factor", 1.0))
    ctx = params.get("_roi_ctx") or {}

    # Custom-palette match takes precedence over the colour-depth count.
    if params.get("quantize_use_palette") and params.get("quantize_palette_path"):
        pal = _load_palette_image(str(params["quantize_palette_path"]))
        if pal is None:
            return out
        quant_fn = lambda im: quantize_to_palette(im, pal, dither)  # noqa: E731
    else:
        n = _QUANTIZE_COLORS.get(str(params.get("quantize_mode", "Full colour")), 0)
        if n <= 0:
            return out                      # full colour, nothing to reduce
        override = ctx.get("quantize_palette")
        if override is not None:
            # ROI crop: reuse the whole-image (low-res) result's palette so the
            # crop's colours match the rest of the image instead of being
            # re-derived from just the crop.
            quant_fn = lambda im: quantize_to_palette(im, override, dither)  # noqa: E731
        else:
            quant_fn = lambda im: quantize_rgba(im, n, dither)     # noqa: E731

    # Dithering only reads as pixel art when applied at the block resolution.
    if dither > 0 and factor > 1:
        return _quantize_at_pixelart_res(out, factor, quant_fn)
    return quant_fn(out)


def process_pil(img: Image.Image,
                params: dict,
                work_dir: Optional[Path] = None,
                ctx: Optional[dict] = None) -> Image.Image:
    """Run the full alpha cleanup pipeline on an in-memory RGBA image and
    return the result as a PIL image.

    No disk I/O except where a subprocess stage (vtracer, ImageMagick effects)
    inherently needs a file. The live preview calls this directly so that
    every parameter change avoids a full PNG encode + decode round-trip; the
    path-based `process_image` is a thin wrapper around it.

    `ctx` is an optional dict of precomputed global context used by the ROI
    full-res preview (see process_crop / compute_roi_ctx) so that a crop matches
    the whole-image result for stages that otherwise depend on global stats
    (currently the quantise palette). None everywhere else — behaviour is then
    identical to before.
    """
    params = {**DEFAULT_PARAMS, **params}
    if ctx:
        params["_roi_ctx"] = ctx

    with _maybe_tempdir(work_dir) as tmp:
        rgb_base, original_alpha = _stage_load(img, params, tmp)
        work_alpha = _stage_build_alpha(rgb_base, original_alpha, params)
        work_alpha = _stage_incong_refine(rgb_base, work_alpha, params)
        work_alpha = _stage_edge_refine(work_alpha, params)
        work_alpha = _stage_snap(work_alpha, params)
        work_alpha = _stage_clamp(work_alpha, params)
        work_alpha = _stage_alpha_shape(work_alpha, params)
        rgb = _stage_rgb_dilate(rgb_base, work_alpha, params)
        rgb = _stage_tone(rgb, params)
        rgb = _stage_premultiply(rgb, work_alpha, params)
        out = _stage_compose(rgb, work_alpha)
        out = _stage_pixelate(out, params)
        out = _stage_soften_edges(out, params)
        out = _stage_fx(out, params)
        out = _stage_colourmatch(out, params)
        out = _stage_quantize(out, params)
        return out


def process_image(input_path,
                  output_path,
                  params: dict,
                  work_dir: Optional[Path] = None) -> Path:
    """Load `input_path`, run the cleanup pipeline, save to `output_path`.
    Returns the output path. Thin wrapper over `process_pil` for batch / file
    saves."""
    params = {**DEFAULT_PARAMS, **params}
    input_path = Path(input_path)
    output_path = Path(output_path)

    with _maybe_tempdir(work_dir) as tmp:
        img = load_rgba(input_path)
        out = process_pil(img, params, work_dir=tmp)

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Clear any existing destination first so the write is a clean
        # overwrite. A stale read-only file (common when outputs were copied
        # from elsewhere or came off a CD/extracted archive) would otherwise
        # make save() raise PermissionError, leaving the OLD file in place.
        _clear_existing_output(output_path)
        suffix = output_path.suffix.lower()
        if suffix == ".dds":
            # DDS save via ImageMagick (Pillow cannot write DDS).
            # A quantised ('P') output is expanded back to RGBA so its alpha
            # survives the DXT encode (the palette has per-entry alpha).
            save_img = out.convert("RGBA") if out.mode == "P" else out
            dds_comp = params.get("dds_compression", "dxt5")
            # If an effect stripped the alpha (e.g. Normal Map), `save_img` is
            # RGB with no alpha to preserve. Encoding RGB as DXT5 (BC3) makes
            # ImageMagick fabricate an all-opaque alpha channel — wasting 4
            # bits/pixel on junk data. Fall back to DXT1 (BC1, RGB) instead.
            if save_img.mode != "RGBA" and dds_comp == "dxt5":
                dds_comp = "dxt1"
            try:
                import post_fx as _pfx
                _pfx.save_dds(save_img, output_path, compression=dds_comp)
            except Exception as _dds_err:
                raise RuntimeError(f"DDS save failed: {_dds_err}") from _dds_err
        else:
            # A quantised ('P') image saves directly to PNG as a true 8-bit
            # (or 4-bit) indexed file with transparency; other formats take it
            # as RGBA (quantised colours, full alpha).
            if out.mode == "P" and suffix != ".png":
                out = out.convert("RGBA")
            out.save(output_path)
        return output_path


# --- ROI (region-of-interest) full-res preview ----------------------------
#
# The live preview runs on a downscaled copy for speed, so any pixel-scale stage
# (reach, blur, erode/dilate, pixelate, soften, ...) looks proportionally
# different from the full-res save. When the user zooms in, we instead process
# only the VISIBLE full-res crop and show that — pixel-accurate where they're
# looking, and cheap because it's just the on-screen region. A couple of global
# stats (currently the quantise palette) are injected from the whole-image
# low-res result so the crop's colours match the rest of the image.

def compute_roi_ctx(last_after: Optional[Image.Image], params: dict) -> dict:
    """Global context for a ROI crop, derived from the already-computed low-res
    result so the crop matches the whole image. Currently: the quantise palette
    (colour-count mode), so crops don't each derive their own palette."""
    params = {**DEFAULT_PARAMS, **params}
    ctx: dict = {}
    if last_after is None:
        return ctx
    if not params.get("quantize_use_palette"):
        n = _QUANTIZE_COLORS.get(str(params.get("quantize_mode", "Full colour")), 0)
        if n > 0:
            try:
                ctx["quantize_palette"] = last_after.convert("RGB").quantize(
                    colors=max(2, min(256, n)))
            except Exception:
                pass
    return ctx


def _roi_margin_align(params: dict) -> "tuple[int, int]":
    """Overscan (px) needed so boundary-sensitive stages match at the crop edge,
    and the grid alignment for pixelate so its blocks line up with the whole
    image. Margin is trimmed off the result; alignment snaps the crop origin."""
    m = 8
    if params.get("incong_enabled"):
        m = max(m, int(params.get("incong_reach", 0)) + 2)
    m = max(m, int(round(float(params.get("blur_radius", 0.0)) * 3)))
    m = max(m, int(params.get("despeckle", 0)))
    m = max(m, int(params.get("alpha_erosion", 0)) + int(params.get("alpha_dilation", 0)))
    m = max(m, int(params.get("edge_dilation", 0)))
    m = max(m, int(params.get("soften_edges", 0)))
    if str(params.get("clamp_mode", "none")) == "ps1":
        m = max(m, int(params.get("ps1_region_min", 0)) * 2)
    if _build_fx_chain(params):
        m = max(m, 32)              # generous pad for fx pixel radii
    m = min(m, 128)                 # cap so the crop stays viewport-cheap
    factor = float(params.get("pixelate_factor", 1.0))
    align = max(1, int(round(factor))) if factor > 1 else 1
    return m, align


def process_crop(full_img: Image.Image, params: dict, rect,
                 ctx: Optional[dict] = None):
    """Process just the full-res region `rect` (x0,y0,x1,y1) of `full_img`,
    with an overscan margin so edge stages match, then trim back to `rect`.
    Returns (rgba_crop, rect) or None if the rect is empty. Local pixel-scale
    stages are exact; `ctx` (compute_roi_ctx) keeps the quantise palette
    consistent with the whole image."""
    params = {**DEFAULT_PARAMS, **params}
    W, H = full_img.size
    x0, y0, x1, y1 = (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))
    x0 = max(0, x0); y0 = max(0, y0); x1 = min(W, x1); y1 = min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return None

    margin, align = _roi_margin_align(params)
    ex0 = max(0, x0 - margin); ey0 = max(0, y0 - margin)
    ex1 = min(W, x1 + margin); ey1 = min(H, y1 + margin)
    if align > 1:
        ex0 -= ex0 % align       # snap origin to the pixelate grid
        ey0 -= ey0 % align

    crop = full_img.crop((ex0, ey0, ex1, ey1))
    res = process_pil(crop, params, ctx=ctx)
    if res.mode != "RGBA":
        res = res.convert("RGBA")
    tx0, ty0 = x0 - ex0, y0 - ey0
    trimmed = res.crop((tx0, ty0, tx0 + (x1 - x0), ty0 + (y1 - y0)))
    return trimmed, (x0, y0, x1, y1)


def process_one_file(task: dict) -> dict:
    """Picklable entry point for parallel batch processing.

    `task` carries plain (picklable) values — {"src", "dst", "params"} — so it
    can cross a ProcessPoolExecutor boundary. Exceptions are caught and
    returned as data rather than propagated, so one bad file can't tear down a
    pool worker; the parent logs the error and carries on.
    """
    src = task["src"]
    dst = task["dst"]
    params = task["params"]
    try:
        process_image(src, dst, params)
        return {"src": src, "dst": dst, "ok": True, "error": None}
    except Exception as e:
        return {"src": src, "dst": dst, "ok": False, "error": str(e)}


# --- Live-preview stage cache ----------------------------------------------

class _PipelineState:
    """Mutable carrier for the intermediate images threaded through the stage
    chain. Fields hold references to immutable (never mutated in place) PIL
    images, so `copy()` only needs to duplicate the container."""
    __slots__ = ("rgb_base", "original_alpha", "work_alpha", "rgb", "out")

    def __init__(self):
        self.rgb_base = None
        self.original_alpha = None
        self.work_alpha = None
        self.rgb = None
        self.out = None

    def copy(self) -> "_PipelineState":
        s = _PipelineState()
        s.rgb_base = self.rgb_base
        s.original_alpha = self.original_alpha
        s.work_alpha = self.work_alpha
        s.rgb = self.rgb
        s.out = self.out
        return s


def _fx_params_key(params: dict) -> tuple:
    """All post-processing params (slot effects, strengths, per-effect extras,
    tile size) as a hashable key for the fx stage."""
    return tuple(sorted(
        (k, params[k]) for k in params if k.startswith("fx_")
    ))


class PreviewPipeline:
    """Stateful, memoised wrapper over the pipeline stages for the live preview.

    On each call it reruns only the stages at and after the first parameter
    change since the previous call, reusing cached intermediate images for
    every stage before it. The output is identical to ``process_pil(img,
    params)`` — this only avoids redoing expensive early stages (vtracer,
    alpha synthesis, morphology, RGB edge-dilation) when a later-stage slider
    (tone, pixelate, fx) is the only thing that moved.

    Not thread-safe: drive it from a single worker thread (the PreviewWorker
    processes one job at a time, so that holds).
    """

    def __init__(self):
        self._source: Optional[Image.Image] = None
        self._token = 0
        self._cache: list = []   # per stage: (key, _PipelineState snapshot)
        self._stages = self._build_stage_table()

    def _build_stage_table(self) -> list:
        # Each entry is (key_fn(params, token) -> hashable, run_fn(state, params,
        # tmp_dir)). A stage is reused when its key is unchanged AND no earlier
        # stage recomputed; otherwise it (and everything after) reruns. Gated
        # stages fold their disabled state into the key so toggling a feature
        # off makes its other params irrelevant to the cache.
        def k_load(p, t):
            if not p["vtracer_enabled"]:
                return (t, False)
            return (t, True, p["vtracer_ignore_alpha"],
                    p["vtracer_pre_quantize"], p["vtracer_mode"],
                    p["vtracer_hierarchical"], p["vtracer_color_precision"],
                    p["vtracer_filter_speckle"], p["vtracer_gradient_step"],
                    p["vtracer_corner_threshold"], p["vtracer_segment_length"],
                    p["vtracer_splice_threshold"])

        def r_load(s, p, tmp):
            s.rgb_base, s.original_alpha = _stage_load(self._source, p, tmp)

        def k_build(p, t):
            if p["alpha_source"] == "fix":
                return ("fix", p["fix_level_low"], p["fix_level_high"])
            if p["alpha_source"] == "fuzzythresh":
                return ("fuzzythresh", p["fuzzythresh_target"],
                        p["fuzzythresh_radius"], p["fuzzythresh_invert"],
                        p["fuzzythresh_seed_mode"], p["fuzzythresh_seed_grow"],
                        p["fuzzythresh_seed_shrink"], p["fuzzythresh_expand_limit"],
                        p["fuzzythresh_alpha_floor"])
            if p["alpha_source"] == "rangethresh":
                return ("rangethresh", p["rangethresh_colorspace"],
                        p["rangethresh_unit"], p["rangethresh_target"],
                        p["rangethresh_r_low"], p["rangethresh_r_high"],
                        p["rangethresh_g_low"], p["rangethresh_g_high"],
                        p["rangethresh_b_low"], p["rangethresh_b_high"],
                        p["rangethresh_seed_mode"], p["rangethresh_seed_shrink"],
                        p["rangethresh_seed_grow"], p["rangethresh_expand_limit"],
                        p["rangethresh_alpha_floor"])
            return ("create", p["create_seed_mode"], p["create_seed_shrink"],
                    p["create_seed_grow"], p["create_expand_limit"],
                    p["create_lum_threshold"], p["create_target"],
                    p["create_lum_mode"], p["create_alpha_floor"],
                    p["create_lum_invert"])

        def r_build(s, p, tmp):
            s.work_alpha = _stage_build_alpha(s.rgb_base, s.original_alpha, p)

        def k_incong(p, t):
            if not p.get("incong_enabled", False):
                return (False,)
            return (True, p["incong_reach"], p["incong_content_thresh"],
                    p["incong_line_thresh"], p["incong_smooth"],
                    p.get("incong_alpha_floor", 16))

        def r_incong(s, p, tmp):
            s.work_alpha = _stage_incong_refine(s.rgb_base, s.work_alpha, p)

        def k_refine(p, t):
            return (p["blur_radius"], p["despeckle"])

        def r_refine(s, p, tmp):
            s.work_alpha = _stage_edge_refine(s.work_alpha, p)

        def k_snap(p, t):
            if not p["snap_enabled"]:
                return (False,)
            return (True, p["snap_levels"], p["snap_low_threshold"],
                    p["snap_high_threshold"], p["snap_mid_tolerance"])

        def r_snap(s, p, tmp):
            s.work_alpha = _stage_snap(s.work_alpha, p)

        def k_clamp(p, t):
            m = str(p.get("clamp_mode", "none"))
            if m == "binary":
                return ("binary", int(p.get("binary_threshold", 128)))
            if m == "tri-clamp":
                return ("tri-clamp", int(p.get("tri_clamp_lo", 64)),
                        int(p.get("tri_clamp_hi", 192)))
            if m == "ps1":
                return ("ps1", int(p.get("ps1_semi_value", 143)),
                        int(p.get("ps1_semi_tolerance", 40)),
                        int(p.get("ps1_region_min", 2)),
                        int(p.get("binary_threshold", 128)))
            return ("none",)

        def r_clamp(s, p, tmp):
            s.work_alpha = _stage_clamp(s.work_alpha, p)

        def k_shape(p, t):
            return (p["alpha_erosion"], p["alpha_dilation"])

        def r_shape(s, p, tmp):
            s.work_alpha = _stage_alpha_shape(s.work_alpha, p)

        def k_rgbdil(p, t):
            return (p["edge_dilation"], p["edge_dilation_threshold"])

        def r_rgbdil(s, p, tmp):
            s.rgb = _stage_rgb_dilate(s.rgb_base, s.work_alpha, p)

        def k_tone(p, t):
            return (p["saturation"], p["contrast"], p["brightness"],
                    p.get("level_r", 0), p.get("level_g", 0), p.get("level_b", 0))

        def r_tone(s, p, tmp):
            s.rgb = _stage_tone(s.rgb, p)

        def k_premult(p, t):
            return (bool(p["premultiply"]),)

        def r_premult(s, p, tmp):
            s.rgb = _stage_premultiply(s.rgb, s.work_alpha, p)

        def k_compose(p, t):
            return ()   # depends only on inputs; recompute is driven upstream

        def r_compose(s, p, tmp):
            s.out = _stage_compose(s.rgb, s.work_alpha)

        def k_pix(p, t):
            return (p["pixelate_factor"],)

        def r_pix(s, p, tmp):
            s.out = _stage_pixelate(s.out, p)

        def k_soften(p, t):
            return (p.get("soften_edges", 0),)

        def r_soften(s, p, tmp):
            s.out = _stage_soften_edges(s.out, p)

        def k_fx(p, t):
            return _fx_params_key(p)

        def r_fx(s, p, tmp):
            s.out = _stage_fx(s.out, p)

        def k_colourmatch(p, t):
            if not p.get("colourmatch_enabled", False):
                return (False,)
            import colour_match
            return (True, p.get("colourmatch_method"),
                    p.get("colourmatch_strength"),
                    colour_match.ref_signature(p.get("colourmatch_ref_stats")))

        def r_colourmatch(s, p, tmp):
            s.out = _stage_colourmatch(s.out, p)

        def k_quant(p, t):
            return (p.get("quantize_mode", "Full colour"),
                    p.get("quantize_dither", 0.0),
                    p.get("quantize_use_palette", False),
                    p.get("quantize_palette_path", ""),
                    p.get("pixelate_factor", 1.0))

        def r_quant(s, p, tmp):
            s.out = _stage_quantize(s.out, p)

        return [
            (k_load, r_load), (k_build, r_build), (k_incong, r_incong),
            (k_refine, r_refine),
            (k_snap, r_snap), (k_clamp, r_clamp), (k_shape, r_shape),
            (k_rgbdil, r_rgbdil), (k_tone, r_tone), (k_premult, r_premult),
            (k_compose, r_compose), (k_pix, r_pix), (k_soften, r_soften),
            (k_fx, r_fx), (k_colourmatch, r_colourmatch),
            (k_quant, r_quant),
        ]

    def run(self, img: Image.Image, params: dict) -> Image.Image:
        """Process `img` with `params`, reusing unchanged cached stages.
        Returns an RGBA image identical to process_pil()."""
        params = {**DEFAULT_PARAMS, **params}
        if img is not self._source:
            # New source image — every stage's input changed; drop the cache.
            self._source = img
            self._token += 1
            self._cache = []

        state = _PipelineState()
        recompute = False
        # Only the vtracer stage needs a temp dir; skip the per-preview mkdir/
        # rmdir churn when it's disabled (the common live-tuning case).
        tmp_cm = _maybe_tempdir(None) if params["vtracer_enabled"] else nullcontext(None)
        with tmp_cm as tmp:
            for i, (key_fn, run_fn) in enumerate(self._stages):
                key = key_fn(params, self._token)
                if (not recompute and i < len(self._cache)
                        and self._cache[i] is not None
                        and self._cache[i][0] == key):
                    # Reuse: restore the cumulative state captured last time.
                    state = self._cache[i][1].copy()
                    continue
                recompute = True
                run_fn(state, params, tmp)
                snapshot = (key, state.copy())
                if i < len(self._cache):
                    self._cache[i] = snapshot
                else:
                    self._cache.append(snapshot)

        out = state.out
        return out if out.mode == "RGBA" else out.convert("RGBA")
