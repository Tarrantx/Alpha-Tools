"""
Alpha Tools — GUI for cleaning alpha channels in upscaled game textures.

Two-tab layout:
  * Tune  — single image, side-by-side Before/After with checkerboard
            background. Live preview updates as parameters change.
  * Batch — folder in / folder out with preset selector, runs the same
            pipeline at full resolution.

Architecture notes:
  * All processing logic lives in alpha_pipeline.py — this file is GUI only.
  * Preview rendering happens on a downscaled copy (max 1024 px) on a worker
    thread, debounced ~200 ms after the last parameter change.
  * Presets are JSON files. Defaults ship under ./presets/; user-created
    presets land in %APPDATA%/Alpha Tools/presets/.
"""

from __future__ import annotations

import concurrent.futures as futures
import json
import os
import queue
import sys
import threading
import time
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw, ImageTk

import alpha_pipeline as pipeline


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# --- Constants -------------------------------------------------------------

APP_TITLE = "Alpha Tools"
APP_VERSION = "1.1.0"
PREVIEW_MAX_DIM = 1024
PREVIEW_DEBOUNCE_MS = 200
SUPPORTED_INPUTS = (
    ".png", ".tga", ".tif", ".tiff", ".bmp", ".webp",
    ".jpg", ".jpeg", ".gif", ".dds",
)
OUTPUT_FORMATS = ["Same as input", "PNG", "TIFF", "WebP", "DDS DXT1", "DDS DXT5"]


# --- Per-control help text -------------------------------------------------

HELP_TEXT = {
    "vtracer_enabled":
        "Optional pre-stage that quantises the palette then re-traces the RGB "
        "with vtracer. Best for cel-shaded / small-palette source textures "
        "where the upscaler has hallucinated noise or fake gradients. The "
        "original alpha is preserved through this step; only RGB is cleaned.",
    "vtracer_ignore_alpha":
        "Don't let the alpha channel influence the trace. By default, "
        "fully-transparent pixels (alpha 0) are forced to white before "
        "tracing so vtracer doesn't trace the garbage RGB the upscaler left "
        "in transparent areas.\n\n"
        "Tick this to trace the RGB exactly as-is — useful when that "
        "white-fill is bleeding into silhouette edges or shifting the traced "
        "shape. Your original alpha channel is preserved either way; this "
        "only changes what RGB vtracer sees.",
    "vtracer_pre_quantize":
        "Reduce the image to this many distinct colours before tracing. "
        "Throws away upscaler noise so vtracer only sees the meaningful "
        "colour regions.\n"
        "  Cel-shaded / small-palette sources: 16-32 is usually fine.\n"
        "  Complex character art with many distinct hues: try 64-128, or "
        "set to 0 to skip pre-quantize entirely.\n"
        "Uses libimagequant (same algorithm as pngquant) when available, "
        "which preserves rare-but-saturated colours much better than the "
        "default median-cut. 0 disables.",
    "vtracer_mode":
        "Curve fitting strategy.\n"
        "  pixel — exact pixel boundaries, no smoothing. The most faithful "
        "choice for pixel art and small-palette game textures.\n"
        "  polygon — straight-line segments. Slight simplification, smaller "
        "SVG; preserves angular edges.\n"
        "  spline — Bezier curves. Smoothest output, smallest SVG; best for "
        "organic shapes, but may round off intentional pixel corners.",
    "vtracer_hierarchical":
        "How nested regions are layered (color mode only).\n"
        "  stacked — regions overlap as a stacked tree. Safer against tiny "
        "seam gaps; slightly larger SVG.\n"
        "  cutout — regions touch edge-to-edge with no overlap. Smaller SVG; "
        "can show seam artefacts when rasterised back.",
    "vtracer_color_precision":
        "vtracer --color_precision (1-8). Higher = finer fidelity to the "
        "source colours. Lower = more aggressive colour grouping.",
    "vtracer_filter_speckle":
        "vtracer --filter_speckle. Pixel regions smaller than this are "
        "discarded before tracing. Higher = cleaner but may eat fine detail.",
    "vtracer_gradient_step":
        "vtracer --gradient_step. Coarser steps produce a more posterised "
        "look; finer steps preserve subtle shading.",
    "vtracer_corner_threshold":
        "Spline mode only. Angle (degrees) above which a vertex stays a "
        "sharp corner rather than being smoothed into a curve. Lower = "
        "smoother curves; higher = more preserved corners. Default 60°.",
    "vtracer_segment_length":
        "Spline mode only. Minimum length (pixels) of a Bezier segment. "
        "Higher = fewer / longer curves (smoother); lower = more detail "
        "preserved. Default 4.",
    "vtracer_splice_threshold":
        "Spline mode only. Angle (degrees) below which adjacent curves are "
        "spliced into one. Lower = more aggressive splicing (cleaner curves); "
        "higher = more individual segments preserved. Default 45°.",

    "alpha_source":
        "'fix' uses the existing alpha channel as the starting point and "
        "cleans it up. 'create' synthesises a fresh alpha from RGB luminance, "
        "anchored to the original alpha so dark interior content (shadows, "
        "outlines) is protected and not incorrectly removed. 'fuzzythresh' "
        "automatically finds an optimal luminance threshold using iterative "
        "histogram analysis (Fred Weinhaus fuzzythresh technique). "
        "'rangethresh' keys alpha on a per-channel colour range in a chosen "
        "colourspace (Fred Weinhaus rangethresh technique): a pixel is opaque "
        "only when each channel falls inside its low/high range.",

    "fix_level_low":
        "Input alpha % below this value gets pushed down to 0 (fully "
        "transparent). Acts as a transparent-side clamp.",
    "fix_level_high":
        "Input alpha % above this value gets pushed up to 255 (fully "
        "opaque). Acts as an opaque-side clamp. Set low=high for a hard "
        "binary threshold.",

    "create_target":
        "Fill value for included pixels.\n"
        "  gray  = 128 — semi-transparent (PS1 gray-alpha convention).\n"
        "  white = 255 — fully opaque.\n"
        "This method produces a uniform fill — mixed white/gray in the same "
        "output is not supported. Pick the mode that matches your source.",
    "create_alpha_floor":
        "Treat original alpha values at or below this number as fully "
        "transparent, regardless of their RGB luminance.\n\n"
        "AI upscalers nudge originally-transparent pixels (alpha=0) to small "
        "non-zero values (typically 1–15). Without this setting those "
        "contaminated pixels pass the > 0 threshold, enter the seed or fringe, "
        "and get stamped opaque — turning upscaler noise into visible halos.\n\n"
        "Set this to just above the highest contamination value you see "
        "(commonly 10–20). Pixels at or below the floor are excluded from the "
        "shape entirely and will always output alpha=0.",
    "create_seed_shrink":
        "Erode the original alpha inward by this many pixels to form the "
        "protected interior seed.\n\n"
        "All seed pixels receive the target fill value unconditionally — their "
        "luminance is never checked. This protects dark interior content "
        "(black outlines, shadows, etc.) from being incorrectly removed.\n\n"
        "0 = no shrink; seed equals the full original alpha.\n"
        "Small values (2–5 px) tighten the boundary on gently upscaled textures. "
        "Large values (20–80 px) pull deep into the interior of complex or "
        "heavily-processed alpha shapes.",
    "create_expand_limit":
        "Dilate the original alpha outward by this many pixels to set the "
        "outer search boundary for luminance-based growth.\n\n"
        "Pixels in this expansion ring that exceed the luminance threshold "
        "are added to the new alpha. Useful for recovering fringe pixels that "
        "the original alpha clipped too tightly.\n\n"
        "0 = no expansion beyond original alpha boundary.\n"
        "Small values (2–6 px) catch normal upscaler fringe. Larger values "
        "reach further out to recover detail the original alpha missed entirely.",
    "create_lum_threshold":
        "Minimum luminance (0-100 %) a pixel must have to be included when "
        "it falls in the fringe zone (between the shrunk seed and the outer "
        "growth boundary).\n\n"
        "Low values (5-15 %) admit most pixels — only near-black ones are "
        "excluded. Use this range when the texture background is pure black.\n"
        "Higher values become more selective: increase if upscaler colour bleed "
        "is contaminating the new alpha with unwanted semi-transparent haze.\n\n"
        "Note: this threshold never applies inside the protected seed region.",
    "create_lum_mode":
        "Formula used to compute luminance from RGB.\n\n"
        "  perceptual — standard weighted sum (0.299 R + 0.587 G + 0.114 B). "
        "Matches human brightness perception; the safe default.\n"
        "  max-channel — max(R, G, B) (HSV 'value'). Catches saturated colours "
        "that perceptual luminance underweights (e.g. a vivid red has "
        "perceptual lum ≈ 30 % but max-channel = 100 %). Use for sprites with "
        "saturated colours against a dark background.",
    "create_lum_invert":
        "Invert the luminance threshold direction.\n\n"
        "Off (default) — pixels at or above the threshold are included. Use "
        "when invalid fringe pixels are dark (near-black) and valid content "
        "is brighter.\n\n"
        "On — pixels at or below the threshold are included. Use when invalid "
        "fringe pixels are bright (near-white) and valid content is darker — "
        "e.g. dark sprites on a white background, or textures where the "
        "upscaler bleeds white into transparent edges.",

    "snap_enabled":
        "Snap the cleaned alpha back to evenly-spaced plateaus (e.g. "
        "0 / 128 / 255) while keeping edge transitions between them. "
        "Essential for upscaled textures whose source used quantised alpha "
        "(transparent / semi-transparent / opaque regions).",
    "snap_levels":
        "Number of evenly-spaced plateau values to snap to. 3 = 0/128/255. "
        "4 = 0/85/170/255. 2 = pure binary 0/255.",
    "snap_low_threshold":
        "Alpha values at or below this snap to 0 (fully transparent). "
        "Compensates for upscaler smear in nominally-transparent regions.",
    "snap_high_threshold":
        "Alpha values at or above this snap to 255 (fully opaque). "
        "Compensates for upscaler softening of solid regions.",
    "snap_mid_tolerance":
        "How close to an intermediate plateau (e.g. 128) a value has to be "
        "before it gets snapped to it. Smaller = tighter snap. Larger = "
        "more aggressive plateau preservation but may flatten anti-aliased "
        "edges.",

    "incong_enabled":
        "Colour-aware halo removal (incongruence refine). Walks inward from "
        "the transparency border and removes edge pixels that don't match the "
        "local interior colour — the dark fringe AI upscalers leave at the "
        "opaque/transparent boundary. Unlike luminance methods it never eats "
        "genuinely dark content: where the edge matches the content (e.g. a "
        "shadowed side) nothing is removed, and a sharp colour line (a real "
        "outline) stops it. Runs before blur/despeckle.",
    "incong_reach":
        "Maximum pixels removed inward from the existing alpha border. Set to "
        "the typical fringe width — 3-5 px for a 4x upscale. This is the "
        "safety cap: no boundary point can lose more than this.",
    "incong_alpha_floor":
        "Alpha above this counts as the subject; the halo is peeled from the "
        "OUTER edge of that subject. Keep it BELOW your subject's alpha — for a "
        "semi-transparent grey subject (PS1 127/143) the floor must be under "
        "that value, or the interior is split into an opaque/transparent "
        "speckle and removal eats inward through the middle. Default 16 (also "
        "ignores upscaler contamination). Raise toward 128 only for a strictly "
        "opaque subject on transparency; if the subject is grey and you set "
        "this above its alpha, halo removal does nothing.",
    "incong_content_thresh":
        "How close (RGB distance) a boundary pixel must be to the local "
        "interior colour to count as content and stop the removal. Higher = "
        "more aggressive (eats deeper into soft fringes); lower = more "
        "conservative. 30-45 is typical.",
    "incong_line_thresh":
        "Sharp-step guard: if the colour step from the last removed pixel "
        "exceeds this, it's treated as a genuine outline and removal stops "
        "before crossing it. Lower values protect fainter outlines; raise it "
        "only if real fringes are surviving because they sit next to strong "
        "edges.",
    "incong_smooth":
        "Smooth the removal decision (2 majority-filter passes) so the "
        "refined silhouette stays clean instead of pixel-jagged. Only "
        "operates where pixels were removed; never adds opacity.",

    "blur_radius":
        "Gaussian blur on the cleaned alpha to soften jagged edges. 0 "
        "disables. 0.2-0.3 is typical; higher just makes edges fuzzy.",
    "despeckle":
        "Number of 3x3 median-filter passes on the alpha to remove orphan "
        "noise pixels (lone bright pixels in transparent areas and vice "
        "versa). 1-2 is usually enough.",

    "alpha_erosion":
        "Shrink the alpha mask by this many pixels around its silhouette. "
        "Use to 'pull the outline back' when the alpha covers more than the "
        "intended visible region (common after upscaling, where edges grow "
        "outward). Each pixel of erosion is one MinFilter pass — start with "
        "1-2 px for subtle tightening.",
    "alpha_dilation":
        "Grow the alpha mask by this many pixels around its silhouette. "
        "Opposite of erosion — use to 'fill out the outline' when the alpha "
        "is tighter than the intended visible region. Distinct from the RGB "
        "edge dilation (which bleeds colour without changing alpha shape).",

    "clamp_mode":
        "Final alpha quantisation applied after snap and shape.\n"
        "  none      — no clamping; keep continuous alpha values.\n"
        "  binary    — force all values to 0 or 255 (alpha-test / 1-bit).\n"
        "  tri-clamp — snap to 0, 128, or 255, preserving the PS1 semi-\n"
        "              transparent state (50% blend) as a distinct level.\n"
        "  ps1       — region-aware PS1 tri-state. Snaps to 0, the chosen\n"
        "              semi value, or 255, but keeps the semi level ONLY on\n"
        "              flat semi-transparent surfaces — cutout edges go binary\n"
        "              so the upscaler's anti-aliased pixels don't become\n"
        "              blended halos on re-injection. Use for PS1 textures.",
    "binary_threshold":
        "Decision point for binary mode. Alpha below this value → 0; "
        "at or above → 255. Default 128. (Also the binary cutoff for non-semi "
        "pixels in PS1 mode.)",
    "tri_clamp_lo":
        "Values strictly below this threshold snap to 0 (fully transparent). "
        "Default 64 — the lower quarter of the range.",
    "tri_clamp_hi":
        "Values strictly above this threshold snap to 255 (fully opaque). "
        "Default 192 — the upper quarter of the range. Values between the "
        "two thresholds snap to 128 (semi-transparent).",
    "ps1_semi_value":
        "The exact alpha value PS1 semi-transparent pixels are snapped to — "
        "and written back — so the round-trip is exact.\n\n"
        "Measured from real dumps:\n"
        "  DuckStation  = 143\n"
        "  Beetle PSX HW = 127 (true 50%)\n"
        "Match this to your emulator. Writing the wrong value can make the "
        "emulator misread which pixels are semi-transparent.",
    "ps1_semi_tolerance":
        "How far from the semi value a pixel can be and still count as a "
        "semi candidate. The upscaler smears the original flat semi value, so "
        "this band has to be wide enough to catch the smear (±40 is a good "
        "start) without reaching into the cutout-edge gradients.",
    "ps1_region_min":
        "Opening radius (px) used to tell flat semi SURFACES from cutout "
        "edges that merely pass through the semi value. Edge bands thinner "
        "than ~2× this are dropped to binary; flat semi blobs survive. Raise "
        "if false semi rings still appear on edges; lower if thin genuine "
        "semi areas are being lost. Default 2.",

    "edge_dilation":
        "Bleed opaque RGB outward into transparent areas by this many "
        "pixels. Prevents the dark/coloured halos that appear when an "
        "upscaled cutout texture is bilinear-filtered or mip-mapped in "
        "engine. 4 px is a good general value; raise for heavy mipmapping.",
    "premultiply":
        "Multiply the RGB by the alpha before saving. Use for engines or "
        "blending modes that expect premultiplied alpha. Leave off for "
        "standard / straight alpha.",

    "perceptual_enabled":
        "Perceptual tone/colour adjust in OKLCh — a hue-linear perceptual "
        "space. Unlike RGB saturation/contrast, these knobs keep lightness, "
        "colourfulness and hue independent (boosting vividness won't shift "
        "brightness; contrast won't drift hue). Off = no conversion, no cost.",
    "perceptual_lightness":
        "Perceptual contrast on lightness (L) only, pivoted around mid-grey. "
        "1.00 = no change, <1 flattens, >1 deepens — without dragging hue or "
        "saturation along the way, unlike the RGB Contrast slider.",
    "perceptual_chroma":
        "Vividness (chroma) multiplier in OKLCh. 1.00 = no change, 0.00 = "
        "grayscale, >1 = more colourful, at constant perceived lightness. "
        "Cleaner than RGB saturation for punchy pixel art.",
    "perceptual_hue":
        "Rotate every hue by this many degrees around the OKLCh wheel "
        "(-180..180). 0 = no change. Hue-linear, so colours shift evenly "
        "without the blue->purple drift CIELAB would give.",
    "perceptual_autolevel":
        "Stretch the lightness (L) channel to the full perceptual range — a "
        "perceptual auto-contrast that leaves hue and saturation untouched. "
        "Runs independently of the OKLCh knobs above.",
    "perceptual_autolevel_clip":
        "Percent of pixels clipped off each end of the lightness histogram "
        "before the auto-level stretch, so a few outlier pixels don't waste "
        "the range. 0.5 = ignore the darkest/brightest 0.5%.",

    "saturation":
        "Colour saturation. 1.00 = no change, 0.00 = full grayscale, "
        "2.00 = doubled. Useful for restoring vibrance to upscaled textures "
        "that the upscaler desaturated.",
    "contrast":
        "Tonal contrast. 1.00 = no change, <1 flattens midtones, >1 deepens "
        "shadows and brightens highlights.",
    "brightness":
        "Overall brightness multiplier. 1.00 = no change, <1 darkens, "
        ">1 brightens. Applies uniformly across the image.",

    "level_colors":
        "Per-channel colour level (Fred Weinhaus levelcolors), one slider each "
        "for Red / Green / Blue. 0 = no change.\n"
        "  > 0 brightens that channel (pulls its white point down) — adds that "
        "colour.\n"
        "  < 0 darkens that channel (raises its black point) — removes that "
        "colour.\n"
        "Use these to correct a colour cast or re-balance the palette. Applied "
        "to RGB only, after the tone sliders; the alpha channel is untouched.",

    "pixelate_factor":
        "Re-pixelate the output: downsample by this factor with "
        "nearest-neighbour then upsample back to the original size. "
        "1 = off. 2-4 is typical for restoring a pixelated look to an "
        "upscaled texture; higher values give a heavier blocky effect. "
        "Fractional factors (1.25, 1.5, 1.75) give a milder effect than the "
        "next whole step — use them when factor 2 is already too blocky. "
        "Affects RGB and alpha together so edges stay aligned.",

    "soften_edges":
        "Gaussian-blur the COLOUR (RGB) in a band along the texture's outline "
        "only — the interior stays sharp and the alpha channel is untouched, so "
        "the cutout shape doesn't change. The outline is taken from the alpha "
        "silhouette; the value (px) sets both the blur strength and how far in "
        "from the edge it reaches. The blur is alpha-weighted so transparent "
        "fill colours don't bleed in as a dark halo. 0 = off.",

    "quantize_mode":
        "Final output colour depth. Reduces the texture to a fixed palette "
        "(via libimagequant, the pngquant engine) as the very last step — the "
        "alpha channel is fully retained.\n\n"
        "  Full colour — no reduction; 8-bit-per-channel RGBA (default).\n"
        "  256 (8-bit) … 2 (1-bit) — palette of that many colours. PS1's CLUT "
        "limits are 256 (8-bit) and 16 (4-bit); the smaller depths "
        "(8/4/2-colour) are for heavier posterised / retro experiments.\n\n"
        "PNG output is written as a true indexed PNG with transparency; other "
        "formats keep the reduced colours as RGBA. Great for seeing how a "
        "texture looks under a fixed palette constraint.",

    "quantize_dither":
        "Error-diffusion (Floyd–Steinberg) dithering applied during colour "
        "reduction. 0 = flat colour bands (default); higher scatters pixels "
        "between palette colours to hide banding at small palettes.\n\n"
        "Only has an effect when a palette is active — set the Palette dropdown "
        "to a colour depth (16 or fewer is where it's obvious) or enable Match "
        "palette image. With Full colour there is nothing to dither, so the "
        "slider is greyed out.\n\n"
        "When Pixelate is on, the dithering is computed at the pixel-art block "
        "resolution and upscaled, so each dither dot is a chunky block (proper "
        "pixel-art dithering) rather than fine full-resolution speckle.",

    "quantize_use_palette":
        "Match the texture to the colours of a reference palette image instead "
        "of reducing to a fixed colour count. Pick any image (a palette strip, "
        "a game screenshot, or a chosen frame) and every colour is remapped to "
        "the nearest colour in it — alpha is preserved.\n\n"
        "Use this to lock a whole animation / texture set onto ONE shared "
        "palette, which removes colour flicker between frames completely. "
        "Overrides the colour-depth dropdown while enabled; the Dither slider "
        "still applies.",

    "fx_effect":
        "Optional post-processing effect applied AFTER all alpha cleanup. "
        "Operates on the RGB channels only — the cleaned alpha is preserved "
        "and reattached, unless the effect produces a different artefact "
        "(e.g. Normal Map → no alpha).\n\n"
        "Up to 4 chained slots: slot #1 runs first, then #2, etc. Slots set "
        "to 'None' are skipped. Typical chains:\n"
        "  • Bilateral Smooth → Unsharp Mask → Saturation Boost (clean+crisp)\n"
        "  • Emboss → Saturation Boost (relief texture, vibrant colour)\n"
        "  • Normal Map alone (output a tangent-space normal map)\n\n"
        "Requires magick.exe for IM-based effects (Normal Map runs in pure "
        "Python).",

    "fx_strength":
        "Effect intensity multiplier. 1.0 = nominal, 0.0 = nearly off, "
        "2.0 = double-strength. Ignored when the slot's effect is 'None'.\n"
        "Hidden when Emboss, Granulate, or Chrome is selected — those use "
        "their own controls.",

    "fx_chrome_smooth":
        "Pre-blur applied before shading. Lower values preserve fine surface "
        "detail in the chrome bands (try 0–2 for sharp chrome). Higher values "
        "produce smoother, rounder highlights. Default 3.",

    "fx_chrome_cycles":
        "Number of sinusoidal oscillations in the chrome lookup table. "
        "1 = one highlight band (soft metallic). 2–4 = classic multi-band "
        "chrome. Higher values give a tighter, more complex ripple.",

    "fx_chrome_intensity":
        "Contrast of the chrome map after the CLUT is applied. 100 = no "
        "change (default). Higher values deepen the chrome highlights/shadows; "
        "lower values flatten them toward grey.",

    "fx_chrome_azimuth":
        "Light source direction for the surface shading, in degrees "
        "counter-clockwise from East. 135° (default) = NW light.",

    "fx_chrome_elevation":
        "Light source height above the surface plane in degrees. "
        "0° = grazing (strong relief). 90° = overhead (minimal relief). "
        "Default 45°.",

    "fx_chrome_blend":
        "How much of the chrome result mixes back over the original colour. "
        "0.0 = original unchanged. 1.0 = full chrome composite. Default 0.6.",

    "fx_chrome_compose":
        "How the chrome map composites over the original colour image.\n"
        "  Screen — chrome highlights brighten the colour (default, additive).\n"
        "  Overlay — balanced darkening/brightening, colour-aware.\n"
        "  HardLight — strong chrome, still colour-retaining.\n"
        "  SoftLight — subtle, preserves most colour.\n"
        "  Lighten — only chrome where it is brighter than the original.\n"
        "  Luminize — chrome controls brightness only; original hue and\n"
        "             saturation are fully preserved.\n"
        "  None — output the raw greyscale chrome map.",

    "fx_chrome_source":
        "What the chrome relief is derived from (Fred's -A).\n"
        "  image — the image's own luminance (default).\n"
        "  alpha — the alpha SHAPE itself, so the silhouette looks chromed "
        "(directional lighting reads much more strongly than on image content). "
        "Falls back to image when the texture has no alpha.",

    "fx_emboss_azimuth":
        "Light source direction in degrees, measured counter-clockwise from "
        "East (0°). Common values: 0=E, 45=NE, 90=N, 135=NW (default), "
        "180=W, 225=SW, 270=S, 315=SE.",

    "fx_emboss_elevation":
        "Light source elevation above the surface plane in degrees. "
        "0° = grazing light (strongest edge shadows), 90° = directly overhead "
        "(minimal relief). Default 45°.",

    "fx_emboss_depth":
        "Depth of the emboss relief. Method 1: scales the polynomial contrast "
        "stretch on the shaded layer (1 = nominal, higher exaggerates relief). "
        "Method 2: the pixel offset distance of the directional difference "
        "(higher = thicker edges). Integer > 0.",

    "fx_emboss_compose":
        "Compose operator used to blend the emboss layer back over the "
        "original colour image.\n"
        "  Overlay / HardLight / SoftLight — general-purpose texture relief.\n"
        "  Multiply / ColorBurn / LinearBurn — darkening blends.\n"
        "  ColorDodge / LinearDodge — brightening blends.\n"
        "  PegtopLight / PinLight / VividLight / LinearLight — strong contrast.\n"
        "  Bumpmap — IM's dedicated bump-map operator.\n"
        "  None — output the raw grayscale emboss layer (no colour blend).",

    "fx_emboss_method":
        "Embossing algorithm (Fred's -m).\n"
        "  1 = GIMP — shade the grayscale by azimuth/elevation, then a "
        "depth-driven contrast stretch. Uses Azimuth, Elevation, Depth.\n"
        "  2 = Photoshop — offset a grayscale copy by Depth px along the "
        "Azimuth and take the signed difference (directional edge relief). "
        "Uses Azimuth, Depth, Intensity (Elevation is ignored).",

    "fx_emboss_intensity":
        "Method 2 only: contrast of the Photoshop emboss relief (−50..50). "
        "0 = nominal; higher deepens the light/dark edges. No effect on "
        "method 1.",

    "fx_gran_sigma":
        "Scale of the granulation strokes in pixels. Low values (1–4) give "
        "fine, tight grain good for subtle seam blending. Higher values "
        "(8–20) produce coarser, more painterly strokes.",

    "fx_gran_angle":
        "Direction of the granulation strokes in degrees. 45° (default) runs "
        "diagonally and crosses both horizontal and vertical tile seams "
        "equally. 0° = horizontal strokes, 90° = vertical.",

    "fx_gran_blend":
        "How much of the granulated result mixes with the original image. "
        "0.0 = original unchanged, 1.0 = full granulation. Default 0.30 — "
        "enough to disguise seam lines without flattening the texture.",

    "fx_gran_compose":
        "How the granulation strokes interact with the image colour before "
        "blending.\n"
        "  Overlay — balanced, darkens darks / brightens lights (default).\n"
        "  Multiply — darkening; original behaviour of the old Sketch effect.\n"
        "  Screen — brightening; useful on dark or underexposed textures.\n"
        "  SoftLight — very subtle; best for barely-visible seam smoothing.\n"
        "  HardLight — strong contrast; most visible grain.",

    "fx_lomo_type":
        "Lomography colour style. Each type selectively reduces one colour "
        "channel to produce a distinct cast:\n"
        "  turquoise — reduces Red, boosting the cyan/teal complement.\n"
        "  purple    — reduces Green, boosting the magenta/purple complement.\n"
        "  lime      — reduces Blue, boosting the yellow/lime complement.\n"
        "All three apply a saturation boost and a dark corner vignette.",

    "fx_frost_spread":
        "Pixel displacement radius for the frosted-glass diffusion. Higher "
        "values scatter pixels further, producing thicker, heavier frosting. "
        "0 = no displacement (effect is just the blur passes). Default 5.",

    "fx_frost_pre_blur":
        "Blur applied before pixel displacement. Controls how much detail is "
        "visible through the glass — lower values preserve surface texture, "
        "higher values produce a more diffuse, milky look. Default 5.",

    "fx_frost_post_blur":
        "Light cleanup blur applied after displacement. Softens the harsh "
        "jagged edges that -spread introduces at colour region boundaries. "
        "Keep low (0–1.5). Default 0.5.",

    "fx_paint_gamma":
        "Midtone brightness applied before all other paint operations. "
        "1.0 = no change (default). >1 brightens midtones (lifts dark "
        "textures before painting). <1 darkens. Range 0.2–3.0.",

    "fx_paint_blur":
        "Selective blur (edge-preserving, 20% threshold) applied before the "
        "oil-paint step. Reduces fine detail within colour regions while "
        "keeping sharp boundaries intact. 0 = no blur. Default 2.0.",

    "fx_paint_radius":
        "Oil-paint neighbourhood radius (-paint). Each pixel is replaced by "
        "the most common colour within this radius. 0 = skip oil-paint step "
        "(edge outlines still apply). Higher = broader colour blocks, more "
        "abstracted. Default 2.",

    "fx_paint_edge":
        "Edge outline gain. A clone of the painted image is sharpened, "
        "morphology edge-detected, amplified by this factor, inverted, and "
        "Multiply-composited back as dark outlines. 0 = no outlines. "
        "8 = moderate painted lines (default). 15–20 = strong outlines.",

    "fx_tile_size":
        "When set, the post-processing effect chain is applied independently "
        "to each N×N tile of the image instead of globally. Useful for "
        "sprite sheets and texture atlases where effects (sharpen, smooth, "
        "saturation) should respect tile boundaries instead of bleeding "
        "across them.\n"
        "  Off — chain runs once on the whole image (default).\n"
        "  16×16 / 32×32 / 64×64 — typical sprite-sheet tile sizes.\n"
        "Each tile spawns its own ImageMagick subprocess for IM-backed "
        "effects, so small tile sizes on big images can be slow. The Grid "
        "toolbar above uses the same sizes — pair them to inspect the "
        "tile boundaries while choosing.",

    # Shared seed controls (create + fuzzythresh paths)
    "seed_mode":
        "How the protected, always-opaque seed is built — the core that is "
        "kept regardless of the luminance/threshold test.\n\n"
        "  shrink — erode the original alpha inward by 'Seed shrink' px. "
        "Spatial: pulls the protected core back from the silhouette boundary. "
        "0 px keeps the whole region as seed.\n"
        "  grow   — keep the brightest 'Seed grow' % of the region by "
        "luminance, grown outward from the luminance peak. 0% protects almost "
        "nothing; 100% protects the whole region.",
    "seed_shrink":
        "Shrink-mode seed size. Erode the original alpha inward by this many "
        "pixels to form the protected interior seed. Larger = smaller seed, so "
        "more of the region is left to the luminance/threshold test. 0 = the "
        "whole region is protected. (Active when Seed mode = shrink.)",
    "seed_grow":
        "Grow-mode seed size. Keep the brightest this-% of the region's pixels "
        "(by luminance) as the protected seed, grown out from the luminance "
        "peak. 0% protects only the peak (the test decides almost everything); "
        "100% protects the whole region. (Active when Seed mode = grow.)",

    # Fuzzythresh path
    "fuzzythresh_target":
        "Fill value for pixels classified as opaque by the threshold.\n"
        "  white = 255 — fully opaque.\n"
        "  gray  = 128 — semi-transparent (PS1 convention).",

    "fuzzythresh_radius":
        "Fred Weinhaus's fuzzythresh radius. The auto-threshold weights each "
        "pixel by how close its graylevel is to its class mean; this radius "
        "Gaussian-blurs that weight map so neighbouring pixels correlate "
        "spatially, removing noise from the threshold. Higher = smoother, more "
        "spatially-coherent split. 0 = no spatial correlation (default).",

    "fuzzythresh_seed_grow":
        "Protected, always-opaque seed grown from the brightest core of the "
        "original alpha region outward by luminance.\n\n"
        "Keeps the brightest N% of the region's pixels as seed (never tested "
        "against the threshold), so the auto-threshold only decides the "
        "remaining dimmer pixels.\n"
        "  0%   — protect only the luminance peak; the threshold decides "
        "almost everything.\n"
        "  100% — protect the whole region; the threshold decides nothing "
        "inside it (the original silhouette is kept as-is).",

    "fuzzythresh_expand_limit":
        "Dilate the original alpha region outward by this many pixels to set "
        "the outer growth boundary. Pixels in this ring that pass the "
        "auto-threshold are added to the alpha — recovers a fringe just "
        "outside the original silhouette. 0 = no growth beyond the original.",

    "fuzzythresh_alpha_floor":
        "Treat original alpha values at or below this number as fully "
        "transparent. AI upscalers nudge originally-transparent pixels "
        "(alpha=0) to small non-zero values (typically 1–15); without this "
        "floor they pass the > 0 region test and get stamped opaque. Pixels "
        "at or below the floor are excluded from the silhouette entirely. "
        "Default 16 covers typical upscaler contamination.",

    "fuzzythresh_invert":
        "When enabled, dark pixels become opaque instead of light pixels. "
        "Use for images with a light background where the subject is darker "
        "than the background.",

    # Rangethresh path
    "rangethresh_target":
        "Fill value for pixels that fall inside the colour range.\n"
        "  white = 255 — fully opaque.\n"
        "  gray  = 128 — semi-transparent (PS1 convention).",

    "rangethresh_colorspace":
        "Colourspace the R/G/B range sliders operate in (Fred Weinhaus "
        "rangethresh).\n"
        "  sRGB — sliders are Red / Green / Blue channels.\n"
        "  HSV  — sliders are Hue / Saturation / Value.\n"
        "  HSL  — sliders are Hue / Saturation / Lightness.\n"
        "All channels are evaluated on a 0-255 scale regardless of "
        "colourspace.",

    "rangethresh_unit":
        "How the slider numbers are interpreted.\n"
        "  8bit    — raw 0-255 channel values.\n"
        "  percent — 0-100% of full scale (internally x2.55). Lets you key "
        "by proportion rather than absolute level.",

    "rangethresh_channels":
        "Per-channel colour key. A pixel is classified opaque only when ALL "
        "three channels fall within their low..high range (inclusive). Widen a "
        "channel to 0-255 to ignore it; narrow it to isolate a specific colour. "
        "Channel meaning follows the selected colourspace (sRGB R/G/B, HSV "
        "H/S/V, HSL H/S/L).",

    "rangethresh_expand_limit":
        "Dilate the original alpha region outward by this many pixels to set "
        "the outer growth boundary. Pixels in this ring that pass the colour "
        "range are added to the alpha. 0 = no growth beyond the original.",

    "rangethresh_alpha_floor":
        "Treat original alpha values at or below this number as fully "
        "transparent. AI upscalers nudge originally-transparent pixels "
        "(alpha=0) to small non-zero values (typically 1–15); without this "
        "floor they pass the > 0 region test and get stamped opaque. Pixels "
        "at or below the floor are excluded from the silhouette entirely. "
        "Default 16 covers typical upscaler contamination.",

    # Crosshatch overlay
    "fx_crosshatch":
        "Overlay a diagonal crosshatch texture on the Unsharp Mask or "
        "Bilateral Smooth result. Two motion-blur passes (+45° and −45°) "
        "are blended to create the hatching pattern, then mixed back against "
        "the original. Does not consume an extra effect slot.",

    "fx_crosshatch_length":
        "Length of the diagonal motion-blur strokes in pixels. Longer "
        "strokes produce coarser hatching. Default 7.",

    "fx_crosshatch_balance":
        "Bias the crosshatch toward one diagonal. −100 = all +45° strokes, "
        "+100 = all −45° strokes, 0 = equal blend (default).",

    "fx_crosshatch_sharp":
        "Secondary unsharp-mask strength applied to the blended crosshatch "
        "before the final mix. Accentuates the hatch lines. 0 = off.",

    "fx_crosshatch_mix":
        "How much of the crosshatched result replaces the base effect output. "
        "100 = full crosshatch (default), 0 = base effect only.",

    # Dave Hill effect
    "fx_davehill_brightness":
        "Brightness multiplier applied before the Dave Hill processing. "
        "1.0 = no change, >1.0 brightens the base image. Default 1.0.",

    "fx_davehill_contrast":
        "Sigmoidal contrast adjustment. Positive = more contrast, "
        "negative = less. Range −10 to 10; 0 = off (default).",

    "fx_davehill_gain":
        "How much of the processed hyper-real result replaces the original. "
        "0 = original unchanged, 100 = full effect. Default 40.",

    # Texturize effect (Fred Weinhaus texturize — synthetic woven texture)
    "fx_texturize_format":
        "Texture style.\n"
        "  bump  = shaded 3-D relief (lit weave; the default).\n"
        "  plain = flat grey weave with no shading.",
    "fx_texturize_threshold":
        "Dot density. Keeps this top percentage of the random-noise pixels as "
        "bright seed dots before they're blurred into the woven pattern. "
        "Higher = denser, finer weave. Default 2%.",
    "fx_texturize_blur":
        "Weave blur. Sigma of the two orthogonal 1-D blurs whose product makes "
        "the cross-hatch threads. Larger = longer, softer threads (coarser "
        "fabric); smaller = tighter weave. Default 9.",
    "fx_texturize_widen":
        "Optional 2-D blur applied to the seed dots before they're streaked, "
        "fattening each thread crossing. 0 = off (default).",
    "fx_texturize_gnoise":
        "Amplitude of additive Gaussian grain mixed into the texture for a "
        "rougher, more organic surface. 0 = smooth. Default 5.",
    "fx_texturize_spread":
        "Random per-pixel jitter (in px) that displaces the texture for a less "
        "regular, hand-made look. 0 = perfectly regular weave (default).",
    "fx_texturize_contrast":
        "Extra contrast applied to the texture around mid-grey before it's "
        "blended in, deepening the weave. 0 = none (default).",
    "fx_texturize_mix":
        "Blend strength — how strongly the grey texture is laid over the "
        "image. 0 = invisible, 100 = full texture. Default 25%.",
    "fx_texturize_azimuth":
        "Light direction (degrees) for the 'bump' relief shading. Ignored in "
        "plain mode. Default 135.",
    "fx_texturize_elevation":
        "Light height (degrees) for the 'bump' relief shading. Higher = flatter "
        "lighting. Ignored in plain mode. Default 45.",
    "fx_texturize_seed":
        "Random seed for the noise pattern. Change it to get a different weave "
        "with the same settings. Default 1.",

    # Watercolor effect (Fred Weinhaus watercolor — painted look)
    "fx_watercolor_smoothing":
        "Edge-preserving smoothing of the image texture (mean-shift window). "
        "0 = off (default). Higher flattens colour regions into watercolor-like "
        "washes. Use ODD values — even ones shift the image by half a pixel. "
        "(Needs ImageMagick; slower at large values.)",
    "fx_watercolor_edge":
        "Edge gain — power applied to the detected (Sobel) edges. >=1; higher "
        "makes the dark outline strokes more prominent. Default 5.",
    "fx_watercolor_mixing":
        "How much of the edge content is blended into the smoothed image. "
        "0-100; larger = stronger outlines over the colour washes. Default 33.",
    "fx_watercolor_contrast":
        "Sigmoidal contrast boost (centred on midtones) applied before "
        "smoothing. 0 = off (default); higher deepens tonal separation.",

    # Screen Effects (Fred Weinhaus screeneffects)
    "fx_screenfx_type":
        "simple = colourise a sinusoidal screen mesh and multiply it over the "
        "image (a tinted screen-door overlay). displace = use the mesh as a "
        "displacement map to warp the image, like textured glass.",
    "fx_screenfx_spacing":
        "Pixel pitch of the sinusoidal screen pattern. Smaller = finer/denser "
        "mesh, larger = coarser. Default 6.",
    "fx_screenfx_rotate":
        "Rotate the screen pattern 45° (diagonal mesh instead of "
        "horizontal/vertical).",
    "fx_screenfx_color":
        "simple type only: the mesh's dark colour (black point). The pattern "
        "runs from this colour up to white, so black gives a neutral "
        "screen-door darkening and a hue gives a tinted mesh.",
    "fx_screenfx_amount":
        "displace type only: maximum pixel shift the mesh applies when warping "
        "the image. Default 6.",
    "fx_screenfx_method":
        "displace type only — which maps drive the warp:\n"
        "  1 = merged map for both x and y (default),\n"
        "  2 = x-map drives x, y-map drives y,\n"
        "  3 = swapped (y-map drives x, x-map drives y).",

    # Bevel effect (Fred Weinhaus bevel — shaped from the ALPHA silhouette)
    "fx_bevel_form":
        "Where the bevel sits relative to the silhouette edge.\n"
        "  inner  — bevel inside the edge; the shape is unchanged.\n"
        "  outer  — bevel in a ring OUTSIDE the edge; the alpha is expanded to "
        "include the ring (filled with the bevel colour).\n"
        "  emboss — a ridge straddling the edge (inner + outer); alpha "
        "expanded.\n"
        "Needs an alpha channel to work — it's a no-op on fully-opaque images.",

    "fx_bevel_width":
        "Width of the bevel band in pixels, measured from the silhouette edge "
        "via a distance transform of the alpha. Larger = a wider, more gradual "
        "bevel. Default 8.",

    "fx_bevel_azimuth":
        "Direction of the simulated light source in degrees, counter-clockwise "
        "from East. 135° = upper-left (default). Sets which edges read as "
        "highlighted vs shadowed.",

    "fx_bevel_elevation":
        "Elevation angle of the light source above the surface (degrees). "
        "Lower values produce a more raking, dramatic bevel. Default 30.",

    "fx_bevel_depth":
        "Relief contrast of the bevel (0–200%). 100 = nominal (default); "
        "higher deepens the highlight/shadow, lower flattens it.",

    "fx_bevel_innermode":
        "Inner-bevel profile (form = inner).\n"
        "  hard   — linear ramp; a crisp chamfer.\n"
        "  smooth — eased (rounded) ramp; a softer, pillowed edge.",

    "fx_bevel_outermode":
        "Outer-bevel relief direction (form = outer).\n"
        "  raised  — the ring rises toward the shape (default).\n"
        "  lowered — the ring sinks away from the shape.",

    "fx_bevel_soften":
        "Gaussian blur (px) applied to the bevel's height field before "
        "shading, smoothing the relief and any stair-stepping. 0 = off.",

    "fx_bevel_taper":
        "Shapes the OUTER ramp profile (form = outer or emboss); 0–3. Higher "
        "values make the outer slope fall off more sharply toward the rim. "
        "No effect on form = inner.",

    "fx_bevel_bcolor":
        "Fill colour for the OUTER bevel ring (the new area added outside the "
        "silhouette for form = outer / emboss). Gray (#808080) is neutral. "
        "Enter as hex or use the colour picker. (Inner bevels keep the "
        "original image colour.)",

    "fx_bevel_colormode":
        "Relief tone when compositing the bevel over the image.\n"
        "  dark  — neutral hardlight relief (default; shadows read recessed).\n"
        "  light — lifts the relief toward highlights (screen).",

    "fx_bevel_ring_alpha":
        "Alpha for the NEW ring of bevel material added outside the silhouette "
        "(form = outer or emboss; ignored for inner). The original shape always "
        "keeps its own alpha.\n"
        "  Opaque       — solid 255 border (default).\n"
        "  Inherit edge — each ring pixel takes the alpha of the nearest "
        "silhouette edge, so the border matches a semi-transparent / mixed "
        "texture.\n"
        "  Fade out     — inherits the edge alpha, then tapers to 0 at the "
        "outer rim for a soft, feathered bevel border.",
}


def app_data_dir() -> Path:
    """Per-user config directory."""
    base = os.environ.get("APPDATA") or str(Path.home() / ".config")
    p = Path(base) / "Alpha Tools"
    p.mkdir(parents=True, exist_ok=True)
    return p


def shipped_presets_dir() -> Path:
    """Read-only presets shipped with the install."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent / "presets"
    return Path(__file__).parent / "presets"


def user_presets_dir() -> Path:
    p = app_data_dir() / "presets"
    p.mkdir(parents=True, exist_ok=True)
    return p


# Per-dialog "last used directory". Tk shares one last-location across every
# file dialog; we instead key it by purpose ("load_image", "load_preset", ...)
# so each button reopens where it was last used, persisted across restarts.
def _dialog_dirs_file() -> Path:
    return app_data_dir() / "dialog_dirs.json"


def load_dialog_dir(key: str, default=None):
    """Last directory used for the dialog `key`, or `default` if none is
    stored or it no longer exists on disk."""
    try:
        data = json.loads(_dialog_dirs_file().read_text(encoding="utf-8"))
        d = data.get(key)
        if d and Path(d).is_dir():
            return d
    except Exception:
        pass
    return str(default) if default is not None else None


def save_dialog_dir(key: str, path) -> None:
    """Remember the directory of `path` (a chosen file or a folder) under
    `key`. Best-effort — never raises into the caller."""
    if not path:
        return
    try:
        p = Path(path)
        d = str(p if p.is_dir() else p.parent)
        f = _dialog_dirs_file()
        data = {}
        if f.exists():
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        data[key] = d
        f.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


# --- Tooltip / help icon (lifted from Vtracer Tools) ----------------------

class _Tooltip:
    """Hover tooltip for any widget. Borderless tk.Toplevel positioned just
    below the widget, shown after a short delay so a stray cursor pass
    doesn't flash a tooltip."""

    def __init__(self, widget, text, delay_ms=400, wraplength=320):
        self._widget = widget
        self._text = text
        self._delay = delay_ms
        self._wraplength = wraplength
        self._after_id = None
        self._popup = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")
        widget.bind("<Destroy>", self._hide, add="+")

    def _schedule(self, event=None):
        self._cancel_pending()
        self._after_id = self._widget.after(self._delay, self._show)

    def _cancel_pending(self):
        if self._after_id is not None:
            try:
                self._widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _show(self):
        if self._popup is not None:
            return
        try:
            x = self._widget.winfo_rootx() + 18
            y = self._widget.winfo_rooty() + self._widget.winfo_height() + 4
        except Exception:
            return
        top = tk.Toplevel(self._widget)
        top.wm_withdraw()
        top.wm_overrideredirect(True)
        try:
            top.wm_attributes("-topmost", True)
        except tk.TclError:
            pass
        tk.Label(
            top,
            text=self._text,
            background="#2b2b2b",
            foreground="#dddddd",
            relief="solid",
            borderwidth=1,
            padx=8,
            pady=6,
            wraplength=self._wraplength,
            justify="left",
            font=("Segoe UI", 9),
        ).pack()
        top.wm_geometry(f"+{x}+{y}")
        top.wm_deiconify()
        self._popup = top

    def _hide(self, event=None):
        self._cancel_pending()
        if self._popup is not None:
            try:
                self._popup.destroy()
            except Exception:
                pass
            self._popup = None


def _help_icon(parent, tooltip_text):
    """A small grey '?' label with a hover tooltip. Caller pack()/grid()s it."""
    lbl = ctk.CTkLabel(
        parent,
        text="?",
        width=18,
        height=18,
        text_color=("#666", "#999"),
        font=ctk.CTkFont(size=11, weight="bold"),
    )
    _Tooltip(lbl, tooltip_text)
    return lbl


# --- Checkerboard helper ---------------------------------------------------

def make_checkerboard(width: int, height: int, square: int = 8) -> Image.Image:
    """Generate a grey checkerboard pattern. Used as the transparency
    backdrop so semi-transparent pixels and fringing are visible."""
    if width < 1 or height < 1:
        return Image.new("RGB", (1, 1), (200, 200, 200))
    yy, xx = np.mgrid[0:height, 0:width]
    pattern = (((xx // square) + (yy // square)) % 2).astype(bool)
    light = np.array([220, 220, 220], dtype=np.uint8)
    dark  = np.array([180, 180, 180], dtype=np.uint8)
    rgb = np.where(pattern[:, :, None], light, dark).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def composite_over_checkerboard(rgba: Image.Image) -> Image.Image:
    """Flatten an RGBA image over a checkerboard backdrop for display."""
    if rgba.mode != "RGBA":
        rgba = rgba.convert("RGBA")
    bg = make_checkerboard(rgba.width, rgba.height)
    return Image.alpha_composite(bg.convert("RGBA"), rgba).convert("RGB")


def composite_over_solid(rgba: Image.Image, color_hex: str) -> Image.Image:
    """Flatten an RGBA image over a solid colour backdrop for display."""
    if rgba.mode != "RGBA":
        rgba = rgba.convert("RGBA")
    h = color_hex.lstrip("#")
    try:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    except (ValueError, IndexError):
        r, g, b = 128, 128, 128
    bg = Image.new("RGB", rgba.size, (r, g, b))
    return Image.alpha_composite(bg.convert("RGBA"), rgba).convert("RGB")


def alpha_as_rgb(rgba: Image.Image) -> Image.Image:
    """Render the alpha channel as a grayscale RGB image for the
    'Show alpha only' inspection mode."""
    if rgba.mode != "RGBA":
        return rgba.convert("RGB")
    a = rgba.getchannel("A")
    return Image.merge("RGB", (a, a, a))


# --- ZoomPanPreview --------------------------------------------------------

class ZoomPanPreview(ctk.CTkFrame):
    """Canvas-based image preview with zoom (wheel) and click-drag pan.

    Adapted from the Image Tools project. Two additions:
      * `link_with(other)`  -> mirror view (zoom/pan) between two panes.
      * `on_view_change`    -> callback fires whenever the view transforms,
                               so the linked partner can mirror without an
                               infinite loop.
    """

    def __init__(self, parent, placeholder="No image", **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)

        self._base_img: Image.Image | None = None
        self._zoom = 1.0
        self._offset_x = 0
        self._offset_y = 0
        self._photo: ImageTk.PhotoImage | None = None
        self._image_id = None
        self._placeholder_id = None
        self._placeholder_text = placeholder

        self._drag_start_x = 0
        self._drag_start_y = 0
        self._drag_start_ox = 0
        self._drag_start_oy = 0

        self._linked: list[ZoomPanPreview] = []
        self._mirroring = False    # re-entry guard

        # Optional callback(coord) fired on cursor move, where coord is the
        # (x, y) pixel under the cursor in this pane's displayed-image space,
        # or None when the cursor is off the image. Used for the coordinate
        # readout. Not mirrored to linked panes (each reports its own cursor).
        self.on_cursor_move = None

        # Grid overlay. None = off; otherwise a dict with keys
        # size (source px), line_width (screen px), color, semi.
        # _grid_overlay_cache stores the most-recently-built overlay so
        # drag-pan events don't rebuild it 60 times a second.
        self._grid: dict | None = None
        self._grid_overlay_cache: tuple | None = None  # (key, RGBA image)
        self._photo_grid: ImageTk.PhotoImage | None = None
        # Source px per base (preview) px. >1 when the preview is downscaled to
        # PREVIEW_MAX_DIM; keeps the grid on SOURCE-pixel boundaries (and aligned
        # to the full-res detail overlay) instead of downscaled-preview pixels.
        self._source_scale: float = 1.0

        # ROI full-res detail overlay: a full-resolution image covering a rect
        # (in BASE-image coords) drawn on top of the low-res crop where the user
        # is zoomed in. Provided by the owner via set_detail() after the view
        # settles. on_view_settle fires ~250 ms after zoom/pan stops.
        self._detail: tuple | None = None          # (base_rect, full_res_image)
        self._photo_detail: ImageTk.PhotoImage | None = None
        self.on_view_settle = None
        self._settle_id = None
        self._SETTLE_MS = 250

        # Manual-edit painting. When active, LEFT-drag paints (via the callbacks,
        # in SOURCE px); right-drag still pans. Owner sets the callbacks + mode.
        self._edit_active = False
        self._brush_size = 4                        # source px, for the cursor
        self._left_paint = False                    # this left-drag is a paint
        self._brush_cursor_id = None
        self.on_paint_begin = None                  # (sx, sy)
        self.on_paint_to = None                     # (sx, sy)
        self.on_paint_end = None                    # ()

        bg = "#2b2b2b" if ctk.get_appearance_mode() == "Dark" else "#dbdbdb"
        self.canvas = tk.Canvas(self, highlightthickness=0, bd=0, bg=bg)
        self.canvas.pack(fill="both", expand=True)

        # Left button paints when edit mode is active, else pans.
        self.canvas.bind("<ButtonPress-1>", self._on_left_press)
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_release)
        # Right-drag ALWAYS pans (right button is otherwise unused), so that
        # while manual-edit mode makes left-drag paint, panning stays available.
        self.canvas.bind("<ButtonPress-3>", self._on_press)
        self.canvas.bind("<B3-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-3>", self._on_release)
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Enter>", lambda e: self.canvas.focus_set())
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Leave>", self._on_leave)

    # -- public API ----------------------------------------------------------

    def set_image(self, pil_img: Image.Image | None, reset_view: bool = True):
        self._base_img = pil_img
        if reset_view:
            self._zoom = 1.0
            self._offset_x = 0
            self._offset_y = 0
        # The base image changed (new source or new params), so any detail
        # overlay is stale — drop it (the owner recomputes via _redraw_previews).
        self._detail = None
        self._photo_detail = None
        self.canvas.configure(cursor="fleur" if pil_img is not None else "")
        self._render()

    def set_detail(self, detail_img: Image.Image | None, base_rect):
        """Overlay a full-resolution `detail_img` covering `base_rect`
        (bx0,by0,bx1,by1 in base-image coords) on top of the low-res crop."""
        self._detail = None if detail_img is None else (tuple(base_rect), detail_img)
        self._render()

    def set_source_scale(self, scale: float):
        """Tell the pane how many SOURCE pixels each base (preview) pixel spans,
        so the grid stays on source-pixel boundaries when the preview is
        downscaled. 1.0 for images at or below PREVIEW_MAX_DIM."""
        new = float(scale) if scale and scale > 0 else 1.0
        if abs(new - self._source_scale) > 1e-9:
            self._source_scale = new
            self._render()

    def _schedule_settle(self):
        if self._settle_id is not None:
            try:
                self.after_cancel(self._settle_id)
            except Exception:
                pass
        self._settle_id = self.after(self._SETTLE_MS, self._fire_settle)

    def _fire_settle(self):
        self._settle_id = None
        if self.on_view_settle is not None:
            self.on_view_settle()

    def set_placeholder(self, text: str):
        self._placeholder_text = text
        self._base_img = None
        self._render()

    def link_with(self, other: "ZoomPanPreview"):
        if other is self or other in self._linked:
            return
        self._linked.append(other)
        other._linked.append(self)

    def set_grid(self, size: int, line_width: int, color: str,
                 semi_transparent: bool):
        """Configure the grid overlay. size=0 disables. line_width is in
        screen pixels; color is one of _GRID_COLORS' keys.

        Propagates to linked panes — same contract as zoom/pan mirroring —
        so callers only need to set it on one pane and both stay in sync.
        """
        if size <= 0:
            self._grid = None
        else:
            self._grid = {
                "size":       int(size),
                "line_width": max(1, int(line_width)),
                "color":      color,
                "semi":       bool(semi_transparent),
            }
        self._grid_overlay_cache = None
        self._render()
        if self._mirroring:
            return
        for partner in self._linked:
            partner._mirroring = True
            try:
                partner.set_grid(size, line_width, color, semi_transparent)
            finally:
                partner._mirroring = False

    # -- internal ------------------------------------------------------------

    def _fit_factor(self) -> float:
        if self._base_img is None:
            return 1.0
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        iw, ih = self._base_img.size
        return min(cw / iw, ch / ih) if (iw and ih) else 1.0

    def _canvas_to_image_px(self, mx: float, my: float):
        """Map canvas coords (mx, my) to integer (x, y) pixel coords in the
        displayed image, or None if outside the image. Inverts the same
        centre-anchored transform used by _render."""
        if self._base_img is None:
            return None
        eff = self._fit_factor() * self._zoom
        if eff <= 0:
            return None
        iw, ih = self._base_img.size
        dw = max(1, int(iw * eff))
        dh = max(1, int(ih * eff))
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        cx = cw // 2 + self._offset_x
        cy = ch // 2 + self._offset_y
        left = cx - dw / 2.0
        top = cy - dh / 2.0
        sx = (mx - left) / eff
        sy = (my - top) / eff
        if 0 <= sx < iw and 0 <= sy < ih:
            return (min(iw - 1, int(sx)), min(ih - 1, int(sy)))
        return None

    def visible_base_rect(self):
        """The base-image region currently visible on the canvas, as
        (x0, y0, x1, y1, eff), or None. `eff` is screen-px per base-px. Same
        transform/overscan as _render, so the owner can request a matching
        full-res crop."""
        if self._base_img is None:
            return None
        eff = self._fit_factor() * self._zoom
        if eff <= 0:
            return None
        iw, ih = self._base_img.size
        dw = iw * eff; dh = ih * eff
        cw = self.canvas.winfo_width() or 1
        ch = self.canvas.winfo_height() or 1
        cx = cw // 2 + self._offset_x
        cy = ch // 2 + self._offset_y
        left = cx - dw / 2.0; top = cy - dh / 2.0
        x0 = max(0, int((0 - left) / eff) - 1)
        y0 = max(0, int((0 - top) / eff) - 1)
        x1 = min(iw, int((cw - left) / eff) + 2)
        y1 = min(ih, int((ch - top) / eff) + 2)
        if x1 <= x0 or y1 <= y0:
            return None
        return (x0, y0, x1, y1, eff)

    def _on_motion(self, event):
        if self.on_cursor_move is not None:
            self.on_cursor_move(self._canvas_to_image_px(event.x, event.y))
        if self._edit_active:
            self._update_brush_cursor(event.x, event.y)

    def _on_leave(self, _event):
        if self.on_cursor_move is not None:
            self.on_cursor_move(None)
        self._clear_brush_cursor()

    # -- manual-edit painting ------------------------------------------------

    def set_edit_mode(self, active: bool, brush_size: int | None = None):
        """Enable/disable painting. When active, left-drag paints via the
        on_paint_* callbacks (right-drag still pans)."""
        self._edit_active = bool(active)
        if brush_size is not None:
            self._brush_size = max(1, int(brush_size))
        if self._base_img is not None:
            self.canvas.configure(cursor="crosshair" if active else "fleur")
        if not active:
            self._clear_brush_cursor()

    def _canvas_to_source_px(self, mx: float, my: float):
        """Cursor position in full SOURCE-image px, or None if off-image."""
        base = self._canvas_to_image_px(mx, my)
        if base is None:
            return None
        scale = self._source_scale if self._source_scale > 0 else 1.0
        return (base[0] * scale, base[1] * scale)

    def _on_left_press(self, event):
        if self._edit_active and self.on_paint_begin is not None:
            self._left_paint = True
            sp = self._canvas_to_source_px(event.x, event.y)
            if sp is not None:
                self.on_paint_begin(sp[0], sp[1])
            self._update_brush_cursor(event.x, event.y)
        else:
            self._left_paint = False
            self._on_press(event)

    def _on_left_drag(self, event):
        if self._left_paint:
            sp = self._canvas_to_source_px(event.x, event.y)
            if sp is not None and self.on_paint_to is not None:
                self.on_paint_to(sp[0], sp[1])
            self._update_brush_cursor(event.x, event.y)
        else:
            self._on_drag(event)

    def _on_left_release(self, event):
        if self._left_paint:
            self._left_paint = False
            if self.on_paint_end is not None:
                self.on_paint_end()
        else:
            self._on_release(event)

    def _update_brush_cursor(self, mx, my):
        """Draw a square outline showing the brush footprint at the cursor."""
        self._clear_brush_cursor()
        if not self._edit_active or self._base_img is None:
            return
        eff = self._fit_factor() * self._zoom
        scale = self._source_scale if self._source_scale > 0 else 1.0
        side = max(2.0, self._brush_size * eff / scale)   # source px -> screen px
        self._brush_cursor_id = self.canvas.create_rectangle(
            mx - side / 2, my - side / 2, mx + side / 2, my + side / 2,
            outline="#ff3030", width=1)

    def _clear_brush_cursor(self):
        if self._brush_cursor_id is not None:
            try:
                self.canvas.delete(self._brush_cursor_id)
            except Exception:
                pass
            self._brush_cursor_id = None

    def _render(self):
        self.canvas.delete("all")
        self._photo = None
        self._image_id = None
        self._placeholder_id = None
        self._brush_cursor_id = None            # cleared by canvas.delete("all")

        if self._base_img is None:
            cw = self.canvas.winfo_width() or 200
            ch = self.canvas.winfo_height() or 100
            self._placeholder_id = self.canvas.create_text(
                cw // 2, ch // 2,
                text=self._placeholder_text,
                fill="#888888",
                font=("Segoe UI", 12),
            )
            return

        fit = self._fit_factor()
        eff = fit * self._zoom
        iw, ih = self._base_img.size
        dw = iw * eff
        dh = ih * eff

        cw = self.canvas.winfo_width() or 1
        ch = self.canvas.winfo_height() or 1
        cx = cw // 2 + self._offset_x
        cy = ch // 2 + self._offset_y
        left = cx - dw / 2.0      # canvas x of the image's left edge
        top = cy - dh / 2.0

        # Viewport rendering: rasterise ONLY the source region currently visible
        # in the canvas, not the whole image scaled to the (possibly enormous)
        # zoomed size. Cost is O(viewport) at any zoom instead of O(image·zoom²),
        # which is what made high-zoom pan/zoom crawl. (A 1–2px overscan avoids
        # edge gaps.) The coordinate transform is unchanged, so the cursor
        # readout and grid stay pixel-accurate.
        src_x0 = max(0, int((0 - left) / eff) - 1)
        src_y0 = max(0, int((0 - top) / eff) - 1)
        src_x1 = min(iw, int((cw - left) / eff) + 2)
        src_y1 = min(ih, int((ch - top) / eff) + 2)
        if src_x1 <= src_x0 or src_y1 <= src_y0:
            return  # image is entirely off-screen (panned away)

        crop = self._base_img.crop((src_x0, src_y0, src_x1, src_y1))
        disp_w = max(1, int(round((src_x1 - src_x0) * eff)))
        disp_h = max(1, int(round((src_y1 - src_y0) * eff)))

        # Pillow resample: NEAREST when zoomed past 1:1 (preserve pixels),
        # LANCZOS otherwise (good quality downscale / mild upscale).
        resample = Image.Resampling.NEAREST if eff >= 1.5 else Image.Resampling.LANCZOS
        scaled = crop.resize((disp_w, disp_h), resample=resample)
        self._photo = ImageTk.PhotoImage(scaled)

        # Place the crop where its source top-left maps to on the canvas.
        self._image_id = self.canvas.create_image(
            left + src_x0 * eff, top + src_y0 * eff,
            image=self._photo, anchor="nw")

        # Full-res detail overlay: draw the accurate full-resolution result on
        # top of the low-res crop wherever it's available and visible.
        self._draw_detail(eff, left, top, src_x0, src_y0, src_x1, src_y1)

        # Grid LAST — on top of BOTH the low-res crop and the full-res detail.
        # (Previously it was composited into the low-res crop and then hidden by
        # the detail overlay whenever you zoomed in on a downscaled preview.)
        self._draw_grid(eff, left, top, src_x0, src_y0, disp_w, disp_h)

    def _draw_detail(self, eff, left, top, vx0, vy0, vx1, vy1):
        if self._detail is None:
            self._photo_detail = None
            return
        (bx0, by0, bx1, by1), dimg = self._detail
        # Intersect the detail's base rect with the visible region.
        ix0 = max(vx0, bx0); iy0 = max(vy0, by0)
        ix1 = min(vx1, bx1); iy1 = min(vy1, by1)
        if ix1 <= ix0 or iy1 <= iy0:
            self._photo_detail = None
            return
        # detail-image pixels per base pixel (detail is full-res for base_rect).
        fs = dimg.width / max(1, (bx1 - bx0))
        dcrop = dimg.crop((int((ix0 - bx0) * fs), int((iy0 - by0) * fs),
                           int((ix1 - bx0) * fs), int((iy1 - by0) * fs)))
        ddw = max(1, int(round((ix1 - ix0) * eff)))
        ddh = max(1, int(round((iy1 - iy0) * eff)))
        # screen px per detail px; NEAREST once we're upscaling the full-res.
        det_scale = eff / fs
        resample = Image.Resampling.NEAREST if det_scale >= 1.5 else Image.Resampling.LANCZOS
        self._photo_detail = ImageTk.PhotoImage(dcrop.resize((ddw, ddh), resample))
        self.canvas.create_image(left + ix0 * eff, top + iy0 * eff,
                                 image=self._photo_detail, anchor="nw")

    _GRID_COLORS = {
        "red":   (255,  64,  64),
        "green": ( 64, 255,  64),
        "blue":  ( 64, 128, 255),
    }

    @staticmethod
    def _grid_axis_lines(src0: float, extent: int, gstep_base: float,
                         eff: float, width: int):
        """(lo, hi) screen-pixel ranges for grid lines along one axis of a crop
        whose base-coord origin is `src0`. Lines lie at base multiples of
        `gstep_base`; positions are crop-local screen px, each `width` px thick.
        Empty when the grid is denser than 2 screen px (would read as noise).
        Uses a fixed integer width so a line always draws, even when it lands on
        an exact pixel boundary (round-half-to-even would otherwise drop it)."""
        out = []
        if gstep_base <= 0:
            return out
        step_screen = gstep_base * eff
        if step_screen < 2:
            return out
        w = max(1, int(round(width)))
        k = int(np.ceil(src0 / gstep_base))          # first line at/after origin
        pos = (k * gstep_base - src0) * eff
        while pos <= extent + 0.5:
            lo = int(round(pos - w / 2.0))
            hi = lo + w - 1
            if hi >= 0 and lo <= extent - 1:
                out.append((max(0, lo), min(extent - 1, hi)))
            pos += step_screen
        return out

    def _build_grid_overlay(self, dw: int, dh: int, eff: float,
                            src_x0: int, src_y0: int) -> Image.Image | None:
        """Transparent RGBA overlay (dw×dh) of grid lines for the current
        viewport crop, or None if the grid is off / too dense to show. Lines sit
        on SOURCE-pixel multiples of cfg['size']: `_source_scale` converts source
        px → base px (>1 when the preview is downscaled), so the grid lands on
        source pixel boundaries and aligns with the full-res detail overlay.
        Line thickness is in SCREEN pixels."""
        cfg = self._grid
        if cfg is None:
            return None
        scale = self._source_scale if self._source_scale > 0 else 1.0
        gstep_base = int(cfg["size"]) / scale        # source px → base px
        width = int(cfg["line_width"])
        xs = self._grid_axis_lines(src_x0, dw, gstep_base, eff, width)
        ys = self._grid_axis_lines(src_y0, dh, gstep_base, eff, width)
        if not xs and not ys:
            return None
        overlay = Image.new("RGBA", (dw, dh), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        fill = self._GRID_COLORS[cfg["color"]] + (110 if cfg["semi"] else 230,)
        for lo, hi in xs:
            draw.rectangle([lo, 0, hi, dh - 1], fill=fill)
        for lo, hi in ys:
            draw.rectangle([0, lo, dw - 1, hi], fill=fill)
        return overlay

    def _draw_grid(self, eff: float, left: float, top: float,
                   src_x0: int, src_y0: int, dw: int, dh: int):
        """Draw the grid overlay as the top-most canvas layer (above the low-res
        crop and any full-res detail)."""
        if self._grid is None:
            self._photo_grid = None
            return
        overlay = self._build_grid_overlay(dw, dh, eff, src_x0, src_y0)
        if overlay is None:
            self._photo_grid = None
            return
        self._photo_grid = ImageTk.PhotoImage(overlay)
        self.canvas.create_image(left + src_x0 * eff, top + src_y0 * eff,
                                 image=self._photo_grid, anchor="nw")

    def _on_resize(self, _event):
        self._render()

    def _on_press(self, event):
        self._drag_start_x = event.x
        self._drag_start_y = event.y
        self._drag_start_ox = self._offset_x
        self._drag_start_oy = self._offset_y

    def _on_drag(self, event):
        if self._base_img is None:
            return
        self._offset_x = self._drag_start_ox + (event.x - self._drag_start_x)
        self._offset_y = self._drag_start_oy + (event.y - self._drag_start_y)
        self._render()
        self._mirror_view()
        self._schedule_settle()
        # <Motion> doesn't fire during a B1 drag, so keep the readout live.
        if self.on_cursor_move is not None:
            self.on_cursor_move(self._canvas_to_image_px(event.x, event.y))

    def _on_release(self, _event):
        pass

    def _on_wheel(self, event):
        if self._base_img is None:
            return
        step = 1.1 if event.delta > 0 else (1 / 1.1)
        self._zoom = max(0.1, min(40.0, self._zoom * step))
        self._render()
        self._mirror_view()
        self._schedule_settle()

    def _mirror_view(self):
        if self._mirroring:
            return
        for partner in self._linked:
            partner._mirroring = True
            try:
                partner._zoom = self._zoom
                partner._offset_x = self._offset_x
                partner._offset_y = self._offset_y
                partner._render()
            finally:
                partner._mirroring = False


# --- Parameter widgets -----------------------------------------------------

class SliderRow(ctk.CTkFrame):
    """A label + value-readout + slider, with two-way wiring to a key in a
    shared params dict."""

    def __init__(self, parent, label: str, key: str, params: dict,
                 from_: float, to: float, step: float = 1,
                 fmt: str = "{:.0f}", help_text: str | None = None,
                 on_change=None, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)
        self.params = params
        self.key = key
        self.fmt = fmt
        self.on_change = on_change

        top = ctk.CTkFrame(self, fg_color="transparent")
        top.pack(fill="x")
        self.label = ctk.CTkLabel(top, text=label, anchor="w")
        self.label.pack(side="left")
        if help_text:
            _help_icon(top, help_text).pack(side="left", padx=(4, 0))
        self.value_label = ctk.CTkLabel(top, text="", anchor="e", width=60)
        self.value_label.pack(side="right")

        steps = max(1, int(round((to - from_) / step)))
        self.slider = ctk.CTkSlider(
            self, from_=from_, to=to, number_of_steps=steps,
            command=self._on_slide,
        )
        self.slider.pack(fill="x", pady=(2, 6))

        self.slider.set(params.get(key, from_))
        self._update_readout(params.get(key, from_))

    def _on_slide(self, value):
        # Coerce ints if the format implies it
        if "{:.0f}" in self.fmt or "{:d}" in self.fmt:
            value = int(round(value))
        self.params[self.key] = value
        self._update_readout(value)
        if self.on_change:
            self.on_change()

    def _update_readout(self, value):
        try:
            self.value_label.configure(text=self.fmt.format(value))
        except (ValueError, TypeError):
            self.value_label.configure(text=str(value))

    def refresh_from_params(self):
        v = self.params.get(self.key, self.slider.get())
        self.slider.set(v)
        self._update_readout(v)

    def set_enabled(self, enabled: bool):
        """Enable/disable the slider and dim its label when the control has no
        effect in the current configuration."""
        self.slider.configure(state="normal" if enabled else "disabled")
        self.label.configure(
            text_color=(("#222", "#ddd") if enabled else ("#999", "#666")))


class ToggleRow(ctk.CTkFrame):
    def __init__(self, parent, label: str, key: str, params: dict,
                 help_text: str | None = None, on_change=None, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)
        self.params = params
        self.key = key
        self.on_change = on_change

        self.var = tk.BooleanVar(value=bool(params.get(key, False)))
        self.checkbox = ctk.CTkCheckBox(
            self, text=label, variable=self.var, command=self._on_toggle,
        )
        self.checkbox.pack(side="left", anchor="w", pady=2)
        if help_text:
            _help_icon(self, help_text).pack(side="left", padx=(4, 0))

    def _on_toggle(self):
        self.params[self.key] = bool(self.var.get())
        if self.on_change:
            self.on_change()

    def refresh_from_params(self):
        self.var.set(bool(self.params.get(self.key, False)))


class OptionRow(ctk.CTkFrame):
    def __init__(self, parent, label: str, key: str, params: dict,
                 values: list[str], help_text: str | None = None,
                 on_change=None, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)
        self.params = params
        self.key = key
        self.values = values
        self.on_change = on_change

        top = ctk.CTkFrame(self, fg_color="transparent")
        top.pack(fill="x")
        ctk.CTkLabel(top, text=label, anchor="w").pack(side="left")
        if help_text:
            _help_icon(top, help_text).pack(side="left", padx=(4, 0))
        self.var = tk.StringVar(value=str(params.get(key, values[0])))
        self.menu = ctk.CTkOptionMenu(
            self, values=values, variable=self.var, command=self._on_pick,
        )
        self.menu.pack(fill="x", pady=(2, 6))

    def _on_pick(self, _value):
        self.params[self.key] = self.var.get()
        if self.on_change:
            self.on_change()

    def refresh_from_params(self):
        v = str(self.params.get(self.key, self.values[0]))
        if v in self.values:
            self.var.set(v)


class ColourPickerRow(ctk.CTkFrame):
    """Label + colour swatch + hex entry + Pick button, wired to params[key]."""

    DEFAULT_COLOR = "#0080FF"

    def __init__(self, parent, key: str, params: dict,
                 label: str = "Tint colour", default_color: str | None = None,
                 on_change=None, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)
        self.params = params
        self.key = key
        self.on_change = on_change
        self._default_color = default_color or self.DEFAULT_COLOR

        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", pady=(2, 6))

        ctk.CTkLabel(row, text=label, anchor="w",
                     width=110).pack(side="left")

        # Small colour swatch (plain tk.Canvas so we can set bg colour)
        self.swatch = tk.Canvas(row, width=22, height=18,
                                highlightthickness=1,
                                highlightbackground="#555555",
                                bd=0)
        self.swatch.pack(side="left", padx=(4, 4))

        # Hex entry
        self.color_var = tk.StringVar(
            value=str(params.get(key, self.DEFAULT_COLOR)))
        self.entry = ctk.CTkEntry(row, textvariable=self.color_var, width=76)
        self.entry.pack(side="left", padx=(0, 6))
        self.entry.bind("<FocusOut>", self._on_entry)
        self.entry.bind("<Return>",   self._on_entry)

        ctk.CTkButton(row, text="Pick…", width=60,
                      command=self._pick).pack(side="left")

        self._update_swatch()

    # ── helpers ──────────────────────────────────────────────────────

    def _normalise(self, raw: str) -> str | None:
        """Return a canonical '#RRGGBB' string, or None if invalid."""
        s = raw.strip().lstrip("#")
        if len(s) == 6:
            try:
                int(s, 16)
                return f"#{s.upper()}"
            except ValueError:
                pass
        return None

    def _update_swatch(self):
        color = self.params.get(self.key, self._default_color)
        try:
            self.swatch.configure(bg=color)
        except Exception:
            self.swatch.configure(bg=self.DEFAULT_COLOR)

    # ── event handlers ───────────────────────────────────────────────

    def _on_entry(self, _event=None):
        normalized = self._normalise(self.color_var.get())
        if normalized:
            self.params[self.key] = normalized
            self.color_var.set(normalized)
            self._update_swatch()
            if self.on_change:
                self.on_change()
        else:
            # Revert to last valid value
            self.color_var.set(str(self.params.get(self.key, self._default_color)))

    def _pick(self):
        from tkinter import colorchooser
        current = self.params.get(self.key, self._default_color)
        result = colorchooser.askcolor(color=current,
                                       title="Choose tint colour",
                                       parent=self)
        if result and result[1]:
            color = result[1].upper()
            self.params[self.key] = color
            self.color_var.set(color)
            self._update_swatch()
            if self.on_change:
                self.on_change()

    def refresh_from_params(self):
        v = str(self.params.get(self.key, self._default_color))
        self.color_var.set(v)
        self._update_swatch()


class SectionHeader(ctk.CTkLabel):
    def __init__(self, parent, text, **kwargs):
        super().__init__(
            parent, text=text, anchor="w",
            font=ctk.CTkFont(size=13, weight="bold"),
            **kwargs,
        )


# --- Parameter panel -------------------------------------------------------

# Section names — each maps to a _build_<name> method on ParameterPanel.
# TuneTab passes the section list it wants per panel; both panels share the
# same params dict so edits from either propagate.
ALPHA_SECTIONS = (
    "alpha_source",
    "fix_path",
    "create_path",
    "fuzzythresh_path",
    "rangethresh_path",
    "edge_refinement",
    "snap",
    "alpha_shape",
    "clamp",
)
IMAGE_SECTIONS = (
    "vtracer",
    "rgb_processing",
    "tone",
    "pixelate",
    "post_processing",
    "quantize",
)


class ParameterPanel(ctk.CTkScrollableFrame):
    """All the cleanup parameter controls, wired to a shared params dict.

    Pass a `sections` tuple to limit which sections this panel renders —
    used by TuneTab to split the controls into an alpha-effects column and
    an image-effects column on either side of the preview area.
    """

    def __init__(self, parent, params: dict, on_change,
                 sections: tuple[str, ...] | None = None, **kwargs):
        kwargs.setdefault("width", 320)
        super().__init__(parent, **kwargs)
        self.params = params
        self.on_change = on_change
        self.rows: list = []
        self.sections = sections or (ALPHA_SECTIONS + IMAGE_SECTIONS)

        self._build()

    # Alpha-source path sections shown one-at-a-time based on the "Source"
    # dropdown, so only the active method's controls are visible.
    _PATH_SECTIONS = {
        "fix_path":         "fix",
        "create_path":      "create",
        "fuzzythresh_path": "fuzzythresh",
        "rangethresh_path": "rangethresh",
    }

    def _build(self):
        dispatch = {
            "vtracer":         self._build_vtracer,
            "alpha_source":    self._build_alpha_source,
            "fix_path":        self._build_fix_path,
            "create_path":     self._build_create_path,
            "fuzzythresh_path": self._build_fuzzythresh_path,
            "rangethresh_path": self._build_rangethresh_path,
            "edge_refinement": self._build_edge_refinement,
            "snap":            self._build_snap,
            "alpha_shape":     self._build_alpha_shape,
            "clamp":           self._build_clamp,
            "rgb_processing":  self._build_rgb_processing,
            "tone":            self._build_tone,
            "pixelate":        self._build_pixelate,
            "post_processing": self._build_post_processing,
            "quantize":        self._build_quantize,
        }
        # The four alpha-source path sections render into their own containers
        # inside a single wrapper, so we can show only the active one.
        self._path_containers: dict = {}
        self._path_wrapper = None
        self._cur_parent = self
        for name in self.sections:
            if name in self._PATH_SECTIONS:
                if self._path_wrapper is None:
                    self._path_wrapper = ctk.CTkFrame(self, fg_color="transparent")
                    self._path_wrapper.pack(fill="x")
                cont = ctk.CTkFrame(self._path_wrapper, fg_color="transparent")
                self._path_containers[self._PATH_SECTIONS[name]] = cont
                self._cur_parent = cont
                dispatch[name]()
                self._cur_parent = self
            else:
                dispatch[name]()
        # Trailing spacer so the last section isn't flush against the bottom
        ctk.CTkLabel(self, text="").pack(pady=10)
        self._update_alpha_path_visibility()

    def _update_alpha_path_visibility(self, *_):
        """Show only the path section matching the current alpha_source; hide
        the others. No-op for panels without path sections (e.g. image panel)."""
        if not getattr(self, "_path_containers", None):
            return
        active = self.params.get("alpha_source", "fix")
        for src, cont in self._path_containers.items():
            if src == active:
                cont.pack(fill="x")
            else:
                cont.pack_forget()

    def _on_alpha_source_change(self):
        self._update_alpha_path_visibility()
        self._fire()

    def _build_vtracer(self):
        H = HELP_TEXT
        # Persistent section wrapper so the options container (toggled below)
        # always re-packs in the right place within this section, not at the
        # bottom of the panel.
        section = ctk.CTkFrame(self, fg_color="transparent")
        section.pack(fill="x")
        SectionHeader(section, text="Vtracer pre-stage").pack(fill="x", pady=(4, 2))
        self.rows.append(ToggleRow(
            section, "Enable vtracer cleanup", "vtracer_enabled",
            self.params, help_text=H["vtracer_enabled"],
            on_change=self._on_vtracer_toggle,
        ))
        self.rows[-1].pack(fill="x")

        # All the vtracer options live in a container that's only shown when
        # the cleanup toggle is on, to keep the panel uncluttered when unused.
        box = ctk.CTkFrame(section, fg_color="transparent")
        box.pack(fill="x")
        self._vtracer_options = box

        self.rows.append(ToggleRow(
            box, "Ignore alpha (trace raw RGB)", "vtracer_ignore_alpha",
            self.params, help_text=H["vtracer_ignore_alpha"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")
        self.rows.append(SliderRow(
            box, "Pre-quantize colours (0 = off)", "vtracer_pre_quantize",
            self.params, 0, 256, 1, "{:.0f}",
            help_text=H["vtracer_pre_quantize"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")
        self.rows.append(OptionRow(
            box, "Curve fitting", "vtracer_mode", self.params,
            ["pixel", "polygon", "spline"],
            help_text=H["vtracer_mode"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")
        self.rows.append(OptionRow(
            box, "Hierarchical", "vtracer_hierarchical", self.params,
            ["stacked", "cutout"],
            help_text=H["vtracer_hierarchical"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")
        self.rows.append(SliderRow(
            box, "Color precision", "vtracer_color_precision",
            self.params, 1, 8, 1, "{:.0f}",
            help_text=H["vtracer_color_precision"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")
        self.rows.append(SliderRow(
            box, "Speckle filter", "vtracer_filter_speckle",
            self.params, 0, 32, 1, "{:.0f}",
            help_text=H["vtracer_filter_speckle"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")
        self.rows.append(SliderRow(
            box, "Gradient step", "vtracer_gradient_step",
            self.params, 1, 64, 1, "{:.0f}",
            help_text=H["vtracer_gradient_step"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")

        # Spline-mode-only controls (vtracer silently ignores them in polygon
        # mode; flagged in help text)
        ctk.CTkLabel(box, text="  (spline-mode only)", anchor="w",
                     text_color=("#888", "#888"),
                     font=ctk.CTkFont(size=10, slant="italic")
                     ).pack(fill="x", pady=(4, 0))
        self.rows.append(SliderRow(
            box, "Corner threshold °", "vtracer_corner_threshold",
            self.params, 0, 180, 1, "{:.0f}",
            help_text=H["vtracer_corner_threshold"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")
        self.rows.append(SliderRow(
            box, "Segment length", "vtracer_segment_length",
            self.params, 0.5, 16.0, 0.5, "{:.1f}",
            help_text=H["vtracer_segment_length"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")
        self.rows.append(SliderRow(
            box, "Splice threshold °", "vtracer_splice_threshold",
            self.params, 0, 180, 1, "{:.0f}",
            help_text=H["vtracer_splice_threshold"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")
        self._update_vtracer_visibility()

    def _update_vtracer_visibility(self, *_):
        """Show the vtracer options only when cleanup is enabled."""
        box = getattr(self, "_vtracer_options", None)
        if box is None:
            return
        if self.params.get("vtracer_enabled"):
            box.pack(fill="x")
        else:
            box.pack_forget()

    def _on_vtracer_toggle(self):
        self._update_vtracer_visibility()
        self._fire()

    def _build_alpha_source(self):
        H = HELP_TEXT
        SectionHeader(self, text="Alpha source").pack(fill="x", pady=(4, 2))
        self.rows.append(OptionRow(
            self, "Source", "alpha_source", self.params,
            ["fix", "create", "fuzzythresh", "rangethresh"],
            help_text=H["alpha_source"], on_change=self._on_alpha_source_change,
        ))
        self.rows[-1].pack(fill="x")

    def _build_fix_path(self):
        H = HELP_TEXT
        P = self._cur_parent
        SectionHeader(P, text="Fix path (use existing alpha)").pack(fill="x", pady=(12, 2))
        self.rows.append(SliderRow(
            P, "Level low %", "fix_level_low",
            self.params, 0, 100, 1, "{:.0f}",
            help_text=H["fix_level_low"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")
        self.rows.append(SliderRow(
            P, "Level high %", "fix_level_high",
            self.params, 0, 100, 1, "{:.0f}",
            help_text=H["fix_level_high"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")

    def _build_seed_controls(self, mode_key: str, shrink_key: str,
                             grow_key: str):
        """Seed-mode dropdown (shrink | grow) + the matching slider, kept in a
        container so toggling slider visibility stays positioned within the
        current section. Shared by the create and fuzzythresh paths."""
        H = HELP_TEXT
        box = ctk.CTkFrame(self._cur_parent, fg_color="transparent")
        box.pack(fill="x")
        mode_row = OptionRow(
            box, "Seed mode", mode_key, self.params, ["shrink", "grow"],
            help_text=H["seed_mode"], on_change=self._fire,
        )
        mode_row.pack(fill="x")
        self.rows.append(mode_row)
        shrink_row = SliderRow(
            box, "Seed shrink (px)", shrink_key, self.params,
            0, 100, 1, "{:.0f}", help_text=H["seed_shrink"], on_change=self._fire,
        )
        self.rows.append(shrink_row)
        grow_row = SliderRow(
            box, "Seed grow %", grow_key, self.params,
            0, 100, 1, "{:.0f}", help_text=H["seed_grow"], on_change=self._fire,
        )
        self.rows.append(grow_row)

        def _update(*_):
            shrink_row.pack_forget()
            grow_row.pack_forget()
            (grow_row if mode_row.var.get() == "grow"
             else shrink_row).pack(fill="x")
        mode_row.var.trace_add("write", lambda *a: _update())
        _update()

    def _build_create_path(self):
        H = HELP_TEXT
        P = self._cur_parent
        SectionHeader(P, text="Create path (luminance from RGB)").pack(fill="x", pady=(12, 2))
        self.rows.append(OptionRow(
            P, "Target fill", "create_target", self.params,
            ["gray", "white"],
            help_text=H["create_target"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")
        self.rows.append(OptionRow(
            P, "Luminance mode", "create_lum_mode", self.params,
            ["perceptual", "max-channel"],
            help_text=H["create_lum_mode"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")
        self.rows.append(ToggleRow(
            P, "Invert threshold (dark = include)", "create_lum_invert",
            self.params, help_text=H["create_lum_invert"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")
        self.rows.append(SliderRow(
            P, "Alpha floor", "create_alpha_floor",
            self.params, 0, 100, 1, "{:.0f}",
            help_text=H["create_alpha_floor"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")
        self._build_seed_controls(
            mode_key="create_seed_mode", shrink_key="create_seed_shrink",
            grow_key="create_seed_grow",
        )
        self.rows.append(SliderRow(
            P, "Expand limit (px)", "create_expand_limit",
            self.params, 0, 100, 1, "{:.0f}",
            help_text=H["create_expand_limit"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")
        self.rows.append(SliderRow(
            P, "Lum threshold %", "create_lum_threshold",
            self.params, 0, 100, 1, "{:.0f}",
            help_text=H["create_lum_threshold"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")

    def _build_fuzzythresh_path(self):
        H = HELP_TEXT
        P = self._cur_parent
        SectionHeader(P, text="Fuzzythresh path (auto histogram)").pack(fill="x", pady=(12, 2))
        self.rows.append(OptionRow(
            P, "Target fill", "fuzzythresh_target", self.params,
            ["white", "gray"],
            help_text=H["fuzzythresh_target"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")
        self.rows.append(SliderRow(
            P, "Radius (spatial weighting)", "fuzzythresh_radius",
            self.params, 0, 20, 1, "{:.0f}",
            help_text=H["fuzzythresh_radius"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")
        self._build_seed_controls(
            mode_key="fuzzythresh_seed_mode",
            shrink_key="fuzzythresh_seed_shrink",
            grow_key="fuzzythresh_seed_grow",
        )
        self.rows.append(SliderRow(
            P, "Expand limit (px)", "fuzzythresh_expand_limit",
            self.params, 0, 100, 1, "{:.0f}",
            help_text=H["fuzzythresh_expand_limit"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")
        self.rows.append(SliderRow(
            P, "Alpha floor", "fuzzythresh_alpha_floor",
            self.params, 0, 100, 1, "{:.0f}",
            help_text=H["fuzzythresh_alpha_floor"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")
        self.rows.append(ToggleRow(
            P, "Invert (dark = opaque)", "fuzzythresh_invert",
            self.params, help_text=H["fuzzythresh_invert"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")

    def _build_rangethresh_path(self):
        H = HELP_TEXT
        P = self._cur_parent
        SectionHeader(P, text="Rangethresh path (colour key)").pack(fill="x", pady=(12, 2))
        self.rows.append(OptionRow(
            P, "Target fill", "rangethresh_target", self.params,
            ["white", "gray"],
            help_text=H["rangethresh_target"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")
        self.rows.append(OptionRow(
            P, "Colourspace", "rangethresh_colorspace", self.params,
            list(pipeline.RANGETHRESH_COLORSPACES),
            help_text=H["rangethresh_colorspace"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")
        self.rows.append(OptionRow(
            P, "Unit", "rangethresh_unit", self.params,
            ["8bit", "percent"],
            help_text=H["rangethresh_unit"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")
        # R/G/B (or H/S/V·L) low/high ranges.
        for ch, hkey in (("R", "rangethresh_r"), ("G", "rangethresh_g"),
                         ("B", "rangethresh_b")):
            self.rows.append(SliderRow(
                P, f"{ch} low", f"{hkey}_low",
                self.params, 0, 255, 1, "{:.0f}",
                help_text=H["rangethresh_channels"] if ch == "R" else None,
                on_change=self._fire,
            ))
            self.rows[-1].pack(fill="x")
            self.rows.append(SliderRow(
                P, f"{ch} high", f"{hkey}_high",
                self.params, 0, 255, 1, "{:.0f}",
                on_change=self._fire,
            ))
            self.rows[-1].pack(fill="x")
        self._build_seed_controls(
            mode_key="rangethresh_seed_mode",
            shrink_key="rangethresh_seed_shrink",
            grow_key="rangethresh_seed_grow",
        )
        self.rows.append(SliderRow(
            P, "Expand limit (px)", "rangethresh_expand_limit",
            self.params, 0, 100, 1, "{:.0f}",
            help_text=H["rangethresh_expand_limit"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")
        self.rows.append(SliderRow(
            P, "Alpha floor", "rangethresh_alpha_floor",
            self.params, 0, 100, 1, "{:.0f}",
            help_text=H["rangethresh_alpha_floor"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")

    def _build_snap(self):
        H = HELP_TEXT
        SectionHeader(self, text="N-level alpha snap").pack(fill="x", pady=(12, 2))
        self.rows.append(ToggleRow(
            self, "Enable alpha snap", "snap_enabled",
            self.params, help_text=H["snap_enabled"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")
        self.rows.append(SliderRow(
            self, "Levels", "snap_levels",
            self.params, 2, 8, 1, "{:.0f}",
            help_text=H["snap_levels"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")
        self.rows.append(SliderRow(
            self, "Transparent threshold", "snap_low_threshold",
            self.params, 0, 127, 1, "{:.0f}",
            help_text=H["snap_low_threshold"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")
        self.rows.append(SliderRow(
            self, "Opaque threshold", "snap_high_threshold",
            self.params, 128, 255, 1, "{:.0f}",
            help_text=H["snap_high_threshold"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")
        self.rows.append(SliderRow(
            self, "Mid-tone tolerance ±", "snap_mid_tolerance",
            self.params, 0, 64, 1, "{:.0f}",
            help_text=H["snap_mid_tolerance"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")

    def _build_edge_refinement(self):
        H = HELP_TEXT
        SectionHeader(self, text="Edge refinement").pack(fill="x", pady=(12, 2))
        # Incongruence refine runs first in the pipeline (pre-blur), so its
        # controls lead the section.
        self.rows.append(ToggleRow(
            self, "Incongruence halo removal", "incong_enabled",
            self.params, help_text=H["incong_enabled"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")
        self.rows.append(SliderRow(
            self, "Reach (px)", "incong_reach",
            self.params, 1, 12, 1, "{:.0f}",
            help_text=H["incong_reach"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")
        self.rows.append(SliderRow(
            self, "Subject alpha floor", "incong_alpha_floor",
            self.params, 0, 128, 1, "{:.0f}",
            help_text=H["incong_alpha_floor"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")
        self.rows.append(SliderRow(
            self, "Content match threshold", "incong_content_thresh",
            self.params, 10, 100, 1, "{:.0f}",
            help_text=H["incong_content_thresh"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")
        self.rows.append(SliderRow(
            self, "Outline guard threshold", "incong_line_thresh",
            self.params, 40, 200, 5, "{:.0f}",
            help_text=H["incong_line_thresh"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")
        self.rows.append(ToggleRow(
            self, "Smooth removal edge", "incong_smooth",
            self.params, help_text=H["incong_smooth"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")
        self.rows.append(SliderRow(
            self, "Gaussian blur radius", "blur_radius",
            self.params, 0.0, 2.0, 0.05, "{:.2f}",
            help_text=H["blur_radius"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")
        self.rows.append(SliderRow(
            self, "Despeckle passes", "despeckle",
            self.params, 0, 5, 1, "{:.0f}",
            help_text=H["despeckle"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")

    def _build_alpha_shape(self):
        H = HELP_TEXT
        SectionHeader(self, text="Alpha shape").pack(fill="x", pady=(12, 2))
        self.rows.append(SliderRow(
            self, "Erosion (px) — pull outline back", "alpha_erosion",
            self.params, 0, 16, 1, "{:.0f}",
            help_text=H["alpha_erosion"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")
        self.rows.append(SliderRow(
            self, "Dilation (px) — fill outline out", "alpha_dilation",
            self.params, 0, 16, 1, "{:.0f}",
            help_text=H["alpha_dilation"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")

    def _build_clamp(self):
        H = HELP_TEXT
        SectionHeader(self, text="Alpha clamp").pack(fill="x", pady=(12, 2))

        mode_row = OptionRow(
            self, "Clamp mode", "clamp_mode",
            self.params, ["none", "binary", "tri-clamp", "ps1"],
            help_text=H["clamp_mode"], on_change=self._fire,
        )
        mode_row.pack(fill="x")
        self.rows.append(mode_row)

        # Binary frame — threshold slider, shown only for "binary" mode
        binary_frame = ctk.CTkFrame(self, fg_color="transparent")
        thresh_row = SliderRow(
            binary_frame, "Binary threshold", "binary_threshold",
            self.params, 0, 255, 1, "{:.0f}",
            help_text=H["binary_threshold"], on_change=self._fire,
        )
        thresh_row.pack(fill="x")
        self.rows.append(thresh_row)

        # Tri-clamp frame — lo/hi cutoff sliders, shown only for "tri-clamp" mode
        tri_frame = ctk.CTkFrame(self, fg_color="transparent")
        lo_row = SliderRow(
            tri_frame, "Low cutoff (→ 0)", "tri_clamp_lo",
            self.params, 0, 127, 1, "{:.0f}",
            help_text=H["tri_clamp_lo"], on_change=self._fire,
        )
        lo_row.pack(fill="x")
        self.rows.append(lo_row)
        hi_row = SliderRow(
            tri_frame, "High cutoff (→ 255)", "tri_clamp_hi",
            self.params, 128, 255, 1, "{:.0f}",
            help_text=H["tri_clamp_hi"], on_change=self._fire,
        )
        hi_row.pack(fill="x")
        self.rows.append(hi_row)

        # PS1 frame — region-aware tri-state, shown only for "ps1" mode
        ps1_frame = ctk.CTkFrame(self, fg_color="transparent")
        semi_row = SliderRow(
            ps1_frame, "Semi value (DS 143 / Beetle 127)", "ps1_semi_value",
            self.params, 1, 254, 1, "{:.0f}",
            help_text=H["ps1_semi_value"], on_change=self._fire,
        )
        semi_row.pack(fill="x")
        self.rows.append(semi_row)
        semi_tol_row = SliderRow(
            ps1_frame, "Semi detect tolerance ±", "ps1_semi_tolerance",
            self.params, 0, 100, 1, "{:.0f}",
            help_text=H["ps1_semi_tolerance"], on_change=self._fire,
        )
        semi_tol_row.pack(fill="x")
        self.rows.append(semi_tol_row)
        region_row = SliderRow(
            ps1_frame, "Min semi region (px)", "ps1_region_min",
            self.params, 0, 8, 1, "{:.0f}",
            help_text=H["ps1_region_min"], on_change=self._fire,
        )
        region_row.pack(fill="x")
        self.rows.append(region_row)

        def _update_clamp_vis(*_):
            m = mode_row.var.get()
            binary_frame.pack_forget()
            tri_frame.pack_forget()
            ps1_frame.pack_forget()
            if m == "binary":
                binary_frame.pack(fill="x")
            elif m == "tri-clamp":
                tri_frame.pack(fill="x")
            elif m == "ps1":
                ps1_frame.pack(fill="x")

        mode_row.var.trace_add("write", lambda *a: _update_clamp_vis())
        _update_clamp_vis()  # set initial visibility

    def _build_rgb_processing(self):
        H = HELP_TEXT
        SectionHeader(self, text="RGB processing").pack(fill="x", pady=(12, 2))
        self.rows.append(SliderRow(
            self, "Edge dilation (px)", "edge_dilation",
            self.params, 0, 16, 1, "{:.0f}",
            help_text=H["edge_dilation"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")
        self.rows.append(ToggleRow(
            self, "Premultiply alpha", "premultiply",
            self.params, help_text=H["premultiply"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")

    def _build_tone(self):
        H = HELP_TEXT
        SectionHeader(self, text="Tone adjustment").pack(fill="x", pady=(12, 2))
        self.rows.append(SliderRow(
            self, "Saturation", "saturation",
            self.params, 0.0, 2.0, 0.05, "{:.2f}",
            help_text=H["saturation"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")
        self.rows.append(SliderRow(
            self, "Contrast", "contrast",
            self.params, 0.0, 2.0, 0.05, "{:.2f}",
            help_text=H["contrast"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")
        self.rows.append(SliderRow(
            self, "Brightness", "brightness",
            self.params, 0.0, 2.0, 0.05, "{:.2f}",
            help_text=H["brightness"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")
        # Per-channel colour level (Fred Weinhaus levelcolors).
        for label, key in (("Level Red", "level_r"),
                           ("Level Green", "level_g"),
                           ("Level Blue", "level_b")):
            self.rows.append(SliderRow(
                self, label, key,
                self.params, -100, 100, 1, "{:.0f}",
                help_text=H["level_colors"] if key == "level_r" else None,
                on_change=self._fire,
            ))
            self.rows[-1].pack(fill="x")

        # Perceptual tone in OKLCh (hue-linear). Auto-level is independent of
        # the enable toggle; both operate on the same OKLab round-trip.
        SectionHeader(self, text="Perceptual (OKLCh)").pack(fill="x", pady=(12, 2))
        self.rows.append(ToggleRow(
            self, "Enable perceptual adjust", "perceptual_enabled",
            self.params, help_text=H["perceptual_enabled"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")
        for label, key, lo, hi, step, fmt in (
                ("Lightness (contrast)", "perceptual_lightness", 0.5, 2.0, 0.05, "{:.2f}"),
                ("Chroma (vividness)",   "perceptual_chroma",    0.0, 2.0, 0.05, "{:.2f}"),
                ("Hue rotate (deg)",     "perceptual_hue",      -180, 180, 1,    "{:.0f}")):
            self.rows.append(SliderRow(
                self, label, key, self.params, lo, hi, step, fmt,
                help_text=H[key], on_change=self._fire,
            ))
            self.rows[-1].pack(fill="x")
        self.rows.append(ToggleRow(
            self, "Auto-level lightness", "perceptual_autolevel",
            self.params, help_text=H["perceptual_autolevel"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")
        self.rows.append(SliderRow(
            self, "Auto-level clip (%)", "perceptual_autolevel_clip",
            self.params, 0.0, 5.0, 0.1, "{:.1f}",
            help_text=H["perceptual_autolevel_clip"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")

    def _build_pixelate(self):
        H = HELP_TEXT
        SectionHeader(self, text="Pixelate").pack(fill="x", pady=(12, 2))
        self.rows.append(SliderRow(
            self, "Pixelate factor (1 = off)", "pixelate_factor",
            self.params, 1.0, 16.0, 0.25, "{:.2f}",
            help_text=H["pixelate_factor"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")
        self.rows.append(SliderRow(
            self, "Soften edges (px, 0 = off)", "soften_edges",
            self.params, 0, 20, 1, "{:.0f}",
            help_text=H["soften_edges"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")

    def _build_quantize(self):
        H = HELP_TEXT
        SectionHeader(self, text="Output colour depth / palette").pack(fill="x", pady=(12, 2))
        mode_row = OptionRow(
            self, "Palette", "quantize_mode", self.params,
            list(pipeline.QUANTIZE_LABELS),
            help_text=H["quantize_mode"], on_change=self._fire,
        )
        mode_row.pack(fill="x")
        self.rows.append(mode_row)
        dither_row = SliderRow(
            self, "Dither", "quantize_dither",
            self.params, 0.0, 1.0, 0.05, "{:.2f}",
            help_text=H["quantize_dither"], on_change=self._fire,
        )
        dither_row.pack(fill="x")
        self.rows.append(dither_row)

        # Custom palette match (persistent wrapper so the path row toggles in
        # place without jumping below the trailing spacer).
        section = ctk.CTkFrame(self, fg_color="transparent")
        section.pack(fill="x")
        pal_toggle = ToggleRow(
            section, "Match palette image", "quantize_use_palette",
            self.params, help_text=H["quantize_use_palette"], on_change=self._fire,
        )
        pal_toggle.pack(fill="x")
        self.rows.append(pal_toggle)

        pal_box = ctk.CTkFrame(section, fg_color="transparent")
        self._quantize_palette_var = tk.StringVar(
            value=self.params.get("quantize_palette_path", ""))
        entry = ctk.CTkEntry(pal_box, textvariable=self._quantize_palette_var)
        entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
        ctk.CTkButton(pal_box, text="Browse…", width=80,
                      command=self._pick_palette_image).pack(side="left")

        def _on_path_change(*_):
            self.params["quantize_palette_path"] = self._quantize_palette_var.get()
            self._fire()
        self._quantize_palette_var.trace_add("write", _on_path_change)

        def _update_pal_vis(*_):
            if pal_toggle.var.get():
                pal_box.pack(fill="x")
            else:
                pal_box.pack_forget()
        pal_toggle.var.trace_add("write", lambda *a: _update_pal_vis())
        _update_pal_vis()

        # Dither only does anything when a palette reduction is active (a colour
        # depth other than "Full colour", or the palette match). Grey the slider
        # out otherwise so it's clear it has no effect.
        full_colour = pipeline.QUANTIZE_LABELS[0]

        def _update_dither_state(*_):
            active = pal_toggle.var.get() or mode_row.var.get() != full_colour
            dither_row.set_enabled(active)
        mode_row.var.trace_add("write", lambda *a: _update_dither_state())
        pal_toggle.var.trace_add("write", lambda *a: _update_dither_state())
        _update_dither_state()

    def _pick_palette_image(self):
        path = filedialog.askopenfilename(
            title="Choose palette reference image",
            initialdir=load_dialog_dir("palette_image"),
            filetypes=[("Images", " ".join(f"*{e}" for e in SUPPORTED_INPUTS)),
                       ("All files", "*.*")],
        )
        if path:
            save_dialog_dir("palette_image", path)
            self._quantize_palette_var.set(path)

    def _build_post_processing(self):
        H = HELP_TEXT
        # Late import keeps pipeline-only consumers (CLI, batch hooks) from
        # needing the post_fx module on the path.
        try:
            import post_fx as _post_fx
            _fx_values = _post_fx.EFFECTS_LIST
            _tile_values = list(_post_fx.TILE_SIZE_OPTIONS.keys())
        except ImportError:
            _fx_values = ["None"]
            _tile_values = ["Off"]

        SectionHeader(self, text="Post-Processing").pack(fill="x", pady=(12, 2))
        self.rows.append(OptionRow(
            self, "Tile size (per-tile application)", "fx_tile_size",
            self.params, _tile_values,
            help_text=H["fx_tile_size"], on_change=self._fire,
        ))
        self.rows[-1].pack(fill="x")
        try:
            _colour_tint_name = _post_fx.EFFECT_COLOUR_TINT
        except AttributeError:
            _colour_tint_name = None

        try:
            _emboss_name = _post_fx.EFFECT_EMBOSS
        except AttributeError:
            _emboss_name = None

        _emboss_compose_values = [
            "Overlay", "HardLight", "SoftLight",
            "PegtopLight", "PinLight", "VividLight", "LinearLight",
            "Multiply", "ColorBurn", "LinearBurn",
            "ColorDodge", "LinearDodge",
            "Bumpmap", "None",
        ]

        try:
            _granulate_name = _post_fx.EFFECT_GRANULATE
        except AttributeError:
            _granulate_name = None

        _gran_compose_values = [
            "Overlay", "Multiply", "Screen", "SoftLight", "HardLight",
        ]

        try:
            _chrome_name = _post_fx.EFFECT_CHROME
        except AttributeError:
            _chrome_name = None

        _chrome_compose_values = [
            "Screen", "Overlay", "HardLight", "SoftLight", "Lighten",
            "Luminize", "None",
        ]

        try:
            _frosted_name = _post_fx.EFFECT_FROSTED
        except AttributeError:
            _frosted_name = None

        try:
            _lomo_name = _post_fx.EFFECT_LOMO
        except AttributeError:
            _lomo_name = None

        try:
            _paint_name = _post_fx.EFFECT_PAINT
        except AttributeError:
            _paint_name = None

        try:
            _bevel_name = _post_fx.EFFECT_BEVEL
        except AttributeError:
            _bevel_name = None

        try:
            _davehill_name = _post_fx.EFFECT_DAVEHILL
        except AttributeError:
            _davehill_name = None

        try:
            _unsharp_name = _post_fx.EFFECT_UNSHARP
        except AttributeError:
            _unsharp_name = None

        try:
            _bilateral_name = _post_fx.EFFECT_BILATERAL
        except AttributeError:
            _bilateral_name = None

        try:
            _texturize_name = _post_fx.EFFECT_TEXTURIZE
        except AttributeError:
            _texturize_name = None

        try:
            _watercolor_name = _post_fx.EFFECT_WATERCOLOR
        except AttributeError:
            _watercolor_name = None

        try:
            _screenfx_name = _post_fx.EFFECT_SCREENFX
        except AttributeError:
            _screenfx_name = None

        for slot in (1, 2, 3, 4, 5):
            # Each slot lives in its own transparent frame so that per-effect
            # extras always appear directly below the effect dropdown and can
            # be shown/hidden cleanly via pack_forget / pack.
            slot_frame = ctk.CTkFrame(self, fg_color="transparent")
            slot_frame.pack(fill="x")

            effect_row = OptionRow(
                slot_frame, f"#{slot} effect", f"fx_effect_{slot}",
                self.params, _fx_values,
                help_text=H["fx_effect"] if slot == 1 else None,
                on_change=self._fire,
            )
            effect_row.pack(fill="x")
            self.rows.append(effect_row)

            # Strength slider — hidden when Emboss or Granulate is selected
            # (those effects use their own controls). Not packed here; the
            # updater below manages visibility.
            strength_row = SliderRow(
                slot_frame, f"#{slot} strength", f"fx_strength_{slot}",
                self.params, 0.0, 4.0, 0.1, "{:.1f}",
                help_text=H["fx_strength"] if slot == 1 else None,
                on_change=self._fire,
            )
            self.rows.append(strength_row)

            # Colour Tint extras — hidden by default, shown only for that effect
            tint_frame = None
            if _colour_tint_name is not None:
                tint_frame = ctk.CTkFrame(slot_frame, fg_color="transparent")

                color_row = ColourPickerRow(
                    tint_frame, f"fx_color_{slot}", self.params,
                    on_change=self._fire,
                )
                color_row.pack(fill="x")
                self.rows.append(color_row)

                opacity_row = SliderRow(
                    tint_frame, "Tint opacity", f"fx_opacity_{slot}",
                    self.params, 0.0, 1.0, 0.01, "{:.2f}",
                    on_change=self._fire,
                )
                opacity_row.pack(fill="x")
                self.rows.append(opacity_row)

            # Emboss extras — hidden by default, shown only for Emboss effect
            emboss_frame = None
            if _emboss_name is not None:
                emboss_frame = ctk.CTkFrame(slot_frame, fg_color="transparent")

                method_row = OptionRow(
                    emboss_frame, "Method", f"fx_emboss_method_{slot}",
                    self.params, ["1", "2"],
                    help_text=H["fx_emboss_method"] if slot == 1 else None,
                    on_change=self._fire,
                )
                method_row.pack(fill="x")
                self.rows.append(method_row)

                az_row = SliderRow(
                    emboss_frame, "Azimuth (light direction °)",
                    f"fx_emboss_azimuth_{slot}",
                    self.params, 0, 360, 5, "{:.0f}",
                    help_text=H["fx_emboss_azimuth"] if slot == 1 else None,
                    on_change=self._fire,
                )
                az_row.pack(fill="x")
                self.rows.append(az_row)

                el_row = SliderRow(
                    emboss_frame, "Elevation (light height °)",
                    f"fx_emboss_elevation_{slot}",
                    self.params, 0, 90, 1, "{:.0f}",
                    help_text=H["fx_emboss_elevation"] if slot == 1 else None,
                    on_change=self._fire,
                )
                el_row.pack(fill="x")
                self.rows.append(el_row)

                dep_row = SliderRow(
                    emboss_frame, "Depth", f"fx_emboss_depth_{slot}",
                    self.params, 1, 15, 1, "{:.0f}",
                    help_text=H["fx_emboss_depth"] if slot == 1 else None,
                    on_change=self._fire,
                )
                dep_row.pack(fill="x")
                self.rows.append(dep_row)

                eint_row = SliderRow(
                    emboss_frame, "Intensity (method 2)",
                    f"fx_emboss_intensity_{slot}",
                    self.params, -50, 50, 1, "{:.0f}",
                    help_text=H["fx_emboss_intensity"] if slot == 1 else None,
                    on_change=self._fire,
                )
                eint_row.pack(fill="x")
                self.rows.append(eint_row)

                comp_row = OptionRow(
                    emboss_frame, "Compose method",
                    f"fx_emboss_compose_{slot}",
                    self.params, _emboss_compose_values,
                    help_text=H["fx_emboss_compose"] if slot == 1 else None,
                    on_change=self._fire,
                )
                comp_row.pack(fill="x")
                self.rows.append(comp_row)

            # Granulate extras — hidden by default, shown only for Granulate
            granulate_frame = None
            if _granulate_name is not None:
                granulate_frame = ctk.CTkFrame(slot_frame, fg_color="transparent")

                gran_sigma_row = SliderRow(
                    granulate_frame, "Sigma (stroke scale)",
                    f"fx_gran_sigma_{slot}",
                    self.params, 1, 20, 1, "{:.0f}",
                    help_text=H["fx_gran_sigma"] if slot == 1 else None,
                    on_change=self._fire,
                )
                gran_sigma_row.pack(fill="x")
                self.rows.append(gran_sigma_row)

                gran_angle_row = SliderRow(
                    granulate_frame, "Angle (stroke direction °)",
                    f"fx_gran_angle_{slot}",
                    self.params, 0, 175, 5, "{:.0f}",
                    help_text=H["fx_gran_angle"] if slot == 1 else None,
                    on_change=self._fire,
                )
                gran_angle_row.pack(fill="x")
                self.rows.append(gran_angle_row)

                gran_blend_row = SliderRow(
                    granulate_frame, "Blend (intensity)",
                    f"fx_gran_blend_{slot}",
                    self.params, 0.0, 1.0, 0.05, "{:.2f}",
                    help_text=H["fx_gran_blend"] if slot == 1 else None,
                    on_change=self._fire,
                )
                gran_blend_row.pack(fill="x")
                self.rows.append(gran_blend_row)

                gran_compose_row = OptionRow(
                    granulate_frame, "Compose method",
                    f"fx_gran_compose_{slot}",
                    self.params, _gran_compose_values,
                    help_text=H["fx_gran_compose"] if slot == 1 else None,
                    on_change=self._fire,
                )
                gran_compose_row.pack(fill="x")
                self.rows.append(gran_compose_row)

            # Chrome extras — hidden by default, shown only for Chrome effect
            chrome_frame = None
            if _chrome_name is not None:
                chrome_frame = ctk.CTkFrame(slot_frame, fg_color="transparent")

                chr_smooth_row = SliderRow(
                    chrome_frame, "Smooth (pre-blur)",
                    f"fx_chrome_smooth_{slot}",
                    self.params, 0.0, 10.0, 0.5, "{:.1f}",
                    help_text=H["fx_chrome_smooth"] if slot == 1 else None,
                    on_change=self._fire,
                )
                chr_smooth_row.pack(fill="x")
                self.rows.append(chr_smooth_row)

                chr_cycles_row = SliderRow(
                    chrome_frame, "Cycles (highlight bands)",
                    f"fx_chrome_cycles_{slot}",
                    self.params, 1, 8, 1, "{:.0f}",
                    help_text=H["fx_chrome_cycles"] if slot == 1 else None,
                    on_change=self._fire,
                )
                chr_cycles_row.pack(fill="x")
                self.rows.append(chr_cycles_row)

                chr_intensity_row = SliderRow(
                    chrome_frame, "Intensity",
                    f"fx_chrome_intensity_{slot}",
                    self.params, 0, 200, 5, "{:.0f}",
                    help_text=H["fx_chrome_intensity"] if slot == 1 else None,
                    on_change=self._fire,
                )
                chr_intensity_row.pack(fill="x")
                self.rows.append(chr_intensity_row)

                chr_az_row = SliderRow(
                    chrome_frame, "Azimuth (light direction °)",
                    f"fx_chrome_azimuth_{slot}",
                    self.params, 0, 360, 5, "{:.0f}",
                    help_text=H["fx_chrome_azimuth"] if slot == 1 else None,
                    on_change=self._fire,
                )
                chr_az_row.pack(fill="x")
                self.rows.append(chr_az_row)

                chr_el_row = SliderRow(
                    chrome_frame, "Elevation (light height °)",
                    f"fx_chrome_elevation_{slot}",
                    self.params, 0, 90, 1, "{:.0f}",
                    help_text=H["fx_chrome_elevation"] if slot == 1 else None,
                    on_change=self._fire,
                )
                chr_el_row.pack(fill="x")
                self.rows.append(chr_el_row)

                chr_blend_row = SliderRow(
                    chrome_frame, "Blend (opacity)",
                    f"fx_chrome_blend_{slot}",
                    self.params, 0.0, 1.0, 0.05, "{:.2f}",
                    help_text=H["fx_chrome_blend"] if slot == 1 else None,
                    on_change=self._fire,
                )
                chr_blend_row.pack(fill="x")
                self.rows.append(chr_blend_row)

                chr_compose_row = OptionRow(
                    chrome_frame, "Compose method",
                    f"fx_chrome_compose_{slot}",
                    self.params, _chrome_compose_values,
                    help_text=H["fx_chrome_compose"] if slot == 1 else None,
                    on_change=self._fire,
                )
                chr_compose_row.pack(fill="x")
                self.rows.append(chr_compose_row)

                chr_source_row = OptionRow(
                    chrome_frame, "Source", f"fx_chrome_source_{slot}",
                    self.params, ["image", "alpha"],
                    help_text=H["fx_chrome_source"] if slot == 1 else None,
                    on_change=self._fire,
                )
                chr_source_row.pack(fill="x")
                self.rows.append(chr_source_row)

            # Frosted Glass extras — hidden by default, shown only for Frosted
            frosted_frame = None
            if _frosted_name is not None:
                frosted_frame = ctk.CTkFrame(slot_frame, fg_color="transparent")

                frost_spread_row = SliderRow(
                    frosted_frame, "Spread (displacement radius)",
                    f"fx_frost_spread_{slot}",
                    self.params, 0, 30, 1, "{:.0f}",
                    help_text=H["fx_frost_spread"] if slot == 1 else None,
                    on_change=self._fire,
                )
                frost_spread_row.pack(fill="x")
                self.rows.append(frost_spread_row)

                frost_pre_row = SliderRow(
                    frosted_frame, "Pre-blur (before spread)",
                    f"fx_frost_pre_blur_{slot}",
                    self.params, 0.0, 10.0, 0.5, "{:.1f}",
                    help_text=H["fx_frost_pre_blur"] if slot == 1 else None,
                    on_change=self._fire,
                )
                frost_pre_row.pack(fill="x")
                self.rows.append(frost_pre_row)

                frost_post_row = SliderRow(
                    frosted_frame, "Post-blur (cleanup)",
                    f"fx_frost_post_blur_{slot}",
                    self.params, 0.0, 3.0, 0.5, "{:.1f}",
                    help_text=H["fx_frost_post_blur"] if slot == 1 else None,
                    on_change=self._fire,
                )
                frost_post_row.pack(fill="x")
                self.rows.append(frost_post_row)

            # Lomo extras — hidden by default, shown only for Lomography effect
            lomo_frame = None
            if _lomo_name is not None:
                lomo_frame = ctk.CTkFrame(slot_frame, fg_color="transparent")

                lomo_type_row = OptionRow(
                    lomo_frame, "Lomo type",
                    f"fx_lomo_type_{slot}",
                    self.params, ["turquoise", "purple", "lime"],
                    help_text=H["fx_lomo_type"] if slot == 1 else None,
                    on_change=self._fire,
                )
                lomo_type_row.pack(fill="x")
                self.rows.append(lomo_type_row)

            # Paint extras — hidden by default, shown only for Paint effect
            paint_frame = None
            if _paint_name is not None:
                paint_frame = ctk.CTkFrame(slot_frame, fg_color="transparent")

                paint_gamma_row = SliderRow(
                    paint_frame, "Gamma (midtone brightness)",
                    f"fx_paint_gamma_{slot}",
                    self.params, 0.2, 3.0, 0.1, "{:.1f}",
                    help_text=H["fx_paint_gamma"] if slot == 1 else None,
                    on_change=self._fire,
                )
                paint_gamma_row.pack(fill="x")
                self.rows.append(paint_gamma_row)

                paint_blur_row = SliderRow(
                    paint_frame, "Blur (selective, pre-paint)",
                    f"fx_paint_blur_{slot}",
                    self.params, 0.0, 10.0, 0.5, "{:.1f}",
                    help_text=H["fx_paint_blur"] if slot == 1 else None,
                    on_change=self._fire,
                )
                paint_blur_row.pack(fill="x")
                self.rows.append(paint_blur_row)

                paint_radius_row = SliderRow(
                    paint_frame, "Paint radius (oil paint)",
                    f"fx_paint_radius_{slot}",
                    self.params, 0, 8, 1, "{:.0f}",
                    help_text=H["fx_paint_radius"] if slot == 1 else None,
                    on_change=self._fire,
                )
                paint_radius_row.pack(fill="x")
                self.rows.append(paint_radius_row)

                paint_edge_row = SliderRow(
                    paint_frame, "Edge gain (outline darkness)",
                    f"fx_paint_edge_{slot}",
                    self.params, 0.0, 20.0, 0.5, "{:.1f}",
                    help_text=H["fx_paint_edge"] if slot == 1 else None,
                    on_change=self._fire,
                )
                paint_edge_row.pack(fill="x")
                self.rows.append(paint_edge_row)

            # Bevel extras — hidden by default, shown only for Bevel effect
            bevel_frame = None
            if _bevel_name is not None:
                bevel_frame = ctk.CTkFrame(slot_frame, fg_color="transparent")

                bevel_form_row = OptionRow(
                    bevel_frame, "Form", f"fx_bevel_form_{slot}",
                    self.params, ["inner", "outer", "emboss"],
                    help_text=H["fx_bevel_form"] if slot == 1 else None,
                    on_change=self._fire,
                )
                bevel_form_row.pack(fill="x")
                self.rows.append(bevel_form_row)

                bevel_width_row = SliderRow(
                    bevel_frame, "Width (px)", f"fx_bevel_width_{slot}",
                    self.params, 1, 50, 1, "{:.0f}",
                    help_text=H["fx_bevel_width"] if slot == 1 else None,
                    on_change=self._fire,
                )
                bevel_width_row.pack(fill="x")
                self.rows.append(bevel_width_row)

                bevel_az_row = SliderRow(
                    bevel_frame, "Azimuth (light direction °)",
                    f"fx_bevel_azimuth_{slot}",
                    self.params, 0, 360, 5, "{:.0f}",
                    help_text=H["fx_bevel_azimuth"] if slot == 1 else None,
                    on_change=self._fire,
                )
                bevel_az_row.pack(fill="x")
                self.rows.append(bevel_az_row)

                bevel_el_row = SliderRow(
                    bevel_frame, "Elevation (light height °)",
                    f"fx_bevel_elevation_{slot}",
                    self.params, 0, 90, 1, "{:.0f}",
                    help_text=H["fx_bevel_elevation"] if slot == 1 else None,
                    on_change=self._fire,
                )
                bevel_el_row.pack(fill="x")
                self.rows.append(bevel_el_row)

                bevel_depth_row = SliderRow(
                    bevel_frame, "Depth %", f"fx_bevel_depth_{slot}",
                    self.params, 0, 200, 5, "{:.0f}",
                    help_text=H["fx_bevel_depth"] if slot == 1 else None,
                    on_change=self._fire,
                )
                bevel_depth_row.pack(fill="x")
                self.rows.append(bevel_depth_row)

                bevel_inner_row = OptionRow(
                    bevel_frame, "Inner mode", f"fx_bevel_innermode_{slot}",
                    self.params, ["hard", "smooth"],
                    help_text=H["fx_bevel_innermode"] if slot == 1 else None,
                    on_change=self._fire,
                )
                bevel_inner_row.pack(fill="x")
                self.rows.append(bevel_inner_row)

                bevel_outer_row = OptionRow(
                    bevel_frame, "Outer mode", f"fx_bevel_outermode_{slot}",
                    self.params, ["raised", "lowered"],
                    help_text=H["fx_bevel_outermode"] if slot == 1 else None,
                    on_change=self._fire,
                )
                bevel_outer_row.pack(fill="x")
                self.rows.append(bevel_outer_row)

                bevel_soften_row = SliderRow(
                    bevel_frame, "Soften (px)", f"fx_bevel_soften_{slot}",
                    self.params, 0, 20, 1, "{:.0f}",
                    help_text=H["fx_bevel_soften"] if slot == 1 else None,
                    on_change=self._fire,
                )
                bevel_soften_row.pack(fill="x")
                self.rows.append(bevel_soften_row)

                bevel_taper_row = SliderRow(
                    bevel_frame, "Taper (outer/emboss)", f"fx_bevel_taper_{slot}",
                    self.params, 0, 3, 1, "{:.0f}",
                    help_text=H["fx_bevel_taper"] if slot == 1 else None,
                    on_change=self._fire,
                )
                bevel_taper_row.pack(fill="x")
                self.rows.append(bevel_taper_row)

                bevel_color_row = ColourPickerRow(
                    bevel_frame, f"fx_bevel_bcolor_{slot}", self.params,
                    label="Bevel colour", default_color="#808080",
                    on_change=self._fire,
                )
                bevel_color_row.pack(fill="x")
                self.rows.append(bevel_color_row)

                bevel_mode_row = OptionRow(
                    bevel_frame, "Colour mode", f"fx_bevel_colormode_{slot}",
                    self.params, ["dark", "light"],
                    help_text=H["fx_bevel_colormode"] if slot == 1 else None,
                    on_change=self._fire,
                )
                bevel_mode_row.pack(fill="x")
                self.rows.append(bevel_mode_row)

                bevel_ring_row = OptionRow(
                    bevel_frame, "Ring alpha (outer/emboss)",
                    f"fx_bevel_ring_alpha_{slot}",
                    self.params, ["Opaque", "Inherit edge", "Fade out"],
                    help_text=H["fx_bevel_ring_alpha"] if slot == 1 else None,
                    on_change=self._fire,
                )
                bevel_ring_row.pack(fill="x")
                self.rows.append(bevel_ring_row)

            # Dave Hill extras — hidden by default, shown only for Dave Hill effect
            davehill_frame = None
            if _davehill_name is not None:
                davehill_frame = ctk.CTkFrame(slot_frame, fg_color="transparent")

                dh_brightness_row = SliderRow(
                    davehill_frame, "Brightness (multiply)",
                    f"fx_davehill_brightness_{slot}",
                    self.params, 0.5, 2.0, 0.05, "{:.2f}",
                    help_text=H["fx_davehill_brightness"] if slot == 1 else None,
                    on_change=self._fire,
                )
                dh_brightness_row.pack(fill="x")
                self.rows.append(dh_brightness_row)

                dh_contrast_row = SliderRow(
                    davehill_frame, "Contrast (sigmoidal)",
                    f"fx_davehill_contrast_{slot}",
                    self.params, -10.0, 10.0, 0.5, "{:.1f}",
                    help_text=H["fx_davehill_contrast"] if slot == 1 else None,
                    on_change=self._fire,
                )
                dh_contrast_row.pack(fill="x")
                self.rows.append(dh_contrast_row)

                dh_gain_row = SliderRow(
                    davehill_frame, "Gain (effect strength %)",
                    f"fx_davehill_gain_{slot}",
                    self.params, 0, 100, 5, "{:.0f}",
                    help_text=H["fx_davehill_gain"] if slot == 1 else None,
                    on_change=self._fire,
                )
                dh_gain_row.pack(fill="x")
                self.rows.append(dh_gain_row)

            # Texturize extras — hidden by default, shown only for Texturize
            texturize_frame = None
            if _texturize_name is not None:
                texturize_frame = ctk.CTkFrame(slot_frame, fg_color="transparent")

                tx_format_row = OptionRow(
                    texturize_frame, "Relief", f"fx_texturize_format_{slot}",
                    self.params, ["bump", "plain"],
                    help_text=H["fx_texturize_format"] if slot == 1 else None,
                    on_change=self._fire,
                )
                tx_format_row.pack(fill="x")
                self.rows.append(tx_format_row)

                tx_thresh_row = SliderRow(
                    texturize_frame, "Dot density %",
                    f"fx_texturize_threshold_{slot}",
                    self.params, 0.0, 100.0, 1.0, "{:.0f}",
                    help_text=H["fx_texturize_threshold"] if slot == 1 else None,
                    on_change=self._fire,
                )
                tx_thresh_row.pack(fill="x")
                self.rows.append(tx_thresh_row)

                tx_blur_row = SliderRow(
                    texturize_frame, "Weave blur (streak)",
                    f"fx_texturize_blur_{slot}",
                    self.params, 0.0, 30.0, 0.5, "{:.1f}",
                    help_text=H["fx_texturize_blur"] if slot == 1 else None,
                    on_change=self._fire,
                )
                tx_blur_row.pack(fill="x")
                self.rows.append(tx_blur_row)

                tx_widen_row = SliderRow(
                    texturize_frame, "Widen (2-D blur)",
                    f"fx_texturize_widen_{slot}",
                    self.params, 0.0, 20.0, 0.5, "{:.1f}",
                    help_text=H["fx_texturize_widen"] if slot == 1 else None,
                    on_change=self._fire,
                )
                tx_widen_row.pack(fill="x")
                self.rows.append(tx_widen_row)

                tx_gnoise_row = SliderRow(
                    texturize_frame, "Grain noise",
                    f"fx_texturize_gnoise_{slot}",
                    self.params, 0.0, 30.0, 1.0, "{:.0f}",
                    help_text=H["fx_texturize_gnoise"] if slot == 1 else None,
                    on_change=self._fire,
                )
                tx_gnoise_row.pack(fill="x")
                self.rows.append(tx_gnoise_row)

                tx_spread_row = SliderRow(
                    texturize_frame, "Spread (jitter px)",
                    f"fx_texturize_spread_{slot}",
                    self.params, 0, 20, 1, "{:.0f}",
                    help_text=H["fx_texturize_spread"] if slot == 1 else None,
                    on_change=self._fire,
                )
                tx_spread_row.pack(fill="x")
                self.rows.append(tx_spread_row)

                tx_contrast_row = SliderRow(
                    texturize_frame, "Contrast",
                    f"fx_texturize_contrast_{slot}",
                    self.params, 0.0, 10.0, 0.5, "{:.1f}",
                    help_text=H["fx_texturize_contrast"] if slot == 1 else None,
                    on_change=self._fire,
                )
                tx_contrast_row.pack(fill="x")
                self.rows.append(tx_contrast_row)

                tx_mix_row = SliderRow(
                    texturize_frame, "Mix % (texture vs image)",
                    f"fx_texturize_mix_{slot}",
                    self.params, 0.0, 100.0, 5.0, "{:.0f}",
                    help_text=H["fx_texturize_mix"] if slot == 1 else None,
                    on_change=self._fire,
                )
                tx_mix_row.pack(fill="x")
                self.rows.append(tx_mix_row)

                tx_az_row = SliderRow(
                    texturize_frame, "Light azimuth ° (bump)",
                    f"fx_texturize_azimuth_{slot}",
                    self.params, 0, 360, 5, "{:.0f}",
                    help_text=H["fx_texturize_azimuth"] if slot == 1 else None,
                    on_change=self._fire,
                )
                tx_az_row.pack(fill="x")
                self.rows.append(tx_az_row)

                tx_el_row = SliderRow(
                    texturize_frame, "Light elevation ° (bump)",
                    f"fx_texturize_elevation_{slot}",
                    self.params, 0, 90, 1, "{:.0f}",
                    help_text=H["fx_texturize_elevation"] if slot == 1 else None,
                    on_change=self._fire,
                )
                tx_el_row.pack(fill="x")
                self.rows.append(tx_el_row)

                tx_seed_row = SliderRow(
                    texturize_frame, "Seed",
                    f"fx_texturize_seed_{slot}",
                    self.params, 1, 100, 1, "{:.0f}",
                    help_text=H["fx_texturize_seed"] if slot == 1 else None,
                    on_change=self._fire,
                )
                tx_seed_row.pack(fill="x")
                self.rows.append(tx_seed_row)

            # Watercolor extras — hidden by default, shown only for Watercolor
            watercolor_frame = None
            if _watercolor_name is not None:
                watercolor_frame = ctk.CTkFrame(slot_frame, fg_color="transparent")

                wc_smooth_row = SliderRow(
                    watercolor_frame, "Smoothing (0 = off)",
                    f"fx_watercolor_smoothing_{slot}",
                    self.params, 0, 15, 1, "{:.0f}",
                    help_text=H["fx_watercolor_smoothing"] if slot == 1 else None,
                    on_change=self._fire,
                )
                wc_smooth_row.pack(fill="x")
                self.rows.append(wc_smooth_row)

                wc_edge_row = SliderRow(
                    watercolor_frame, "Edge gain",
                    f"fx_watercolor_edge_{slot}",
                    self.params, 1.0, 12.0, 0.5, "{:.1f}",
                    help_text=H["fx_watercolor_edge"] if slot == 1 else None,
                    on_change=self._fire,
                )
                wc_edge_row.pack(fill="x")
                self.rows.append(wc_edge_row)

                wc_mix_row = SliderRow(
                    watercolor_frame, "Mixing % (edge content)",
                    f"fx_watercolor_mixing_{slot}",
                    self.params, 0, 100, 1, "{:.0f}",
                    help_text=H["fx_watercolor_mixing"] if slot == 1 else None,
                    on_change=self._fire,
                )
                wc_mix_row.pack(fill="x")
                self.rows.append(wc_mix_row)

                wc_contrast_row = SliderRow(
                    watercolor_frame, "Contrast (0 = off)",
                    f"fx_watercolor_contrast_{slot}",
                    self.params, 0.0, 10.0, 0.5, "{:.1f}",
                    help_text=H["fx_watercolor_contrast"] if slot == 1 else None,
                    on_change=self._fire,
                )
                wc_contrast_row.pack(fill="x")
                self.rows.append(wc_contrast_row)

            # Screen Effects extras — hidden by default, shown only for ScreenFX
            screenfx_frame = None
            if _screenfx_name is not None:
                screenfx_frame = ctk.CTkFrame(slot_frame, fg_color="transparent")

                sc_type_row = OptionRow(
                    screenfx_frame, "Type", f"fx_screenfx_type_{slot}",
                    self.params, ["simple", "displace"],
                    help_text=H["fx_screenfx_type"] if slot == 1 else None,
                    on_change=self._fire,
                )
                sc_type_row.pack(fill="x")
                self.rows.append(sc_type_row)

                sc_spacing_row = SliderRow(
                    screenfx_frame, "Spacing (px)",
                    f"fx_screenfx_spacing_{slot}",
                    self.params, 1.0, 40.0, 0.5, "{:.1f}",
                    help_text=H["fx_screenfx_spacing"] if slot == 1 else None,
                    on_change=self._fire,
                )
                sc_spacing_row.pack(fill="x")
                self.rows.append(sc_spacing_row)

                sc_rotate_row = ToggleRow(
                    screenfx_frame, "Rotate 45°", f"fx_screenfx_rotate_{slot}",
                    self.params,
                    help_text=H["fx_screenfx_rotate"] if slot == 1 else None,
                    on_change=self._fire,
                )
                sc_rotate_row.pack(fill="x")
                self.rows.append(sc_rotate_row)

                sc_color_row = ColourPickerRow(
                    screenfx_frame, f"fx_screenfx_color_{slot}", self.params,
                    label="Screen colour (simple)", default_color="#000000",
                    on_change=self._fire,
                )
                sc_color_row.pack(fill="x")
                self.rows.append(sc_color_row)

                sc_amount_row = SliderRow(
                    screenfx_frame, "Amount (displace)",
                    f"fx_screenfx_amount_{slot}",
                    self.params, 0.0, 30.0, 0.5, "{:.1f}",
                    help_text=H["fx_screenfx_amount"] if slot == 1 else None,
                    on_change=self._fire,
                )
                sc_amount_row.pack(fill="x")
                self.rows.append(sc_amount_row)

                sc_method_row = OptionRow(
                    screenfx_frame, "Method (displace)",
                    f"fx_screenfx_method_{slot}",
                    self.params, ["1", "2", "3"],
                    help_text=H["fx_screenfx_method"] if slot == 1 else None,
                    on_change=self._fire,
                )
                sc_method_row.pack(fill="x")
                self.rows.append(sc_method_row)

            # Crosshatch extras — shown alongside strength for Unsharp/Bilateral
            crosshatch_frame = None
            if _unsharp_name is not None or _bilateral_name is not None:
                crosshatch_frame = ctk.CTkFrame(slot_frame, fg_color="transparent")

                xh_enable_row = ToggleRow(
                    crosshatch_frame, "Enable crosshatch overlay",
                    f"fx_crosshatch_{slot}",
                    self.params,
                    help_text=H["fx_crosshatch"] if slot == 1 else None,
                    on_change=self._fire,
                )
                xh_enable_row.pack(fill="x")
                self.rows.append(xh_enable_row)

                xh_length_row = SliderRow(
                    crosshatch_frame, "Stroke length (px)",
                    f"fx_crosshatch_length_{slot}",
                    self.params, 1, 30, 1, "{:.0f}",
                    help_text=H["fx_crosshatch_length"] if slot == 1 else None,
                    on_change=self._fire,
                )
                xh_length_row.pack(fill="x")
                self.rows.append(xh_length_row)

                xh_balance_row = SliderRow(
                    crosshatch_frame, "Diagonal balance",
                    f"fx_crosshatch_balance_{slot}",
                    self.params, -100, 100, 5, "{:.0f}",
                    help_text=H["fx_crosshatch_balance"] if slot == 1 else None,
                    on_change=self._fire,
                )
                xh_balance_row.pack(fill="x")
                self.rows.append(xh_balance_row)

                xh_sharp_row = SliderRow(
                    crosshatch_frame, "Secondary sharpen",
                    f"fx_crosshatch_sharp_{slot}",
                    self.params, 0, 30, 1, "{:.0f}",
                    help_text=H["fx_crosshatch_sharp"] if slot == 1 else None,
                    on_change=self._fire,
                )
                xh_sharp_row.pack(fill="x")
                self.rows.append(xh_sharp_row)

                xh_mix_row = SliderRow(
                    crosshatch_frame, "Mix % (effect vs original)",
                    f"fx_crosshatch_mix_{slot}",
                    self.params, 0, 100, 5, "{:.0f}",
                    help_text=H["fx_crosshatch_mix"] if slot == 1 else None,
                    on_change=self._fire,
                )
                xh_mix_row.pack(fill="x")
                self.rows.append(xh_mix_row)

            # Unified show/hide updater: manages strength, tint, emboss,
            # granulate, chrome, frosted, lomo, paint, bevel, davehill, and
            # crosshatch visibility whenever the effect dropdown changes.
            # the effect dropdown changes. All frames are pack_forget'd and
            # re-packed in order each call so stacking order stays consistent.
            def _make_updater(erow, srow, tf, emf, grf, chf, frf, lmf, ptf,
                              bvf, dhf, xhf, txf, wcf, scf,
                              ct=_colour_tint_name,
                              em=_emboss_name, gr=_granulate_name,
                              ch=_chrome_name, fr=_frosted_name,
                              lm=_lomo_name, pt=_paint_name,
                              bv=_bevel_name, dh=_davehill_name,
                              us=_unsharp_name, bi=_bilateral_name,
                              tx=_texturize_name, wc=_watercolor_name,
                              sc=_screenfx_name):
                def _update(*_):
                    eff = erow.var.get()
                    srow.pack_forget()
                    for frame in (tf, emf, grf, chf, frf, lmf, ptf, bvf, dhf,
                                  xhf, txf, wcf, scf):
                        if frame is not None:
                            frame.pack_forget()
                    if eff not in (em, gr, ch, fr, lm, pt, bv, dh, tx, wc, sc):
                        srow.pack(fill="x")
                    if tf is not None and eff == ct:
                        tf.pack(fill="x")
                    if emf is not None and eff == em:
                        emf.pack(fill="x")
                    if grf is not None and eff == gr:
                        grf.pack(fill="x")
                    if chf is not None and eff == ch:
                        chf.pack(fill="x")
                    if frf is not None and eff == fr:
                        frf.pack(fill="x")
                    if lmf is not None and eff == lm:
                        lmf.pack(fill="x")
                    if ptf is not None and eff == pt:
                        ptf.pack(fill="x")
                    if bvf is not None and eff == bv:
                        bvf.pack(fill="x")
                    if dhf is not None and eff == dh:
                        dhf.pack(fill="x")
                    if xhf is not None and eff in (us, bi):
                        xhf.pack(fill="x")
                    if txf is not None and eff == tx:
                        txf.pack(fill="x")
                    if wcf is not None and eff == wc:
                        wcf.pack(fill="x")
                    if scf is not None and eff == sc:
                        scf.pack(fill="x")
                return _update

            upd = _make_updater(
                effect_row, strength_row,
                tint_frame, emboss_frame, granulate_frame, chrome_frame,
                frosted_frame, lomo_frame, paint_frame,
                bevel_frame, davehill_frame, crosshatch_frame,
                texturize_frame, watercolor_frame, screenfx_frame,
            )
            effect_row.var.trace_add("write", lambda *a, u=upd: u())
            upd()   # set initial visibility

    def _fire(self):
        if self.on_change:
            self.on_change()

    def refresh_all(self):
        for row in self.rows:
            row.refresh_from_params()
        self._update_alpha_path_visibility()
        self._update_vtracer_visibility()


# --- Preset I/O ------------------------------------------------------------

def list_presets() -> list[tuple[str, Path]]:
    """Return [(display_name, path), ...] from shipped + user dirs."""
    out: dict[str, Path] = {}
    for d in (shipped_presets_dir(), user_presets_dir()):
        if not d.exists():
            continue
        for f in sorted(d.glob("*.json")):
            out[f.stem] = f   # user dir overrides shipped on name collision
    return [(name, path) for name, path in sorted(out.items())]


def load_preset(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("params", data)


def load_preset_description(path: Path) -> str:
    """Return the human-readable description from a preset's JSON, or '' if
    none / unreadable."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("description", "") or ""
    except Exception:
        return ""


def find_preset_path(name: str) -> Path | None:
    """Look up a preset's path by display name (matches list_presets entries)."""
    for n, path in list_presets():
        if n == name:
            return path
    return None


def save_preset(path: Path, params: dict):
    payload = {"version": 1, "params": params}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# --- Preview worker (debounced background processing) ---------------------

class PreviewWorker:
    """Runs the pipeline on a downscaled copy of the source on a worker
    thread. Coalesces rapid parameter changes into the most recent set.
    Calls `on_result(rgba_image)` on the Tk main thread when done."""

    def __init__(self, tk_root: tk.Misc, on_result, on_error):
        self.root = tk_root
        self.on_result = on_result
        self.on_error = on_error

        self._lock = threading.Lock()
        self._pending: dict | None = None    # (src_img, params)
        self._busy = False
        self._stopping = False
        # Persistent stage cache: only the pipeline stages affected by a given
        # parameter change are recomputed between previews. Touched solely by
        # the worker thread (one job at a time), so no extra locking needed.
        self._pipeline = pipeline.PreviewPipeline()

    def request(self, src_img: Image.Image, params: dict):
        with self._lock:
            self._pending = {"src": src_img, "params": dict(params)}
            if not self._busy:
                self._busy = True
                threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self._stopping = True

    def _run(self):
        while True:
            with self._lock:
                if self._pending is None:
                    self._busy = False
                    return
                job = self._pending
                self._pending = None

            try:
                result = self._process(job["src"], job["params"])
            except Exception:
                err = traceback.format_exc()
                # Mirror the success-path guard: don't touch the Tk root after
                # the app has begun shutting down, or root.after() raises a
                # TclError on the destroyed widget.
                if not self._stopping:
                    self.root.after(0, lambda e=err: self.on_error(e))
                continue

            if not self._stopping:
                self.root.after(0, lambda r=result: self.on_result(r))

    def _process(self, src_img: Image.Image, params: dict) -> Image.Image:
        """Run the pipeline on an in-memory image and return the resulting
        RGBA image. Uses a persistent stage cache (no filesystem round-trip),
        so each debounced parameter change recomputes only the stages at and
        after whatever the user actually changed — a tone/fx tweak no longer
        re-runs alpha synthesis or morphology."""
        return self._pipeline.run(src_img, params)


# --- Tune tab --------------------------------------------------------------

# Grid dropdown options — keyed by display label so the StringVar carries
# the label and we look up the int directly instead of re-parsing the label.
_GRID_SIZE_OPTIONS = {
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
_GRID_WIDTH_OPTIONS = {
    "1 px":  1,
    "2 px":  2,
    "4 px":  4,
    "8 px":  8,
    "16 px": 16,
}


def _hex_to_rgb(h: str):
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _rgb_to_hex(rgb):
    return "#{:02X}{:02X}{:02X}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


class ManualEditsPanel(ctk.CTkFrame):
    """Manual touch-up controls (top-right, under Save image). This UI is the
    source of truth for the tool / brush / values; it pushes them onto the
    TuneTab's per-image EditController. Editing itself is gated by the view mode
    (exactly one of show-alpha-only / show-texture-only)."""
    _TOOL_MAP = {"Pencil": "pencil", "Eraser": "eraser", "Picker": "picker"}

    def __init__(self, parent, tune, **kwargs):
        super().__init__(parent, **kwargs)
        self.tune = tune
        self._rgb_hex = "#FF0000"
        self._build()

    def _build(self):
        SectionHeader(self, text="Manual edits").pack(fill="x", pady=(6, 2))
        self._hint = ctk.CTkLabel(
            self, text="", wraplength=280, justify="left", anchor="w",
            font=ctk.CTkFont(size=11), text_color=("#666", "#aaa"))
        self._hint.pack(fill="x", padx=6, pady=(0, 4))

        self._tool_var = tk.StringVar(value="Pencil")
        self._tools = ctk.CTkSegmentedButton(
            self, values=["Pencil", "Eraser", "Picker"],
            variable=self._tool_var, command=lambda _v: self._push())
        self._tools.pack(fill="x", padx=6, pady=2)

        size_row = ctk.CTkFrame(self, fg_color="transparent")
        size_row.pack(fill="x", padx=6, pady=2)
        ctk.CTkLabel(size_row, text="Brush (px):", anchor="w").pack(side="left")
        self._size_var = tk.IntVar(value=4)
        self._size_minus = ctk.CTkButton(size_row, text="−", width=28,
                                         command=lambda: self._bump_size(-1))
        self._size_minus.pack(side="left", padx=(6, 2))
        ctk.CTkLabel(size_row, textvariable=self._size_var, width=40).pack(side="left")
        self._size_plus = ctk.CTkButton(size_row, text="+", width=28,
                                        command=lambda: self._bump_size(1))
        self._size_plus.pack(side="left", padx=(2, 0))

        a_row = ctk.CTkFrame(self, fg_color="transparent")
        a_row.pack(fill="x", padx=6, pady=2)
        ctk.CTkLabel(a_row, text="Alpha:", anchor="w", width=48).pack(side="left")
        self._alpha_var = tk.IntVar(value=255)
        ctk.CTkLabel(a_row, textvariable=self._alpha_var, width=34).pack(side="right")
        self._alpha_slider = ctk.CTkSlider(a_row, from_=0, to=255,
                                          number_of_steps=255, command=self._on_alpha)
        self._alpha_slider.set(255)
        self._alpha_slider.pack(side="left", fill="x", expand=True, padx=6)

        c_row = ctk.CTkFrame(self, fg_color="transparent")
        c_row.pack(fill="x", padx=6, pady=2)
        ctk.CTkLabel(c_row, text="Colour:", anchor="w", width=48).pack(side="left")
        self._colour_btn = ctk.CTkButton(
            c_row, text="", width=44, height=24, fg_color=self._rgb_hex,
            hover_color=self._rgb_hex, command=self._pick_colour)
        self._colour_btn.pack(side="left", padx=6)

        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(fill="x", padx=6, pady=(6, 6))
        ctk.CTkButton(btn_row, text="Stop editing", width=88,
                      command=lambda: self.tune._stop_editing()).pack(side="left", padx=(0, 4))
        ctk.CTkButton(btn_row, text="Save edits", width=84,
                      command=lambda: self.tune._save_edits()).pack(side="left", padx=(0, 4))
        ctk.CTkButton(btn_row, text="Delete all", width=80,
                      fg_color=("#b04a4a", "#7a3030"), hover_color=("#c25a5a", "#8a3838"),
                      command=lambda: self.tune._delete_all_edits()).pack(side="left")

        self.set_enabled(False, None)

    def _bump_size(self, d):
        self._size_var.set(max(1, min(256, self._size_var.get() + d)))
        self._push()

    def _on_alpha(self, v):
        self._alpha_var.set(int(round(float(v))))
        self._push()

    def _pick_colour(self):
        from tkinter import colorchooser
        r = colorchooser.askcolor(color=self._rgb_hex, title="Edit colour", parent=self)
        if r and r[1]:
            self._rgb_hex = r[1].upper()
            self._colour_btn.configure(fg_color=self._rgb_hex, hover_color=self._rgb_hex)
            self._push()

    def _push(self):
        """Push the UI selections onto the current controller."""
        e = self.tune.edit
        if e is None:
            return
        e.tool = self._TOOL_MAP.get(self._tool_var.get(), "pencil")
        e.brush = int(self._size_var.get())
        e.alpha_value = int(self._alpha_var.get())
        e.rgb_value = _hex_to_rgb(self._rgb_hex)
        self.tune._update_edit_mode()          # refresh brush-cursor size + gating

    def sync_to_controller(self):
        """Called when a new per-image controller is created."""
        self._push()

    def reflect_picked_value(self):
        """The value picker set a value and reverted to pencil — mirror it."""
        e = self.tune.edit
        if e is None:
            return
        self._tool_var.set("Pencil")
        self._tools.set("Pencil")
        if e.channel == "alpha":
            self._alpha_var.set(int(e.alpha_value))
            self._alpha_slider.set(int(e.alpha_value))
        else:
            self._rgb_hex = _rgb_to_hex(e.rgb_value)
            self._colour_btn.configure(fg_color=self._rgb_hex, hover_color=self._rgb_hex)

    def set_enabled(self, active: bool, channel):
        """Enable controls per edit state; the alpha slider and colour picker
        follow the active channel."""
        def _cfg(w, on):
            try:
                w.configure(state="normal" if on else "disabled")
            except Exception:
                pass
        for w in (self._tools, self._size_minus, self._size_plus):
            _cfg(w, active)
        _cfg(self._alpha_slider, active and channel == "alpha")
        _cfg(self._colour_btn, active and channel == "rgb")
        if not active:
            self._hint.configure(
                text="Tick “Show alpha only” or “Show texture only” to edit.")
        else:
            ch = "Alpha" if channel == "alpha" else "RGB colour"
            self._hint.configure(
                text=f"Editing {ch} — left-drag paints, right-drag pans, Ctrl+Z undo.")


class TuneTab(ctk.CTkFrame):
    def __init__(self, parent, params: dict, **kwargs):
        super().__init__(parent, **kwargs)
        self.params = params
        self.source_image: Image.Image | None = None       # full-res RGBA
        self.source_path: str | None = None                # path it was loaded from
        self.preview_source: Image.Image | None = None     # downscaled RGBA
        self.last_after: Image.Image | None = None

        self._debounce_after_id: str | None = None
        self._show_alpha_only = tk.BooleanVar(value=False)
        self._show_texture_only = tk.BooleanVar(value=False)
        self._bg_solid = tk.BooleanVar(value=False)
        self._bg_color: str = "#808080"   # initial solid background colour

        # ROI full-res detail: a coalescing background worker recomputes the
        # visible crop at full resolution when the view settles.
        self._detail_lock = threading.Lock()
        self._detail_pending: dict | None = None
        self._detail_busy = False
        self._detail_token = 0        # bumps on every settle to drop stale jobs
        # Trigger detail when zoomed past ~1:1 of the full-res source; skip when
        # the visible full-res region is bigger than this (zoomed out, and the
        # low-res preview is fine there anyway).
        self._DETAIL_MAX_PX = 4_000_000

        self._build()

        self.worker = PreviewWorker(
            self, on_result=self._on_preview_result, on_error=self._on_preview_error,
        )

    def _build(self):
        # Three-column layout: [alpha-effect panel] [preview centre] [image-effect panel]
        self.columnconfigure(0, weight=0)   # alpha panel — fixed width
        self.columnconfigure(1, weight=1)   # preview area — expands
        self.columnconfigure(2, weight=0)   # image panel — fixed width
        self.rowconfigure(0, weight=1)

        # Left column: alpha-modifying controls
        self.alpha_panel = ParameterPanel(
            self, self.params, on_change=self._on_param_change,
            sections=ALPHA_SECTIONS, width=300,
        )
        self.alpha_panel.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=8)

        # Centre column: preview area (toolbars, side-by-side panes, status)
        left = ctk.CTkFrame(self)
        left.grid(row=0, column=1, sticky="nsew", padx=4, pady=8)
        left.rowconfigure(3, weight=1)  # preview row expands
        left.columnconfigure(0, weight=1)

        # Toolbar (row 1: image + settings buttons)
        toolbar = ctk.CTkFrame(left, fg_color="transparent")
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ctk.CTkButton(toolbar, text="Load image…", width=110,
                      command=self._load_image).pack(side="left", padx=(0, 6))
        ctk.CTkButton(toolbar, text="Save preset…", width=110,
                      command=self._save_preset).pack(side="left", padx=(0, 6))
        ctk.CTkButton(toolbar, text="Reset to defaults", width=130,
                      command=self._reset_defaults).pack(side="left", padx=(0, 6))
        ctk.CTkCheckBox(toolbar, text="Show alpha only",
                        variable=self._show_alpha_only,
                        command=lambda: self._on_view_mode_changed("alpha")
                        ).pack(side="left", padx=(12, 0))
        ctk.CTkCheckBox(toolbar, text="Show texture only",
                        variable=self._show_texture_only,
                        command=lambda: self._on_view_mode_changed("texture")
                        ).pack(side="left", padx=(8, 0))
        ctk.CTkCheckBox(toolbar, text="Solid bg",
                        variable=self._bg_solid,
                        command=self._on_bg_changed,
                        ).pack(side="left", padx=(8, 0))
        self._bg_swatch = ctk.CTkButton(
            toolbar, text="", width=26, height=26,
            fg_color=self._bg_color, hover_color=self._bg_color,
            command=self._pick_bg_color,
        )
        self._bg_swatch.pack(side="left", padx=(4, 0))
        ctk.CTkButton(toolbar, text="Save image…", width=110,
                      command=self._save_image).pack(side="right")

        # Toolbar (row 2: preset picker + description)
        preset_row = ctk.CTkFrame(left, fg_color="transparent")
        preset_row.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        preset_row.columnconfigure(3, weight=1)

        ctk.CTkLabel(preset_row, text="Preset:", width=50, anchor="w"
                     ).grid(row=0, column=0, sticky="w")
        self._tune_preset_var = tk.StringVar(value="(Current settings)")
        self._tune_preset_menu = ctk.CTkOptionMenu(
            preset_row, variable=self._tune_preset_var,
            values=self._preset_dropdown_values(),
            command=self._on_tune_preset_pick, width=270,
        )
        self._tune_preset_menu.grid(row=0, column=1, sticky="w", padx=(4, 6))
        ctk.CTkButton(preset_row, text="Load from file…", width=120,
                      command=self._load_preset).grid(row=0, column=2, padx=(0, 12))
        self._tune_preset_desc = ctk.CTkLabel(
            preset_row,
            text="Pick a preset to see its description here.",
            wraplength=600, justify="left", anchor="w",
            font=ctk.CTkFont(size=11),
            text_color=("#666", "#aaa"),
        )
        self._tune_preset_desc.grid(row=0, column=3, sticky="ew")

        # Toolbar (row 3: grid overlay controls — for tile-seam inspection)
        grid_row = ctk.CTkFrame(left, fg_color="transparent")
        grid_row.grid(row=2, column=0, sticky="ew", pady=(0, 6))

        ctk.CTkLabel(grid_row, text="Grid:", width=40, anchor="w"
                     ).pack(side="left")
        self._grid_size_var = tk.StringVar(value="Off")
        ctk.CTkOptionMenu(
            grid_row, variable=self._grid_size_var,
            values=list(_GRID_SIZE_OPTIONS.keys()),
            command=lambda _v: self._apply_grid_to_panes(), width=90,
        ).pack(side="left", padx=(0, 8))

        ctk.CTkLabel(grid_row, text="Line:", anchor="w").pack(side="left")
        self._grid_width_var = tk.StringVar(value="1 px")
        ctk.CTkOptionMenu(
            grid_row, variable=self._grid_width_var,
            values=list(_GRID_WIDTH_OPTIONS.keys()),
            command=lambda _v: self._apply_grid_to_panes(), width=70,
        ).pack(side="left", padx=(4, 8))

        ctk.CTkLabel(grid_row, text="Colour:", anchor="w").pack(side="left")
        self._grid_color_var = tk.StringVar(value="red")
        for label in ZoomPanPreview._GRID_COLORS:
            ctk.CTkRadioButton(
                grid_row, text=label, value=label,
                variable=self._grid_color_var,
                command=self._apply_grid_to_panes,
            ).pack(side="left", padx=(6, 0))

        self._grid_semi_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            grid_row, text="Semi-transparent",
            variable=self._grid_semi_var,
            command=self._apply_grid_to_panes,
        ).pack(side="left", padx=(16, 0))

        # Preview row
        preview_row = ctk.CTkFrame(left, fg_color="transparent")
        preview_row.grid(row=3, column=0, sticky="nsew")
        preview_row.columnconfigure(0, weight=1)
        preview_row.columnconfigure(1, weight=1)
        preview_row.rowconfigure(1, weight=1)

        ctk.CTkLabel(preview_row, text="Before",
                     font=ctk.CTkFont(size=12, weight="bold")
                     ).grid(row=0, column=0, sticky="w", padx=4)
        ctk.CTkLabel(preview_row, text="After",
                     font=ctk.CTkFont(size=12, weight="bold")
                     ).grid(row=0, column=1, sticky="w", padx=4)

        # Cursor coordinate / pixel readout, right-aligned in the header row.
        self.coord_label = ctk.CTkLabel(
            preview_row, text="", anchor="e",
            font=ctk.CTkFont(family="Consolas", size=11),
            text_color=("#555", "#aaa"),
        )
        self.coord_label.grid(row=0, column=1, sticky="e", padx=4)

        self.before_view = ZoomPanPreview(preview_row, placeholder="Load an image to begin")
        self.before_view.grid(row=1, column=0, sticky="nsew", padx=2)
        self.after_view = ZoomPanPreview(preview_row, placeholder="Live preview")
        self.after_view.grid(row=1, column=1, sticky="nsew", padx=2)
        self.before_view.link_with(self.after_view)
        self.before_view.on_cursor_move = (
            lambda c: self._update_coord_readout(c, "before"))
        self.after_view.on_cursor_move = (
            lambda c: self._update_coord_readout(c, "after"))
        # ROI full-res detail: recompute the on-screen crop at full resolution
        # when the view settles after a zoom/pan (either pane fires; both are
        # updated). Only the interacted pane schedules, so no double work.
        self.before_view.on_view_settle = self._on_view_settle
        self.after_view.on_view_settle = self._on_view_settle

        # Manual editing happens on the After pane (the result being touched up).
        self.edit = None                        # manual_edits.EditController per image
        self.after_view.on_paint_begin = self._edit_begin
        self.after_view.on_paint_to = self._edit_to
        self.after_view.on_paint_end = self._edit_end
        for seq in ("<Control-z>", "<Control-Z>"):
            self.after_view.canvas.bind(seq, lambda e: self._edit_undo())

        # Status line
        self.status_label = ctk.CTkLabel(left, text="", anchor="w",
                                         font=ctk.CTkFont(size=11))
        self.status_label.grid(row=4, column=0, sticky="ew", pady=(4, 0))

        # Right column: manual-edits panel (top) + image-modifying controls.
        right = ctk.CTkFrame(self, fg_color="transparent")
        right.grid(row=0, column=2, sticky="nsew", padx=(4, 8), pady=8)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)     # image panel expands
        self.manual_edits_panel = ManualEditsPanel(right, self, width=300)
        self.manual_edits_panel.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.image_panel = ParameterPanel(
            right, self.params, on_change=self._on_param_change,
            sections=IMAGE_SECTIONS, width=300,
        )
        self.image_panel.grid(row=1, column=0, sticky="nsew")

    # -- file ops ------------------------------------------------------------

    def _load_image(self):
        path = filedialog.askopenfilename(
            title="Load texture",
            initialdir=load_dialog_dir("load_image"),
            filetypes=[
                ("Images", " ".join(f"*{e}" for e in SUPPORTED_INPUTS)),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        save_dialog_dir("load_image", path)
        try:
            img = pipeline.load_rgba(path)
        except Exception as e:
            messagebox.showerror("Load failed", f"Couldn't open image:\n{e}")
            return
        self.source_image = img
        self.source_path = path
        self.preview_source = self._make_preview_source(img)
        self._init_edit_layer()             # fresh edit layer for this image
        self._redraw_previews()
        self._schedule_preview()
        self.status_label.configure(
            text=f"Loaded {Path(path).name} — {img.width}×{img.height}px",
        )

    def _save_image(self):
        if self.last_after is None:
            messagebox.showinfo("Nothing to save",
                                "Load an image and let the preview render first.")
            return
        # A single "DDS" filter — asksaveasfilename can't report which filter
        # the user picked, so two DDS entries (DXT1/DXT5) couldn't be told
        # apart. The compression is chosen explicitly below once the path is
        # known to be .dds.
        # Default the filename to the loaded image's name (stem only, so the
        # user still picks the output format via the dialog's type filter).
        initialfile = ""
        if self.source_path:
            initialfile = Path(self.source_path).stem
        path = filedialog.asksaveasfilename(
            title="Save cleaned texture",
            initialdir=load_dialog_dir("save_image"),
            initialfile=initialfile,
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("TIFF", "*.tif;*.tiff"),
                       ("WebP", "*.webp"),
                       ("DDS", "*.dds"),
                       ("All files", "*.*")],
        )
        if not path:
            return
        save_dialog_dir("save_image", path)
        # For the saved file, run the pipeline at full resolution (not the
        # downscaled preview).
        try:
            if self.source_image is None:
                raise RuntimeError("No source image loaded.")
            # Derive DDS compression by asking the user, since the file dialog
            # cannot tell us which compression they intended.
            save_params = dict(self.params)
            if Path(path).suffix.lower() == ".dds":
                use_dxt5 = messagebox.askyesno(
                    "DDS compression",
                    "Save as DXT5 (BC3, full 8-bit alpha)?\n\n"
                    "Yes → DXT5 — keeps the alpha channel.\n"
                    "No  → DXT1 — BC1, RGB, no/1-bit alpha, smaller file.",
                )
                save_params["dds_compression"] = "dxt5" if use_dxt5 else "dxt1"
            import tempfile
            with tempfile.TemporaryDirectory(prefix="alphasave_") as tmp:
                tmp = Path(tmp)
                in_path = tmp / "src.png"
                self.source_image.save(in_path, compress_level=1)
                pipeline.process_image(in_path, path, save_params, work_dir=tmp)
            self.status_label.configure(text=f"Saved {path}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def _save_preset(self):
        path = filedialog.asksaveasfilename(
            title="Save preset",
            initialdir=load_dialog_dir("save_preset", user_presets_dir()),
            defaultextension=".json",
            filetypes=[("Preset JSON", "*.json")],
        )
        if not path:
            return
        save_dialog_dir("save_preset", path)
        try:
            save_preset(Path(path), self.params)
            self.status_label.configure(text=f"Preset saved: {Path(path).name}")
        except Exception as e:
            messagebox.showerror("Save preset failed", str(e))

    def _refresh_all_panels(self):
        """Refresh both parameter panels (split into alpha / image columns)."""
        self.alpha_panel.refresh_all()
        self.image_panel.refresh_all()

    def _reset_defaults(self):
        for k, v in pipeline.DEFAULT_PARAMS.items():
            self.params[k] = v
        self._refresh_all_panels()
        self._schedule_preview()
        self._tune_preset_var.set("(Current settings)")
        self._tune_preset_desc.configure(
            text="Settings reset to defaults.",
            text_color=("#666", "#aaa"),
        )
        self.status_label.configure(text="Settings reset to defaults")

    def _load_preset(self):
        path = filedialog.askopenfilename(
            title="Load preset",
            initialdir=load_dialog_dir(
                "load_preset",
                shipped_presets_dir() if shipped_presets_dir().exists()
                else user_presets_dir()),
            filetypes=[("Preset JSON", "*.json")],
        )
        if not path:
            return
        save_dialog_dir("load_preset", path)
        try:
            loaded = load_preset(Path(path))
            desc = load_preset_description(Path(path))
        except Exception as e:
            messagebox.showerror("Load preset failed", str(e))
            return
        for k, v in loaded.items():
            self.params[k] = v
        self._refresh_all_panels()
        self._schedule_preview()
        self._tune_preset_var.set(Path(path).stem)
        self._tune_preset_desc.configure(
            text=desc or "(No description in preset file.)",
            text_color=("#222", "#ddd") if desc else ("#666", "#aaa"),
        )
        self.status_label.configure(text=f"Preset loaded: {Path(path).stem}")

    # -- preset dropdown wiring (Tune) --------------------------------------

    def _preset_dropdown_values(self) -> list[str]:
        return ["(Current settings)"] + [name for name, _ in list_presets()]

    def _refresh_tune_presets(self):
        self._tune_preset_menu.configure(values=self._preset_dropdown_values())

    def _on_tune_preset_pick(self, choice: str):
        if choice == "(Current settings)":
            self._tune_preset_desc.configure(
                text="No preset applied — Tune sliders hold the current values.",
                text_color=("#666", "#aaa"),
            )
            return
        path = find_preset_path(choice)
        if path is None:
            self._tune_preset_desc.configure(
                text=f"Preset '{choice}' not found.",
                text_color=("#a33", "#f88"),
            )
            return
        try:
            loaded = load_preset(path)
        except Exception as e:
            messagebox.showerror("Load preset failed", str(e))
            return
        for k, v in loaded.items():
            self.params[k] = v
        self._refresh_all_panels()
        self._schedule_preview()
        desc = load_preset_description(path)
        self._tune_preset_desc.configure(
            text=desc or "(No description in preset file.)",
            text_color=("#222", "#ddd") if desc else ("#666", "#aaa"),
        )
        self.status_label.configure(text=f"Preset loaded: {choice}")

    # -- preview pipeline ----------------------------------------------------

    def _make_preview_source(self, full: Image.Image) -> Image.Image:
        """Downscale to PREVIEW_MAX_DIM for snappy live updates."""
        w, h = full.size
        if max(w, h) <= PREVIEW_MAX_DIM:
            return full.copy()
        scale = PREVIEW_MAX_DIM / max(w, h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        return full.resize((nw, nh), resample=Image.Resampling.LANCZOS)

    def _on_param_change(self):
        self._schedule_preview()

    def _schedule_preview(self):
        if self.preview_source is None:
            return
        if self._debounce_after_id is not None:
            try:
                self.after_cancel(self._debounce_after_id)
            except tk.TclError:
                pass
        self._debounce_after_id = self.after(
            PREVIEW_DEBOUNCE_MS, self._kick_preview,
        )

    def _kick_preview(self):
        self._debounce_after_id = None
        if self.preview_source is None:
            return
        self.status_label.configure(text="Updating preview…")
        self.worker.request(self.preview_source, self.params)

    def _on_preview_result(self, rgba: Image.Image):
        self.last_after = rgba
        self._redraw_previews()
        if self.source_image is not None:
            self.status_label.configure(
                text=f"Preview at {rgba.width}×{rgba.height} "
                     f"(source {self.source_image.width}×{self.source_image.height})",
            )

    def _on_preview_error(self, err: str):
        self.status_label.configure(text="Preview error (see dialog)")
        messagebox.showerror("Preview error", err)

    def _apply_grid_to_panes(self):
        """Push the toolbar's grid settings to the preview panes. Only
        before_view is touched directly — set_grid mirrors to linked panes
        the same way zoom/pan does."""
        self.before_view.set_grid(
            size=_GRID_SIZE_OPTIONS[self._grid_size_var.get()],
            line_width=_GRID_WIDTH_OPTIONS[self._grid_width_var.get()],
            color=self._grid_color_var.get(),
            semi_transparent=bool(self._grid_semi_var.get()),
        )

    def _on_view_mode_changed(self, which: str):
        """Keep the two view-mode checkboxes mutually exclusive."""
        if which == "alpha" and self._show_alpha_only.get():
            self._show_texture_only.set(False)
        elif which == "texture" and self._show_texture_only.get():
            self._show_alpha_only.set(False)
        self._redraw_previews()
        self._update_edit_mode()            # after redraw, so the cursor wins

    # -- manual edits --------------------------------------------------------

    def _init_edit_layer(self):
        """Fresh edit layer/controller for the current source image (loading a
        new image clears prior edits)."""
        import manual_edits
        if self.source_image is None:
            self.edit = None
            self.params.pop("_manual_edits", None)
            return
        w, h = self.source_image.width, self.source_image.height
        layer = manual_edits.EditLayer(w, h)
        # Auto-load a matching sidecar if one sits next to the image.
        if self.source_path:
            sc = manual_edits.EditLayer.sidecar_path(self.source_path)
            if sc.exists():
                try:
                    loaded = manual_edits.EditLayer.load(sc)
                    if (loaded.w, loaded.h) == (w, h):
                        layer = loaded
                        self.status_label.configure(text=f"Loaded edits: {sc.name}")
                except Exception:
                    pass
        self.edit = manual_edits.EditController(layer, on_change=self._on_edit_change)
        self.params["_manual_edits"] = layer
        if hasattr(self, "manual_edits_panel"):
            self.manual_edits_panel.sync_to_controller()
        self._update_edit_mode()

    def _update_edit_mode(self):
        """Editing is enabled only when exactly one of the alpha-only / texture-
        only view modes is on; that mode selects the target channel."""
        alpha_only = bool(self._show_alpha_only.get())
        texture_only = bool(self._show_texture_only.get())
        active = self.edit is not None and (alpha_only != texture_only)
        channel = None
        if active:
            channel = "alpha" if alpha_only else "rgb"
            self.edit.channel = channel
        brush = self.edit.brush if self.edit is not None else 4
        self.after_view.set_edit_mode(active, brush)
        if hasattr(self, "manual_edits_panel"):
            self.manual_edits_panel.set_enabled(active, channel)

    def _edit_begin(self, sx, sy):
        if self.edit is None:
            return
        if self.edit.tool == "picker":
            self._pick_value(sx, sy)
        else:
            self.edit.begin(sx, sy)

    def _edit_to(self, sx, sy):
        if self.edit is not None and self.edit.tool != "picker":
            self.edit.to(sx, sy)

    def _edit_end(self):
        if self.edit is not None and self.edit.tool != "picker":
            self.edit.end()

    def _edit_undo(self):
        if self.edit is not None:
            self.edit.undo()

    def _on_edit_change(self):
        """The edit layer changed — refresh the preview (debounced)."""
        self._schedule_preview()

    def _pick_value(self, sx, sy):
        """Value picker: sample the After result at source (sx, sy), set the
        current channel's value, then revert to the pencil (one-shot)."""
        if self.edit is None or self.last_after is None:
            return
        pw = self.preview_source.width if self.preview_source else self.last_after.width
        scale = (self.source_image.width / pw) if (self.source_image and pw) else 1.0
        bx = int(min(self.last_after.width - 1, max(0, sx / scale)))
        by = int(min(self.last_after.height - 1, max(0, sy / scale)))
        try:
            px = self.last_after.convert("RGBA").getpixel((bx, by))
        except Exception:
            return
        if self.edit.channel == "alpha":
            self.edit.alpha_value = int(px[3])
        else:
            self.edit.rgb_value = (int(px[0]), int(px[1]), int(px[2]))
        self.edit.tool = "pencil"           # one-shot
        self._on_edit_value_picked()        # Phase 3 hook: sync the UI widgets

    def _on_edit_value_picked(self):
        """Reflect a value the picker sampled back into the panel widgets."""
        if hasattr(self, "manual_edits_panel"):
            self.manual_edits_panel.reflect_picked_value()

    def _stop_editing(self):
        """Leave edit mode by clearing both view-mode toggles."""
        self._show_alpha_only.set(False)
        self._show_texture_only.set(False)
        self._redraw_previews()
        self._update_edit_mode()

    def _save_edits(self):
        """Write the manual edit layer to a sidecar next to the source image."""
        if self.edit is None or self.edit.layer.is_empty():
            messagebox.showinfo("No edits", "There are no manual edits to save.")
            return
        import manual_edits
        if self.source_path:
            path = str(manual_edits.EditLayer.sidecar_path(self.source_path))
        else:
            path = filedialog.asksaveasfilename(
                title="Save manual edits", defaultextension=".npz",
                filetypes=[("Edit layer", "*.npz")])
            if not path:
                return
        try:
            self.edit.layer.save(path)
            self.status_label.configure(text=f"Saved edits: {Path(path).name}")
        except Exception as e:
            messagebox.showerror("Save edits failed", str(e))

    def _delete_all_edits(self):
        if self.edit is None or self.edit.layer.is_empty():
            return
        if messagebox.askyesno(
                "Delete all manual edits",
                "Remove all manual edits? You can still undo with Ctrl+Z."):
            self.edit.clear()

    def _on_bg_changed(self):
        """Redraw both panes when the solid-background toggle changes."""
        self._redraw_previews()

    def _pick_bg_color(self):
        """Open the system colour picker to choose the solid canvas background."""
        from tkinter import colorchooser
        result = colorchooser.askcolor(
            color=self._bg_color,
            title="Canvas background colour",
            parent=self,
        )
        if result and result[1]:
            self._bg_color = result[1].upper()
            self._bg_swatch.configure(
                fg_color=self._bg_color,
                hover_color=self._bg_color,
            )
            if self._bg_solid.get():
                self._redraw_previews()

    def _render_for_view(self, rgba: Image.Image) -> Image.Image:
        """Render an RGBA image according to the current view-mode tick boxes."""
        if self._show_alpha_only.get():
            return alpha_as_rgb(rgba)
        if self._show_texture_only.get():
            # Texture only: RGB ignoring the alpha (no checkerboard).
            return rgba.convert("RGB")
        if self._bg_solid.get():
            return composite_over_solid(rgba, self._bg_color)
        return composite_over_checkerboard(rgba)

    def _update_coord_readout(self, coord, which: str):
        """Show the source-texture pixel coordinate (and that pixel's RGBA)
        under the cursor. `coord` is (x, y) in the displayed preview-image
        space, or None when the cursor leaves the image.

        The preview may be downscaled (PREVIEW_MAX_DIM); coordinates are
        reported in the full source-resolution space so they match the saved
        texture. The 'before' pane samples the original RGBA at full
        resolution; the 'after' pane samples the (preview-res) result.
        """
        if coord is None or self.preview_source is None:
            self.coord_label.configure(text="")
            return
        bx, by = coord
        base_w, base_h = self.preview_source.size

        # Map preview-space → full source-space.
        if self.source_image is not None and base_w > 0:
            scale = self.source_image.width / base_w
            fx = min(self.source_image.width - 1, int(bx * scale))
            fy = min(self.source_image.height - 1, int(by * scale))
        else:
            fx, fy = bx, by

        # Sample the true RGBA: full source for 'before', result for 'after'.
        px = None
        try:
            if which == "before" and self.source_image is not None:
                px = self.source_image.convert("RGBA").getpixel((fx, fy))
            elif which == "after" and self.last_after is not None:
                px = self.last_after.convert("RGBA").getpixel((bx, by))
        except Exception:
            px = None

        text = f"x {fx}  y {fy}"
        if px is not None:
            text += f"   RGBA {px[0]} {px[1]} {px[2]} {px[3]}"
        self.coord_label.configure(text=text)

    def _redraw_previews(self):
        if self.preview_source is None:
            self.before_view.set_placeholder("Load an image to begin")
            self.after_view.set_placeholder("Live preview")
            return
        # Tell both panes the source→preview downscale so the grid overlay lands
        # on source-pixel boundaries (set before set_image so the first render
        # already uses it). >1 only when the image exceeds PREVIEW_MAX_DIM.
        scale = 1.0
        if self.source_image is not None and self.preview_source.width > 0:
            scale = self.source_image.width / self.preview_source.width
        self.before_view.set_source_scale(scale)
        self.after_view.set_source_scale(scale)
        self.before_view.set_image(
            self._render_for_view(self.preview_source), reset_view=False,
        )
        if self.last_after is not None:
            self.after_view.set_image(
                self._render_for_view(self.last_after), reset_view=False,
            )
        else:
            self.after_view.set_placeholder("Computing…")
        # Refresh the full-res detail overlay for the new params / view mode.
        self._on_view_settle()

    # -- ROI full-res detail preview ----------------------------------------

    def _on_view_settle(self):
        """After the view settles (or the preview refreshes), recompute the
        visible on-screen crop at FULL resolution so what's shown matches the
        save exactly. Only fires when zoomed in on a downscaled preview; the
        low-res preview is left as-is when zoomed out."""
        src, ps = self.source_image, self.preview_source
        if src is None or ps is None or self.last_after is None:
            return
        if max(src.size) <= PREVIEW_MAX_DIM:
            return          # preview is already full-res — nothing to refine
        vr = self.before_view.visible_base_rect()
        if vr is None:
            return
        bx0, by0, bx1, by1, eff = vr
        if eff <= 1.05:     # zoomed out — low-res preview is fine; drop detail
            self.before_view.set_detail(None, (0, 0, 0, 0))
            self.after_view.set_detail(None, (0, 0, 0, 0))
            return
        full_scale = src.width / ps.width
        fx0 = int(bx0 * full_scale); fy0 = int(by0 * full_scale)
        fx1 = min(src.width, int(bx1 * full_scale) + 1)
        fy1 = min(src.height, int(by1 * full_scale) + 1)
        if (fx1 - fx0) * (fy1 - fy0) > self._DETAIL_MAX_PX:
            return          # visible region too large — keep the fast preview
        self._detail_token += 1
        job = {"token": self._detail_token, "rect": (fx0, fy0, fx1, fy1),
               "base_rect": (bx0, by0, bx1, by1), "params": dict(self.params),
               "source": src, "last_after": self.last_after}
        with self._detail_lock:
            self._detail_pending = job
            if not self._detail_busy:
                self._detail_busy = True
                threading.Thread(target=self._run_detail, daemon=True).start()

    def _run_detail(self):
        while True:
            with self._detail_lock:
                if self._detail_pending is None:
                    self._detail_busy = False
                    return
                job = self._detail_pending
                self._detail_pending = None
            try:
                ctx = pipeline.compute_roi_ctx(
                    job["last_after"], job["params"], source=job["source"])
                out = pipeline.process_crop(
                    job["source"], job["params"], job["rect"], ctx=ctx)
            except Exception:
                out = None
            if out is None:
                continue
            after_crop, _ = out
            before_crop = job["source"].crop(job["rect"])
            self.after(0, lambda t=job["token"], b=before_crop,
                       a=after_crop, r=job["base_rect"]:
                       self._apply_detail(t, b, a, r))

    def _apply_detail(self, token, before_crop, after_crop, base_rect):
        if token != self._detail_token:
            return          # a newer view superseded this result
        try:
            self.before_view.set_detail(
                self._render_for_view(before_crop), base_rect)
            self.after_view.set_detail(
                self._render_for_view(after_crop), base_rect)
        except Exception:
            pass


# --- Batch tab -------------------------------------------------------------

class BatchTab(ctk.CTkFrame):
    def __init__(self, parent, params: dict, on_params_changed=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.params = params  # shared with Tune tab
        # Called after this tab mutates the shared params dict (e.g. loading a
        # preset) so the Tune tab can re-sync its widgets — otherwise the Tune
        # sliders keep showing stale values that silently overwrite the loaded
        # preset on the next edit.
        self.on_params_changed = on_params_changed
        self._worker_thread: threading.Thread | None = None
        self._stop_flag = threading.Event()
        self._log_queue: queue.Queue[str] = queue.Queue()

        self._build()
        self.after(100, self._drain_log)

    def _build(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(5, weight=1)

        # Input row
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 4))
        row.columnconfigure(1, weight=1)
        ctk.CTkLabel(row, text="Input folder:", width=110, anchor="w"
                     ).grid(row=0, column=0, sticky="w")
        self.input_var = tk.StringVar()
        ctk.CTkEntry(row, textvariable=self.input_var
                     ).grid(row=0, column=1, sticky="ew", padx=6)
        ctk.CTkButton(row, text="Browse…", width=90,
                      command=self._pick_input).grid(row=0, column=2)

        # Output row
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.grid(row=1, column=0, sticky="ew", padx=10, pady=4)
        row.columnconfigure(1, weight=1)
        ctk.CTkLabel(row, text="Output folder:", width=110, anchor="w"
                     ).grid(row=0, column=0, sticky="w")
        self.output_var = tk.StringVar()
        ctk.CTkEntry(row, textvariable=self.output_var
                     ).grid(row=0, column=1, sticky="ew", padx=6)
        ctk.CTkButton(row, text="Browse…", width=90,
                      command=self._pick_output).grid(row=0, column=2)

        # Options row
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.grid(row=2, column=0, sticky="ew", padx=10, pady=4)
        ctk.CTkLabel(row, text="Preset:", width=110, anchor="w"
                     ).grid(row=0, column=0, sticky="w")
        self.preset_var = tk.StringVar(value="(Use current Tune settings)")
        self.preset_menu = ctk.CTkOptionMenu(
            row, variable=self.preset_var, values=self._preset_choices(),
            command=self._on_preset_pick, width=260,
        )
        self.preset_menu.grid(row=0, column=1, sticky="w", padx=6)
        ctk.CTkButton(row, text="Refresh", width=80,
                      command=self._refresh_presets).grid(row=0, column=2)

        self.recurse_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(row, text="Recurse into subfolders",
                        variable=self.recurse_var).grid(row=0, column=3, padx=(20, 0))

        self.autopick_var = tk.BooleanVar(value=False)
        autopick_cb = ctk.CTkCheckBox(
            row, text="Auto-pick preset per file (overrides dropdown)",
            variable=self.autopick_var,
        )
        autopick_cb.grid(row=1, column=1, columnspan=4, sticky="w", padx=6, pady=(4, 0))
        _Tooltip(autopick_cb,
                 "Per-file alpha-histogram classifier. For each input the "
                 "batch checks the alpha plane and picks:\n"
                 "  • No alpha / all transparent / fully opaque  → "
                 "preset 01 (Pass-through)\n"
                 "  • Any meaningful alpha                       → "
                 "preset 07 (Preserve plateaus)\n"
                 "The log shows which preset each file got and which plateaus "
                 "(0 / 128 / 255 / graded) were detected, so you can spot "
                 "misclassifications.")

        self.skip_existing_var = tk.BooleanVar(value=False)
        skip_cb = ctk.CTkCheckBox(
            row, text="Skip files already present in the output folder",
            variable=self.skip_existing_var,
        )
        skip_cb.grid(row=2, column=1, columnspan=4, sticky="w", padx=6, pady=(4, 0))
        _Tooltip(skip_cb,
                 "When ticked, any input whose output file already exists is "
                 "left untouched (logged as SKIP) instead of being "
                 "reprocessed — handy for resuming an interrupted batch.\n\n"
                 "When unticked, existing files are overwritten; if any already "
                 "exist you'll be asked to confirm before the batch starts.")

        self.include_edits_var = tk.BooleanVar(value=False)
        edits_cb = ctk.CTkCheckBox(
            row, text="Include manual edits (from the Tune tab)",
            variable=self.include_edits_var,
        )
        edits_cb.grid(row=3, column=1, columnspan=4, sticky="w", padx=6, pady=(4, 0))
        _Tooltip(edits_cb,
                 "Apply the manual pixel edits currently painted on the Tune tab "
                 "to every file, at the SAME x,y coordinates (no scaling). For "
                 "batches of same-size images. Load an image on the Tune tab and "
                 "paint first; leave off to ignore manual edits.")

        self.output_format_var = tk.StringVar(value=OUTPUT_FORMATS[0])
        ctk.CTkLabel(row, text="Format:", anchor="w"
                     ).grid(row=0, column=4, padx=(20, 4))
        ctk.CTkOptionMenu(row, variable=self.output_format_var,
                          values=OUTPUT_FORMATS, width=130
                          ).grid(row=0, column=5)

        # Preset description row
        desc_row = ctk.CTkFrame(self, fg_color="transparent")
        desc_row.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 4))
        desc_row.columnconfigure(1, weight=1)
        ctk.CTkLabel(desc_row, text="Description:", width=110, anchor="nw"
                     ).grid(row=0, column=0, sticky="nw")
        self._batch_preset_desc = ctk.CTkLabel(
            desc_row,
            text="Uses the parameters currently dialled in on the Tune tab.",
            wraplength=900, justify="left", anchor="w",
            font=ctk.CTkFont(size=11),
            text_color=("#666", "#aaa"),
        )
        self._batch_preset_desc.grid(row=0, column=1, sticky="ew", padx=6)

        # Colour match to reference (kills animation colour flicker)
        self._build_colourmatch(row_index=4)

        # Log
        self.log_box = ctk.CTkTextbox(self, height=300)
        self.log_box.grid(row=5, column=0, sticky="nsew", padx=10, pady=8)
        self.log_box.configure(state="disabled")

        # Run / stop
        runrow = ctk.CTkFrame(self, fg_color="transparent")
        runrow.grid(row=6, column=0, sticky="ew", padx=10, pady=(0, 10))
        runrow.columnconfigure(1, weight=1)
        self.run_btn = ctk.CTkButton(runrow, text="Run batch", width=140,
                                     command=self._run)
        self.run_btn.grid(row=0, column=0, sticky="w")
        self.stop_btn = ctk.CTkButton(runrow, text="Stop", width=80,
                                      command=self._stop, state="disabled")
        self.stop_btn.grid(row=0, column=2, sticky="e")
        self.progress = ctk.CTkProgressBar(runrow)
        self.progress.grid(row=0, column=1, sticky="ew", padx=12)
        self.progress.set(0)

    def _build_colourmatch(self, row_index: int):
        try:
            import colour_match
            methods = colour_match.COLOURMATCH_METHODS
            spaces = colour_match.COLOURMATCH_SPACES
        except Exception:
            methods, spaces = ["Reinhard (mean/std)"], ["LAB", "RGB"]

        frame = ctk.CTkFrame(self)
        frame.grid(row=row_index, column=0, sticky="ew", padx=10, pady=(0, 4))
        frame.columnconfigure(1, weight=1)

        self.cm_enabled_var = tk.BooleanVar(value=False)
        cb = ctk.CTkCheckBox(
            frame, text="Colour match to reference (fix animation flicker)",
            variable=self.cm_enabled_var, command=self._update_cm_state,
        )
        cb.grid(row=0, column=0, columnspan=4, sticky="w", padx=6, pady=(6, 2))
        _Tooltip(cb,
                 "Force every texture in the batch to match one reference "
                 "image's colours, removing the frame-to-frame colour flicker "
                 "in an animation. Only the RGB is changed — the alpha channel "
                 "is preserved. Statistics are taken from opaque pixels only.")

        # Reference image picker
        ctk.CTkLabel(frame, text="Reference:", width=90, anchor="w"
                     ).grid(row=1, column=0, sticky="w", padx=6)
        self.cm_ref_var = tk.StringVar()
        self._cm_ref_entry = ctk.CTkEntry(frame, textvariable=self.cm_ref_var)
        self._cm_ref_entry.grid(row=1, column=1, sticky="ew", padx=6)
        self._cm_ref_btn = ctk.CTkButton(frame, text="Browse…", width=90,
                                         command=self._pick_colourmatch_ref)
        self._cm_ref_btn.grid(row=1, column=2, padx=(0, 6))

        # Method / colourspace / strength
        opts = ctk.CTkFrame(frame, fg_color="transparent")
        opts.grid(row=2, column=0, columnspan=4, sticky="ew", padx=6, pady=(2, 6))
        ctk.CTkLabel(opts, text="Method:", anchor="w").pack(side="left")
        self.cm_method_var = tk.StringVar(value=methods[0])
        self._cm_method_menu = ctk.CTkOptionMenu(
            opts, variable=self.cm_method_var, values=list(methods), width=190)
        self._cm_method_menu.pack(side="left", padx=(4, 14))

        ctk.CTkLabel(opts, text="Space:", anchor="w").pack(side="left")
        self.cm_space_var = tk.StringVar(value=spaces[0])
        self._cm_space_menu = ctk.CTkOptionMenu(
            opts, variable=self.cm_space_var, values=list(spaces), width=80)
        self._cm_space_menu.pack(side="left", padx=(4, 14))

        ctk.CTkLabel(opts, text="Strength:", anchor="w").pack(side="left")
        self.cm_strength_var = tk.DoubleVar(value=1.0)
        self._cm_strength_lbl = ctk.CTkLabel(opts, text="1.00", width=36)
        self._cm_strength_slider = ctk.CTkSlider(
            opts, from_=0.0, to=1.0, number_of_steps=100,
            variable=self.cm_strength_var, width=140,
            command=lambda v: self._cm_strength_lbl.configure(text=f"{float(v):.2f}"))
        self._cm_strength_slider.pack(side="left", padx=(4, 4))
        self._cm_strength_lbl.pack(side="left")

        self._cm_widgets = [self._cm_ref_entry, self._cm_ref_btn,
                            self._cm_method_menu, self._cm_space_menu,
                            self._cm_strength_slider]
        self._update_cm_state()

    def _update_cm_state(self):
        state = "normal" if self.cm_enabled_var.get() else "disabled"
        for w in getattr(self, "_cm_widgets", []):
            try:
                w.configure(state=state)
            except Exception:
                pass

    def _pick_colourmatch_ref(self):
        path = filedialog.askopenfilename(
            title="Choose colour-match reference image",
            initialdir=load_dialog_dir("colourmatch_ref"),
            filetypes=[("Images", " ".join(f"*{e}" for e in SUPPORTED_INPUTS)),
                       ("All files", "*.*")],
        )
        if path:
            save_dialog_dir("colourmatch_ref", path)
            self.cm_ref_var.set(path)

    # -- preset wiring -------------------------------------------------------

    def _preset_choices(self) -> list[str]:
        names = [name for name, _ in list_presets()]
        return ["(Use current Tune settings)"] + names

    def _refresh_presets(self):
        self.preset_menu.configure(values=self._preset_choices())

    def _on_preset_pick(self, choice: str):
        if choice == "(Use current Tune settings)":
            self._batch_preset_desc.configure(
                text="Uses the parameters currently dialled in on the Tune tab.",
                text_color=("#666", "#aaa"),
            )
            return
        path = find_preset_path(choice)
        if path is None:
            self._batch_preset_desc.configure(
                text=f"Preset '{choice}' not found.",
                text_color=("#a33", "#f88"),
            )
            return
        try:
            loaded = load_preset(path)
        except Exception as e:
            self._log(f"Failed to load preset {choice}: {e}")
            return
        for k, v in loaded.items():
            self.params[k] = v
        # Re-sync the Tune tab so its sliders reflect the loaded preset rather
        # than the previous (now stale) values.
        if self.on_params_changed is not None:
            self.on_params_changed()
        desc = load_preset_description(path)
        self._batch_preset_desc.configure(
            text=desc or "(No description in preset file.)",
            text_color=("#222", "#ddd") if desc else ("#666", "#aaa"),
        )
        self._log(f"Loaded preset: {choice}")

    # -- folder pickers ------------------------------------------------------

    def _pick_input(self):
        d = filedialog.askdirectory(
            title="Choose input folder",
            initialdir=load_dialog_dir("batch_input"))
        if d:
            save_dialog_dir("batch_input", d)
            self.input_var.set(d)
            if not self.output_var.get():
                self.output_var.set(str(Path(d) / "cleaned"))

    def _pick_output(self):
        d = filedialog.askdirectory(
            title="Choose output folder",
            initialdir=load_dialog_dir("batch_output"))
        if d:
            save_dialog_dir("batch_output", d)
            self.output_var.set(d)

    # -- logging -------------------------------------------------------------

    def _log(self, msg: str):
        self._log_queue.put(msg)

    def _drain_log(self):
        try:
            while True:
                msg = self._log_queue.get_nowait()
                self.log_box.configure(state="normal")
                self.log_box.insert("end", msg + "\n")
                self.log_box.see("end")
                self.log_box.configure(state="disabled")
        except queue.Empty:
            pass
        self.after(100, self._drain_log)

    # -- run -----------------------------------------------------------------

    def _enumerate(self, root: Path, recurse: bool,
                   out_root: "Path | None" = None) -> list[Path]:
        """Supported inputs under `root`. When recursing and the output folder
        lives INSIDE the input tree (e.g. <input>\\cleaned), its subtree is
        excluded — otherwise a second run would re-process the previous run's
        outputs as inputs. No exclusion when out_root IS the input root."""
        skip_out = None
        if recurse and out_root is not None:
            out_res = out_root.resolve()
            if out_res != root.resolve():
                skip_out = out_res

        def _under_out(p: Path) -> bool:
            if skip_out is None:
                return False
            try:
                p.resolve().relative_to(skip_out)
                return True
            except ValueError:
                return False

        out = []
        if recurse:
            for p in root.rglob("*"):
                if (p.is_file() and p.suffix.lower() in SUPPORTED_INPUTS
                        and not _under_out(p)):
                    out.append(p)
        else:
            for p in root.iterdir():
                if p.is_file() and p.suffix.lower() in SUPPORTED_INPUTS:
                    out.append(p)
        return sorted(out)

    def _resolve_output_path(self, src: Path, in_root: Path,
                             out_root: Path, fmt: str) -> Path:
        rel = src.relative_to(in_root)
        dst = out_root / rel
        ext_map = {
            "Same as input": src.suffix,
            "PNG":      ".png",
            "TIFF":     ".tif",
            "WebP":     ".webp",
            "DDS DXT1": ".dds",
            "DDS DXT5": ".dds",
        }
        dst = dst.with_suffix(ext_map.get(fmt, src.suffix))
        return dst

    def _run(self):
        in_dir = self.input_var.get().strip()
        out_dir = self.output_var.get().strip()
        if not in_dir or not Path(in_dir).is_dir():
            messagebox.showerror("Batch", "Please choose a valid input folder.")
            return
        if not out_dir:
            messagebox.showerror("Batch", "Please choose an output folder.")
            return
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return

        in_root = Path(in_dir)
        out_root = Path(out_dir)
        recurse = bool(self.recurse_var.get())
        fmt = self.output_format_var.get()

        files = self._enumerate(in_root, recurse, out_root)
        if not files:
            messagebox.showinfo("Batch", "No supported images found in the input folder.")
            return

        skip_existing = bool(self.skip_existing_var.get())
        # Existing-output handling: when not skipping, warn before clobbering.
        existing = [
            s for s in files
            if self._resolve_output_path(s, in_root, out_root, fmt).exists()
        ]
        if existing and not skip_existing:
            if not messagebox.askyesno(
                "Overwrite existing files?",
                f"{len(existing)} of {len(files)} output file(s) already exist "
                f"in:\n{out_root}\n\nOverwrite them?\n\n"
                "Tick “Skip files already present in the output folder” to keep "
                "them instead.",
            ):
                self._log("Batch cancelled — existing output files would be "
                          "overwritten.")
                return

        params_snapshot = dict(self.params)

        # Colour match: compute the reference statistics once, up front, and
        # inject them into the params every task carries.
        if self.cm_enabled_var.get():
            ref_path = self.cm_ref_var.get().strip()
            if not ref_path or not Path(ref_path).is_file():
                messagebox.showerror(
                    "Batch", "Colour match is enabled but no valid reference "
                    "image is set.")
                return
            try:
                import colour_match
                ref_img = pipeline.load_rgba(ref_path)
                space = self.cm_space_var.get()
                ref_stats = colour_match.compute_reference_stats(ref_img, space)
            except Exception as e:
                messagebox.showerror(
                    "Batch", f"Couldn't read the reference image:\n{e}")
                return
            params_snapshot["colourmatch_enabled"] = True
            params_snapshot["colourmatch_method"] = self.cm_method_var.get()
            params_snapshot["colourmatch_strength"] = float(self.cm_strength_var.get())
            params_snapshot["colourmatch_colorspace"] = space
            params_snapshot["colourmatch_ref_stats"] = ref_stats
            self._log(
                f"Colour match ON: reference {Path(ref_path).name}, "
                f"{self.cm_method_var.get()} in {space}, "
                f"strength {float(self.cm_strength_var.get()):.2f}")
        else:
            params_snapshot["colourmatch_enabled"] = False

        # Manual edits: apply the Tune tab's painted layer to every file at fixed
        # x,y (flatten() drops the undo history for a smaller per-worker pickle).
        if self.include_edits_var.get():
            layer = self.params.get("_manual_edits")
            if layer is None or layer.is_empty():
                messagebox.showerror(
                    "Batch", "“Include manual edits” is on, but there are no "
                    "manual edits. Load an image on the Tune tab and paint first.")
                return
            params_snapshot["_manual_edits"] = layer.flatten()
            params_snapshot["_manual_edits_fit"] = "place"
            self._log("Manual edits ON: applying at fixed x,y (same-size images "
                      "assumed; out-of-bounds is clipped).")
        else:
            params_snapshot.pop("_manual_edits", None)
            params_snapshot.pop("_manual_edits_fit", None)

        autopick = bool(self.autopick_var.get())
        self._stop_flag.clear()
        self.run_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.progress.set(0)

        mode = "auto-pick per file" if autopick else "fixed preset"
        skip_note = ", skipping existing" if skip_existing else ""
        self._log(f"Starting batch: {len(files)} files ({mode}{skip_note})")
        self._worker_thread = threading.Thread(
            target=self._worker, daemon=True,
            args=(files, in_root, out_root, fmt, params_snapshot, autopick,
                  skip_existing),
        )
        self._worker_thread.start()

    def _stop(self):
        self._stop_flag.set()
        self._log("Stop requested — finishing in-flight files and halting.")

    @staticmethod
    def _worker_count(n_tasks: int) -> int:
        """Pick a parallel worker count: one per core, leaving a core free for
        the UI + dispatcher, capped by the number of files. The pipeline is
        numpy/Pillow/subprocess heavy, so separate processes sidestep the GIL
        entirely."""
        cores = os.cpu_count() or 2
        return max(1, min(cores - 1, n_tasks))

    def _build_tasks(self, files: list[Path], in_root: Path, out_root: Path,
                     fmt: str, params: dict, autopick: bool,
                     skip_existing: bool) -> list[dict]:
        """Resolve the destination path and the effective params for every
        file up front (incl. auto-pick classification), so the pool workers
        receive fully self-contained, picklable jobs. Files whose output
        already exists are flagged for skipping when skip_existing is set."""
        autopick_cache: dict[str, dict] = {}
        tasks: list[dict] = []
        for src in files:
            dst = self._resolve_output_path(src, in_root, out_root, fmt)
            rel = str(src.relative_to(in_root))
            if skip_existing and dst.exists():
                tasks.append({"src": str(src), "dst": str(dst), "params": None,
                              "rel": rel, "reldst": "", "skip": True})
                continue
            try:
                if autopick:
                    preset_stem, reason = pipeline.classify_for_preset(src)
                    if preset_stem not in autopick_cache:
                        path = find_preset_path(preset_stem)
                        if path is None:
                            raise RuntimeError(
                                f"auto-pick wants preset '{preset_stem}' but "
                                "it's not on disk")
                        autopick_cache[preset_stem] = load_preset(path)
                    file_params = autopick_cache[preset_stem]
                    self._log(f"  → {rel}: auto-pick → {preset_stem}  ({reason})")
                else:
                    file_params = params

                # Colour-match keys are a batch-wide setting; merge them into
                # every file's params (incl. auto-pick presets, which lack them).
                if params.get("colourmatch_enabled"):
                    file_params = {
                        **file_params,
                        **{k: v for k, v in params.items()
                           if k.startswith("colourmatch_")},
                    }

                # Manual edits are batch-wide too; merge them in (auto-pick
                # presets lack these keys), so every file gets the same layer.
                if params.get("_manual_edits") is not None:
                    file_params = {
                        **file_params,
                        "_manual_edits": params["_manual_edits"],
                        "_manual_edits_fit": params.get("_manual_edits_fit", "place"),
                    }

                if dst.suffix.lower() == ".dds":
                    dds_comp = "dxt1" if fmt == "DDS DXT1" else "dxt5"
                    file_params = {**file_params, "dds_compression": dds_comp}

                tasks.append({
                    "src": str(src), "dst": str(dst), "params": file_params,
                    "rel": rel, "reldst": str(dst.relative_to(out_root)),
                })
            except Exception as e:
                # Surface classification / preset errors now; mark as a failed
                # task so the totals still add up.
                tasks.append({"src": str(src), "dst": str(dst), "params": None,
                              "rel": rel, "reldst": "", "error": str(e)})
        return tasks

    def _worker(self, files: list[Path], in_root: Path, out_root: Path,
                fmt: str, params: dict, autopick: bool, skip_existing: bool):
        total = len(files)
        done = 0
        errors = 0
        skipped = 0
        start = time.monotonic()

        tasks = self._build_tasks(files, in_root, out_root, fmt, params,
                                  autopick, skip_existing)

        def report(ok: bool, rel: str, reldst: str, error: str | None):
            nonlocal done, errors
            done += 1
            if ok:
                self._log(f"  OK  {rel} -> {reldst}")
            else:
                errors += 1
                self._log(f"  ERR {rel}: {error}")
            self.after(0, lambda d=done, t=total: self.progress.set(d / t))

        # Separate skipped (output already present) and build-failed tasks from
        # the runnable ones.
        runnable = []
        for tk in tasks:
            if tk.get("skip"):
                done += 1
                skipped += 1
                self._log(f"  SKIP {tk['rel']} (output already exists)")
                self.after(0, lambda d=done, t=total: self.progress.set(d / t))
            elif tk["params"] is None:
                report(False, tk["rel"], "", tk.get("error", "task build failed"))
            else:
                runnable.append(tk)

        workers = self._worker_count(len(runnable))
        if workers <= 1 or len(runnable) <= 1:
            # Sequential — avoids process-spawn overhead for tiny batches.
            for tk in runnable:
                if self._stop_flag.is_set():
                    break
                res = pipeline.process_one_file(
                    {"src": tk["src"], "dst": tk["dst"], "params": tk["params"]})
                report(res["ok"], tk["rel"], tk["reldst"], res["error"])
        else:
            self._log(f"Processing with {workers} parallel workers…")
            try:
                with futures.ProcessPoolExecutor(max_workers=workers) as ex:
                    fut_map = {}
                    for tk in runnable:
                        if self._stop_flag.is_set():
                            break
                        fut = ex.submit(pipeline.process_one_file, {
                            "src": tk["src"], "dst": tk["dst"],
                            "params": tk["params"],
                        })
                        fut_map[fut] = tk
                    for fut in futures.as_completed(fut_map):
                        tk = fut_map[fut]
                        try:
                            res = fut.result()
                        except futures.CancelledError:
                            continue
                        report(res["ok"], tk["rel"], tk["reldst"], res["error"])
                        if self._stop_flag.is_set():
                            # Cancel everything that hasn't started yet; let
                            # in-flight files finish (the with-block waits).
                            for pending in fut_map:
                                if not pending.done():
                                    pending.cancel()
            except Exception as e:
                self._log(f"Batch pool error: {e}")

        elapsed = time.monotonic() - start
        skip_note = f", {skipped} skipped" if skipped else ""
        self._log(f"Done: {done}/{total} processed, {errors} errors"
                  f"{skip_note}, {elapsed:.1f}s elapsed")
        self.after(0, self._on_worker_done)

    def _on_worker_done(self):
        self.run_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")


# --- Main window -----------------------------------------------------------

class AlphaToolsApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title(f"{APP_TITLE} {APP_VERSION}")
        self.geometry("1400x900")
        self.minsize(1100, 700)

        # Shared params dict — both tabs read/write the same instance.
        self.params: dict = dict(pipeline.DEFAULT_PARAMS)

        # Tabview
        self.tabs = ctk.CTkTabview(self)
        self.tabs.pack(fill="both", expand=True, padx=8, pady=8)
        self.tabs.add("Tune")
        self.tabs.add("Batch")

        self.tune = TuneTab(self.tabs.tab("Tune"), self.params)
        self.tune.pack(fill="both", expand=True)

        self.batch = BatchTab(
            self.tabs.tab("Batch"), self.params,
            on_params_changed=self._sync_tune_from_params,
        )
        self.batch.pack(fill="both", expand=True)

    def _sync_tune_from_params(self):
        """Refresh the Tune tab's widgets + preview after another tab mutates
        the shared params dict (e.g. the Batch tab loading a preset)."""
        self.tune._refresh_all_panels()
        self.tune._schedule_preview()


def main():
    app = AlphaToolsApp()
    app.mainloop()


if __name__ == "__main__":
    # Must run before any ProcessPoolExecutor spawns a child. In the frozen
    # cx_Freeze build a spawned child re-executes the bundled exe; without this
    # guard each worker would relaunch the whole GUI (a fork bomb). It is a
    # no-op in the parent process and when running from source.
    import multiprocessing
    multiprocessing.freeze_support()
    main()
