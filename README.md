# Alpha Tools

A Windows desktop app for cleaning and reconstructing the **alpha channel of AI‑upscaled game textures**, with a full colour / pixel‑art finishing pipeline on top. Built with Python + CustomTkinter.

Upscalers (ESRGAN and friends) do a great job on RGB but wreck transparency — they smear the alpha edge, bleed the (usually black) fill colour into the outline, and turn crisp cut‑outs into muddy halos. Alpha Tools rebuilds a clean alpha from the RGB and/or the existing alpha, keeps semi‑transparency intact (including the PS1 tri‑state convention), and then lets you re‑pixelate, re‑palette, and stylise the result.

It has a live **Tune** tab (single image, side‑by‑side before/after with zoom/pan) and a parallel **Batch** tab for whole folders.

## Features

**Alpha creation / cleanup**
- Four alpha sources: `fix` (level the existing alpha), `create` (synthesise from RGB luminance, anchored to the source), `fuzzythresh` (auto histogram threshold), and `rangethresh` (per‑channel colour‑range key, in sRGB / HSV / HSL).
- **Incongruence halo removal** — colour‑aware edge cleanup that eats the upscaler fringe while protecting genuine dark content and outlines.
- N‑level alpha snap, alpha shape (erode / dilate), and a **PS1 region‑aware tri‑state clamp** with the DuckStation (143) and Beetle PSX HW (127) semi‑transparency values.

**Colour / tone**
- Saturation, contrast, brightness, and per‑channel **colour levels** (R/G/B).
- **Perceptual tone (OKLCh)** — a hue‑linear perceptual space where lightness, colourfulness and hue stay independent: a *perceptual contrast* on lightness, a *chroma/vividness* boost that doesn't shift brightness, an even *hue rotation* (no CIELAB blue→purple drift), and an optional *auto‑level* that stretches lightness to full range. Independent of any reference image or palette.
- Optional [vtracer](https://github.com/visioncortex/vtracer) raster→vector pre‑stage (needs Inkscape).

**Manual edits (Tune tab)**
- Paint touch‑ups the automatic pipeline can't reach, directly on the preview. An explicit **Enable editing** toggle (off by default, so the view modes stay for inspection) reveals the tools; a **Channel** selector (Alpha / RGB) is independent of the view, so you can paint alpha *while looking at the texture*.
- Tools: **Pencil**, **Eraser** (reverts to the pipeline result), **Picker** (eyedropper — or **Ctrl+left‑click** any time), **Granulate** (per‑pixel colour jitter around the underlying colour, to break up flat upscale banding), and **Smudge** (a soft round brush that blends colours so an edit merges seamlessly — one *Strength* dial from gentle feather to strong drag).
- Brush size in **source pixels**, so edits are zoom/pan/resolution‑invariant and land identically on the full‑res save and in batch. Left‑drag paints, right‑drag pans, **Ctrl+Z** undo. Edits flow through the whole pipeline (RGB early, alpha late), and **Save Image bakes them in**.
- **Save edits… / Load edits…** persist the edit layer to a `*.edits.npz` sidecar and re‑apply it to another image (same‑size verbatim, different‑size placed at the same x,y and clipped). A matching sidecar auto‑loads with its image. (See [MANUAL_EDITS.md](MANUAL_EDITS.md) for the architecture.)

**Finishing**
- **Pixelate** (fractional factors) and **soften edges** (alpha‑outline blur, RGB only), with an optional **dither** on top of the blur — a zero‑mean speckle confined to the same softened band, to break up the smooth gradient.
- Output **colour depth / palette**: reduce via libimagequant, with **dithering** (rendered at pixel‑art block resolution) and **custom‑palette matching** to any reference image. **Protect vivid colours** reserves palette slots for small saturated accents (e.g. red eyes on a grey body) so they survive aggressive reduction instead of merging into the dominant range. Reducing colour depth **preserves the transparent base colour** (black / white / whatever it is) instead of leaving an arbitrary fringe.
- A 5‑slot **post‑processing effects chain**: unsharp, bilateral smooth, saturation boost, emboss, chrome, bevel, Dave Hill, texturize, watercolor, screen effects (mesh / displace), colour tint, film grain, frosted glass, lomo, oil paint, granulate, turbulence, normal map, CRT scanlines.

**Preview (Tune tab)**
- Live before/after with linked zoom/pan and a cursor pixel readout (source x/y + RGBA under the mouse). Wheel or **Ctrl +/‑** to zoom (about the viewport centre), left‑ or right‑drag to pan.
- Stays smooth at any zoom level (only the visible region is rasterised).
- **Full‑resolution accuracy** — when you zoom in on a large image the visible region is recomputed at full resolution, so what you see matches the saved output exactly instead of a scaled‑down approximation. (See [ROI_PREVIEW_IMPLEMENTATION.md](ROI_PREVIEW_IMPLEMENTATION.md) for how this works.)
- A **tile‑seam grid** overlay (size / colour / line width / semi‑transparent) that lands on true **source‑pixel** boundaries and stays put through zoom, pan and the full‑res detail overlay — for spotting upscale tile seams.

**Batch**
- Parallel processing, presets (with per‑file auto‑pick), skip/overwrite handling, **colour‑match to a reference frame** (Reinhard or histogram, in LAB/RGB) to kill colour flicker across animation frames, and an optional **Include manual edits** (applies the Tune tab's painted layer to every file at fixed x,y).

**Export:** PNG (true indexed PNG with transparency), TIFF, WebP, DDS (DXT1 / DXT5).

## Running from source

```bash
pip install -r requirements.txt
py -3 main.py
```

Requires Python 3.12+ and: `pillow`, `numpy`, `customtkinter`, `imagequant` (libimagequant). Some post‑processing effects call **ImageMagick** (`magick` on PATH or bundled in a build); the vtracer stage needs `vtracer` + Inkscape.

## Building the installer (Windows)

Packaged with cx_Freeze into an MSI:

```bash
py -3 setup.py bdist_msi     # -> .\dist\
py -3 setup.py build         # quick frozen-exe test -> .\build\
```

The build bundles a portable ImageMagick — `setup.py` expects it at the path set by `IMAGEMAGICK_DIR` (edit that constant for your machine).

## Credits

- Many of the post‑processing effects and colour operations are ports / reimplementations of **Fred Weinhaus's** ImageMagick scripts — <http://www.fmwconcepts.com/imagemagick/> (fuzzythresh, rangethresh, bevel, chrome, emboss, texturize, watercolor, screeneffects, levelcolors, Dave Hill effect).
- Palette reduction + dithering approach inspired by [sd‑webui‑pixelart](https://github.com/mrreplicart/sd-webui-pixelart).
- Colour reduction via [libimagequant / pngquant](https://pngquant.org/lib/); vectoriser via [vtracer](https://github.com/visioncortex/vtracer); effects via [ImageMagick](https://imagemagick.org/).

## License

[MIT](LICENSE) © 2026 Jed Kelly
