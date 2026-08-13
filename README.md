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

**Finishing**
- **Pixelate** (fractional factors) and **soften edges** (alpha‑outline blur, RGB only).
- Output **colour depth / palette**: reduce via libimagequant, with **dithering** (rendered at pixel‑art block resolution) and **custom‑palette matching** to any reference image.
- A 5‑slot **post‑processing effects chain**: unsharp, bilateral smooth, saturation boost, emboss, chrome, bevel, Dave Hill, texturize, watercolor, screen effects (mesh / displace), colour tint, film grain, frosted glass, lomo, oil paint, granulate, turbulence, normal map, CRT scanlines.

**Preview (Tune tab)**
- Live before/after with linked zoom/pan and a cursor pixel readout (source x/y + RGBA under the mouse).
- Stays smooth at any zoom level (only the visible region is rasterised).
- **Full‑resolution accuracy** — when you zoom in on a large image the visible region is recomputed at full resolution, so what you see matches the saved output exactly instead of a scaled‑down approximation. (See [ROI_PREVIEW_IMPLEMENTATION.md](ROI_PREVIEW_IMPLEMENTATION.md) for how this works.)

**Batch**
- Parallel processing, presets (with per‑file auto‑pick), skip/overwrite handling, and **colour‑match to a reference frame** (Reinhard or histogram, in LAB/RGB) to kill colour flicker across animation frames.

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
