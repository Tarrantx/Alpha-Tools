# Full-resolution preview — implementation guide

How the "zoom/pan is fast **and** the preview matches the full-res save" behaviour
was built, in enough detail to reproduce from scratch in another session.

There are **two independent changes**. Part A is small and low-risk; Part B builds
on it. Both are now **merged into `main`** (the `roi-preview` branch used during
development has been fast-forwarded in). Original commits, for reference:

- **Part A — viewport rendering** → `8bdff3e`.
- **Part B — ROI full-res preview** → `fc20f35` (developed on branch `roi-preview`,
  now on `main`).

Files touched: `main.py` (the `ZoomPanPreview` widget + the `TuneTab`) and
`alpha_pipeline.py` (the pipeline). Nothing in the batch/save path changes — the
new pipeline argument defaults to "off", so `process_image`/`process_one_file`
behave exactly as before.

---

## The problem (why this was needed)

The live preview runs the pipeline on a **downscaled** copy of the source
(`_make_preview_source` → `PREVIEW_MAX_DIM = 1024`), for speed. Two consequences:

1. **Mismatch.** Every *pixel-scale* stage (incongruence reach, blur, erode/dilate,
   edge-dilation, **pixelate factor**, soften-edges, PS1 region-min, and fx pixel
   params) is applied at the wrong scale on a downscaled image, so the preview
   doesn't match the full-res save. On a 2048px image the preview is at 0.5×, so a
   4px reach eats 8 source-px worth, pixelate blocks come out 2× too big, etc.
2. **Zoom/pan lag.** The old `ZoomPanPreview._render()` resized the **whole** base
   image to the zoomed size (`dw = iw * eff`). At 40× zoom on a 1024px base that's
   a ~31,000 × 31,000 image (~1 billion px, ~4 GB) rebuilt **every frame** — 99.99%
   of it off-screen. That's the crawl.

Key realisation: these are **two separate problems**. Part A fixes the lag; Part B
fixes the mismatch. GPU is the wrong tool — profiling showed the pipeline cost is
two stages with per-radius loops (`soften_edges`, `incongruence`), not numpy
throughput, and half the pipeline (ImageMagick fx, libimagequant, vtracer) can't go
on a GPU anyway.

---

## Invariant to preserve throughout

`ZoomPanPreview` maps a base-image pixel `(sx, sy)` to canvas position
`canvas_x = left + sx*eff`, where `eff = fit_factor * zoom`,
`left = (cw//2 + offset_x) - (iw*eff)/2`, `top = ...`. **Do not change this
transform.** Both parts only change *what pixels get rasterised*, never *where a
given source pixel lands*. This is what keeps the cursor-coordinate readout and the
grid overlay accurate.

---

## Part A — Viewport rendering (fixes zoom/pan lag)

Rasterise only the **visible** source region and scale that, instead of scaling the
whole image. Cost becomes O(viewport) (~constant ~22 ms) at any zoom.

### A1. Rewrite `ZoomPanPreview._render()`

Replace the whole-image resize with a visible-crop resize placed with `anchor="nw"`:

```python
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

# Rasterise ONLY the source region visible in the canvas (1–2px overscan
# avoids edge gaps). The transform is unchanged.
src_x0 = max(0, int((0 - left) / eff) - 1)
src_y0 = max(0, int((0 - top) / eff) - 1)
src_x1 = min(iw, int((cw - left) / eff) + 2)
src_y1 = min(ih, int((ch - top) / eff) + 2)
if src_x1 <= src_x0 or src_y1 <= src_y0:
    return  # entirely off-screen (panned away)

crop = self._base_img.crop((src_x0, src_y0, src_x1, src_y1))
disp_w = max(1, int(round((src_x1 - src_x0) * eff)))
disp_h = max(1, int(round((src_y1 - src_y0) * eff)))
resample = Image.Resampling.NEAREST if eff >= 1.5 else Image.Resampling.LANCZOS
scaled = crop.resize((disp_w, disp_h), resample=resample)
self._photo = ImageTk.PhotoImage(scaled)

# Place the crop where its source top-left maps to on the canvas.
self._image_id = self.canvas.create_image(
    left + src_x0 * eff, top + src_y0 * eff, image=self._photo, anchor="nw")
# Part B adds the detail-overlay call here, THEN the grid on top (see A2).
self._draw_detail(eff, left, top, src_x0, src_y0, src_x1, src_y1)
self._draw_grid(eff, left, top, src_x0, src_y0, disp_w, disp_h)
```

### A2. The grid overlay — crop-aware, source-scaled, drawn on top

The grid is **no longer composited into `scaled`** — it's drawn as the **top-most
canvas layer** (`_draw_grid`) *after* the low-res crop and the full-res detail
overlay, so zooming a downscaled preview can't bury it under the detail image.

- `_build_grid_overlay(dw, dh, eff, src_x0, src_y0)` returns a transparent RGBA
  overlay (or `None` when off / too dense), and `_draw_grid` places it as its own
  `create_image` on top. Keep the ref alive (`self._photo_grid`).
- Lines sit on **SOURCE-pixel** multiples of `size`. Because `_base_img` may be a
  downscaled preview, the widget is told the source→preview scale via
  `set_source_scale()`; the base-pixel step is `size / _source_scale` so a "16"
  grid lands on real 16-px source seams (and aligns with the detail overlay),
  not on downscaled-preview pixels.
- `_grid_axis_lines(src0, extent, gstep_base, eff, width)` is a pure/testable
  generator: first line at `k*gstep_base` (`k = ceil(src0 / gstep_base)` via
  `np.ceil`), crop-local screen pos `= (k*gstep_base - src0)*eff`, step
  `gstep_base*eff`; empty when `< 2` screen px. Uses a fixed integer line width so
  a 1-px line always draws even on an exact pixel boundary (round-half-to-even
  used to drop it). The old `_apply_grid` / `_grid_overlay_cache` / `_stripes`
  helpers are gone.

### A3. Result

Render time went from "resize ~1 billion px/frame at 40×" to a **flat ~22 ms** at
any zoom. Verified with a headless bench (see Verification).

---

## Part B — ROI full-res preview (fixes the mismatch)

When zoomed past 1:1 on a downscaled preview, recompute **only the visible full-res
crop** through the real pipeline and overlay it on the low-res base. Pixel-accurate
where you're looking; cost bounded to the on-screen region (capped ~4 MP) regardless
of image size.

Three sub-parts: **B1 pipeline** (process a crop), **B2 widget** (draw a detail
overlay + a settle callback), **B3 TuneTab** (background worker computes the crop).

### B1. Pipeline — `alpha_pipeline.py`

**(a) Thread an optional `ctx` through `process_pil`.** `ctx` is precomputed global
context (currently just the quantise palette). `None` everywhere except the ROI
path, so all existing behaviour is byte-identical.

```python
def process_pil(img, params, work_dir=None, ctx=None):
    ...
    params = {**DEFAULT_PARAMS, **params}
    if ctx:
        params["_roi_ctx"] = ctx      # stash so stages can read it, no signature churn
    ...
```

**(b) Let `_stage_quantize` reuse the whole-image palette.** In colour-count mode,
if `ctx["quantize_palette"]` is present, quantise the crop *to that palette* instead
of deriving a fresh one — so crops don't each pick a different palette:

```python
ctx = params.get("_roi_ctx") or {}
...
else:                                  # colour-count branch
    n = _QUANTIZE_COLORS.get(str(params.get("quantize_mode", "Full colour")), 0)
    if n <= 0:
        return out
    override = ctx.get("quantize_palette")
    if override is not None:
        quant_fn = lambda im: quantize_to_palette(im, override, dither)
    else:
        quant_fn = lambda im: quantize_rgba(im, n, dither)
```

**(c) Add `compute_roi_ctx`, `_roi_margin_align`, `process_crop`.** Verbatim:

```python
def compute_roi_ctx(last_after, params):
    """Global context for a crop, from the already-computed low-res result."""
    params = {**DEFAULT_PARAMS, **params}
    ctx = {}
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


def _roi_margin_align(params):
    """Overscan px so boundary stages match at the crop edge, + pixelate grid
    alignment. Margin is trimmed off; alignment snaps the crop origin."""
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
        m = max(m, 32)
    m = min(m, 128)                    # cap so the crop stays viewport-cheap
    factor = float(params.get("pixelate_factor", 1.0))
    align = max(1, int(round(factor))) if factor > 1 else 1
    return m, align


def process_crop(full_img, params, rect, ctx=None):
    """Process only the full-res region `rect` (x0,y0,x1,y1) with an overscan
    margin so edge stages match, then trim back to `rect`. Returns
    (rgba_crop, rect) or None."""
    params = {**DEFAULT_PARAMS, **params}
    W, H = full_img.size
    x0, y0, x1, y1 = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
    x0 = max(0, x0); y0 = max(0, y0); x1 = min(W, x1); y1 = min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    margin, align = _roi_margin_align(params)
    ex0 = max(0, x0 - margin); ey0 = max(0, y0 - margin)
    ex1 = min(W, x1 + margin); ey1 = min(H, y1 + margin)
    if align > 1:
        ex0 -= ex0 % align            # snap origin to the pixelate grid
        ey0 -= ey0 % align
    crop = full_img.crop((ex0, ey0, ex1, ey1))
    res = process_pil(crop, params, ctx=ctx)
    if res.mode != "RGBA":
        res = res.convert("RGBA")
    tx0, ty0 = x0 - ex0, y0 - ey0
    trimmed = res.crop((tx0, ty0, tx0 + (x1 - x0), ty0 + (y1 - y0)))
    return trimmed, (x0, y0, x1, y1)
```

> `pixelate()` also grew an optional `block=(bw,bh)` arg during development to try
> to make pixelate crop-exact, but injecting it was net-negative for non-dividing
> factors, so it's left unused by `process_crop`. Keep the arg (harmless) or drop it.

### B2. Widget — `ZoomPanPreview` in `main.py`

**Add to `__init__`:**

```python
self._detail = None                   # (base_rect, full_res_image)
self._photo_detail = None
self.on_view_settle = None            # owner sets this
self._settle_id = None
self._SETTLE_MS = 250
```

**`set_image`:** drop the stale detail whenever the base changes (add
`self._detail = None; self._photo_detail = None` before `self._render()`). Do **not**
schedule a settle here — the owner drives refresh from `_redraw_previews`.

**New methods:**

```python
def set_detail(self, detail_img, base_rect):
    self._detail = None if detail_img is None else (tuple(base_rect), detail_img)
    self._render()

def _schedule_settle(self):
    if self._settle_id is not None:
        try: self.after_cancel(self._settle_id)
        except Exception: pass
    self._settle_id = self.after(self._SETTLE_MS, self._fire_settle)

def _fire_settle(self):
    self._settle_id = None
    if self.on_view_settle is not None:
        self.on_view_settle()

def visible_base_rect(self):
    """Visible base region as (x0,y0,x1,y1,eff) or None — SAME transform/overscan
    as _render, so the owner can request a matching full-res crop."""
    if self._base_img is None: return None
    eff = self._fit_factor() * self._zoom
    if eff <= 0: return None
    iw, ih = self._base_img.size
    dw = iw * eff; dh = ih * eff
    cw = self.canvas.winfo_width() or 1
    ch = self.canvas.winfo_height() or 1
    cx = cw // 2 + self._offset_x; cy = ch // 2 + self._offset_y
    left = cx - dw / 2.0; top = cy - dh / 2.0
    x0 = max(0, int((0 - left) / eff) - 1)
    y0 = max(0, int((0 - top) / eff) - 1)
    x1 = min(iw, int((cw - left) / eff) + 2)
    y1 = min(ih, int((ch - top) / eff) + 2)
    if x1 <= x0 or y1 <= y0: return None
    return (x0, y0, x1, y1, eff)

def _draw_detail(self, eff, left, top, vx0, vy0, vx1, vy1):
    """Overlay the full-res detail on top of the low-res crop where visible.
    `_detail` = ((bx0,by0,bx1,by1) base coords, full_res_image)."""
    if self._detail is None:
        self._photo_detail = None; return
    (bx0, by0, bx1, by1), dimg = self._detail
    ix0 = max(vx0, bx0); iy0 = max(vy0, by0)
    ix1 = min(vx1, bx1); iy1 = min(vy1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        self._photo_detail = None; return
    fs = dimg.width / max(1, (bx1 - bx0))      # detail px per base px (= full_scale)
    dcrop = dimg.crop((int((ix0 - bx0) * fs), int((iy0 - by0) * fs),
                       int((ix1 - bx0) * fs), int((iy1 - by0) * fs)))
    ddw = max(1, int(round((ix1 - ix0) * eff)))
    ddh = max(1, int(round((iy1 - iy0) * eff)))
    det_scale = eff / fs
    resample = Image.Resampling.NEAREST if det_scale >= 1.5 else Image.Resampling.LANCZOS
    self._photo_detail = ImageTk.PhotoImage(dcrop.resize((ddw, ddh), resample))
    self.canvas.create_image(left + ix0 * eff, top + iy0 * eff,
                             image=self._photo_detail, anchor="nw")
```

**Hook `_draw_detail` into `_render`** — after `create_image` of the base crop:

```python
self._draw_detail(eff, left, top, src_x0, src_y0, src_x1, src_y1)
```

**Schedule a settle after interaction** — add `self._schedule_settle()` at the end of
both `_on_drag` and `_on_wheel` (only the interacted pane fires; linked panes are
just mirrored).

### B3. Owner wiring — `TuneTab` in `main.py`

**Add to `__init__`** (before `_build()`):

```python
self._detail_lock = threading.Lock()
self._detail_pending = None
self._detail_busy = False
self._detail_token = 0
self._DETAIL_MAX_PX = 4_000_000       # skip detail if visible region bigger
```

**Wire the panes** (where `before_view`/`after_view` are created and linked):

```python
self.before_view.on_view_settle = self._on_view_settle
self.after_view.on_view_settle = self._on_view_settle
```

**Drive a refresh from `_redraw_previews`** — after the two `set_image` calls, add
`self._on_view_settle()` so the detail recomputes on param / view-mode changes.

**Add the three methods** (`_on_view_settle`, `_run_detail`, `_apply_detail`):

```python
def _on_view_settle(self):
    src, ps = self.source_image, self.preview_source
    if src is None or ps is None or self.last_after is None:
        return
    if max(src.size) <= PREVIEW_MAX_DIM:
        return                          # preview already full-res
    vr = self.before_view.visible_base_rect()
    if vr is None:
        return
    bx0, by0, bx1, by1, eff = vr
    if eff <= 1.05:                     # zoomed out — low-res is fine
        self.before_view.set_detail(None, (0, 0, 0, 0))
        self.after_view.set_detail(None, (0, 0, 0, 0))
        return
    full_scale = src.width / ps.width
    fx0 = int(bx0 * full_scale); fy0 = int(by0 * full_scale)
    fx1 = min(src.width, int(bx1 * full_scale) + 1)
    fy1 = min(src.height, int(by1 * full_scale) + 1)
    if (fx1 - fx0) * (fy1 - fy0) > self._DETAIL_MAX_PX:
        return                          # region too large — keep fast preview
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
            ctx = pipeline.compute_roi_ctx(job["last_after"], job["params"])
            out = pipeline.process_crop(job["source"], job["params"],
                                        job["rect"], ctx=ctx)
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
    if token != self._detail_token:     # a newer view superseded this
        return
    try:
        self.before_view.set_detail(self._render_for_view(before_crop), base_rect)
        self.after_view.set_detail(self._render_for_view(after_crop), base_rect)
    except Exception:
        pass
```

Notes:
- `base_rect` handed to `set_detail` is in **base (preview_source) coords**; the
  detail image is full-res, so `fs = dimg.width/(bx1-bx0)` recovers `full_scale`.
- The `after(0, ...)` cross-thread callback is the same pattern the existing
  `PreviewWorker` uses (works under a running Tk mainloop).
- `_render_for_view` is applied to the crops so the detail honours the current
  view mode (checkerboard / alpha-only / solid bg).

---

## Key design decisions / rationale

- **Two layers, not one base swap.** The pane keeps its low-res `_base_img` for the
  overview and fast pan; the full-res detail is a *separate overlay* drawn on top
  only where visible. Simpler and keeps zoom-out instant.
- **Global context from the low-res run, not a full-res run.** `compute_roi_ctx`
  derives from the already-computed `last_after`, so no extra whole-image work. The
  expensive stages (soften/incong loops) are never run at full size.
- **Coalescing worker + token.** Rapid zoom/pan coalesces to the latest job
  (`_detail_pending`), and `_detail_token` drops results for superseded views.
- **Cost cap (`_DETAIL_MAX_PX`).** When zoomed out the visible full-res region is the
  whole image (huge) → skip and keep the fast low-res preview (you can't see
  pixel-level mismatch when zoomed out anyway).

---

## Known limitations (accepted for v1 — documented on purpose)

Exact (0.00% vs the whole-image result, verified): incongruence, blur, erode/dilate,
edge-dilation, snap, soften-edges, tone, colour levels, and the `create`/`fix` alpha
methods. Two residuals:

1. **Quantise palette.** A crop can't know the full-image libimagequant palette
   without a full-res run, so `compute_roi_ctx` reuses the **low-res** palette. It's
   *consistent* with the zoomed-out preview but ~mean-11 off the actual save's
   palette. Only the colour choices drift; structure/edges are exact.
2. **Pixelate.** Blocks are the **right size** (the real fix — the low-res preview
   showed them oversized), but nearest-resize block *phase* can be ~1px off the save.
   Making it byte-exact needs a formula-based grid pixelate that matches PIL's
   nearest-resize exactly — a rabbit hole not worth it for a preview.

Auto-threshold alpha with a *grow* seed (percentile over the whole region) would also
drift slightly on a crop; `create` with the default *shrink* seed is exact.

---

## Verification (reproduce the checks)

Run from the project root. Expected results in comments.

**Part A — render is flat at any zoom** (headless, needs a real canvas geometry):
build a `ZoomPanPreview`, set a 1024px image, time `_render()` at zoom 1/5/20/40 →
~22 ms flat (vs old code resizing to 269 MP @20×, 1076 MP @40×).

**Part B — crop is pixel-exact for local stages:**
```python
# process_crop(src, P, rect) vs process_pil(src, P).crop(rect)
# alpha_source='fix' or 'create', quantize_mode='Full colour',
# with incong/blur/erode/edge_dilation/snap/soften/tone/levels ON
#   -> max=0  mean=0.0000  mismatch=0.00%   (EXACT)
# add quantize_mode='16 (4-bit)'  -> palette differs (documented; structure exact)
```

**Regression — the ctx=None path is unchanged:**
```python
# process_pil(img, params) must equal PreviewPipeline().run(img, params)
# and _smoke_test.py must still PASS
```

Full test snippets used during development are in the session transcript; the two
that matter are the "crop vs full-region" pixel-diff and the "process_pil ==
PreviewPipeline" regression.

---

## Git workflow used

Part B was built on a throwaway branch so it stayed revertible during development,
then fast-forward-merged into `main` and pushed once it was confirmed working. Both
parts are now on `main`; the `roi-preview` branch is fully merged (safe to delete).

```bash
# Part A committed to main (verified, low-risk):
git add main.py
git commit -m "Preview: viewport rendering to fix zoom/pan lag at high zoom"

# Fork for the experiment so it's revertible:
git checkout -b roi-preview
git add main.py alpha_pipeline.py
git commit -m "ROI full-res preview: show the accurate full-res result where zoomed in"

# Merged back once verified:
git checkout main && git merge --ff-only roi-preview && git push origin main
```

---

## If reproducing from scratch in a new session

1. Start from a clean `main`. Do **Part A** first, verify zoom is flat, commit.
2. `git checkout -b roi-preview`.
3. Do **B1** (pipeline) and test `process_crop` vs `process_pil().crop` in isolation
   until local stages are 0.00% — this is the correctness anchor.
4. Do **B2** (widget) then **B3** (TuneTab wiring).
5. Confirm the regression (`process_pil == PreviewPipeline`, smoke test) and that
   `TuneTab` still builds headless.
6. The one thing not verifiable headless is the live interactive feel (settle
   timing, overlay smoothness) — run `py -3 main.py`, load a >1024px image, zoom in.
