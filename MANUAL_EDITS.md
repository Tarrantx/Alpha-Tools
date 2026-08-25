# Manual edits — implementation notes

Paint per-pixel touch-ups directly on the Tune preview, for the few spots no
automatic setting can fix. This is a reference for maintaining/merging the
feature; the user-facing summary lives in the README.

Files: `manual_edits.py` (all the model/logic, GUI-free) and the
`ManualEditsPanel` + `TuneTab` glue in `main.py`. Two pipeline stages live in
`alpha_pipeline.py`. Tests: `_test_manual_edits.py` (model/pipeline, no GUI) and
`_test_edit_widget.py` (widget + panel, GUI in isolated subprocesses).

## Data model — `EditLayer` (source resolution)

Four arrays sized to the **source** image: `rgb` (HxWx3) + `rgb_mask` (HxW bool),
`alpha` (HxW) + `alpha_mask` (HxW bool). A pixel is "edited" where its mask is
True. Coordinates and brush sizes are in **source pixels**, so edits are
zoom/pan/resolution-invariant and land identically on the full-res save and in
batch.

- `apply_dab(cx,cy,size,channel,mode,alpha_value,rgb_value)` — pencil (paint) or
  eraser (clears the mask → falls back to the pipeline result), square brush.
- `apply_granulate_dab(cx,cy,size,base_rgb,amount)` — per-pixel colour jitter
  around `base_rgb`, writes RGB.
- `apply_smudge_dab(cx,cy,size,eff_rgb,eff_alpha,carried,strength)` — soft round
  brush; blends a carried colour along the stroke, gated to opaque pixels;
  returns the updated carried colour.
- **Version bumps per DAB** (not per stroke) so the live-preview cache
  invalidates mid-drag; **undo granularity is per stroke** (`begin_stroke` /
  `end_stroke` bracket a stroke; `undo()` reverse-replays that stroke's square
  snapshots — correct across overlapping dabs). `clear()` is one undoable step.
- Persistence: `save`/`load` a compressed `*.edits.npz`; `sidecar_path(img)` =
  `<img>.edits.npz`. `flatten()` drops the undo history (small pickle for batch).
  `placed_into(w,h)` = a copy sized to another image (verbatim if equal, else
  pasted at (0,0) and clipped) — used by Load edits and batch.

## Controller — `EditController` (GUI-free)

Holds the current tool / channel / brush / values and turns paint events into
layer ops. `begin`/`to`/`end` bracket a stroke and **interpolate dabs** along a
drag (no gaps). Tools: `pencil`, `eraser`, `picker` (handled upstream), plus the
RGB-only `granulate` and `smudge`.

- `source_rgb` / `source_alpha` are set per image (what granulate jitters around
  and smudge blends; alpha gates smudge).
- `_effective_rgb()` / `_effective_alpha()` = source with existing manual edits
  composited in. Granulate freezes this per stroke (`_gran_base`) so it **builds
  on** pencil edits and can't compound its own jitter. Smudge freezes both
  (`_eff_rgb`/`_eff_alpha`) and carries a colour (`_smudge_carry`) along the
  stroke.

## Pipeline placement — **hybrid** (in `process_pil`)

Manual edits are a live object in `params["_manual_edits"]`; the stages no-op
when it's absent/empty.

- **RGB edits — EARLY**: `_stage_manual_rgb` runs after RGB edge-dilate, *before*
  tone, so tone / perceptual / FX / quantise reprocess them (stays colour-
  consistent with the image).
- **Alpha edits — LATE**: `_stage_manual_alpha` runs after snap/clamp/shape (so
  those can't destroy painted alpha) but before compose (post-processing still
  applies).

Resolution handling via `manual_edits.apply_rgb` / `apply_alpha` with a `fit`:
| Path | region | fit | result |
|------|--------|-----|--------|
| Full-res save | whole | `scale` | 1:1, exact |
| Live preview (≤1024) | whole | `scale` | nearest downscale |
| ROI full-res crop | crop rect | `scale` | sliced region, pixel-exact |
| Batch | whole | `place` | fixed x,y, clipped (no scaling) |

`process_crop` injects `manual_src_region` into `_roi_ctx` so a zoomed crop
slices the right region. `PreviewPipeline` has cache-keyed stages
(`k_manual_rgb`/`k_manual_alpha`, keyed on `layer.version()`), so edits repaint
live. `_manual_edits_fit` selects `scale` (interactive) vs `place` (batch).

## Widget + panel (in `main.py`)

- `ZoomPanPreview`: LEFT-drag paints when `_edit_active` (via `on_paint_*`
  callbacks in SOURCE px, mapped through `_source_scale`), else pans; RIGHT-drag
  always pans; **Ctrl+left-click** = eyedropper (`on_paint_pick`); square brush
  cursor. `set_edit_mode(active, brush)`.
- `ManualEditsPanel` (top-right, above the image-effects panel) is the source of
  truth for tool/brush/values and pushes them onto the per-image controller.
  Editing is gated by an explicit **Enable editing** toggle (off by default —
  the view modes are *not* what enables it); the **Channel** selector is
  independent of the view (paint alpha while viewing texture). The RGB-only tools
  (granulate/smudge) lock the channel to RGB and reveal their Variation/Strength
  slider. All the tool controls live in `self._tool_box` which is
  `pack`/`pack_forget`-collapsed until editing is enabled. **NB:** the attribute
  must not be named `_options` — tkinter reserves `Misc._options` and shadowing
  it breaks `.grid()`.
- `TuneTab`: `_edit_enabled` + `_set_edit_enabled`; `_update_edit_mode` (view-
  independent); `_init_edit_layer` (fresh controller per image, auto-loads a
  matching sidecar, sets `source_rgb`/`source_alpha`); `_edit_begin/to/end`,
  `_pick_value(...,reset_tool)`, `_save_edits`/`_load_edits`/`_delete_all_edits`.

## Batch

`BatchTab` "Include manual edits" checkbox. On run it puts `layer.flatten()` +
`_manual_edits_fit="place"` into the params snapshot (or strips `_manual_edits`
when off); `_build_tasks` merges them into every file's params (so auto-pick
presets carry them too). Batch uses a `ProcessPoolExecutor`, so the flattened
layer is pickled to each worker — `flatten()` keeps that small.

## Presets

Manual edits are **not** part of presets. `save_preset` drops all `"_"`-prefixed
private keys (`_manual_edits`, `_manual_edits_fit`, `_roi_ctx`) — they aren't
JSON-serialisable and don't belong in a preset. Preset load only sets the keys a
file contains, so the live edit layer survives a load.
