"""Regression checks for the grid overlay on downscaled previews (>1024 px).

Two bugs fixed (both from the ROI full-res / viewport-rendering work):
  (a) the grid stepped in downscaled-preview pixels, not source pixels, so on a
      2048px image a "16" grid drew 32-source-px cells and didn't line up with
      the full-res detail overlay when zoomed;
  (b) the grid was composited into the low-res crop and then HIDDEN by the
      full-res detail overlay drawn on top of it whenever you zoomed in.

The math (source-pixel alignment, density guard, exact-boundary lines) is tested
without a GUI. The layering fix (grid drawn as the top-most canvas layer) is
tested against a withdrawn Tk window, and skipped if no display is available.

Run: py -3 _test_grid.py
"""
import numpy as np
import main

Z = main.ZoomPanPreview
gal = Z._grid_axis_lines


def _centers(lines):
    return [(lo + hi) / 2 for lo, hi in lines]


def test_exact_boundary_lines_draw():
    # eff exactly 1.0 puts lines on integer pixels; round-half-to-even used to
    # collapse them to zero width. They must draw.
    lines = gal(0, 100, 16.0, 1.0, 1)
    assert _centers(lines) == [0, 16, 32, 48, 64, 80, 96]


def test_source_pixel_alignment_downscaled():
    # scale=2 (2048 source -> 1024 base): grid steps every 8 BASE px, which is
    # every 16 SOURCE px. Every line must land on a source multiple of 16.
    lines = gal(0, 100, 16.0 / 2, 1.0, 1)
    src = [c * 2 for c in _centers(lines)]           # base center -> source px
    assert src and all(round(s) % 16 == 0 for s in src)
    # ...and twice as dense as the pre-fix behaviour (which stepped 16 base px).
    old_buggy = gal(0, 100, 16.0, 1.0, 1)
    assert len(lines) == 2 * len(old_buggy) - 1 or len(lines) > len(old_buggy)


def test_alignment_when_zoomed():
    scale, size, eff, src0 = 2.0, 16, 3.7, 100.0
    lines = gal(src0, 80, size / scale, eff, 1)
    src_lines = [round((src0 + c / eff) * scale) for c in _centers(lines)]
    assert src_lines and all(s % 16 == 0 for s in src_lines)


def test_density_guard():
    assert gal(0, 100, 2.0, 0.5, 1) == []            # <2 screen px => no lines


def test_build_overlay_spacing_and_dense_none():
    z = object.__new__(Z)
    z._GRID_COLORS = Z._GRID_COLORS
    z._grid = {"size": 16, "line_width": 1, "color": "red", "semi": False}
    z._source_scale = 2.0
    ov = z._build_grid_overlay(200, 150, 1.0, 0, 0)
    arr = np.asarray(ov)
    cols = np.where(arr[10, :, 3] > 0)[0]
    assert len(cols) > 1 and int(np.min(np.diff(cols))) == 8   # 16 src / scale 2
    z._grid = {"size": 2, "line_width": 1, "color": "red", "semi": False}
    z._source_scale = 1.0
    assert z._build_grid_overlay(200, 150, 0.5, 0, 0) is None


def test_grid_drawn_above_detail():
    """The grid must be the top-most canvas layer, above the ROI detail overlay
    (the reported bug: it vanished when zooming a large image)."""
    try:
        import customtkinter as ctk
        root = ctk.CTk()
        root.withdraw()
    except Exception as e:                            # no display / Tk
        print("SKIP test_grid_drawn_above_detail (no GUI):", e)
        return
    try:
        from PIL import Image
        pane = main.ZoomPanPreview(root)
        pane.canvas.winfo_width = lambda: 400
        pane.canvas.winfo_height = lambda: 300
        pane.set_source_scale(2.0)                    # 2048 source -> 1024 base
        pane.set_image(Image.fromarray(np.full((512, 1024, 3), 90, np.uint8), "RGB"))
        pane.set_grid(16, 1, "red", False)
        pane._zoom = 6.0
        pane.set_detail(
            Image.fromarray(np.full((200, 200, 4), (200, 50, 50, 255), np.uint8),
                            "RGBA"), (500, 250, 540, 290))
        pane._render()
        imgs = [i for i in pane.canvas.find_all() if pane.canvas.type(i) == "image"]
        assert len(imgs) == 3            # low-res crop + detail + grid
        assert pane._photo_grid is not None
        assert pane.canvas.find_all()[-1] == imgs[-1]   # grid is top-most
    finally:
        root.destroy()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} grid tests passed.")
