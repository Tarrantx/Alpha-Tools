"""Tests for the optional dither on top of the soften-edges blur.

Run: py -3 _test_soften_dither.py
"""
import numpy as np
from PIL import Image
import alpha_pipeline as ap


def _edge():
    """Left half opaque grey, right half transparent — a vertical alpha edge."""
    H = W = 48
    a = np.zeros((H, W, 4), np.uint8)
    a[:, :24, :3] = 120
    a[:, :24, 3] = 255
    return (Image.fromarray(a[..., :3], "RGB"),
            Image.fromarray(a[..., 3], "L"))


def test_dither_confined_to_band():
    rgb, alpha = _edge()
    base = np.asarray(ap.soften_edges(rgb, alpha, 6), np.int16)
    dith = np.asarray(ap.soften_edges(rgb, alpha, 6, dither=True, strength=60), np.int16)
    d = dith - base
    assert (d != 0).any(), "dither did nothing"
    # Changes sit in the ~6px band just inside the edge (x < 24, near 18-23).
    cols = set(np.where((d != 0).any(2).any(0))[0].tolist())
    assert cols and max(cols) < 24 and min(cols) >= 18
    assert not (d[:, 2] != 0).any(), "deep interior was dithered"
    assert not (d[:, 40] != 0).any(), "transparent area was dithered"


def test_dither_deterministic():
    rgb, alpha = _edge()
    a = np.asarray(ap.soften_edges(rgb, alpha, 6, dither=True, strength=60))
    b = np.asarray(ap.soften_edges(rgb, alpha, 6, dither=True, strength=60))
    assert np.array_equal(a, b)                      # stable -> no preview flicker


def test_dither_off_is_plain_blur():
    rgb, alpha = _edge()
    plain = np.asarray(ap.soften_edges(rgb, alpha, 6))
    off = np.asarray(ap.soften_edges(rgb, alpha, 6, dither=True, strength=0))
    assert np.array_equal(plain, off)                # strength 0 == no dither
    off2 = np.asarray(ap.soften_edges(rgb, alpha, 6, dither=False, strength=60))
    assert np.array_equal(plain, off2)               # toggle off == no dither


def test_dither_position_parity():
    """The dither noise is keyed on ABSOLUTE source coords, so a crop matches the
    same region of the full image (ROI full-res detail parity)."""
    full = ap._edge_dither_noise(40, 40, (0, 0))
    sub = ap._edge_dither_noise(10, 10, (12, 8))     # origin x=12, y=8
    assert np.allclose(sub, full[8:18, 12:22])


def test_stage_pipeline_roi_parity():
    """process_crop of the soften-dither region matches the full render."""
    H = W = 96
    a = np.zeros((H, W, 4), np.uint8)
    yy, xx = np.mgrid[0:H, 0:W]
    disc = (xx - 48) ** 2 + (yy - 48) ** 2 <= 34 ** 2
    a[disc, :3] = 130
    a[disc, 3] = 255
    src = Image.fromarray(a, "RGBA")
    p = {"soften_edges": 6, "soften_dither": True, "soften_dither_strength": 60,
         "snap_enabled": False}
    full = np.asarray(ap.process_pil(src, p).convert("RGBA"))
    rect = (20, 20, 76, 76)
    crop, _ = ap.process_crop(src, p, rect)
    crop = np.asarray(crop.convert("RGBA"))
    ref = full[rect[1]:rect[3], rect[0]:rect[2]]
    assert crop.shape == ref.shape and np.array_equal(crop, ref)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} soften-dither tests passed.")
