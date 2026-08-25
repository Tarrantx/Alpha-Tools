"""Regression tests for two fixes:
  1. Saving a preset while a live EditLayer sits in params (the manual-edit
     layer isn't JSON-serialisable — presets must drop private keys).
  2. Colour-depth quantise turning the transparent 'base' colour green
     (libimagequant gives the transparent palette entry an arbitrary RGB).
"""
import os
import json
import numpy as np
from PIL import Image
import alpha_pipeline as ap
import manual_edits as me


def _img(base_rgb, n_content=True):
    H = W = 48
    a = np.zeros((H, W, 4), np.uint8)
    a[..., :3] = base_rgb                       # transparent pixels carry the base
    if n_content:
        a[16:32, 16:32] = (200, 30, 30, 255)    # opaque content
    return Image.fromarray(a, "RGBA")


def test_quantize_preserves_base_colour():
    for base in [(0, 0, 0), (255, 255, 255), (20, 40, 90)]:
        for n in (256, 16, 4):
            q = ap.quantize_rgba(_img(base), n)
            px = np.asarray(q.convert("RGBA"))
            corner = tuple(int(x) for x in px[0, 0])
            assert corner[3] == 0, "transparent pixel became opaque"
            assert corner[:3] == base, f"base {base} not preserved at n={n}: {corner[:3]}"


def test_quantize_content_still_reduced():
    # The visible content is still quantised (base fix doesn't disable it).
    src = _img((0, 0, 0))
    q = ap.quantize_rgba(src, 4)
    ncolors = len(q.convert("RGBA").getcolors(maxcolors=100000))
    assert ncolors <= 6                          # ~4 palette entries (+ base)


def test_save_preset_drops_private_keys():
    import main
    params = {**ap.DEFAULT_PARAMS}
    params["_manual_edits"] = me.EditLayer(10, 10)   # not JSON-serialisable
    params["_manual_edits_fit"] = "scale"
    params["_roi_ctx"] = {"x": 1}
    p = "_preset_bugfix_test.json"
    try:
        main.save_preset(p, params)                   # must not raise
        saved = json.loads(open(p, encoding="utf-8").read())["params"]
        assert "_manual_edits" not in saved and "_manual_edits_fit" not in saved
        assert "_roi_ctx" not in saved
        assert "saturation" in saved and "quantize_mode" in saved
    finally:
        if os.path.exists(p):
            os.remove(p)


def _wolf(seed=3):
    """Grey body (many shades) + tiny red eyes, fully opaque."""
    rng = np.random.default_rng(seed)
    H = W = 72
    a = np.zeros((H, W, 4), np.uint8)
    a[..., :3] = np.clip(120 + rng.integers(-40, 40, (H, W, 1)), 0, 255)
    a[..., 3] = 255
    yy, xx = np.mgrid[0:H, 0:W]
    for ex in (28, 44):
        eye = ((xx - ex) ** 2 + (yy - 28) ** 2) <= 5
        a[eye, :3] = (205, 25, 25)
    return Image.fromarray(a, "RGBA")


def _eye(img):
    return tuple(int(x) for x in np.asarray(img.convert("RGB"))[28, 28])


def test_protect_vivid_keeps_accent():
    src = _wolf()
    p = {"quantize_mode": "4 (2-bit)", "snap_enabled": False}
    normal = _eye(ap.process_pil(src, p))
    prot = _eye(ap.process_pil(src, {**p, "quantize_protect_vivid": True,
                                     "quantize_reserve": 2, "quantize_sat_thresh": 45}))
    # normal loses the red (eye goes grey/pink); protected keeps it near source.
    assert abs(prot[0] - 205) < 40 and prot[1] < 80 and prot[2] < 80, f"not protected: {prot}"
    assert not (abs(normal[0] - 205) < 40 and normal[1] < 80), f"normal unexpectedly kept red: {normal}"


def test_protect_off_by_default():
    src = _wolf()
    a = np.asarray(ap.process_pil(src, {"quantize_mode": "4 (2-bit)"}).convert("RGBA"))
    b = np.asarray(ap.process_pil(src, {"quantize_mode": "4 (2-bit)",
                                        "quantize_protect_vivid": False}).convert("RGBA"))
    assert np.array_equal(a, b)                      # default == explicit-off


def test_protect_roi_parity():
    """A crop containing the eyes matches the same region of the full render
    (the reserved-accent palette is injected via compute_roi_ctx)."""
    src = _wolf()
    p = {"quantize_mode": "4 (2-bit)", "snap_enabled": False,
         "quantize_protect_vivid": True, "quantize_reserve": 2,
         "quantize_sat_thresh": 45}
    full = np.asarray(ap.process_pil(src, p).convert("RGBA"))
    rect = (16, 16, 56, 40)
    crop, _ = ap.process_crop(src, p, rect)
    crop = np.asarray(crop.convert("RGBA"))
    assert crop.shape[:2] == (rect[3] - rect[1], rect[2] - rect[0])
    # The eye colours in the crop match the full render (both red, same palette).
    fe = tuple(int(x) for x in full[28, 28])
    ce = tuple(int(x) for x in crop[28 - 16, 28 - 16])
    assert fe == ce and fe[0] > 150 and fe[1] < 80


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} bugfix tests passed.")
