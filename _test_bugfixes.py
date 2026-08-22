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


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} bugfix tests passed.")
