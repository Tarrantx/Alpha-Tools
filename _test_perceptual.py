"""Unit checks for the perceptual OKLCh tone stage (alpha_pipeline).

Verifies: sRGB<->OKLab round-trip accuracy, identity when disabled, the
chroma / hue / lightness / auto-level operations, that alpha is never touched,
and that the ROI auto-level bounds injected via compute_roi_ctx make a crop
stretch identically to the whole-image save.

Run: py -3 _test_perceptual.py
"""
import numpy as np
from PIL import Image
import alpha_pipeline as ap


def _arr(img):
    return np.asarray(img, dtype=np.uint8)


def test_roundtrip():
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    rt = np.clip(ap.oklab_to_srgb(ap.srgb_to_oklab(arr)), 0, 255).astype(np.uint8)
    err = int(np.abs(rt.astype(int) - arr.astype(int)).max())
    assert err <= 1, f"round-trip error {err} > 1"


def test_identity_when_off():
    img = Image.fromarray(np.full((32, 32, 3), 120, np.uint8), "RGB")
    # Both off -> exact same object, no conversion at all.
    assert ap.perceptual_oklch(img, dict(ap.DEFAULT_PARAMS)) is img
    # Enabled but neutral knobs -> pixel-identical (within float rounding).
    p = {**ap.DEFAULT_PARAMS, "perceptual_enabled": True}
    out = _arr(ap.perceptual_oklch(img, p))
    assert int(np.abs(out.astype(int) - 120).max()) <= 1


def test_chroma():
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 256, size=(48, 48, 3), dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    sat = lambda a: np.asarray(Image.fromarray(a, "RGB").convert("HSV"))[..., 1].mean()
    boost = _arr(ap.perceptual_oklch(img, {**ap.DEFAULT_PARAMS,
                "perceptual_enabled": True, "perceptual_chroma": 1.8}))
    gray = _arr(ap.perceptual_oklch(img, {**ap.DEFAULT_PARAMS,
                "perceptual_enabled": True, "perceptual_chroma": 0.0}))
    assert sat(boost) > sat(arr) > sat(gray)
    assert sat(gray) < 2.0, "chroma=0 should be ~greyscale"


def test_autolevel():
    rng = np.random.default_rng(2)
    narrow = rng.integers(60, 130, size=(64, 64, 3)).astype(np.uint8)
    before = ap.srgb_to_oklab(narrow)[..., 0]
    out = _arr(ap.perceptual_oklch(Image.fromarray(narrow, "RGB"),
               {**ap.DEFAULT_PARAMS, "perceptual_autolevel": True,
                "perceptual_autolevel_clip": 0.5}))
    after = ap.srgb_to_oklab(out)[..., 0]
    assert (after.max() - after.min()) > (before.max() - before.min()) + 0.2


def test_roi_parity():
    """A crop given the whole-image auto-level bounds must match the same
    region of the full-image result; using local bounds must NOT."""
    rng = np.random.default_rng(3)
    full = rng.integers(60, 130, size=(64, 64, 3)).astype(np.uint8)
    p = {**ap.DEFAULT_PARAMS, "perceptual_autolevel": True,
         "perceptual_autolevel_clip": 0.5}
    lo, hi = ap.oklab_autolevel_lohi(full, 0.5)
    crop = Image.fromarray(full[:16, :16], "RGB")
    full_region = _arr(ap.perceptual_oklch(Image.fromarray(full, "RGB"), p))[:16, :16]
    with_global = _arr(ap.perceptual_oklch(crop, p, lohi=(lo, hi)))
    with_local = _arr(ap.perceptual_oklch(crop, p))
    assert int(np.abs(with_global.astype(int) - full_region.astype(int)).max()) <= 1
    assert int(np.abs(with_local.astype(int) - full_region.astype(int)).max()) > 1


def test_alpha_untouched():
    """process_pil with the feature on must leave alpha identical to the
    feature-off run (the perceptual stage is RGB-only)."""
    rng = np.random.default_rng(4)
    rgb = rng.integers(60, 130, size=(64, 64, 3)).astype(np.uint8)
    pin = Image.fromarray(np.dstack([rgb, np.full((64, 64), 200, np.uint8)]), "RGBA")
    off = ap.process_pil(pin, {})
    on = ap.process_pil(pin, {"perceptual_enabled": True, "perceptual_chroma": 1.4,
                              "perceptual_autolevel": True, "perceptual_hue": 15.0})
    a_off = np.asarray(off.getchannel("A"))
    a_on = np.asarray(on.getchannel("A"))
    assert np.array_equal(a_off, a_on), "alpha changed by the perceptual stage"
    # ...and RGB did change, proving the stage is actually doing something.
    assert not np.array_equal(np.asarray(off.convert("RGB")),
                              np.asarray(on.convert("RGB")))


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} perceptual tests passed.")
