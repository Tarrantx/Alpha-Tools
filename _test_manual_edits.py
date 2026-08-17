"""Phase 1 checks for the manual-edits layer + its two pipeline stages.

Covers the edit-layer model (paint/erase/undo/clear/sidecar), the apply helpers
at every resolution mode (full 1:1, downscaled 'scale', batch 'place'), the
hybrid pipeline placement (RGB early = reprocessed by tone; alpha late =
survives snap), ROI-crop parity with the full render, and PreviewPipeline
parity + cache invalidation on edits.

Run: py -3 _test_manual_edits.py
"""
import numpy as np
from PIL import Image
import alpha_pipeline as ap
import manual_edits as me


def _img(w=64, h=64, rgb=(120, 90, 60), a=255):
    arr = np.zeros((h, w, 4), np.uint8)
    arr[..., 0], arr[..., 1], arr[..., 2] = rgb
    arr[..., 3] = a
    return Image.fromarray(arr, "RGBA")


# -- model -------------------------------------------------------------------
def test_paint_and_state():
    L = me.EditLayer(64, 64)
    assert L.is_empty() and L.version() == 0
    L.begin_stroke(); L.apply_dab(10, 10, 6, "alpha", "paint", alpha_value=128); L.end_stroke()
    assert L.has_alpha() and not L.has_rgb()
    assert L.version() == 1
    L.begin_stroke(); L.apply_dab(40, 40, 6, "rgb", "paint", rgb_value=(1, 2, 3)); L.end_stroke()
    assert L.has_rgb() and L.version() == 2
    assert L.alpha_mask[10, 10] and L.alpha[10, 10] == 128
    assert L.rgb_mask[40, 40] and tuple(L.rgb[40, 40]) == (1, 2, 3)


def test_undo_and_overlap():
    L = me.EditLayer(64, 64)
    L.begin_stroke(); L.apply_dab(20, 20, 10, "alpha", "paint", alpha_value=100); L.end_stroke()
    L.begin_stroke(); L.apply_dab(22, 22, 10, "alpha", "paint", alpha_value=200); L.end_stroke()
    # Undo the 2nd (overlapping) stroke -> the overlap returns to stroke 1's 100.
    assert L.undo()
    assert L.alpha_mask[20, 20] and L.alpha[20, 20] == 100
    # Undo the 1st -> empty again.
    assert L.undo()
    assert L.is_empty()
    assert not L.undo()             # nothing left


def test_erase_reverts():
    L = me.EditLayer(64, 64)
    L.begin_stroke(); L.apply_dab(30, 30, 12, "alpha", "paint", alpha_value=150); L.end_stroke()
    L.begin_stroke(); L.apply_dab(30, 30, 4, "alpha", "erase"); L.end_stroke()
    assert not L.alpha_mask[30, 30]          # erased pixel falls back to pipeline
    assert L.alpha_mask[30, 25]              # outside the eraser still painted


def test_clear_is_undoable():
    L = me.EditLayer(64, 64)
    L.begin_stroke(); L.apply_dab(30, 30, 8, "rgb", "paint", rgb_value=(9, 9, 9)); L.end_stroke()
    L.clear()
    assert L.is_empty()
    assert L.undo() and L.has_rgb()          # clear can be undone


def test_sidecar_roundtrip(tmp="_me_side.npz"):
    import os
    L = me.EditLayer(48, 32)
    L.begin_stroke(); L.apply_dab(10, 10, 6, "alpha", "paint", alpha_value=77); L.end_stroke()
    L.begin_stroke(); L.apply_dab(20, 15, 6, "rgb", "paint", rgb_value=(5, 6, 7)); L.end_stroke()
    L.save(tmp)
    try:
        R = me.EditLayer.load(tmp)
        assert (R.w, R.h) == (48, 32)
        assert np.array_equal(R.alpha_mask, L.alpha_mask)
        assert np.array_equal(R.rgb, L.rgb)
    finally:
        os.remove(tmp)


# -- apply helpers at each resolution mode -----------------------------------
def test_apply_full_exact():
    L = me.EditLayer(64, 64)
    L.begin_stroke(); L.apply_dab(32, 32, 8, "alpha", "paint", alpha_value=42); L.end_stroke()
    a = Image.new("L", (64, 64), 255)
    out = np.asarray(me.apply_alpha(a, L))          # region None, size==layer -> exact
    assert out[32, 32] == 42 and out[0, 0] == 255


def test_apply_scale_downsampled():
    L = me.EditLayer(64, 64)
    L.begin_stroke(); L.apply_dab(32, 32, 16, "alpha", "paint", alpha_value=99); L.end_stroke()
    a = Image.new("L", (32, 32), 255)               # half-res preview
    out = np.asarray(me.apply_alpha(a, L, fit="scale"))
    assert out.shape == (32, 32) and (out == 99).any()   # edit present, resampled


def test_apply_region_crop_exact():
    L = me.EditLayer(64, 64)
    L.begin_stroke(); L.apply_dab(40, 40, 8, "alpha", "paint", alpha_value=55); L.end_stroke()
    # Crop region (32,32)-(56,56) at full res -> 24x24; painted px 40,40 -> local 8,8.
    a = Image.new("L", (24, 24), 255)
    out = np.asarray(me.apply_alpha(a, L, region=(32, 32, 56, 56), fit="scale"))
    assert out[8, 8] == 55 and out[0, 0] == 255


def test_apply_place_batch_clip():
    L = me.EditLayer(64, 64)
    L.begin_stroke(); L.apply_dab(10, 10, 6, "rgb", "paint", rgb_value=(7, 8, 9)); L.end_stroke()
    # Batch onto a SMALLER image: fixed x,y, clipped, no scaling.
    rgb = Image.new("RGB", (32, 32), (0, 0, 0))
    out = np.asarray(me.apply_rgb(rgb, L, fit="place"))
    assert tuple(out[10, 10]) == (7, 8, 9)          # same coords, not rescaled


# -- pipeline placement (hybrid) ---------------------------------------------
def test_alpha_edit_survives_snap():
    """Alpha painted 200 must survive the default N-level snap (edit is applied
    AFTER snap/clamp)."""
    L = me.EditLayer(64, 64)
    L.begin_stroke(); L.apply_dab(32, 32, 10, "alpha", "paint", alpha_value=200); L.end_stroke()
    src = _img(a=180)                                # 180 would snap to 255
    out = np.asarray(ap.process_pil(src, {"_manual_edits": L}).getchannel("A"))
    assert out[32, 32] == 200                        # not snapped
    assert out[0, 0] in (0, 128, 255)                # elsewhere: snapped


def test_rgb_edit_reprocessed_by_tone():
    """RGB painted early is reprocessed by tone (saturation changes it), and is
    applied verbatim when tone is neutral."""
    L = me.EditLayer(64, 64)
    L.begin_stroke(); L.apply_dab(32, 32, 10, "rgb", "paint", rgb_value=(150, 40, 40)); L.end_stroke()
    neutral = np.asarray(ap.process_pil(_img(), {"_manual_edits": L}).convert("RGB"))
    assert tuple(neutral[32, 32]) == (150, 40, 40)   # verbatim when tone neutral
    boosted = np.asarray(ap.process_pil(
        _img(), {"_manual_edits": L, "saturation": 2.0}).convert("RGB"))
    assert tuple(boosted[32, 32]) != (150, 40, 40)   # tone reprocessed it


# -- ROI crop parity ---------------------------------------------------------
def test_roi_crop_parity():
    """process_crop of a region containing edits matches the same region of the
    full render (validates region slicing + the ctx plumbing)."""
    L = me.EditLayer(128, 128)
    L.begin_stroke(); L.apply_dab(70, 70, 12, "alpha", "paint", alpha_value=200); L.end_stroke()
    L.begin_stroke(); L.apply_dab(60, 60, 10, "rgb", "paint", rgb_value=(10, 220, 30)); L.end_stroke()
    src = _img(128, 128)
    params = {"_manual_edits": L}
    full = np.asarray(ap.process_pil(src, params).convert("RGBA"))
    rect = (48, 48, 96, 96)
    crop, r = ap.process_crop(src, params, rect)
    crop = np.asarray(crop.convert("RGBA"))
    ref = full[rect[1]:rect[3], rect[0]:rect[2]]
    assert crop.shape == ref.shape
    assert np.array_equal(crop, ref)


# -- PreviewPipeline parity + cache invalidation -----------------------------
def test_preview_parity_and_invalidation():
    L = me.EditLayer(80, 80)
    L.begin_stroke(); L.apply_dab(40, 40, 12, "alpha", "paint", alpha_value=200); L.end_stroke()
    src = _img(80, 80)
    pp = ap.PreviewPipeline()

    def a(im):
        return np.asarray(im.convert("RGBA"), np.uint8)

    p = {"_manual_edits": L}
    assert np.array_equal(a(pp.run(src, p)), a(ap.process_pil(src, p)))
    before = a(pp.run(src, p))
    # A new stroke bumps the version -> the cached stage must invalidate.
    L.begin_stroke(); L.apply_dab(20, 20, 12, "rgb", "paint", rgb_value=(0, 255, 0)); L.end_stroke()
    after = a(pp.run(src, p))
    assert not np.array_equal(before, after), "preview didn't update after an edit"
    assert np.array_equal(after, a(ap.process_pil(src, p)))


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} manual-edit tests passed.")
