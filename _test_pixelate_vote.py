"""Tests for the pixelate "Majority vote" block colour.

Run: py -3 _test_pixelate_vote.py
"""
import numpy as np
from PIL import Image
import alpha_pipeline as ap

FACTORS = (2, 3, 4, 6, 8, 1.25, 1.5, 2.5, 5.5)
VOTE = "Majority vote"


def _rand_rgba(h, w, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 256, (h, w, 4), dtype=np.uint8)
    a[..., 3] = rng.choice(np.array([0, 127, 255], np.uint8), (h, w))
    return Image.fromarray(a, "RGBA")


def _block_ids(h, w, factor):
    sh, sw = max(1, round(h / factor)), max(1, round(w / factor))
    cy = ap._nearest_block_index(h, sh)
    cx = ap._nearest_block_index(w, sw)
    return cy[:, None] * sw + cx[None, :]


def test_centre_sample_unchanged():
    """The default mode is still the plain two-resize nearest pixelate."""
    img = _rand_rgba(29, 37)
    for f in FACTORS:
        sw, sh = max(1, round(37 / f)), max(1, round(29 / f))
        old = img.resize((sw, sh), Image.Resampling.NEAREST).resize(
            (37, 29), Image.Resampling.NEAREST)
        assert np.array_equal(np.asarray(ap.pixelate(img, f)), np.asarray(old)), f


def test_vote_uses_the_centre_sample_grid():
    """Vote blocks are exactly the centre-sample blocks, fractional factors too."""
    h, w = 29, 37
    # Every pixel unique, so each centre-sample block shows one distinct value.
    idx = np.arange(h * w, dtype=np.uint32).reshape(h, w)
    a = np.zeros((h, w, 4), np.uint8)
    a[..., 0], a[..., 1], a[..., 2] = idx & 255, idx >> 8, 7
    a[..., 3] = 255
    img = Image.fromarray(a, "RGBA")
    for f in FACTORS:
        ids = _block_ids(h, w, f)
        for out in (ap.pixelate(img, f), ap.pixelate(img, f, block_colour=VOTE)):
            o = np.asarray(out).reshape(-1, 4)
            for b in np.unique(ids):
                vals = np.unique(o[ids.reshape(-1) == b], axis=0)
                assert len(vals) == 1, (f, b)
        centre = np.asarray(ap.pixelate(img, f)).reshape(-1, 4)
        firsts = [centre[ids.reshape(-1) == b][0] for b in np.unique(ids)]
        assert len(np.unique(np.array(firsts), axis=0)) == len(firsts), f


def test_vote_picks_majority():
    """10 noisy red pixels beat 6 blue ones, even with blue at the centre."""
    rng = np.random.default_rng(1)
    a = np.zeros((4, 4, 4), np.uint8)
    a[..., 3] = 255
    blue = np.zeros((4, 4), bool)
    blue[1:4, 2:4] = True                              # 6 px, covers centre (2, 2)
    a[blue, :3] = (20, 40, 200)
    a[~blue, :3] = (200, 30, 30) + rng.integers(-3, 4, (10, 3))
    img = Image.fromarray(a, "RGBA")
    centre = np.asarray(ap.pixelate(img, 4))
    vote = np.asarray(ap.pixelate(img, 4, block_colour=VOTE))
    assert tuple(centre[0, 0, :3]) == (20, 40, 200)
    assert (vote == vote[0, 0]).all()
    assert any((vote[0, 0] == p).all() for p in a[~blue])    # a real red pixel


def test_vote_picks_majority_large_block():
    """Factor 8 (candidate subset): a 40 px noisy red ring beats a 24 px blue
    middle that holds the centre pixel."""
    rng = np.random.default_rng(5)
    a = np.zeros((8, 8, 4), np.uint8)
    a[..., 3] = 255
    blue = np.zeros((8, 8), bool)
    blue[2:6, 1:7] = True                              # 24 px, covers centre (4, 4)
    a[blue, :3] = (20, 40, 200)
    a[~blue, :3] = (200, 30, 30) + rng.integers(-3, 4, (40, 3))
    img = Image.fromarray(a, "RGBA")
    assert tuple(np.asarray(ap.pixelate(img, 8))[0, 0, :3]) == (20, 40, 200)
    vote = np.asarray(ap.pixelate(img, 8, block_colour=VOTE))
    assert (vote == vote[0, 0]).all()
    assert any((vote[0, 0] == p).all() for p in a[~blue])


def test_vote_never_invents_values():
    """Every output colour is a pixel of its block; alpha stays in {0,127,255}."""
    h, w = 31, 41
    rng = np.random.default_rng(2)
    base = np.array([[200, 30, 30], [20, 40, 200], [240, 240, 240]], np.int16)
    a = np.zeros((h, w, 4), np.uint8)
    pick = rng.integers(0, 3, (h, w))
    a[..., :3] = np.clip(base[pick] + rng.integers(-6, 7, (h, w, 3)), 0, 255)
    a[..., 3] = rng.choice(np.array([0, 127, 255], np.uint8), (h, w), p=[.2, .2, .6])
    img = Image.fromarray(a, "RGBA")
    flat = a.reshape(-1, 4)
    for f in (3, 2.5, 4, 8):
        o = np.asarray(ap.pixelate(img, f, block_colour=VOTE)).reshape(-1, 4)
        ids = _block_ids(h, w, f).reshape(-1)
        for b in np.unique(ids):
            m = ids == b
            assert any((o[m][0] == p).all() for p in flat[m]), (f, b)
        assert set(np.unique(o[:, 3]).tolist()) <= {0, 127, 255}


def test_transparent_pixels_vote_together():
    """9 transparent pixels with junk RGB outvote 7 opaque ones (the RGB under
    alpha 0 must not split their vote)."""
    rng = np.random.default_rng(3)
    a = np.zeros((4, 4, 4), np.uint8)
    a[..., :3] = rng.integers(0, 256, (4, 4, 3))
    opaque = np.zeros((4, 4), bool)
    opaque[1:4, 1:4] = True
    opaque[1, 1] = opaque[3, 3] = False               # 7 px, covers centre (2, 2)
    a[opaque] = (40, 160, 60, 255)
    img = Image.fromarray(a, "RGBA")
    assert np.asarray(ap.pixelate(img, 4))[0, 0, 3] == 255
    vote = np.asarray(ap.pixelate(img, 4, block_colour=VOTE))
    assert (vote[..., 3] == 0).all()


def test_no_majority_falls_back_to_centre():
    """All 16 colours far apart (beyond tolerance): the vote ties -> centre."""
    a = np.zeros((4, 4, 4), np.uint8)
    a[..., 3] = 255
    for i in range(16):
        a[i // 4, i % 4, :3] = ((i * 53) % 256, (i * 97) % 256, (i * 151) % 256)
    img = Image.fromarray(a, "RGBA")
    centre = np.asarray(ap.pixelate(img, 4))
    vote = np.asarray(ap.pixelate(img, 4, block_colour=VOTE, tolerance=8))
    assert np.array_equal(centre, vote)


def test_rgb_mode_round_trips():
    img = _rand_rgba(20, 20).convert("RGB")
    out = ap.pixelate(img, 4, block_colour=VOTE)
    assert out.mode == "RGB" and out.size == img.size


def _noisy_disc(H=96, W=96):
    rng = np.random.default_rng(4)
    a = np.zeros((H, W, 4), np.uint8)
    yy, xx = np.mgrid[0:H, 0:W]
    disc = (xx - 48) ** 2 + (yy - 48) ** 2 <= 34 ** 2
    a[disc, :3] = np.clip(130 + rng.integers(-8, 9, (int(disc.sum()), 3)), 0, 255)
    a[disc, 3] = 255
    return Image.fromarray(a, "RGBA")


def test_preview_cache_picks_up_mode_change():
    """PreviewPipeline reruns pixelate when only the block colour / tolerance
    changes, and still matches process_pil."""
    src = _noisy_disc()
    pp = ap.PreviewPipeline()
    base = {"pixelate_factor": 4, "snap_enabled": False}
    for p in (base,
              {**base, "pixelate_block_colour": VOTE},
              {**base, "pixelate_block_colour": VOTE, "pixelate_vote_tolerance": 4}):
        got = np.asarray(pp.run(src, p).convert("RGBA"))
        ref = np.asarray(ap.process_pil(src, p).convert("RGBA"))
        assert np.array_equal(got, ref)
    a = np.asarray(ap.process_pil(src, base).convert("RGBA"))
    b = np.asarray(ap.process_pil(src, {**base, "pixelate_block_colour": VOTE}).convert("RGBA"))
    assert not np.array_equal(a, b), "vote changed nothing on a noisy image"


def test_stage_pipeline_roi_parity():
    """process_crop with majority vote matches the full render."""
    src = _noisy_disc()
    p = {"pixelate_factor": 4, "pixelate_block_colour": VOTE, "snap_enabled": False}
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
    print(f"\nAll {len(tests)} pixelate-vote tests passed.")
