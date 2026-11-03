"""Tests for the Normal Map effect: the new Sprite (alpha-aware) method and the
kept Original (luminance) method.

Run: py -3 _test_normal_map.py
"""
import numpy as np
from PIL import Image
import alpha_pipeline as ap
import post_fx as pf

NM = pf.EFFECT_NORMAL_MAP
ORIGINAL, SPRITE = pf.NORMAL_METHODS[1], pf.NORMAL_METHODS[0]


def _disc_rgba(H=48, W=48, cx=24, cy=24, r=14, fill=(0, 0, 0), colour=(120, 90, 60)):
    a = np.zeros((H, W, 4), np.uint8)
    a[..., :3] = fill
    yy, xx = np.mgrid[0:H, 0:W]
    disc = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    a[disc, :3] = colour
    a[disc, 3] = 255
    return a, disc


def _sprite(a, **kw):
    img = Image.fromarray(a, "RGBA")
    return np.asarray(pf._sprite_normal_map(img.convert("RGB"), img.getchannel("A"), **kw))


def test_original_method_unchanged():
    """Original = the previous _normal_map, RGB out, and it strips alpha."""
    a, _ = _disc_rgba()
    img = Image.fromarray(a, "RGBA").convert("RGB")
    ref = np.asarray(pf._normal_map(img, 1.3))
    out = pf.apply_effect(img, NM, 1.3, normal_method=ORIGINAL)
    assert out.mode == "RGB" and np.array_equal(np.asarray(out), ref)
    assert pf.effect_strips_alpha(NM, {"normal_method": ORIGINAL})
    assert not pf.effect_strips_alpha(NM, {"normal_method": SPRITE})


def test_edt_exact_vs_brute_force():
    rng = np.random.default_rng(0)
    for h, w in ((9, 13), (17, 5), (1, 8), (12, 12)):
        m = rng.random((h, w)) < 0.75
        m[0, 0] = False                                # at least one background
        d = pf._edt(m)
        bg = np.argwhere(~m)
        for y in range(h):
            for x in range(w):
                ref = np.sqrt(((bg - (y, x)) ** 2).sum(1)).min()
                assert abs(d[y, x] - ref) < 1e-6, (h, w, y, x, d[y, x], ref)


def _flood_labels(m):
    """Brute-force 8-connected labelling (reference)."""
    h, w = m.shape
    lab = np.full((h, w), -1)
    n = 0
    for y in range(h):
        for x in range(w):
            if m[y, x] and lab[y, x] < 0:
                stack = [(y, x)]
                lab[y, x] = n
                while stack:
                    cy, cx = stack.pop()
                    for ny in range(max(0, cy - 1), min(h, cy + 2)):
                        for nx in range(max(0, cx - 1), min(w, cx + 2)):
                            if m[ny, nx] and lab[ny, nx] < 0:
                                lab[ny, nx] = n
                                stack.append((ny, nx))
                n += 1
    return lab, n


def test_islands_match_flood_fill():
    rng = np.random.default_rng(1)
    for _ in range(20):
        m = rng.random((23, 31)) < 0.45
        lab, boxes = pf._islands(m)
        ref, n = _flood_labels(m)
        assert len(boxes) == n
        # Same partition: a one-to-one map between label ids.
        pairs = set(zip(lab[m].tolist(), ref[m].tolist()))
        assert len(pairs) == n and len({p[0] for p in pairs}) == n
        for k, (y0, y1, x0, x1) in enumerate(boxes):
            ys, xs = np.nonzero(lab == k)
            assert (y0, y1, x0, x1) == (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)


def test_flat_sprite_gets_volume_and_points_outward():
    a, disc = _disc_rgba()
    n = _sprite(a)
    assert np.array_equal(n[..., 3], a[..., 3])                 # alpha kept
    assert (n[~disc, :3] == (128, 128, 255)).all()               # flat outside
    assert abs(int(n[24, 24, 0]) - 128) <= 1 and abs(int(n[24, 24, 1]) - 128) <= 1
    assert n[24, 11, 0] < 110 and n[24, 37, 0] > 146             # left / right edges
    assert n[11, 24, 1] > 146 and n[37, 24, 1] < 110             # top / bottom (OpenGL)


def test_hidden_fill_colour_is_ignored():
    a1, _ = _disc_rgba(fill=(0, 0, 0))
    a2, _ = _disc_rgba(fill=(200, 30, 250))
    assert np.array_equal(_sprite(a1), _sprite(a2))


def test_directx_flips_green_only():
    a, disc = _disc_rgba()
    gl = _sprite(a).astype(int)
    dx = _sprite(a, fmt=pf.NORMAL_FORMATS[1]).astype(int)
    assert np.array_equal(gl[..., [0, 2, 3]], dx[..., [0, 2, 3]])
    assert set(np.unique(gl[disc, 1] + dx[disc, 1]).tolist()) <= {255, 256}


def test_islands_processed_independently():
    """Two discs 2 px apart: each one's normals equal its normals when alone."""
    a1, d1 = _disc_rgba(W=64, cx=16, r=12)
    a2, d2 = _disc_rgba(W=64, cx=43, r=12)                  # gap of 2 px at x=29..30
    both = a1.copy()
    both[d2] = a2[d2]
    nb, n1, n2 = _sprite(both), _sprite(a1), _sprite(a2)
    assert np.array_equal(nb[d1], n1[d1]) and np.array_equal(nb[d2], n2[d2])


def test_pixel_size_matches_native():
    """A x4 nearest upscale with pixel size 4 gives (close to) the native
    sprite's normals at the block centres."""
    rng = np.random.default_rng(2)
    a, disc = _disc_rgba(H=32, W=32, cx=16, cy=16, r=11)
    a[disc, :3] = np.clip(a[disc, :3] + rng.integers(-30, 31, (int(disc.sum()), 3)), 0, 255)
    native = _sprite(a).astype(int)
    up = _sprite(a.repeat(4, 0).repeat(4, 1), pixel_size=4).astype(int)[2::4, 2::4]
    diff = np.abs(native[disc, :3] - up[disc, :3])
    assert diff.mean() < 4 and np.percentile(diff, 95) < 12, (diff.mean(), np.percentile(diff, 95))


def test_chain_keeps_alpha_and_tiles_isolate():
    a, _ = _disc_rgba(H=64, W=64, cx=32, cy=32, r=26)
    img = Image.fromarray(a, "RGBA")
    chain = [(NM, 1.0, {"normal_method": SPRITE})]
    out = pf.apply_effect_chain(img.convert("RGB"), chain, alpha=img.getchannel("A"))
    assert out.mode == "RGBA" and np.array_equal(np.asarray(out)[..., 3], a[..., 3])
    tiled = np.asarray(pf.apply_effect_chain(img.convert("RGB"), chain, tile_size=32,
                                             alpha=img.getchannel("A")))
    for y0 in (0, 32):
        for x0 in (0, 32):
            tile = a[y0:y0 + 32, x0:x0 + 32]
            assert np.array_equal(tiled[y0:y0 + 32, x0:x0 + 32], _sprite(tile))
    orig = pf.apply_effect_chain(img.convert("RGB"), [(NM, 1.0, {"normal_method": ORIGINAL})],
                                 tile_size=32, alpha=img.getchannel("A"))
    assert orig.mode == "RGB"


def test_pipeline_methods_preview_and_roi():
    a, _ = _disc_rgba(H=96, W=96, cx=48, cy=48, r=34, colour=(150, 100, 70))
    src = Image.fromarray(a, "RGBA")
    base = {"fx_effect_1": NM, "snap_enabled": False}
    sprite = ap.process_pil(src, base)
    original = ap.process_pil(src, {**base, "fx_normal_method_1": ORIGINAL})
    assert sprite.mode == "RGBA" and original.mode == "RGB"
    pp = ap.PreviewPipeline()
    for p in (base, {**base, "fx_normal_method_1": ORIGINAL},
              {**base, "fx_normal_pixel_1": 3}, {**base, "fx_normal_format_1": pf.NORMAL_FORMATS[1]}):
        assert np.array_equal(np.asarray(pp.run(src, p).convert("RGBA")),
                              np.asarray(ap.process_pil(src, p).convert("RGBA")))
    full = np.asarray(sprite.convert("RGBA"))
    rect = (20, 20, 76, 76)
    crop, _ = ap.process_crop(src, base, rect)
    ref = full[rect[1]:rect[3], rect[0]:rect[2]]
    assert np.array_equal(np.asarray(crop.convert("RGBA")), ref)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} normal-map tests passed.")
