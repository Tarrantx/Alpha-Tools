"""Tests for the pixel-exact power-of-two output scale (Batch tab).

Run: py -3 _test_output_scale.py
"""
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
import alpha_pipeline as ap

VOTE, CENTRE = "Majority vote", "Centre sample"


def _rand_rgba(h, w, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 256, (h, w, 4), dtype=np.uint8)
    a[..., 3] = rng.choice(np.array([0, 127, 255], np.uint8), (h, w))
    return a


def test_one_is_a_no_op():
    img = Image.fromarray(_rand_rgba(9, 7), "RGBA")
    assert ap.scale_pow2(img, 1.0) is img


def test_rejects_non_power_of_two():
    img = Image.fromarray(_rand_rgba(8, 8), "RGBA")
    for bad in (3.0, 1 / 3, 1.5, 6.0):
        try:
            ap.scale_pow2(img, bad)
        except ValueError:
            continue
        raise AssertionError(f"scale {bad} was accepted")


def test_upscale_duplicates_pixels():
    a = _rand_rgba(9, 7)
    for k in (2, 4, 8):
        out = np.asarray(ap.scale_pow2(Image.fromarray(a, "RGBA"), float(k)))
        assert np.array_equal(out, a.repeat(k, 0).repeat(k, 1)), k


def test_round_trip_is_exact():
    """native -> up k -> down k gives the native image back, both methods."""
    a = _rand_rgba(13, 11)
    img = Image.fromarray(a, "RGBA")
    for k in (2, 4, 8):
        up = ap.scale_pow2(img, float(k))
        for m in (VOTE, CENTRE):
            back = np.asarray(ap.scale_pow2(up, 1.0 / k, m))
            assert np.array_equal(back, a), (k, m)


def test_centre_matches_block_centre():
    a = _rand_rgba(32, 24)
    for k in (2, 4, 8):
        out = np.asarray(ap.scale_pow2(Image.fromarray(a, "RGBA"), 1.0 / k, CENTRE))
        assert np.array_equal(out, a[k // 2::k, k // 2::k]), k


def test_vote_picks_block_majority():
    """4x4 block: 10 noisy red pixels beat 6 blue ones holding the centre."""
    rng = np.random.default_rng(1)
    a = np.zeros((4, 8, 4), np.uint8)
    a[..., 3] = 255
    blue = np.zeros((4, 8), bool)
    blue[1:4, 2:4] = True
    a[blue, :3] = (20, 40, 200)
    a[~blue, :3] = np.clip(np.array([200, 30, 30]) + rng.integers(-3, 4, (26, 3)), 0, 255)
    img = Image.fromarray(a, "RGBA")
    centre = np.asarray(ap.scale_pow2(img, 0.25, CENTRE))
    vote = np.asarray(ap.scale_pow2(img, 0.25, VOTE))
    assert centre.shape == (1, 2, 4) and tuple(centre[0, 0, :3]) == (20, 40, 200)
    assert any((vote[0, 0] == p).all() for p in a[:, :4][~blue[:, :4]])


def test_uneven_size_rounds_up_without_cropping():
    """10x7 at 1/4 -> 3x2; every output pixel comes from its own (partial)
    block, and alpha stays in {0,127,255}."""
    a = _rand_rgba(7, 10, seed=3)
    img = Image.fromarray(a, "RGBA")
    for m in (VOTE, CENTRE):
        out = np.asarray(ap.scale_pow2(img, 0.25, m))
        assert out.shape == (2, 3, 4), m
        for by in range(2):
            for bx in range(3):
                blk = a[by * 4:by * 4 + 4, bx * 4:bx * 4 + 4].reshape(-1, 4)
                assert any((out[by, bx] == p).all() for p in blk), (m, by, bx)
        assert set(np.unique(out[..., 3]).tolist()) <= {0, 127, 255}


def test_indexed_image_keeps_its_palette():
    """A quantised ('P') output stays indexed with the SAME palette, and its
    pixels are exactly the RGBA image's scaled pixels."""
    rgba = Image.fromarray(_rand_rgba(16, 16, seed=4), "RGBA")
    q = ap.quantize_rgba(rgba, 16)
    assert q.mode == "P"
    for s in (0.5, 0.25, 2.0):
        for m in (VOTE, CENTRE):
            out = ap.scale_pow2(q, s, m)
            assert out.mode == "P", (s, m)
            assert out.getpalette(rawmode="RGBA") == q.getpalette(rawmode="RGBA")
            ref = ap.scale_pow2(q.convert("RGBA"), s, m)
            assert np.array_equal(np.asarray(out.convert("RGBA")), np.asarray(ref)), (s, m)


def test_process_image_applies_scale():
    """End to end through the batch save path, incl. a 16-colour indexed PNG."""
    src = Image.fromarray(_rand_rgba(32, 48, seed=5), "RGBA")
    with tempfile.TemporaryDirectory(prefix="ascale_") as t:
        t = Path(t)
        src.save(t / "in.png")
        for scale, size in ((0.5, (24, 16)), (0.125, (6, 4)), (2.0, (96, 64)), (1.0, (48, 32))):
            ap.process_image(t / "in.png", t / "out.png",
                             {"output_scale": scale, "snap_enabled": False})
            with Image.open(t / "out.png") as im:
                assert im.size == size, (scale, im.size)
        ap.process_image(t / "in.png", t / "out.png",
                         {"output_scale": 0.5, "quantize_mode": "16 (4-bit)",
                          "snap_enabled": False})
        with Image.open(t / "out.png") as im:
            assert im.size == (24, 16) and im.mode == "P"
            assert len(im.convert("RGBA").getcolors(256)) <= 16


def test_tune_save_image_scale():
    """Tune tab "Save image": the Save scale / Downscale row reaches the saved
    file, and Downscale is only enabled below 1×. Drives the real handler with
    the file dialog stubbed (and the remembered-folder setting left alone).
    Skipped if no display."""
    try:
        import main
        app = main.AlphaToolsApp()
        app.withdraw(); app.update_idletasks()
    except Exception as e:
        print("SKIP tune save (no GUI):", e)
        return
    saved = {}
    orig = (main.filedialog.asksaveasfilename, main.save_dialog_dir,
            main.load_dialog_dir, main.messagebox.showerror)
    main.save_dialog_dir = lambda *a, **k: None
    main.load_dialog_dir = lambda *a, **k: ""

    def _err(title, msg):
        raise AssertionError(f"{title}: {msg}")
    main.messagebox.showerror = _err
    try:
        t = app.tune
        src = Image.fromarray(_rand_rgba(40, 56, seed=6), "RGBA")
        t.source_image, t.source_path, t.last_after = src, "synthetic.png", src
        assert str(t._save_scale_down_menu.cget("state")) == "disabled"   # 1× (off)
        with tempfile.TemporaryDirectory(prefix="atune_") as d:
            d = Path(d)
            for label, mode, size in (("1/4×", CENTRE, (14, 10)), ("1/2×", VOTE, (28, 20)),
                                      ("2×", VOTE, (112, 80)), ("1× (off)", VOTE, (56, 40))):
                out = d / f"out_{len(saved)}.png"
                main.filedialog.asksaveasfilename = lambda **k: str(out)
                t._save_scale_var.set(label)
                t._update_save_scale_state()
                state = str(t._save_scale_down_menu.cget("state"))
                assert state == ("normal" if label.startswith("1/") else "disabled"), label
                t._save_scale_down_var.set(mode)
                t._save_image()
                with Image.open(out) as im:
                    got = np.asarray(im.convert("RGBA"))
                assert got.shape[1::-1] == size, (label, got.shape)
                # Same pixels as the pipeline's own save at that scale.
                ref = d / "ref.png"
                src.save(d / "src.png")
                ap.process_image(d / "src.png", ref, {**t.params, "output_scale":
                                 dict(ap.OUTPUT_SCALES)[label], "output_scale_down": mode})
                with Image.open(ref) as im:
                    assert np.array_equal(got, np.asarray(im.convert("RGBA"))), label
                saved[label] = size
        assert "scaled" not in t.status_label.cget("text")              # last save was 1×
    finally:
        (main.filedialog.asksaveasfilename, main.save_dialog_dir,
         main.load_dialog_dir, main.messagebox.showerror) = orig
        app.destroy()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} output-scale tests passed.")
