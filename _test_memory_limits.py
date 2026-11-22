"""Tests for the RAM-based image size limit and the batch worker RAM cap.

Run: py -3 _test_memory_limits.py
"""
import tempfile
from pathlib import Path

from PIL import Image
import alpha_pipeline as ap

GiB = 2 ** 30
PILLOW = 2 * 89_478_485


def test_memory_status_is_sane():
    st = ap.memory_status()
    assert st is not None, "RAM not detected"
    total, avail = st
    assert 1 * GiB < total < 4096 * GiB and 0 < avail <= total


def test_limit_from_ram_never_below_pillow():
    assert ap.image_pixel_limit(None) == PILLOW               # unknown RAM
    assert ap.image_pixel_limit(16 * GiB) == PILLOW           # 147 MP < Pillow's 179
    lim64 = ap.image_pixel_limit(64 * GiB)
    assert lim64 == int(64 * GiB * ap.RAM_FRACTION / ap.BYTES_PER_PIXEL)
    assert 580e6 < lim64 < 600e6                              # ~24k x 24k
    assert ap.image_pixel_limit(128 * GiB) > lim64


def test_pillow_uses_our_limit():
    assert ap.IMAGE_PIXEL_LIMIT >= PILLOW
    assert ap.IMAGE_PIXEL_LIMIT <= 2 * Image.MAX_IMAGE_PIXELS <= ap.IMAGE_PIXEL_LIMIT + 1


class _Limit:
    """Temporarily lower the limit to `px` pixels (Pillow + the message)."""
    def __init__(self, px):
        self.px = px

    def __enter__(self):
        self.saved = (Image.MAX_IMAGE_PIXELS, ap.IMAGE_PIXEL_LIMIT)
        Image.MAX_IMAGE_PIXELS, ap.IMAGE_PIXEL_LIMIT = (self.px + 1) // 2, self.px

    def __exit__(self, *exc):
        Image.MAX_IMAGE_PIXELS, ap.IMAGE_PIXEL_LIMIT = self.saved


def test_too_large_gives_plain_message_everywhere():
    with tempfile.TemporaryDirectory(prefix="amem_") as d:
        p = Path(d) / "huge_sheet.png"
        Image.new("RGBA", (100, 50), (1, 2, 3, 128)).save(p)
        with _Limit(2000):
            for fn in (ap.open_image, ap.load_rgba, ap.classify_for_preset):
                try:
                    fn(p)
                except ap.ImageTooLargeError as e:
                    msg = str(e)
                    assert "huge_sheet.png" in msg and "megapixels" in msg, msg
                    assert "decompression bomb" not in msg.lower()
                    continue
                raise AssertionError(f"{fn.__name__} opened an over-limit image")
            try:                                    # the batch save path too
                ap.process_image(p, Path(d) / "out.png", {})
            except ap.ImageTooLargeError:
                pass
            else:
                raise AssertionError("process_image opened an over-limit image")
        assert ap.load_rgba(p).size == (100, 50)    # fine at the real limit


def test_batch_worker_cap():
    saved = (ap.memory_status, ap.BYTES_PER_PIXEL, ap.WORKER_OVERHEAD)
    with tempfile.TemporaryDirectory(prefix="amem_") as d:
        small, big = Path(d) / "s.png", Path(d) / "b.png"
        Image.new("RGBA", (10, 10)).save(small)
        Image.new("RGBA", (100, 100)).save(big)            # 10,000 px = the largest
        junk = Path(d) / "junk.png"
        junk.write_bytes(b"not an image")
        try:
            ap.BYTES_PER_PIXEL, ap.WORKER_OVERHEAD = 100_000, 0   # 1 GB per worker
            ap.memory_status = lambda: (64 * GiB, 32 * GiB)        # 25.6 GB usable
            assert ap.batch_worker_cap([small, big, junk], 7) == (7, "")
            ap.memory_status = lambda: (16 * GiB, 4 * GiB)         # 3.4 GB usable
            n, note = ap.batch_worker_cap([small, big, junk], 7)
            assert n == 3 and "limited to 3" in note and "free RAM" in note, (n, note)
            ap.memory_status = lambda: (16 * GiB, GiB // 4)        # not even one fits
            assert ap.batch_worker_cap([big], 7)[0] == 1
            ap.memory_status = lambda: None                        # unknown RAM
            assert ap.batch_worker_cap([big], 7) == (7, "")
            assert ap.batch_worker_cap([big], 1) == (1, "")
        finally:
            ap.memory_status, ap.BYTES_PER_PIXEL, ap.WORKER_OVERHEAD = saved


def test_tune_load_shows_too_large_dialog():
    """Tune tab: an over-limit image gives the "Image too large" dialog and
    leaves the current image alone. Skipped if no display."""
    try:
        import main
        app = main.AlphaToolsApp()
        app.withdraw(); app.update_idletasks()
    except Exception as e:
        print("SKIP tune load (no GUI):", e)
        return
    shown = []
    orig = (main.filedialog.askopenfilename, main.save_dialog_dir,
            main.load_dialog_dir, main.messagebox.showerror)
    try:
        with tempfile.TemporaryDirectory(prefix="amem_") as d:
            p = Path(d) / "big.png"
            Image.new("RGBA", (100, 50)).save(p)
            main.filedialog.askopenfilename = lambda **k: str(p)
            main.save_dialog_dir = lambda *a, **k: None
            main.load_dialog_dir = lambda *a, **k: ""
            main.messagebox.showerror = lambda title, msg: shown.append((title, msg))
            with _Limit(2000):
                app.tune._load_image()
            assert shown and shown[0][0] == "Image too large" and "big.png" in shown[0][1]
            assert app.tune.source_image is None
    finally:
        (main.filedialog.askopenfilename, main.save_dialog_dir,
         main.load_dialog_dir, main.messagebox.showerror) = orig
        app.destroy()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} memory-limit tests passed.")
