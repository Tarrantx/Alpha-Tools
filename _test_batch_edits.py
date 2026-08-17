"""Phase 4 checks: batch "Include manual edits" — flatten/pickle, the fixed-xy
(fit='place') pipeline behaviour, and the BatchTab param plumbing.

Run: py -3 _test_batch_edits.py
"""
import pickle
import numpy as np
from PIL import Image
import alpha_pipeline as ap
import manual_edits as me


def _img(w, h, a=180):
    arr = np.zeros((h, w, 4), np.uint8)
    arr[..., :3] = (120, 90, 60)
    arr[..., 3] = a
    return Image.fromarray(arr, "RGBA")


def test_flatten_drops_undo_and_pickles():
    L = me.EditLayer(100, 100)
    L.begin_stroke(); L.apply_dab(20, 20, 8, "alpha", "paint", alpha_value=200); L.end_stroke()
    assert L.can_undo()
    f = L.flatten()
    assert not f.can_undo() and f._pending is None      # history dropped
    assert np.array_equal(f.alpha_mask, L.alpha_mask)    # data preserved
    assert f.version() == L.version()
    # Must survive pickling (ProcessPoolExecutor sends it to workers).
    r = pickle.loads(pickle.dumps(f))
    assert np.array_equal(r.alpha, f.alpha) and np.array_equal(r.alpha_mask, f.alpha_mask)


def test_place_vs_scale_via_pipeline():
    """'place' applies at fixed x,y; 'scale' resamples to the target size."""
    L = me.EditLayer(100, 100)
    L.begin_stroke(); L.apply_dab(20, 20, 8, "alpha", "paint", alpha_value=200); L.end_stroke()
    src = _img(60, 60)                                    # smaller than the layer

    place = np.asarray(ap.process_pil(
        src, {"_manual_edits": L, "_manual_edits_fit": "place"}).getchannel("A"))
    scale = np.asarray(ap.process_pil(
        src, {"_manual_edits": L, "_manual_edits_fit": "scale"}).getchannel("A"))

    # place: edit stays at (20,20); the scaled position (12,12) is untouched.
    assert place[20, 20] == 200 and place[12, 12] != 200
    # scale: 100->60 (x0.6) moves the edit to ~(12,12); (20,20) is untouched.
    assert scale[12, 12] == 200 and scale[20, 20] != 200


def test_process_image_place_writes_fixed_coords(tmp="_be_out.png"):
    import os
    L = me.EditLayer(100, 100)
    L.begin_stroke(); L.apply_dab(30, 30, 6, "rgb", "paint", rgb_value=(0, 255, 0)); L.end_stroke()
    src_path = "_be_src.png"
    _img(60, 60, a=255).save(src_path)
    try:
        ap.process_image(src_path, tmp,
                         {"_manual_edits": L.flatten(), "_manual_edits_fit": "place"})
        out = np.asarray(Image.open(tmp).convert("RGB"))
        assert tuple(out[30, 30]) == (0, 255, 0)         # fixed x,y, in-bounds
    finally:
        for f in (src_path, tmp):
            if os.path.exists(f):
                os.remove(f)


def test_batch_build_tasks_includes_edits():
    """BatchTab._build_tasks merges the layer + fit into each file's params
    (so auto-pick presets get them too); absent when off. Skipped if no GUI."""
    try:
        import os, tempfile
        import customtkinter as ctk
        import main
        app = main.AlphaToolsApp(); app.withdraw(); app.update_idletasks()
    except Exception as e:
        print("SKIP batch build-tasks (no GUI):", e)
        return
    try:
        bt = app.batch
        with tempfile.TemporaryDirectory() as d:
            ind = os.path.join(d, "in"); outd = os.path.join(d, "out")
            os.makedirs(ind); os.makedirs(outd)
            _img(40, 40, a=255).save(os.path.join(ind, "a.png"))
            from pathlib import Path
            files = [Path(ind) / "a.png"]
            fmt = bt.output_format_var.get()

            L = me.EditLayer(40, 40)
            L.begin_stroke(); L.apply_dab(10, 10, 4, "alpha", "paint", alpha_value=200); L.end_stroke()
            params_on = {"_manual_edits": L.flatten(), "_manual_edits_fit": "place"}
            tasks = bt._build_tasks(files, Path(ind), Path(outd), fmt,
                                    params_on, autopick=False, skip_existing=False)
            tp = tasks[0]["params"]
            assert tp.get("_manual_edits") is not None
            assert tp.get("_manual_edits_fit") == "place"

            # With no layer in params, tasks carry none.
            tasks_off = bt._build_tasks(files, Path(ind), Path(outd), fmt,
                                        {}, autopick=False, skip_existing=False)
            assert tasks_off[0]["params"].get("_manual_edits") is None
    finally:
        app.destroy()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} batch-edit tests passed.")
