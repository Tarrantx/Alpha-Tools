"""Synthetic smoke test for the pipeline. Run with: py -3 _smoke_test.py"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

import alpha_pipeline as pipeline


def make_test_image(w=128, h=128):
    """Build a tri-level alpha test texture:
       - Top stripe: alpha=255 with red RGB
       - Middle stripe: alpha=128 with green RGB
       - Bottom stripe: alpha=0 with garbage RGB (would normally be invisible
         but typical of textures that haven't had RGB padding applied).
       Then add simulated upscale noise: blur the alpha so values smear off
       the plateaus, and the edges become gradients.
    """
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    a   = np.zeros((h, w),    dtype=np.uint8)

    rgb[0:h//3,           :] = (220, 60, 60)     # red
    rgb[h//3:2*h//3,      :] = (60, 200, 80)     # green
    rgb[2*h//3:,          :] = (40, 40, 200)     # blue (the "transparent" area)

    a[0:h//3,         :] = 255
    a[h//3:2*h//3,    :] = 128
    a[2*h//3:,        :] = 0

    img = Image.fromarray(np.dstack([rgb, a]), mode="RGBA")

    # Simulate upscale: alpha gets smeared by a few pixels of blur
    from PIL import ImageFilter
    alpha_ch = img.getchannel("A").filter(ImageFilter.GaussianBlur(radius=3))
    img.putalpha(alpha_ch)
    return img


def histogram_summary(img: Image.Image, label: str):
    a = np.asarray(img.getchannel("A"), dtype=np.int32)
    print(f"  {label:40s}  shape={a.shape}, "
          f"min={a.min():3d}, max={a.max():3d}, "
          f"%0={(a == 0).mean() * 100:5.1f}, "
          f"%128={(a == 128).mean() * 100:5.1f}, "
          f"%255={(a == 255).mean() * 100:5.1f}, "
          f"mean={a.mean():.1f}")


def test_preset(test_img: Image.Image, preset_path: Path, tmp: Path):
    with open(preset_path) as f:
        data = json.load(f)
    params = data["params"]
    in_path = tmp / "in.png"
    out_path = tmp / f"out_{preset_path.stem}.png"
    test_img.save(in_path)
    pipeline.process_image(in_path, out_path, params, work_dir=tmp)
    return Image.open(out_path).convert("RGBA").copy()


def main():
    test_img = make_test_image(128, 128)
    print("Source alpha histogram:")
    histogram_summary(test_img, "source (post simulated upscale)")
    print()

    presets_dir = Path("presets")
    print(f"Running each shipped preset (excl. vtracer for headless test):")
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        for preset_path in sorted(presets_dir.glob("*.json")):
            # Skip vtracer preset in smoke test — vtracer is slow and we test
            # the pipeline logic separately.
            with open(preset_path) as f:
                if json.load(f).get("params", {}).get("vtracer_enabled"):
                    print(f"  SKIP {preset_path.stem} (vtracer)")
                    continue
            try:
                out_img = test_preset(test_img, preset_path, tmp)
                histogram_summary(out_img, preset_path.stem)
            except Exception as e:
                print(f"  ERR  {preset_path.stem}: {e}")

    # Specifically verify N-level snap rounds to 0/128/255 plateaus
    print()
    print("Direct N-level snap test (n=3, defaults):")
    a = test_img.getchannel("A")
    snapped = pipeline.n_level_snap(a, 3, 25, 230, 40)
    snapped_img = Image.merge("RGBA", (test_img.getchannel("R"),
                                       test_img.getchannel("G"),
                                       test_img.getchannel("B"),
                                       snapped))
    histogram_summary(snapped_img, "after snap")

    # Verify edge dilation actually fills transparent pixels. Use a clean
    # binary alpha test image (no upscale smear) so we can see the bleed
    # against a known-bad RGB background.
    print()
    print("Edge dilation test (clean binary alpha):")
    w = h = 64
    rgb_arr = np.full((h, w, 3), [255, 0, 255], dtype=np.uint8)  # magenta "bad RGB"
    rgb_arr[:h // 2, :, :] = [50, 200, 80]                       # green opaque half
    alpha_arr = np.zeros((h, w), dtype=np.uint8)
    alpha_arr[:h // 2, :] = 255                                  # top half opaque
    rgb_img   = Image.fromarray(rgb_arr,   mode="RGB")
    alpha_img = Image.fromarray(alpha_arr, mode="L")

    dilated = pipeline.dilate_rgb(rgb_img, alpha_img, 6, source_threshold=16)
    dil_arr = np.asarray(dilated).astype(np.int32)

    print(f"  row {h // 2 + 1} (1 px below boundary): "
          f"({dil_arr[h//2+1, w//2, 0]}, "
          f"{dil_arr[h//2+1, w//2, 1]}, "
          f"{dil_arr[h//2+1, w//2, 2]})  expected ~(50,200,80)")
    print(f"  row {h // 2 + 5} (5 px below boundary): "
          f"({dil_arr[h//2+5, w//2, 0]}, "
          f"{dil_arr[h//2+5, w//2, 1]}, "
          f"{dil_arr[h//2+5, w//2, 2]})  expected ~(50,200,80)")
    print(f"  row {h // 2 + 10} (10 px below — beyond reach): "
          f"({dil_arr[h//2+10, w//2, 0]}, "
          f"{dil_arr[h//2+10, w//2, 1]}, "
          f"{dil_arr[h//2+10, w//2, 2]})  expected (255,0,255) magenta")
    near_filled = dil_arr[h//2 + 1, w//2].tolist() == [50, 200, 80]
    far_unfilled = dil_arr[h//2 + 10, w//2].tolist() == [255, 0, 255]
    print(f"  {'PASS' if near_filled and far_unfilled else 'FAIL'}: dilation reached "
          f"{'near band' if near_filled else 'NOT near band'}, "
          f"{'preserved far band' if far_unfilled else 'CORRUPTED far band'}")

    print()
    print("Smoke test complete.")


if __name__ == "__main__":
    main()
