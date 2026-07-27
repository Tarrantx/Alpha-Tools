"""Comprehensive test pass on a real upscaled-texture image.

For each shipped preset and a bunch of parameter combinations, runs the
pipeline and checks for crashes, all-zero alpha, all-zero RGB, wrong output
dimensions, etc. Reports a punch list of anything suspicious.

Run with: py -3 _test_real.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import traceback
from itertools import product
from pathlib import Path

import numpy as np
from PIL import Image

import alpha_pipeline as p


# Use whatever PNG is currently in the test-images folder. Lets the user swap
# the test image without editing this script.
_test_dir = Path(r"C:\Scripts\Test-Images")
_candidates = sorted(_test_dir.glob("*.png")) if _test_dir.is_dir() else []
TEST_IMAGE = _candidates[0] if _candidates else _test_dir / "missing.png"


class Issue:
    def __init__(self, label, problem):
        self.label = label
        self.problem = problem

    def __repr__(self):
        return f"  FAIL  {self.label}\n        {self.problem}"


def assess(name: str, src: Image.Image, out: Image.Image,
           expect_binary: bool = False,
           expect_alpha_unchanged_from: Image.Image | None = None,
           expect_rgb_unchanged_from: Image.Image | None = None,
           expect_pixelate_factor: int | None = None) -> list[Issue]:
    issues = []

    # Dimensions
    if out.size != src.size:
        issues.append(Issue(name,
            f"output size {out.size} != input size {src.size}"))
        return issues
    if out.mode != "RGBA":
        issues.append(Issue(name,
            f"output mode is {out.mode}, expected RGBA"))

    a = np.asarray(out.getchannel("A"), dtype=np.int32)
    rgb = np.asarray(out.convert("RGB"), dtype=np.int32)

    # Catastrophic alpha failure: all zero output
    if a.max() == 0:
        issues.append(Issue(name, "output alpha is ALL ZERO"))
    # Sanity: alpha should be reachable up to 255 somewhere if input had any
    src_a = np.asarray(src.getchannel("A"), dtype=np.int32)
    if src_a.max() > 0 and a.max() == 0:
        issues.append(Issue(name,
            "input had visible content but output is fully transparent"))

    # All-black RGB (often a sign of premature alpha-blending)
    if rgb.max() == 0:
        issues.append(Issue(name, "output RGB is ALL ZERO"))

    # Binary check
    if expect_binary:
        unique_a = np.unique(a)
        if not set(unique_a.tolist()).issubset({0, 255}):
            issues.append(Issue(name,
                f"expected binary alpha, got {len(unique_a)} unique values "
                f"(sample: {list(unique_a[:10])}...)"))

    # Alpha-unchanged check (for presets that should preserve the original)
    if expect_alpha_unchanged_from is not None:
        src_alpha = np.asarray(expect_alpha_unchanged_from.getchannel("A"),
                               dtype=np.int32)
        diff = np.abs(a - src_alpha).mean()
        if diff > 2.0:    # tolerance for rounding through PNG save/load
            issues.append(Issue(name,
                f"alpha drifted from source by mean {diff:.1f} (expected near 0)"))

    # Pixelate check — output should have at most (W/factor) x (H/factor)
    # distinct positions
    if expect_pixelate_factor and expect_pixelate_factor > 1:
        # Count distinct pixel values in a single row
        unique_per_row = len({tuple(rgb[0, x]) for x in range(rgb.shape[1])})
        max_expected = (rgb.shape[1] // expect_pixelate_factor) + 2
        if unique_per_row > max_expected:
            issues.append(Issue(name,
                f"pixelate factor {expect_pixelate_factor}: expected "
                f"<={max_expected} distinct colours in top row, got "
                f"{unique_per_row}"))

    return issues


def run_one(name: str, params: dict, src: Image.Image,
            assess_kwargs: dict | None = None) -> list[Issue]:
    assess_kwargs = assess_kwargs or {}
    try:
        with tempfile.TemporaryDirectory(prefix="atest_") as t:
            t = Path(t)
            in_path = t / "in.png"
            out_path = t / "out.png"
            src.save(in_path)
            p.process_image(in_path, out_path, params, work_dir=t)
            out = Image.open(out_path).convert("RGBA").copy()
        return assess(name, src, out, **assess_kwargs)
    except Exception:
        return [Issue(name, "EXCEPTION:\n" +
                      "        ".join(traceback.format_exc().splitlines(True)))]


def main():
    if not TEST_IMAGE.exists():
        print(f"Test image not found: {TEST_IMAGE}")
        sys.exit(1)

    src = Image.open(TEST_IMAGE).convert("RGBA")
    src_a = np.asarray(src.getchannel("A"))
    print(f"Test image: {TEST_IMAGE.name}  size {src.size}")
    print(f"  alpha range: {src_a.min()}..{src_a.max()}, "
          f"distinct: {len(np.unique(src_a))}")
    print()

    all_issues: list[Issue] = []

    # -- 1. Every shipped preset --
    print("=== Running every shipped preset ===")
    for preset_path in sorted(Path("presets").glob("*.json")):
        data = json.loads(preset_path.read_text(encoding="utf-8"))
        params = data["params"]
        name = preset_path.stem

        kwargs = {}
        if params.get("binary_output"):
            kwargs["expect_binary"] = True
        # Preset 01 (pass-through) preserves the original alpha exactly.
        if name in {"01 Default (pass-through)"}:
            kwargs["expect_alpha_unchanged_from"] = src

        issues = run_one(name, params, src, kwargs)
        if issues:
            all_issues.extend(issues)
            for i in issues:
                print(i)
        else:
            print(f"  PASS  {name}")
    print()

    # -- 2. Parameter sweeps --
    print("=== Parameter sweeps (orthogonal to presets) ===")
    base = dict(p.DEFAULT_PARAMS)

    sweep_cases = []
    # Alpha source x snap on/off x binary on/off
    for asrc, snap, binary in product(["fix", "create"], [False, True],
                                       [False, True]):
        params = dict(base)
        params["alpha_source"] = asrc
        params["snap_enabled"] = snap
        params["binary_output"] = binary
        if binary:
            params["binary_threshold"] = 64
        sweep_cases.append((f"sweep alpha={asrc} snap={snap} bin={binary}",
                            params, {"expect_binary": binary}))

    # Tone adjustment extremes
    for sat, con, bri in [(0.0, 1.0, 1.0), (2.0, 1.0, 1.0),
                          (1.0, 0.0, 1.0), (1.0, 2.0, 1.0),
                          (1.0, 1.0, 0.0), (1.0, 1.0, 2.0)]:
        params = dict(base)
        params["saturation"] = sat
        params["contrast"] = con
        params["brightness"] = bri
        sweep_cases.append((f"sweep sat={sat} con={con} bri={bri}",
                            params, {}))

    # Pixelate factors
    for pf in [1, 2, 4, 8, 16]:
        params = dict(base)
        params["pixelate_factor"] = pf
        sweep_cases.append((f"sweep pixelate={pf}", params,
                            {"expect_pixelate_factor": pf}))

    # Edge dilation extremes
    for ed in [0, 1, 4, 8, 16]:
        params = dict(base)
        params["edge_dilation"] = ed
        sweep_cases.append((f"sweep edge_dilation={ed}", params, {}))

    # Snap level counts
    for n in [2, 3, 4, 5, 8]:
        params = dict(base)
        params["snap_enabled"] = True
        params["snap_levels"] = n
        sweep_cases.append((f"sweep snap_levels={n}", params, {}))

    # Fix-path level extremes
    for low, high in [(0, 100), (10, 90), (40, 60), (45, 45), (50, 50)]:
        params = dict(base)
        params["fix_level_low"] = low
        params["fix_level_high"] = high
        sweep_cases.append((f"sweep level={low}-{high}", params, {}))

    # Create-path with each supported target fill
    for fill in ["gray", "white"]:
        params = dict(base)
        params["alpha_source"] = "create"
        params["create_target"] = fill
        sweep_cases.append((f"sweep create_target={fill}", params, {}))

    for name, params, kwargs in sweep_cases:
        issues = run_one(name, params, src, kwargs)
        if issues:
            all_issues.extend(issues)
            for i in issues:
                print(i)
        else:
            print(f"  PASS  {name}")
    print()

    # -- 3. Vtracer combinations (slower; one of each) --
    print("=== Vtracer combinations ===")
    vtracer_cases = []
    for mode in ["pixel", "polygon", "spline"]:
        for hier in ["stacked", "cutout"]:
            for asrc in ["fix", "create"]:
                params = dict(base)
                params["vtracer_enabled"] = True
                params["vtracer_pre_quantize"] = 0
                params["vtracer_mode"] = mode
                params["vtracer_hierarchical"] = hier
                params["alpha_source"] = asrc
                vtracer_cases.append((
                    f"vtracer mode={mode} hier={hier} alpha={asrc}",
                    params, {}))

    for name, params, kwargs in vtracer_cases:
        issues = run_one(name, params, src, kwargs)
        if issues:
            all_issues.extend(issues)
            for i in issues:
                print(i)
        else:
            print(f"  PASS  {name}")
    print()

    # -- Summary --
    print("=" * 60)
    if all_issues:
        print(f"FOUND {len(all_issues)} ISSUE(S):")
        for i in all_issues:
            print(i)
    else:
        print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
