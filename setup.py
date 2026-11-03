r"""
cx_Freeze build script for Alpha Tools.

Build commands:
    pip install cx_Freeze
    python setup.py build           # freeze to .\build\exe.win-...\
    python setup.py bdist_msi       # build MSI installer in .\dist\

Notes:
    * ImageMagick is bundled from C:\Scripts\Dependencies\Imagemagick, with
      unused delegates / docs / sample images stripped to keep the MSI small.
    * vtracer.exe is bundled alongside the app exe (sourced from Vtracer Tools
      project so we share one binary).
    * Default presets ship under presets/ and are loaded at startup.
    * upgrade_code is a fixed GUID for this product — keep constant across
      versions. Different from Image Tools / Vtracer Tools GUIDs so the MSIs
      don't collide.
"""

import os
import re
import sys
import tempfile
from pathlib import Path
from cx_Freeze import setup, Executable


def collect_files(src_root, dst_prefix, exclude_prefixes=()):
    """Return [(abs_src_file, rel_dst_file), ...] for every file under
    src_root, excluding any file whose POSIX-relative path starts with one of
    exclude_prefixes."""
    src_root = Path(src_root)
    out = []
    for path in src_root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(src_root).as_posix()
        if any(rel.startswith(p) for p in exclude_prefixes):
            continue
        out.append((str(path), f"{dst_prefix}/{rel}"))
    return out


def patch_imagemagick_delegates(src_xml: Path) -> Path:
    """Disable ImageMagick's external rsvg-convert delegate so SVG decoding
    falls back to the bundled CORE_RL_librsvg_.dll instead of shelling out
    to an rsvg-convert.exe that user machines don't have. Without this patch
    every install that lacks rsvg-convert on PATH will fail vectorisation.

    Returns the path to a temp file containing the patched XML, suitable for
    cx_Freeze's include_files.
    """
    text = src_xml.read_text(encoding="utf-8")
    patched = re.sub(
        r'<delegate\s+decode="svg"[^>]*rsvg-convert[^>]*/>',
        '<!-- svg delegate disabled by Alpha Tools build: '
        'use bundled CORE_RL_librsvg_.dll -->',
        text,
    )
    if patched == text:
        # Source file didn't have the offending line — nothing to do, but
        # warn so it's obvious if upstream ImageMagick ever changes the
        # delegate name and our regex misses it.
        sys.stderr.write(
            "WARNING: delegates.xml had no rsvg-convert delegate to patch — "
            "verify the regex still matches the source file.\n"
        )
    tmp = Path(tempfile.gettempdir()) / "alphatools_delegates.xml"
    tmp.write_text(patched, encoding="utf-8")
    return tmp


APP_NAME = "Alpha Tools"
DIST_NAME = "Alpha-Tools"
APP_VERSION = "1.2.0"
# Version history (most recent first):
#   1.2.0 — preview viewport rendering + ROI full-res detail; perceptual
#           OKLCh tone stage; manual pixel edits (pencil / eraser / picker /
#           granulate / smudge, sidecars, batch "Include manual edits");
#           "Protect vivid colours" for colour-depth reduction; soften-edge
#           dither; pixelate "Majority vote" block colour; Batch pixel-exact
#           power-of-two output scale. Fixes: grid overlay on large images,
#           preset save with live edits, quantise transparent base colour.
#   1.1.0 — promoted the former "Code-Review" line to be the main version
#           (the 1.0.x source was retired/archived). Headline addition:
#           incongruence-based alpha edge refinement that removes dark
#           AI-upscale halos at the transparency boundary by colour
#           incongruence (not luminance), with a px reach cap. Supersedes the
#           released 1.0.2.
#   1.0.1 — set MAGICK_HOME/MAGICK_CODER_MODULE_PATH/etc at startup so the
#           bundled magick.exe finds its codec DLLs on machines without
#           ImageMagick installed (was failing with 'RegistryKeyLookupFailed
#           CoderModulesPath' / 'no decode delegate for SVG').
#   1.0.0 — initial release.
ICON_PATH = r"C:\Scripts\ESRGAN-Tools\app.ico"
VTRACER_PATH = r"C:\Scripts\Vtracer-Tools\vtracer.exe"
IMAGEMAGICK_DIR = r"C:\Scripts\Dependencies\Imagemagick"
PRESETS_DIR = "presets"

# Fresh stable GUID for Alpha Tools. Do NOT change between versions.
UPGRADE_CODE = "{7B8E4F6A-2D3C-4B9A-A1F5-9E3D8C7B6A4E}"

# --- ImageMagick strip list ------------------------------------------------
# Keep: magick.exe, CORE_RL_*.dll (libpng/libjpeg/librsvg/etc), modules/
#       (codec DLLs), VC++ runtime DLLs, sRGB.icc, *.xml configs.
# Drop: ffmpeg.exe, dcraw.exe, hp2xx.exe, imdisplay.exe, uninstaller, sample
#       images, HTML docs, license/readme files.
IMAGEMAGICK_EXCLUDES = (
    "ffmpeg.exe",
    "dcraw.exe",
    "hp2xx.exe",
    "imdisplay.exe",
    "unins000.exe",
    "unins000.dat",
    "uninstall",
    "www",
    "images",
    "ChangeLog.txt",
    "License.txt",
    "NOTICE.txt",
    "QuickStart.txt",
    "README.txt",
    "ImageMagick.ico",
    "ImageMagick.rdf",
    "index.html",
    # Skip the upstream delegates.xml — we ship a patched version that
    # disables the external rsvg-convert delegate (see patch_imagemagick_delegates).
    "delegates.xml",
)

patched_delegates = patch_imagemagick_delegates(
    Path(IMAGEMAGICK_DIR) / "delegates.xml"
)

include_files = [
    (VTRACER_PATH, "vtracer.exe"),
    (ICON_PATH, "app.ico"),
    (str(patched_delegates), "vendor/imagemagick/delegates.xml"),
]
include_files += collect_files(IMAGEMAGICK_DIR, "vendor/imagemagick", IMAGEMAGICK_EXCLUDES)
include_files += collect_files(PRESETS_DIR, "presets")

build_exe_options = {
    "packages": [
        "customtkinter",
        "PIL",
        "numpy",
        "tkinter",
        "imagequant",   # libimagequant binding for high-quality palette reduction
        "cffi",         # imagequant transitive dep (C foreign function interface)
    ],
    "excludes": ["test", "unittest", "pydoc_data"],
    "include_files": include_files,
    "include_msvcr": True,
    "optimize": 1,
}

bdist_msi_options = {
    "upgrade_code": UPGRADE_CODE,
    "add_to_path": False,
    "initial_target_dir": r"[ProgramFiles64Folder]\Alpha Tools",
    "install_icon": ICON_PATH,
    "summary_data": {
        "author": "Alpha Tools",
        "comments": "Alpha-channel cleanup utility for upscaled game textures",
    },
    "all_users": True,
}

executables = [
    Executable(
        script="main.py",
        base="gui",
        target_name="AlphaTools.exe",
        icon=ICON_PATH,
        shortcut_name=APP_NAME,
        shortcut_dir="ProgramMenuFolder",
        copyright="Alpha Tools",
    )
]

setup(
    name=DIST_NAME,
    version=APP_VERSION,
    description="Alpha-channel cleanup utility for upscaled game textures",
    options={
        "build_exe": build_exe_options,
        "bdist_msi": bdist_msi_options,
    },
    executables=executables,
)
