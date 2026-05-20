#!/usr/bin/env python3
"""
CR3 Bayer Data Extractor  (with calibration pipeline)
======================================================
Extracts raw Bayer sensor data from Canon CR3 files before demosaicing,
producing a true monochrome image from the actual sensor readout.

Calibration frames are all optional.  The script applies whichever it
receives and falls back gracefully when any are absent, printing a
summary of what was and was not applied.

Calibration pipeline (standard CCD/CMOS reduction order):
    Calibrated = (Light - MasterBias - MasterDark) / NormalisedMasterFlat

  where MasterFlat was itself cleaned:
    MasterFlat_clean = (RawFlat - MasterBias - MasterDarkFlat) / mean(...)

Each "master" can be supplied as:
  * A single pre-stacked master frame  (e.g. master_dark.CR3 / .tiff / .npy)
  * A directory of individual frames   (median-stacked automatically)

Usage examples:
    # Bare minimum - no calibration
    python cr3_bayer_extract.py light.CR3

    # With a master dark only
    python cr3_bayer_extract.py light.CR3 --dark master_dark.CR3

    # With a directory of individual flat frames and a master dark
    python cr3_bayer_extract.py light.CR3 --dark master_dark.CR3 --flat ./flats/

    # Full pipeline
    python cr3_bayer_extract.py light.CR3 \\
        --bias  ./bias/          \\
        --dark  master_dark.CR3  \\
        --flat  ./flats/         \\
        --dark-flat master_darkflat.CR3

    # Inspect file metadata only
    python cr3_bayer_extract.py light.CR3 --show-info
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import rawpy
import tifffile
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

RAW_EXTENSIONS = {".cr3", ".cr2", ".nef", ".arw", ".orf", ".raf", ".dng", ".rw2"}
CALIBRATION_REPORT: list = []   # populated during run, printed at end


# ─────────────────────────────────────────────────────────────────────────────
#  RAW I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def open_raw(path: Path) -> rawpy.RawPy:
    raw = rawpy.imread(str(path))
    return raw


def load_raw_bayer(path: Path) -> np.ndarray:
    """Return the visible Bayer array from a RAW file as float32."""
    with rawpy.imread(str(path)) as raw:
        raw.unpack()
        return raw.raw_image_visible.astype(np.float32)


def load_frame(path: Path) -> np.ndarray:
    """
    Load a single calibration frame from a RAW file, TIFF, or .npy array.
    Always returns float32.
    """
    suffix = path.suffix.lower()
    if suffix in RAW_EXTENSIONS:
        return load_raw_bayer(path)
    elif suffix == ".npy":
        return np.load(str(path)).astype(np.float32)
    elif suffix in (".tif", ".tiff"):
        img = Image.open(str(path))
        return np.array(img, dtype=np.float32)
    elif suffix == ".png":
        img = Image.open(str(path)).convert("I")
        return np.array(img, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported calibration frame format: {path.suffix!r}")


def collect_frames(source: Path) -> list:
    """
    Return a list of frame paths from either a single file or a directory.
    Directories are searched for all supported RAW/image types.
    """
    if source.is_file():
        return [source]
    elif source.is_dir():
        frames = []
        for ext in list(RAW_EXTENSIONS) + [".tif", ".tiff", ".npy", ".png"]:
            frames.extend(source.glob(f"*{ext}"))
            frames.extend(source.glob(f"*{ext.upper()}"))
        frames = sorted(set(frames))
        if not frames:
            raise FileNotFoundError(
                f"No supported frames found in directory: {source}"
            )
        return frames
    else:
        raise FileNotFoundError(f"Calibration source not found: {source}")


# ─────────────────────────────────────────────────────────────────────────────
#  Master frame stacking
# ─────────────────────────────────────────────────────────────────────────────

def build_master(source: Path, label: str) -> np.ndarray:
    """
    Load one or more frames and median-stack them into a master calibration frame.
    A single pre-stacked file is returned as-is.
    """
    paths = collect_frames(source)

    if len(paths) == 1:
        print(f"  {label:12s}: {paths[0].name}  (single master)")
        CALIBRATION_REPORT.append(f"  {label}: {paths[0].name} (single frame, no stacking)")
        return load_frame(paths[0])

    print(f"  {label:12s}: median-stacking {len(paths)} frames ...", end="", flush=True)
    stack = np.stack([load_frame(p) for p in paths], axis=0)
    master = np.median(stack, axis=0).astype(np.float32)
    print(" done")
    CALIBRATION_REPORT.append(
        f"  {label}: median stack of {len(paths)} frames from '{source}'"
    )
    return master


# ─────────────────────────────────────────────────────────────────────────────
#  Calibration pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_calibration_masters(args) -> dict:
    """
    Load and stack whichever calibration sources were provided.
    Returns a dict with keys: bias, dark, flat, dark_flat  (values: array or None).
    """
    masters = {"bias": None, "dark": None, "flat": None, "dark_flat": None}

    if args.bias:
        masters["bias"] = build_master(args.bias, "Bias")
    if args.dark:
        masters["dark"] = build_master(args.dark, "Dark")
    if args.flat:
        masters["flat"] = build_master(args.flat, "Flat")
    if args.dark_flat:
        masters["dark_flat"] = build_master(args.dark_flat, "Dark-flat")

    return masters


def _check_shape(cal: np.ndarray, data: np.ndarray, name: str) -> None:
    if cal.shape != data.shape:
        raise ValueError(
            f"{name} frame shape {cal.shape} does not match "
            f"light frame shape {data.shape}. "
            "Calibration frames must come from the same camera body."
        )


def _normalise_flat_per_channel(flat: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    """
    Normalise flat so each colour channel has a mean response of 1.0.
    This prevents any residual colour-channel brightness difference from
    introducing a bias into the monochrome output.
    """
    norm = flat.copy()
    for row in range(2):
        for col in range(2):
            plane = flat[row::2, col::2]
            valid = plane[plane > 0]
            mean_val = valid.mean() if valid.size > 0 else 1.0
            norm[row::2, col::2] = plane / mean_val if mean_val > 0 else plane
    return norm


def apply_calibration(
    bayer: np.ndarray,
    masters: dict,
    pattern: np.ndarray,
    white_level: float,
    black_levels: np.ndarray,
) -> np.ndarray:
    """
    Apply bias, dark, and flat-field calibration to a raw Bayer array.

    Steps are applied only when the relevant master is available:
      1. Camera black-level subtraction (always — built from EXIF metadata)
      2. Bias subtraction
      3. Dark subtraction  (bias is also removed from the dark before use)
      4. Flat-field division  (flat is cleaned with bias + dark-flat first)

    Returns float32, still in Bayer layout, clipped to [0, effective_white].
    """
    h, w = bayer.shape
    data = bayer.copy()

    # ── Step 1: Per-channel black level (always applied from EXIF) ────────
    bl_map = np.zeros((h, w), dtype=np.float32)
    for row in range(2):
        for col in range(2):
            ch = pattern[row, col]
            bl_map[row::2, col::2] = black_levels[ch]
    data -= bl_map
    CALIBRATION_REPORT.append("  [1] Camera black-level subtracted (from EXIF metadata)")

    steps_skipped = []

    # ── Step 2: Bias subtraction ──────────────────────────────────────────
    bias = masters.get("bias")
    if bias is not None:
        _check_shape(bias, data, "Bias")
        data -= bias
        CALIBRATION_REPORT.append("  [2] Bias subtracted")
    else:
        steps_skipped.append("bias (not provided)")

    # ── Step 3: Dark subtraction ──────────────────────────────────────────
    dark = masters.get("dark")
    if dark is not None:
        _check_shape(dark, data, "Dark")
        dark_corrected = dark.copy()
        if bias is not None:
            dark_corrected -= bias      # remove bias from dark before applying
        data -= dark_corrected
        CALIBRATION_REPORT.append("  [3] Dark subtracted" +
                                   (" (bias-corrected dark)" if bias is not None else ""))
    else:
        steps_skipped.append("dark (not provided)")

    # ── Step 4: Flat-field division ───────────────────────────────────────
    flat_raw = masters.get("flat")
    if flat_raw is not None:
        _check_shape(flat_raw, data, "Flat")
        flat = flat_raw.copy()

        # Clean the flat frame
        if bias is not None:
            flat -= bias
        dark_flat = masters.get("dark_flat")
        if dark_flat is not None:
            _check_shape(dark_flat, data, "Dark-flat")
            df = dark_flat.copy()
            if bias is not None:
                df -= bias
            flat -= df

        flat_norm = _normalise_flat_per_channel(flat, pattern)
        # Protect against dead pixels / underexposed flat regions
        flat_norm = np.where(flat_norm > 0.05, flat_norm, 1.0)
        data /= flat_norm
        CALIBRATION_REPORT.append(
            "  [4] Flat-field divided" +
            (" + dark-flat corrected" if dark_flat is not None else "")
        )
    else:
        steps_skipped.append("flat-field (not provided)")

    if steps_skipped:
        CALIBRATION_REPORT.append(
            "  Skipped: " + ", ".join(steps_skipped)
        )

    effective_white = white_level - float(bl_map.mean())
    return np.clip(data, 0, effective_white)


# ─────────────────────────────────────────────────────────────────────────────
#  Bayer -> monochrome extraction
# ─────────────────────────────────────────────────────────────────────────────

def get_pattern_str(raw: rawpy.RawPy) -> str:
    pattern = raw.raw_pattern.flatten()
    desc = raw.color_desc.decode()
    return "".join(desc[c] for c in pattern)


def _extract_green_planes(bayer: np.ndarray, pattern: np.ndarray):
    """Return (G1_half, G2_half) sub-arrays by pattern position."""
    g1 = g2 = None
    for row in range(2):
        for col in range(2):
            ch = pattern[row, col]
            if ch == 1 and g1 is None:
                g1 = bayer[row::2, col::2]
            elif ch == 3 and g2 is None:
                g2 = bayer[row::2, col::2]
    if g1 is None:
        g1 = g2
    if g2 is None:
        g2 = g1
    return g1, g2


def _upscale_half_to_full(half: np.ndarray, full_h: int, full_w: int) -> np.ndarray:
    """Nearest-neighbour upscale from half-res Bayer plane to full sensor size."""
    return np.repeat(np.repeat(half, 2, axis=0), 2, axis=1)[:full_h, :full_w]


def _channel_index(raw: rawpy.RawPy, target: str) -> int:
    mapping = {"R": 0, "G": 1, "G1": 1, "B": 2, "G2": 3}
    if target not in mapping:
        raise ValueError(f"Unknown channel {target!r}")
    return mapping[target]


def apply_gamma(arr: np.ndarray, gamma: float, max_val: float) -> np.ndarray:
    """Apply gamma encoding. gamma=2.2 is standard monitor gamma."""
    if gamma == 1.0:
        return arr
    normed = np.clip(arr / max_val, 0.0, 1.0)
    return np.power(normed, 1.0 / gamma) * max_val


FILM_CURVES = {
    "negative":   dict(black_lift=0.04, gamma=0.60, shoulder_start=0.75, shoulder_str=0.7,  display_gamma=2.2),
    "negative+":  dict(black_lift=0.03, gamma=0.70, shoulder_start=0.80, shoulder_str=0.5,  display_gamma=2.2),
    "reversal":   dict(black_lift=0.01, gamma=1.90, shoulder_start=0.85, shoulder_str=0.4,  display_gamma=1.0),
    "aerochrome": dict(black_lift=0.01, gamma=1.85, shoulder_start=0.82, shoulder_str=0.45, display_gamma=1.0),
    "hp5":        dict(black_lift=0.05, gamma=0.62, shoulder_start=0.72, shoulder_str=0.8,  display_gamma=2.2),
    "velvia":     dict(black_lift=0.00, gamma=2.10, shoulder_start=0.80, shoulder_str=0.3,  display_gamma=1.0),
    "portra":     dict(black_lift=0.05, gamma=0.58, shoulder_start=0.78, shoulder_str=0.9,  display_gamma=2.2),
}


def apply_film_curve(arr: np.ndarray, curve_name: str, max_val: float) -> np.ndarray:
    """Apply a film-emulation S-curve with toe lift, shoulder roll-off,
    and a display gamma pass to bring negative stocks to correct screen brightness."""
    c = FILM_CURVES.get(curve_name)
    if c is None:
        raise ValueError(f"Unknown film curve: {curve_name!r}. "
                         f"Options: {list(FILM_CURVES)}")
    x = np.clip(arr / max_val, 0.0, 1.0).astype(np.float32)
    y = np.power(x, 1.0 / c["gamma"])
    ss, sw = c["shoulder_start"], c["shoulder_str"]
    if sw > 0:
        mask = x > ss
        if np.any(mask):
            t = np.clip((x[mask] - ss) / (1.0 - ss), 0.0, 1.0)
            cosine_shoulder = 1.0 - (np.cos(t * np.pi) + 1.0) / 2.0
            straight = y[mask]
            rounded  = ss ** (1.0 / c["gamma"]) + \
                       (1.0 - ss ** (1.0 / c["gamma"])) * cosine_shoulder
            y[mask]  = straight * (1.0 - sw) + rounded * sw
    bl = c["black_lift"]
    y = y * (1.0 - bl) + bl
    dg = c.get("display_gamma", 1.0)
    if dg != 1.0:
        y = np.power(np.clip(y, 0.0, 1.0), 1.0 / dg)
    return np.clip(y * max_val, 0.0, max_val)


def _apply_tone(arr: np.ndarray, gamma: float, curve: str, max_val: float) -> np.ndarray:
    if curve:
        return apply_film_curve(arr, curve, max_val)
    return apply_gamma(arr, gamma, max_val)


def extract_bayer_monochrome(
    calibrated_bayer: np.ndarray,
    raw: rawpy.RawPy,
    mode: str = "green-avg",
    normalize: bool = True,
    bit_depth: int = 16,
    gamma: float = 2.2,
    curve: str = "",
    black_clip: float = 0.1,
    white_clip: float = 0.1,
) -> np.ndarray:
    """
    Convert a calibrated float32 Bayer array to a 2-D monochrome image.

    mode:
      'green-avg'       Average both green channels — closest to luminance
      'average'         Average all four RGGB channels
      'R'/'G'/'G1'/'G2'/'B'   Single colour plane, upscaled to full size
    """
    pattern = raw.raw_pattern
    h, w = calibrated_bayer.shape
    dynamic_range = float(raw.white_level) - np.array(
        raw.black_level_per_channel, dtype=np.float32
    ).mean()

    if mode == "green-avg":
        g1, g2 = _extract_green_planes(calibrated_bayer, pattern)
        # Trim to common shape — odd sensor dimensions can cause a 1-px difference
        gh = min(g1.shape[0], g2.shape[0])
        gw = min(g1.shape[1], g2.shape[1])
        mono = _upscale_half_to_full((g1[:gh, :gw] + g2[:gh, :gw]) / 2.0, h, w)

    elif mode == "average":
        planes = [calibrated_bayer[r::2, c::2] for r in range(2) for c in range(2)]
        min_h = min(pl.shape[0] for pl in planes)
        min_w = min(pl.shape[1] for pl in planes)
        planes = [pl[:min_h, :min_w] for pl in planes]
        mono = _upscale_half_to_full(np.mean(planes, axis=0), h, w)

    else:
        target = mode.upper()
        ch_idx = _channel_index(raw, target)
        plane = None
        for row in range(2):
            for col in range(2):
                if pattern[row, col] == ch_idx:
                    plane = calibrated_bayer[row::2, col::2]
                    break
            if plane is not None:
                break
        if plane is None:
            raise ValueError(
                f"Channel {target!r} not found in pattern {get_pattern_str(raw)}"
            )
        mono = _upscale_half_to_full(plane, h, w)

    dtype = np.uint16 if bit_depth == 16 else np.uint8
    max_val = (2 ** bit_depth) - 1

    if normalize:
        lo = np.percentile(mono, black_clip)
        hi = np.percentile(mono, 100.0 - white_clip)
        mono = np.clip((mono - lo) / (hi - lo) * max_val, 0, max_val) \
               if hi > lo else np.zeros_like(mono)
    else:
        mono = mono / dynamic_range * max_val

    mono = _apply_tone(mono, gamma, curve, max_val)
    return np.clip(mono, 0, max_val).astype(dtype)


# ─────────────────────────────────────────────────────────────────────────────
#  Aerochrome composite
# ─────────────────────────────────────────────────────────────────────────────

def extract_aerochrome(
    calibrated_bayer: np.ndarray,
    raw: rawpy.RawPy,
    normalize: bool = True,
    bit_depth: int = 16,
    gamma: float = 2.2,
    curve: str = "",
    black_clip: float = 0.1,
    white_clip: float = 0.1,
) -> np.ndarray:
    """
    Build a false-colour Aerochrome IR film simulation.

    Classic Kodak Aerochrome channel mapping:
      Output R  ←  Sensor red   (IR proxy — long-wavelength tail)
      Output G  ←  Sensor red   (visible red shifted to green slot)
      Output B  ←  Sensor green (visible green shifted to blue slot)
      Sensor blue is dropped (as in the original film's layer design)

    Each channel is independently normalised so exposure differences
    between channels don't collapse the colour rendering.

    Returns (H, W, 3) uint8 or uint16 in RGB order.
    """
    pattern = raw.raw_pattern
    h, w = calibrated_bayer.shape
    dtype = np.uint16 if bit_depth == 16 else np.uint8
    max_val = float((2 ** bit_depth) - 1)
    dynamic_range = float(raw.white_level) - np.array(
        raw.black_level_per_channel, dtype=np.float32
    ).mean()

    def _extract_plane(ch_idx):
        for row in range(2):
            for col in range(2):
                if pattern[row, col] == ch_idx:
                    plane = calibrated_bayer[row::2, col::2]
                    return _upscale_half_to_full(plane, h, w)
        return None

    def _norm(plane):
        lo = np.percentile(plane, black_clip)
        hi = np.percentile(plane, 100.0 - white_clip)
        return np.clip((plane - lo) / (hi - lo) * max_val, 0, max_val) \
               if hi > lo else np.zeros_like(plane)

    red_plane   = _extract_plane(0)
    green_plane = _extract_plane(1)

    if red_plane is None or green_plane is None:
        raise ValueError("Could not locate R and G channels in Bayer pattern.")

    if normalize:
        r_out = _norm(red_plane)
        g_out = _norm(red_plane)
        b_out = _norm(green_plane)
    else:
        r_out = red_plane   / dynamic_range * max_val
        g_out = red_plane   / dynamic_range * max_val
        b_out = green_plane / dynamic_range * max_val

    # Trim to common shape — upscaling odd-dimension sensors produces
    # a 1-pixel discrepancy between planes at different row/col offsets
    out_h = min(r_out.shape[0], b_out.shape[0])
    out_w = min(r_out.shape[1], b_out.shape[1])

    r_out = _apply_tone(r_out[:out_h, :out_w], gamma, curve, max_val)
    g_out = _apply_tone(g_out[:out_h, :out_w], gamma, curve, max_val)
    b_out = _apply_tone(b_out[:out_h, :out_w], gamma, curve, max_val)

    rgb = np.stack([
        np.clip(r_out, 0, max_val).astype(dtype),
        np.clip(g_out, 0, max_val).astype(dtype),
        np.clip(b_out, 0, max_val).astype(dtype),
    ], axis=2)

    return rgb


# ─────────────────────────────────────────────────────────────────────────────
#  Output
# ─────────────────────────────────────────────────────────────────────────────

def save_output(arr: np.ndarray, out_path: Path, fmt: str, **kwargs) -> None:
    """Save a monochrome (2-D) or colour (H×W×3) array."""
    fmt = fmt.lower()
    is_colour = arr.ndim == 3

    if fmt == "npy":
        p = out_path.with_suffix(".npy")
        np.save(str(p), arr)
        print(f"  Saved numpy array  -> {p}")

    elif fmt in ("tiff", "tif"):
        if is_colour:
            tifffile.imwrite(str(out_path), arr,
                             photometric="rgb", compression="deflate")
            print(f"  Saved colour TIFF  -> {out_path}")
        else:
            tifffile.imwrite(str(out_path), arr,
                             photometric="minisblack", compression="deflate")
            print(f"  Saved 16-bit TIFF  -> {out_path}")

    elif fmt == "png":
        if is_colour:
            if arr.dtype == np.uint16:
                Image.fromarray((arr >> 8).astype(np.uint8), mode="RGB").save(
                    str(out_path), format="PNG")
            else:
                Image.fromarray(arr, mode="RGB").save(str(out_path), format="PNG")
        else:
            if arr.dtype == np.uint16:
                tifffile.imwrite(str(out_path), arr)
            else:
                Image.fromarray(arr, mode="L").save(str(out_path), format="PNG")
        print(f"  Saved PNG          -> {out_path}")

    elif fmt in ("jpg", "jpeg"):
        quality = kwargs.get("jpeg_quality", 95)
        if is_colour:
            arr8 = (arr >> 8).astype(np.uint8) if arr.dtype == np.uint16 else arr
            Image.fromarray(arr8, mode="RGB").save(
                str(out_path), format="JPEG", quality=quality, subsampling=0)
        else:
            arr8 = (arr >> 8).astype(np.uint8) if arr.dtype == np.uint16 else arr
            Image.fromarray(arr8, mode="L").save(
                str(out_path), format="JPEG", quality=quality)
        print(f"  Saved JPEG (q={quality})  -> {out_path}")

    else:
        raise ValueError(f"Unknown output format {fmt!r}. Use tiff, png, jpg, or npy.")


def print_info(raw: rawpy.RawPy, path: Path) -> None:
    sizes = raw.sizes
    other = raw.other
    bl = raw.black_level_per_channel
    print(f"\n{'─'*54}")
    print(f"  File          : {path.name}")
    print(f"  Raw size      : {sizes.raw_width} x {sizes.raw_height} px")
    print(f"  Visible size  : {sizes.width} x {sizes.height} px")
    print(f"  Bayer pattern : {raw.color_desc.decode()} / {get_pattern_str(raw)}")
    print(f"  White level   : {raw.white_level}")
    print(f"  Black levels  : R={bl[0]}  G1={bl[1]}  B={bl[2]}  G2={bl[3]}")
    print(f"  ISO           : {other.iso_speed}")
    if other.shutter > 0:
        if other.shutter < 1:
            print(f"  Shutter       : 1/{1/other.shutter:.0f}s")
        else:
            print(f"  Shutter       : {other.shutter}s")
    print(f"  Aperture      : f/{other.aperture}")
    print(f"  Focal length  : {other.focal_len} mm")
    print(f"  Camera WB     : {[round(v, 4) for v in raw.camera_whitebalance]}")
    print(f"{'─'*54}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Extract raw Bayer monochrome image from Canon CR3 files,\n"
            "with an optional bias / dark / flat calibration pipeline."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument("input", type=Path,
                   help="Input CR3 (or any LibRaw-supported RAW) light frame")

    cal = p.add_argument_group(
        "calibration frames",
        "All optional — omit any you do not have. "
        "Each accepts either a single master file or a directory of frames "
        "(which will be median-stacked automatically)."
    )
    cal.add_argument("--bias", type=Path, metavar="FILE_OR_DIR",
                     help="Bias master or directory of bias frames")
    cal.add_argument("--dark", type=Path, metavar="FILE_OR_DIR",
                     help="Dark master or dir  (same ISO & shutter as light)")
    cal.add_argument("--flat", type=Path, metavar="FILE_OR_DIR",
                     help="Flat master or dir  (uniform illumination frames)")
    cal.add_argument("--dark-flat", type=Path, metavar="FILE_OR_DIR",
                     dest="dark_flat",
                     help="Dark-flat master or dir  (same shutter speed as flats)")

    out = p.add_argument_group("output")
    out.add_argument("-o", "--output", type=Path, default=None,
                     help="Output file path (default: <input>_bayer.<ext>)")
    out.add_argument("-f", "--format", default="tiff",
                     choices=["tiff", "tif", "png", "jpg", "jpeg", "npy"],
                     help="Output format (default: tiff)")
    out.add_argument("--jpeg-quality", type=int, default=95, metavar="1-95",
                     help="JPEG quality 1–95 (default: 95, only used with --format jpg)")
    out.add_argument("--bit-depth", type=int, default=16, choices=[8, 16],
                     help="Output bit depth — 8 or 16 (default: 16)")

    ext = p.add_argument_group("extraction")
    ext.add_argument(
        "--mode", default="green-avg",
        choices=["green-avg", "average", "aerochrome", "R", "G", "G1", "G2", "B"],
        help=(
            "Extraction mode (default: green-avg):\n"
            "  green-avg   Average both green channels (best luminance proxy)\n"
            "  average     Average all four RGGB channels\n"
            "  aerochrome  False-colour IR film simulation (colour TIFF/PNG output)\n"
            "  R/G/B/G2    Extract a single Bayer colour plane"
        )
    )
    ext.add_argument("--normalize", action="store_true", default=True,
                     help="Stretch output to full bit-depth range (default: on)")
    ext.add_argument("--no-normalize", dest="normalize", action="store_false",
                     help="Keep relative ADC counts (do not stretch)")
    ext.add_argument("--black-clip", type=float, default=0.1, metavar="PCT",
                     dest="black_clip",
                     help=(
                         "Percentile of pixels crushed to black before stretching "
                         "(default: 0.1). Eliminates noise floor that lifts shadows. "
                         "Raise to 0.5–1.0 if blacks still look grey."
                     ))
    ext.add_argument("--white-clip", type=float, default=0.1, metavar="PCT",
                     dest="white_clip",
                     help=(
                         "Percentile of pixels clipped to white before stretching "
                         "(default: 0.1). Prevents a single specular highlight from "
                         "compressing the entire tonal range."
                     ))
    ext.add_argument("--gamma", type=float, default=2.2, metavar="G",
                     help=(
                         "Gamma encoding to apply (default: 2.2). "
                         "Ignored if --curve is set."
                     ))
    ext.add_argument("--curve", default="", metavar="PRESET",
                     choices=list(FILM_CURVES.keys()) + [""],
                     help=(
                         "Film curve preset (overrides --gamma if set):\n"
                         "  negative   Generic negative (Portra/Ektar feel)\n"
                         "  negative+  Pushed negative (higher contrast)\n"
                         "  reversal   Slide film (Kodachrome feel)\n"
                         "  aerochrome Aerochrome IR reversal\n"
                         "  hp5        Ilford HP5/Delta mono negative\n"
                         "  velvia     Fujichrome Velvia (punchy slide)\n"
                         "  portra     Kodak Portra (smooth shadows)\n"
                     ))

    p.add_argument("--show-info", action="store_true",
                   help="Print RAW metadata and exit")
    p.add_argument("--pattern", action="store_true",
                   help="Print Bayer pattern string and exit")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # ── Open light frame ──────────────────────────────────────────────────
    print(f"\nOpening: {args.input.name}")
    raw = open_raw(args.input)
    raw.unpack()

    if args.pattern:
        print(f"  Bayer pattern: {get_pattern_str(raw)}")
        raw.close()
        return

    if args.show_info:
        print_info(raw, args.input)
        raw.close()
        return

    print_info(raw, args.input)

    # ── Load calibration masters (all optional) ───────────────────────────
    has_any_cal = any([args.bias, args.dark, args.flat, args.dark_flat])

    if has_any_cal:
        print("Loading calibration frames …")
        try:
            masters = build_calibration_masters(args)
        except (FileNotFoundError, ValueError) as exc:
            print(f"\nCalibration error: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        print("No calibration frames provided — running with camera black-level only.")
        CALIBRATION_REPORT.append("  No calibration frames supplied.")
        masters = {"bias": None, "dark": None, "flat": None, "dark_flat": None}

    # ── Apply calibration ─────────────────────────────────────────────────
    print("\nApplying calibration …")
    bayer_raw = raw.raw_image_visible.astype(np.float32)

    calibrated = apply_calibration(
        bayer=bayer_raw,
        masters=masters,
        pattern=raw.raw_pattern,
        white_level=float(raw.white_level),
        black_levels=np.array(raw.black_level_per_channel, dtype=np.float32),
    )

    # ── Extract ───────────────────────────────────────────────────────────
    if args.mode == "aerochrome":
        print(f"Extracting Aerochrome composite  "
              f"({args.bit_depth}-bit, normalize={args.normalize}, "
              f"tone={'curve:'+args.curve if args.curve else 'gamma:'+str(args.gamma)}, "
              f"clip={args.black_clip}/{args.white_clip}%) …")
        result = extract_aerochrome(
            calibrated_bayer=calibrated,
            raw=raw,
            normalize=args.normalize,
            bit_depth=args.bit_depth,
            gamma=args.gamma,
            curve=args.curve,
            black_clip=args.black_clip,
            white_clip=args.white_clip,
        )
        raw.close()
        print(f"  Shape         : {result.shape[1]} x {result.shape[0]} px  (RGB colour)")
    else:
        print(f"Extracting monochrome  "
              f"(mode={args.mode}, {args.bit_depth}-bit, normalize={args.normalize}, "
              f"tone={'curve:'+args.curve if args.curve else 'gamma:'+str(args.gamma)}, "
              f"clip={args.black_clip}/{args.white_clip}%) …")
        result = extract_bayer_monochrome(
            calibrated_bayer=calibrated,
            raw=raw,
            mode=args.mode,
            normalize=args.normalize,
            bit_depth=args.bit_depth,
            gamma=args.gamma,
            curve=args.curve,
            black_clip=args.black_clip,
            white_clip=args.white_clip,
        )
        raw.close()
        print(f"  Shape         : {result.shape[1]} x {result.shape[0]} px")
        print(f"  Value range   : {result.min()} – {result.max()}")

    # ── Save output ───────────────────────────────────────────────────────
    ext_map = {"tiff": ".tiff", "tif": ".tiff", "png": ".png",
               "jpg": ".jpg", "jpeg": ".jpg", "npy": ".npy"}
    suffix = "_aerochrome" if args.mode == "aerochrome" else "_bayer"
    if args.output is None:
        out_path = args.input.parent / f"{args.input.stem}{suffix}{ext_map[args.format]}"
    else:
        out_path = args.output

    save_output(result, out_path, args.format,
                jpeg_quality=args.jpeg_quality)

    # ── Calibration summary ───────────────────────────────────────────────
    print(f"\n{'─'*54}")
    print("Calibration report:")
    for line in CALIBRATION_REPORT:
        print(line)
    print(f"{'─'*54}")
    print("Done.\n")


if __name__ == "__main__":
    main()
