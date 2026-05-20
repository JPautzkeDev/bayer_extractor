#!/usr/bin/env python3
"""
Bayer Monochrome Extractor — PySide6 GUI
=========================================
A cross-platform desktop application (Windows / macOS / Linux) for
extracting true monochrome images from raw camera files (CR3, CR2, NEF,
ARW, RAF, DNG, …) before demosaicing, with an optional calibration
pipeline (bias / dark / flat / dark-flat).

Requirements:
    pip install PySide6 rawpy numpy pillow tifffile

Run:
    python bayer_extractor_gui.py
"""

import sys
import traceback
from pathlib import Path

import numpy as np
import rawpy
import tifffile
from PIL import Image

from PySide6.QtCore import (
    Qt, QThread, Signal, QObject, QRunnable, QThreadPool, Slot
)
from PySide6.QtGui import (
    QPixmap, QImage, QFont, QColor, QPalette, QIcon, QAction
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QComboBox, QCheckBox,
    QFileDialog, QTextEdit, QProgressBar, QGroupBox, QSizePolicy,
    QSplitter, QFrame, QScrollArea, QStatusBar, QToolBar, QMessageBox,
    QSpacerItem
)


# ─────────────────────────────────────────────────────────────────────────────
#  Core processing logic  (same algorithms as the CLI script)
# ─────────────────────────────────────────────────────────────────────────────

RAW_EXTENSIONS = {".cr3", ".cr2", ".nef", ".arw", ".orf", ".raf", ".dng", ".rw2"}


def get_pattern_str(raw: rawpy.RawPy) -> str:
    pattern = raw.raw_pattern.flatten()
    desc = raw.color_desc.decode()
    return "".join(desc[c] for c in pattern)


def is_xtrans(raw: rawpy.RawPy) -> bool:
    """Return True if the sensor uses a 6×6 Fujifilm X-Trans CFA."""
    return raw.raw_pattern.shape == (6, 6)


def get_raw_info(path: Path) -> dict:
    raw = rawpy.imread(str(path))
    try:
        raw.unpack()
        sizes  = raw.sizes
        other  = raw.other
        bl     = raw.black_level_per_channel

        shutter = getattr(other, "shutter", 0) or 0
        if 0 < shutter < 1:
            shutter_str = f"1/{1/shutter:.0f}s"
        elif shutter >= 1:
            shutter_str = f"{shutter:.1f}s"
        else:
            shutter_str = "—"

        aperture   = getattr(other, "aperture",   0) or 0
        focal_len  = getattr(other, "focal_len",  0) or 0
        iso_speed  = getattr(other, "iso_speed",  0) or 0

        return {
            "file":         path.name,
            "raw_size":     f"{sizes.raw_width} × {sizes.raw_height}",
            "visible_size": f"{sizes.width} × {sizes.height}",
            "pattern":      get_pattern_str(raw),
            "white_level":  raw.white_level,
            "black_levels": f"R={bl[0]}  G1={bl[1]}  B={bl[2]}  G2={bl[3]}",
            "iso":          iso_speed if iso_speed else "—",
            "shutter":      shutter_str,
            "aperture":     f"f/{aperture}" if aperture else "—",
            "focal_len":    f"{focal_len} mm" if focal_len else "—",
            "xtrans":       is_xtrans(raw),
        }
    finally:
        raw.close()


def load_raw_bayer(path: Path) -> np.ndarray:
    raw = rawpy.imread(str(path))
    try:
        raw.unpack()
        return raw.raw_image_visible.astype(np.float32)
    finally:
        raw.close()


def load_frame(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in RAW_EXTENSIONS:
        return load_raw_bayer(path)
    elif suffix == ".npy":
        return np.load(str(path)).astype(np.float32)
    elif suffix in (".tif", ".tiff", ".png"):
        img = Image.open(str(path))
        return np.array(img, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported format: {path.suffix!r}")


def collect_frames(source: Path) -> list:
    if source.is_file():
        return [source]
    elif source.is_dir():
        frames = []
        all_exts = list(RAW_EXTENSIONS) + [".tif", ".tiff", ".npy", ".png"]
        for ext in all_exts:
            frames.extend(source.glob(f"*{ext}"))
            frames.extend(source.glob(f"*{ext.upper()}"))
        frames = sorted(set(frames))
        if not frames:
            raise FileNotFoundError(f"No frames found in: {source}")
        return frames
    raise FileNotFoundError(f"Not found: {source}")


def build_master(source: Path, label: str, log_fn=None) -> np.ndarray:
    paths = collect_frames(source)
    if len(paths) == 1:
        if log_fn:
            log_fn(f"  {label}: {paths[0].name} (single master)")
        return load_frame(paths[0])
    if log_fn:
        log_fn(f"  {label}: median-stacking {len(paths)} frames …")
    stack = np.stack([load_frame(p) for p in paths], axis=0)
    master = np.median(stack, axis=0).astype(np.float32)
    if log_fn:
        log_fn(f"  {label}: stacking complete ({len(paths)} frames)")
    return master


def _check_shape(cal, data, name):
    if cal.shape != data.shape:
        raise ValueError(
            f"{name} shape {cal.shape} ≠ light shape {data.shape}. "
            "Frames must be from the same camera."
        )


def _normalise_flat(flat: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    norm = flat.copy()
    tile = pattern.shape[0]   # 2 for Bayer, 6 for X-Trans
    for row in range(tile):
        for col in range(tile):
            plane = flat[row::tile, col::tile]
            valid = plane[plane > 0]
            mean_val = valid.mean() if valid.size > 0 else 1.0
            norm[row::tile, col::tile] = plane / mean_val if mean_val > 0 else plane
    return norm


def apply_calibration(bayer, masters, pattern, white_level, black_levels, log_fn=None):
    h, w = bayer.shape
    data = bayer.copy()
    tile = pattern.shape[0]

    # Black level map
    bl_map = np.zeros((h, w), dtype=np.float32)
    unique_positions = {}
    for row in range(tile):
        for col in range(tile):
            ch = pattern[row, col]
            if ch not in unique_positions:
                unique_positions[ch] = (row, col)
            bl_map[row::tile, col::tile] = black_levels[min(ch, len(black_levels) - 1)]
    data -= bl_map

    if log_fn:
        log_fn("  [1] Camera black-level subtracted")

    bias = masters.get("bias")
    if bias is not None:
        _check_shape(bias, data, "Bias")
        data -= bias
        if log_fn:
            log_fn("  [2] Bias subtracted")
    else:
        if log_fn:
            log_fn("  [2] Bias — not provided, skipped")

    dark = masters.get("dark")
    if dark is not None:
        _check_shape(dark, data, "Dark")
        dark_c = dark.copy()
        if bias is not None:
            dark_c -= bias
        data -= dark_c
        if log_fn:
            log_fn("  [3] Dark subtracted" + (" (bias-corrected)" if bias is not None else ""))
    else:
        if log_fn:
            log_fn("  [3] Dark — not provided, skipped")

    flat_raw = masters.get("flat")
    if flat_raw is not None:
        _check_shape(flat_raw, data, "Flat")
        flat = flat_raw.copy()
        if bias is not None:
            flat -= bias
        dark_flat = masters.get("dark_flat")
        if dark_flat is not None:
            _check_shape(dark_flat, data, "Dark-flat")
            df = dark_flat.copy()
            if bias is not None:
                df -= bias
            flat -= df
        flat_norm = _normalise_flat(flat, pattern)
        flat_norm = np.where(flat_norm > 0.05, flat_norm, 1.0)
        data /= flat_norm
        if log_fn:
            log_fn("  [4] Flat-field corrected" +
                   (" + dark-flat" if dark_flat is not None else ""))
    else:
        if log_fn:
            log_fn("  [4] Flat — not provided, skipped")

    effective_white = white_level - float(bl_map.mean())
    return np.clip(data, 0, effective_white)


def _extract_green_planes(bayer, pattern):
    g1 = g2 = None
    tile = pattern.shape[0]
    for row in range(tile):
        for col in range(tile):
            ch = pattern[row, col]
            if ch == 1 and g1 is None:
                g1 = bayer[row::tile, col::tile]
            elif ch == 3 and g2 is None:
                g2 = bayer[row::tile, col::tile]
    if g1 is None:
        g1 = g2
    if g2 is None:
        g2 = g1
    return g1, g2


def _upscale(half, full_h, full_w, tile):
    return np.repeat(np.repeat(half, tile, axis=0), tile, axis=1)[:full_h, :full_w]


def apply_gamma(arr: np.ndarray, gamma: float, max_val: float) -> np.ndarray:
    """
    Apply gamma encoding to a linear float array.
    gamma=1.0  → no change (keep linear)
    gamma=2.2  → standard monitor gamma (sRGB approximation)

    Internally applies as a power of (1/gamma) so passing the intuitive
    display gamma value (e.g. 2.2) gives the correct brightening effect.
    """
    if gamma == 1.0:
        return arr
    normed = np.clip(arr / max_val, 0.0, 1.0)
    return np.power(normed, 1.0 / gamma) * max_val


# ── Film curve definitions ────────────────────────────────────────────────────
# Each curve is a dict with parameters for a piecewise tone mapping:
#   black_lift    : fraction of max_val added to shadow floor (simulates film base+fog)
#   gamma         : power applied to the midtone region (the film characteristic curve)
#   shoulder_start: normalised value [0–1] where highlight roll-off begins
#   shoulder_str  : 0.0 = hard clip, 1.0 = fully rounded shoulder
#   display_gamma : second-pass display gamma applied after the film curve.
#                   Negative stocks use a compressive gamma (<1.0) for the film curve,
#                   which makes the output dark. The display_gamma re-expands it for
#                   screen viewing, simulating the print/scan stage in a darkroom.
#                   Slide stocks already have a bright gamma (>1.0) so display_gamma=1.0.
#
# The combined effective gamma is approximately film_gamma × display_gamma.
# e.g. portra: 0.58 × 2.2 ≈ 1.28  (slightly bright, with lifted shadows/rolled highlights) (I am using display gamma of 1.8 to lessen washed out look)
#
# The curve is applied in normalised [0,1] space then scaled to max_val.

FILM_CURVES = {
    "negative":   dict(black_lift=0.04, gamma=0.60, shoulder_start=0.75, shoulder_str=0.7,  display_gamma=1.8),
    "negative+":  dict(black_lift=0.03, gamma=0.70, shoulder_start=0.80, shoulder_str=0.5,  display_gamma=1.8),
    "reversal":   dict(black_lift=0.01, gamma=1.90, shoulder_start=0.85, shoulder_str=0.4,  display_gamma=1.0),
    "aerochrome": dict(black_lift=0.01, gamma=1.85, shoulder_start=0.82, shoulder_str=0.45, display_gamma=1.0),
    "hp5":        dict(black_lift=0.05, gamma=0.62, shoulder_start=0.72, shoulder_str=0.8,  display_gamma=1.8),
    "velvia":     dict(black_lift=0.00, gamma=2.10, shoulder_start=0.80, shoulder_str=0.3,  display_gamma=1.0),
    "portra":     dict(black_lift=0.05, gamma=0.58, shoulder_start=0.78, shoulder_str=0.9,  display_gamma=1.8),
}

FILM_CURVE_LABELS = {
    "negative":   "Negative film  (generic — Portra/Ektar feel)",
    "negative+":  "Negative film+ (pushed — higher contrast)",
    "reversal":   "Reversal / slide  (Kodachrome feel)",
    "aerochrome": "Aerochrome IR  (reversal, slightly warmer)",
    "hp5":        "Ilford HP5/Delta  (classic mono negative)",
    "velvia":     "Fujichrome Velvia  (punchy saturated slide)",
    "portra":     "Kodak Portra  (smooth shadow roll-off)",
}


def apply_film_curve(arr: np.ndarray, curve_name: str, max_val: float) -> np.ndarray:
    """
    Apply a film-emulation tone curve to a linear float array.

    The curve has three regions:
      Toe      — gentle lift of deep shadows (simulates film base density + fog)
      Mids     — power-law gamma applied to the core exposure range
      Shoulder — soft highlight roll-off (prevents hard clipping, like film)

    For negative stocks the film gamma is compressive (<1.0), making the output
    dark when viewed directly. A display_gamma pass is applied afterward to
    re-expand the tones for screen viewing — simulating what a darkroom print
    or scan would do. Slide stocks have a bright film gamma so display_gamma=1.0.

    All processing in normalised [0,1] space; result scaled back to max_val.
    """
    c = FILM_CURVES.get(curve_name)
    if c is None:
        raise ValueError(f"Unknown film curve: {curve_name!r}. "
                         f"Options: {list(FILM_CURVES)}")

    x = np.clip(arr / max_val, 0.0, 1.0).astype(np.float32)

    # ── Midtone gamma (film characteristic curve) ─────────────────────────
    gamma = c["gamma"]
    y = np.power(x, 1.0 / gamma)

    # ── Shoulder roll-off ─────────────────────────────────────────────────
    # Above shoulder_start, blend from the power curve toward a smooth
    # asymptote at 1.0 using a cosine ease-in. shoulder_str controls how
    # much of the highlight is rounded (0=none, 1=full cosine blend).
    ss = c["shoulder_start"]
    sw = c["shoulder_str"]
    if sw > 0:
        mask = x > ss
        if np.any(mask):
            t = np.clip((x[mask] - ss) / (1.0 - ss), 0.0, 1.0)
            cosine_shoulder = 1.0 - (np.cos(t * np.pi) + 1.0) / 2.0
            straight = y[mask]
            rounded  = ss ** (1.0 / gamma) + (1.0 - ss ** (1.0 / gamma)) * cosine_shoulder
            y[mask]  = straight * (1.0 - sw) + rounded * sw

    # ── Black lift (film base + fog) ──────────────────────────────────────
    bl = c["black_lift"]
    y = y * (1.0 - bl) + bl

    # ── Display gamma (print/scan stage simulation) ───────────────────────
    # Applied after the film curve to bring negative stocks to correct
    # screen brightness. Slide stocks leave this at 1.0 (no second pass).
    dg = c.get("display_gamma", 1.0)
    if dg != 1.0:
        y = np.clip(y, 0.0, 1.0)
        y = np.power(y, 1.0 / dg)

    return np.clip(y * max_val, 0.0, max_val)


def _apply_tone(arr: np.ndarray, gamma: float, curve: str, max_val: float) -> np.ndarray:
    """Apply either a film curve or a simple gamma depending on which is set."""
    if curve:
        return apply_film_curve(arr, curve, max_val)
    return apply_gamma(arr, gamma, max_val)


def extract_monochrome(calibrated, raw, mode="green-avg", normalize=True,
                       bit_depth=16, gamma=2.2, curve="",
                       black_clip=0.1, white_clip=0.1):
    pattern = raw.raw_pattern
    h, w = calibrated.shape
    tile = pattern.shape[0]
    dynamic_range = float(raw.white_level) - np.array(
        raw.black_level_per_channel, dtype=np.float32
    ).mean()

    if mode == "green-avg":
        g1, g2 = _extract_green_planes(calibrated, pattern)
        # Trim to common shape before averaging — odd sensor dimensions can
        # cause g1 and g2 to differ by 1 pixel depending on their row/col offset
        gh = min(g1.shape[0], g2.shape[0])
        gw = min(g1.shape[1], g2.shape[1])
        mono = _upscale((g1[:gh, :gw] + g2[:gh, :gw]) / 2.0, h, w, tile)

    elif mode == "average":
        planes = [calibrated[r::tile, c::tile]
                  for r in range(tile) for c in range(tile)]
        # Trim to same shape (X-Trans tiles may differ by 1px at edges)
        min_h = min(p.shape[0] for p in planes)
        min_w = min(p.shape[1] for p in planes)
        planes = [p[:min_h, :min_w] for p in planes]
        mono = _upscale(np.mean(planes, axis=0), h, w, tile)

    else:
        ch_map = {"R": 0, "G": 1, "G1": 1, "B": 2, "G2": 3}
        target_ch = ch_map.get(mode.upper())
        if target_ch is None:
            raise ValueError(f"Unknown mode: {mode!r}")
        plane = None
        for row in range(tile):
            for col in range(tile):
                if pattern[row, col] == target_ch:
                    plane = calibrated[row::tile, col::tile]
                    break
            if plane is not None:
                break
        if plane is None:
            raise ValueError(f"Channel {mode!r} not in pattern.")
        mono = _upscale(plane, h, w, tile)

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


def extract_aerochrome(calibrated, raw, normalize=True, bit_depth=16,
                       gamma=2.2, curve="", black_clip=0.1, white_clip=0.1):
    """
    Extract a false-colour Aerochrome composite from Bayer data.

    Channel mapping (classic Kodak Aerochrome IR film simulation):
      Output R  ←  Sensor red channel   (IR proxy — long tail to ~720nm)
      Output G  ←  Sensor red channel   (visible red)
      Output B  ←  Sensor green channel (visible green)

    Wait — that's only two source channels used twice. The full mapping is:
      Output R  ←  Sensor red   (IR/red)
      Output G  ←  Sensor red   (red shifted to green position)
      Output B  ←  Sensor green (green shifted to blue position)

    The authentic Aerochrome shift is: IR→R, Red→G, Green→B, Blue→dropped.
    Each channel is independently normalised before compositing so exposure
    differences between channels don't collapse the colour rendering.

    Returns a (H, W, 3) uint8 or uint16 array in RGB order.
    """
    pattern = raw.raw_pattern
    h, w = calibrated.shape
    tile = pattern.shape[0]
    dtype = np.uint16 if bit_depth == 16 else np.uint8
    max_val = float((2 ** bit_depth) - 1)
    dynamic_range = float(raw.white_level) - np.array(
        raw.black_level_per_channel, dtype=np.float32
    ).mean()

    def _extract_plane(ch_idx):
        for row in range(tile):
            for col in range(tile):
                if pattern[row, col] == ch_idx:
                    return _upscale(calibrated[row::tile, col::tile], h, w, tile)
        return None

    def _normalise(plane):
        lo = np.percentile(plane, black_clip)
        hi = np.percentile(plane, 100.0 - white_clip)
        if hi > lo:
            return np.clip((plane - lo) / (hi - lo) * max_val, 0, max_val)
        return np.zeros_like(plane)

    # Pull the three source planes
    red_plane   = _extract_plane(0)   # sensor red   → IR proxy
    green_plane = _extract_plane(1)   # sensor green → visible green
    # blue_plane intentionally dropped (maps to nothing in Aerochrome)

    if red_plane is None or green_plane is None:
        raise ValueError("Could not locate R and G channels in Bayer pattern.")

    # Aerochrome mapping: IR(red)→R, Red→G, Green→B
    if normalize:
        r_out = _normalise(red_plane)    # IR/red → red output
        g_out = _normalise(red_plane)    # red    → green output  (same source)
        b_out = _normalise(green_plane)  # green  → blue output
    else:
        r_out = red_plane   / dynamic_range * max_val
        g_out = red_plane   / dynamic_range * max_val
        b_out = green_plane / dynamic_range * max_val

    # Trim to common shape — upscaling odd-dimension sensors can produce
    # a 1-pixel discrepancy between the red and green planes
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


def save_image(arr: np.ndarray, out_path: Path, fmt: str, **kwargs) -> None:
    """
    Save a monochrome (2-D) or colour (H×W×3) array.

    TIFF output uses tifffile which produces standard 16-bit grayscale or
    RGB TIFFs readable by Photoshop, Lightroom, Windows Photo Viewer, GIMP,
    and most raw editors. Pillow's 16-bit TIFF mode ("I") writes 32-bit
    signed int which most viewers cannot open.

    JPEG is 8-bit only and lossy. Pass jpeg_quality=95 (1-95) via kwargs.
    """
    fmt = fmt.lower()
    is_colour = arr.ndim == 3

    if fmt == "npy":
        np.save(str(out_path.with_suffix(".npy")), arr)

    elif fmt in ("tiff", "tif"):
        # tifffile writes proper uint16 grayscale and RGB — universally readable
        if is_colour:
            # (H, W, 3) uint16 → RGB TIFF
            tifffile.imwrite(str(out_path), arr,
                             photometric="rgb", compression="deflate")
        else:
            # (H, W) uint16 → grayscale TIFF
            tifffile.imwrite(str(out_path), arr,
                             photometric="minisblack", compression="deflate")

    elif fmt == "png":
        if is_colour:
            if arr.dtype == np.uint16:
                # Pillow RGB PNG needs uint8; downscale to 8-bit for PNG
                arr8 = (arr >> 8).astype(np.uint8)
                Image.fromarray(arr8, mode="RGB").save(str(out_path), format="PNG")
            else:
                Image.fromarray(arr, mode="RGB").save(str(out_path), format="PNG")
        else:
            if arr.dtype == np.uint16:
                # 16-bit grayscale PNG via tifffile
                tifffile.imwrite(str(out_path), arr)
            else:
                Image.fromarray(arr, mode="L").save(str(out_path), format="PNG")

    elif fmt in ("jpg", "jpeg"):
        # JPEG is 8-bit only and lossy — downscale uint16 if needed
        quality = kwargs.get("jpeg_quality", 95)
        if is_colour:
            arr8 = (arr >> 8).astype(np.uint8) if arr.dtype == np.uint16 else arr
            Image.fromarray(arr8, mode="RGB").save(
                str(out_path), format="JPEG", quality=quality, subsampling=0)
        else:
            arr8 = (arr >> 8).astype(np.uint8) if arr.dtype == np.uint16 else arr
            Image.fromarray(arr8, mode="L").save(
                str(out_path), format="JPEG", quality=quality)

    else:
        raise ValueError(f"Unknown format: {fmt!r}")


def mono_to_qimage(arr: np.ndarray) -> QImage:
    """Convert a uint16/uint8 mono (2-D) or colour (H×W×3) array to QImage."""
    if arr.ndim == 3:
        # Colour — downscale to 8-bit for display
        if arr.dtype == np.uint16:
            display = (arr >> 8).astype(np.uint8)
        else:
            display = arr.astype(np.uint8)
        h, w = display.shape[:2]
        bytes_per_line = w * 3
        return QImage(display.tobytes(), w, h, bytes_per_line,
                      QImage.Format.Format_RGB888)
    else:
        # Monochrome
        if arr.dtype == np.uint16:
            display = (arr >> 8).astype(np.uint8)
        else:
            display = arr
        h, w = display.shape
        rgb = np.stack([display, display, display], axis=2).astype(np.uint8)
        bytes_per_line = w * 3
        return QImage(rgb.tobytes(), w, h, bytes_per_line,
                      QImage.Format.Format_RGB888)


# ─────────────────────────────────────────────────────────────────────────────
#  Worker  (runs processing in a background thread)
# ─────────────────────────────────────────────────────────────────────────────

class WorkerSignals(QObject):
    log      = Signal(str)
    progress = Signal(int)        # 0–100
    preview  = Signal(np.ndarray) # mono array for display
    finished = Signal(str)        # output path
    error    = Signal(str)


class ProcessWorker(QRunnable):
    def __init__(self, params: dict):
        super().__init__()
        self.params = params
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        try:
            p = self.params
            log = self.signals.log.emit

            log("━" * 48)
            log(f"Opening: {Path(p['light']).name}")
            self.signals.progress.emit(5)

            raw = rawpy.imread(str(p["light"]))
            raw.unpack()
            log(f"  Pattern : {get_pattern_str(raw)}")
            log(f"  Size    : {raw.sizes.width} × {raw.sizes.height} px")
            log(f"  ISO     : {raw.other.iso_speed}")
            self.signals.progress.emit(15)

            # Build calibration masters
            masters = {"bias": None, "dark": None, "flat": None, "dark_flat": None}
            cal_paths = {
                "bias": p.get("bias"), "dark": p.get("dark"),
                "flat": p.get("flat"), "dark_flat": p.get("dark_flat"),
            }
            has_cal = any(v for v in cal_paths.values())

            if has_cal:
                log("\nLoading calibration frames …")
                for key, src in cal_paths.items():
                    if src:
                        masters[key] = build_master(Path(src), key.title(), log)
                log("Calibration frames loaded.")
            else:
                log("\nNo calibration frames — using camera black-level only.")

            self.signals.progress.emit(45)

            # Apply calibration
            log("\nApplying calibration pipeline …")
            bayer_raw = raw.raw_image_visible.astype(np.float32)
            calibrated = apply_calibration(
                bayer=bayer_raw,
                masters=masters,
                pattern=raw.raw_pattern,
                white_level=float(raw.white_level),
                black_levels=np.array(raw.black_level_per_channel, dtype=np.float32),
                log_fn=log,
            )
            self.signals.progress.emit(65)

            # Extract mono or aerochrome
            if p["mode"] == "aerochrome":
                log(f"\nExtracting Aerochrome composite …")
                result = extract_aerochrome(
                    calibrated=calibrated,
                    raw=raw,
                    normalize=p["normalize"],
                    bit_depth=p["bit_depth"],
                    gamma=p["gamma"],
                    curve=p.get("curve", ""),
                    black_clip=p.get("black_clip", 0.1),
                    white_clip=p.get("white_clip", 0.1),
                )
                tone_desc = p["curve"] if p.get("curve") else f"gamma {p['gamma']}"
                log(f"  Output  : {result.shape[1]} × {result.shape[0]} px  (RGB colour)")
                log(f"  Tone    : {tone_desc}")
                log(f"  Clip    : black {p.get('black_clip', 0.1)}%  "
                    f"white {p.get('white_clip', 0.1)}%")
            else:
                log(f"\nExtracting monochrome (mode={p['mode']}) …")
                result = extract_monochrome(
                    calibrated=calibrated,
                    raw=raw,
                    mode=p["mode"],
                    normalize=p["normalize"],
                    bit_depth=p["bit_depth"],
                    gamma=p["gamma"],
                    curve=p.get("curve", ""),
                    black_clip=p.get("black_clip", 0.1),
                    white_clip=p.get("white_clip", 0.1),
                )
                tone_desc = p["curve"] if p.get("curve") else f"gamma {p['gamma']}"
                log(f"  Output  : {result.shape[1]} × {result.shape[0]} px")
                log(f"  Range   : {result.min()} – {result.max()}")
                log(f"  Tone    : {tone_desc}")
                log(f"  Clip    : black {p.get('black_clip', 0.1)}%  "
                    f"white {p.get('white_clip', 0.1)}%")

            raw.close()
            self.signals.progress.emit(80)

            # Emit preview
            self.signals.preview.emit(result)
            self.signals.progress.emit(88)

            # Save
            out_path = Path(p["output"])
            save_image(result, out_path, p["format"],
                       jpeg_quality=p.get("jpeg_quality", 90))
            self.signals.progress.emit(100)

            log(f"\nSaved → {out_path}")
            log("━" * 48)
            self.signals.finished.emit(str(out_path))

        except Exception as exc:
            self.signals.error.emit(
                f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
            )


# ─────────────────────────────────────────────────────────────────────────────
#  Reusable path-picker row widget
# ─────────────────────────────────────────────────────────────────────────────

class PathPicker(QWidget):
    """A label + line-edit + browse button row for picking a file or folder."""

    def __init__(self, label: str, placeholder: str,
                 pick_dir: bool = False, optional: bool = True, parent=None):
        super().__init__(parent)
        self._pick_dir = pick_dir
        self._optional = optional

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        lbl = QLabel(label)
        lbl.setFixedWidth(90)
        lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(lbl)

        self.path_label = QLabel(placeholder)
        self.path_label.setStyleSheet(
            "color: #888; background: #1e1e1e; border: 1px solid #3a3a3a; "
            "border-radius: 4px; padding: 3px 6px;"
        )
        self.path_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.path_label.setMinimumHeight(28)
        layout.addWidget(self.path_label)

        btn = QPushButton("Browse")
        btn.setFixedWidth(72)
        btn.clicked.connect(self._browse)
        layout.addWidget(btn)

        if optional:
            self.clear_btn = QPushButton("✕")
            self.clear_btn.setFixedWidth(28)
            self.clear_btn.setToolTip("Clear")
            self.clear_btn.clicked.connect(self.clear)
            layout.addWidget(self.clear_btn)

        self._path: str = ""

    def _browse(self):
        if self._pick_dir:
            path = QFileDialog.getExistingDirectory(self, "Select folder")
        else:
            raw_filter = (
                "RAW / Image files "
                "(*.CR3 *.cr3 *.CR2 *.cr2 *.NEF *.nef *.ARW *.arw "
                "*.RAF *.raf *.DNG *.dng *.RW2 *.rw2 "
                "*.tif *.tiff *.TIF *.TIFF *.npy)"
            )
            path, _ = QFileDialog.getOpenFileName(
                self, "Select file", "", raw_filter
            )
        if path:
            self._path = path
            self.path_label.setText(Path(path).name)
            self.path_label.setStyleSheet(
                "color: #e0e0e0; background: #1e1e1e; border: 1px solid #3a3a3a; "
                "border-radius: 4px; padding: 3px 6px;"
            )
            self.path_label.setToolTip(path)

    def clear(self):
        self._path = ""
        self.path_label.setText("— not set —")
        self.path_label.setStyleSheet(
            "color: #888; background: #1e1e1e; border: 1px solid #3a3a3a; "
            "border-radius: 4px; padding: 3px 6px;"
        )
        self.path_label.setToolTip("")

    def path(self) -> str:
        return self._path


# ─────────────────────────────────────────────────────────────────────────────
#  Info panel
# ─────────────────────────────────────────────────────────────────────────────

class InfoPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "background: #1a1a2e; border: 1px solid #2a2a4a; border-radius: 6px;"
        )
        layout = QGridLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(10, 8, 10, 8)

        self._fields = {}
        rows = [
            ("file", "File"),
            ("visible_size", "Size"),
            ("pattern", "Pattern"),
            ("iso", "ISO"),
            ("shutter", "Shutter"),
            ("aperture", "Aperture"),
            ("focal_len", "Focal length"),
            ("white_level", "White level"),
            ("black_levels", "Black levels"),
        ]
        for i, (key, display) in enumerate(rows):
            key_lbl = QLabel(display + ":")
            key_lbl.setStyleSheet("color: #7a8fa6; font-size: 11px;")
            key_lbl.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            val_lbl = QLabel("—")
            val_lbl.setStyleSheet("color: #d0d8e8; font-size: 11px;")
            layout.addWidget(key_lbl, i, 0)
            layout.addWidget(val_lbl, i, 1)
            self._fields[key] = val_lbl

    def update_info(self, info: dict):
        for key, lbl in self._fields.items():
            lbl.setText(str(info.get(key, "—")))


# ─────────────────────────────────────────────────────────────────────────────
#  Preview panel
# ─────────────────────────────────────────────────────────────────────────────

class PreviewPanel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(300, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(
            "background: #0d0d0d; border: 1px solid #2a2a2a; border-radius: 6px;"
        )
        self.setText("No preview")
        self.setStyleSheet(
            "background: #0d0d0d; border: 1px solid #2a2a2a; border-radius: 6px; "
            "color: #444; font-size: 13px;"
        )
        self._pixmap_original: QPixmap | None = None

    def set_mono(self, mono: np.ndarray):
        qimg = mono_to_qimage(mono)
        self._pixmap_original = QPixmap.fromImage(qimg)
        self._fit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()

    def _fit(self):
        if self._pixmap_original:
            scaled = self._pixmap_original.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.setPixmap(scaled)


# ─────────────────────────────────────────────────────────────────────────────
#  Main window
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bayer Monochrome Extractor")
        self.resize(1100, 740)
        self._thread_pool = QThreadPool()
        self._last_mono: np.ndarray | None = None
        self._is_xtrans: bool = False

        self._apply_dark_theme()
        self._build_ui()

        # Accept file drops anywhere on the window
        self.setAcceptDrops(True)

    # ── Drag-and-drop ─────────────────────────────────────────────────────

    def dragEnterEvent(self, event):
        """Accept drag if it contains at least one supported RAW file."""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if any(Path(u.toLocalFile()).suffix.lower() in RAW_EXTENSIONS
                   for u in urls):
                event.acceptProposedAction()
                return
        event.ignore()

    def dropEvent(self, event):
        """Load the first dropped RAW file as the light frame."""
        urls = event.mimeData().urls()
        for url in urls:
            path = Path(url.toLocalFile())
            if path.suffix.lower() in RAW_EXTENSIONS and path.is_file():
                self._load_light(path)
                event.acceptProposedAction()
                return
        event.ignore()

    # ── Dark theme ────────────────────────────────────────────────────────

    def _apply_dark_theme(self):
        app = QApplication.instance()
        app.setStyle("Fusion")
        palette = QPalette()
        bg      = QColor("#121212")
        surface = QColor("#1e1e1e")
        border  = QColor("#2e2e2e")
        accent  = QColor("#4a90d9")
        text    = QColor("#e0e0e0")
        muted   = QColor("#888888")

        palette.setColor(QPalette.ColorRole.Window,          bg)
        palette.setColor(QPalette.ColorRole.WindowText,      text)
        palette.setColor(QPalette.ColorRole.Base,            surface)
        palette.setColor(QPalette.ColorRole.AlternateBase,   QColor("#252525"))
        palette.setColor(QPalette.ColorRole.Text,            text)
        palette.setColor(QPalette.ColorRole.Button,          QColor("#2a2a2a"))
        palette.setColor(QPalette.ColorRole.ButtonText,      text)
        palette.setColor(QPalette.ColorRole.Highlight,       accent)
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
        palette.setColor(QPalette.ColorRole.PlaceholderText, muted)
        app.setPalette(palette)

        app.setStyleSheet("""
            QGroupBox {
                border: 1px solid #2e2e2e;
                border-radius: 6px;
                margin-top: 10px;
                font-weight: 600;
                color: #a0b4c8;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QPushButton {
                background: #2a2a2a;
                border: 1px solid #3a3a3a;
                border-radius: 5px;
                padding: 5px 12px;
                color: #e0e0e0;
            }
            QPushButton:hover  { background: #353535; border-color: #4a90d9; }
            QPushButton:pressed { background: #1e1e1e; }
            QPushButton#run_btn {
                background: #1a5fa8;
                border-color: #2878cc;
                font-weight: 700;
                font-size: 13px;
                padding: 8px 20px;
            }
            QPushButton#run_btn:hover  { background: #2070be; }
            QPushButton#run_btn:pressed { background: #134a84; }
            QPushButton#run_btn:disabled { background: #1e2a38; color: #4a6080; }
            QComboBox {
                background: #1e1e1e;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                padding: 3px 6px;
                color: #e0e0e0;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background: #252525;
                border: 1px solid #3a3a3a;
                selection-background-color: #2a5080;
            }
            QCheckBox { color: #c0c8d8; spacing: 6px; }
            QCheckBox::indicator {
                width: 16px; height: 16px;
                border: 1px solid #4a4a4a;
                border-radius: 3px;
                background: #1e1e1e;
            }
            QCheckBox::indicator:checked {
                background: #4a90d9;
                border-color: #4a90d9;
            }
            QTextEdit {
                background: #0e0e0e;
                border: 1px solid #2a2a2a;
                border-radius: 4px;
                color: #b0c4b8;
                font-family: "Consolas", "Menlo", "Courier New", monospace;
                font-size: 11px;
            }
            QProgressBar {
                border: 1px solid #2e2e2e;
                border-radius: 4px;
                background: #1a1a1a;
                height: 8px;
                text-align: center;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1a5fa8, stop:1 #4a90d9);
                border-radius: 3px;
            }
            QSplitter::handle { background: #2a2a2a; }
            QScrollArea { border: none; }
        """)

    # ── UI construction ───────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(8)

        # ── Toolbar-style header ──────────────────────────────────────────
        header = QHBoxLayout()
        title = QLabel("Bayer Monochrome Extractor")
        title.setStyleSheet("font-size: 16px; font-weight: 700; color: #7ab4e8;")
        header.addWidget(title)
        header.addStretch()
        root.addLayout(header)

        # ── Main horizontal splitter ──────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ── Left panel (controls) ─────────────────────────────────────────
        left = QWidget()
        left.setMinimumWidth(340)
        left.setMaximumWidth(420)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 6, 0)
        left_layout.setSpacing(8)

        # Light frame
        light_group = QGroupBox("Light Frame  (or drag & drop here)")
        light_layout = QVBoxLayout(light_group)
        self.light_picker = PathPicker(
            "File:", "— select a RAW file —",
            pick_dir=False, optional=False
        )
        self.light_picker.path_label.setMinimumWidth(180)
        # Override browse for light to also load info
        self.light_picker._browse_original = self.light_picker._browse
        self.light_picker.path_label  # just referencing to keep lint happy

        browse_btn = self.light_picker.findChild(QPushButton)
        if browse_btn:
            browse_btn.clicked.disconnect()
            browse_btn.clicked.connect(self._pick_light)

        light_layout.addWidget(self.light_picker)

        # X-Trans incompatibility warning — hidden until a file is loaded
        self.xtrans_warning = QLabel(
            "⚠  X-Trans sensor detected — not supported. "
            "This tool requires a Bayer (RGGB) sensor."
        )
        self.xtrans_warning.setWordWrap(True)
        self.xtrans_warning.setStyleSheet(
            "color: #f0a000; background: #2a1f00; border: 1px solid #7a5000; "
            "border-radius: 4px; padding: 5px 8px; font-size: 11px;"
        )
        self.xtrans_warning.setVisible(False)
        light_layout.addWidget(self.xtrans_warning)

        left_layout.addWidget(light_group)

        # Info panel
        info_group = QGroupBox("File Info")
        info_layout = QVBoxLayout(info_group)
        self.info_panel = InfoPanel()
        info_layout.addWidget(self.info_panel)
        left_layout.addWidget(info_group)

        # Calibration frames
        cal_group = QGroupBox("Calibration Frames  (all optional)")
        cal_layout = QVBoxLayout(cal_group)
        cal_layout.setSpacing(4)

        self.bias_picker = PathPicker("Bias:", "— not set —", pick_dir=False)
        self.dark_picker = PathPicker("Dark:", "— not set —", pick_dir=False)
        self.flat_picker = PathPicker("Flat:", "— not set —", pick_dir=True)
        self.darkflat_picker = PathPicker("Dark-flat:", "— not set —", pick_dir=True)

        for picker in [self.bias_picker, self.dark_picker,
                       self.flat_picker, self.darkflat_picker]:
            cal_layout.addWidget(picker)

        left_layout.addWidget(cal_group)

        # Options
        opt_group = QGroupBox("Options")
        opt_layout = QGridLayout(opt_group)
        opt_layout.setSpacing(6)

        opt_layout.addWidget(QLabel("Mode:"), 0, 0,
                             Qt.AlignmentFlag.AlignRight)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "green-avg  (best luminance)",
            "average    (all channels)",
            "aerochrome (IR false colour)",
            "R", "G / G1", "G2", "B",
        ])
        opt_layout.addWidget(self.mode_combo, 0, 1)

        opt_layout.addWidget(QLabel("Format:"), 1, 0,
                             Qt.AlignmentFlag.AlignRight)
        self.fmt_combo = QComboBox()
        self.fmt_combo.addItems(["TIFF (16-bit)", "PNG", "JPEG", "NumPy (.npy)"])
        self.fmt_combo.setToolTip(
            "TIFF: 16-bit lossless — best for editing\n"
            "PNG:  lossless, widely compatible\n"
            "JPEG: 8-bit lossy — sharing/preview only\n"
            "NumPy: raw array for further processing"
        )
        self.fmt_combo.currentTextChanged.connect(self._on_format_changed)
        opt_layout.addWidget(self.fmt_combo, 1, 1)

        # JPEG quality row — visible only when JPEG is selected
        self.jpeg_quality_row = QWidget()
        jpeg_q_layout = QHBoxLayout(self.jpeg_quality_row)
        jpeg_q_layout.setContentsMargins(0, 0, 0, 0)
        jpeg_q_layout.addWidget(QLabel("JPEG quality:"))
        from PySide6.QtWidgets import QSlider
        self.jpeg_quality_slider = QSlider(Qt.Orientation.Horizontal)
        self.jpeg_quality_slider.setRange(50, 95)
        self.jpeg_quality_slider.setValue(90)
        self.jpeg_quality_slider.setTickInterval(5)
        self.jpeg_quality_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.jpeg_quality_label = QLabel("90")
        self.jpeg_quality_label.setFixedWidth(28)
        self.jpeg_quality_slider.valueChanged.connect(
            lambda v: self.jpeg_quality_label.setText(str(v))
        )
        jpeg_q_layout.addWidget(self.jpeg_quality_slider)
        jpeg_q_layout.addWidget(self.jpeg_quality_label)
        self.jpeg_quality_row.setVisible(False)
        opt_layout.addWidget(self.jpeg_quality_row, 2, 0, 1, 2)

        opt_layout.addWidget(QLabel("Bit depth:"), 3, 0,
                             Qt.AlignmentFlag.AlignRight)
        self.depth_combo = QComboBox()
        self.depth_combo.addItems(["16-bit", "8-bit"])
        opt_layout.addWidget(self.depth_combo, 3, 1)

        self.normalize_chk = QCheckBox("Normalize output to full range")
        self.normalize_chk.setChecked(True)
        opt_layout.addWidget(self.normalize_chk, 4, 0, 1, 2)

        # Black clip — percentile of pixels crushed to zero before stretch
        opt_layout.addWidget(QLabel("Black clip %:"), 5, 0,
                             Qt.AlignmentFlag.AlignRight)
        from PySide6.QtWidgets import QDoubleSpinBox as _DSB
        self.black_clip_spin = _DSB()
        self.black_clip_spin.setRange(0.0, 5.0)
        self.black_clip_spin.setSingleStep(0.05)
        self.black_clip_spin.setValue(0.1)
        self.black_clip_spin.setDecimals(2)
        self.black_clip_spin.setToolTip(
            "Percentile of pixels clipped to black before stretching.\n"
            "0.0 = use absolute minimum (shadows may lift).\n"
            "0.1 = crush the bottom 0.1% — removes noise floor without\n"
            "      touching real shadow detail. Raise if blacks still look grey."
        )
        opt_layout.addWidget(self.black_clip_spin, 5, 1)

        opt_layout.addWidget(QLabel("White clip %:"), 6, 0,
                             Qt.AlignmentFlag.AlignRight)
        self.white_clip_spin = _DSB()
        self.white_clip_spin.setRange(0.0, 5.0)
        self.white_clip_spin.setSingleStep(0.05)
        self.white_clip_spin.setValue(0.1)
        self.white_clip_spin.setDecimals(2)
        self.white_clip_spin.setToolTip(
            "Percentile of pixels clipped to white before stretching.\n"
            "0.0 = use absolute maximum (one hot pixel can compress everything).\n"
            "0.1 = clip the top 0.1% — removes specular hotspots without\n"
            "      crushing real highlight detail."
        )
        opt_layout.addWidget(self.white_clip_spin, 6, 1)

        opt_layout.addWidget(QLabel("Tone curve:"), 7, 0,
                             Qt.AlignmentFlag.AlignRight)
        self.gamma_combo = QComboBox()
        self.gamma_combo.addItems([
            "2.2  (standard monitor)",
            "1.8  (legacy Mac / print)",
            "1.0  (linear — no gamma)",
            "Film curve …",
            "Custom gamma …",
        ])
        self.gamma_combo.setToolTip(
            "Raw sensor data is linear — choose a tone curve to encode it.\n"
            "2.2: correct for standard monitors and most viewers.\n"
            "Film curve: applies a toe/shoulder S-curve to imitate film stock.\n"
            "1.0: keep linear for editors that apply their own gamma."
        )
        opt_layout.addWidget(self.gamma_combo, 7, 1)

        # Film preset selector — shown only when "Film curve" is selected
        self.film_curve_row = QWidget()
        film_curve_layout = QHBoxLayout(self.film_curve_row)
        film_curve_layout.setContentsMargins(0, 0, 0, 0)
        film_curve_layout.addWidget(QLabel("Preset:"))
        self.film_curve_combo = QComboBox()
        for key, label in FILM_CURVE_LABELS.items():
            self.film_curve_combo.addItem(label, userData=key)
        film_curve_layout.addWidget(self.film_curve_combo)
        self.film_curve_row.setVisible(False)
        opt_layout.addWidget(self.film_curve_row, 8, 0, 1, 2)

        # Custom gamma spinbox — shown only when "Custom gamma" is selected
        self.gamma_custom_row = QWidget()
        gamma_custom_layout = QHBoxLayout(self.gamma_custom_row)
        gamma_custom_layout.setContentsMargins(0, 0, 0, 0)
        gamma_custom_layout.addWidget(QLabel("Value:"))
        from PySide6.QtWidgets import QDoubleSpinBox
        self.gamma_spinbox = QDoubleSpinBox()
        self.gamma_spinbox.setRange(0.1, 4.0)
        self.gamma_spinbox.setSingleStep(0.1)
        self.gamma_spinbox.setValue(2.2)
        self.gamma_spinbox.setDecimals(2)
        gamma_custom_layout.addWidget(self.gamma_spinbox)
        self.gamma_custom_row.setVisible(False)
        opt_layout.addWidget(self.gamma_custom_row, 9, 0, 1, 2)
        self.gamma_combo.currentTextChanged.connect(self._on_gamma_changed)

        left_layout.addWidget(opt_group)

        # Output path
        out_group = QGroupBox("Output")
        out_layout = QVBoxLayout(out_group)
        self.out_picker = PathPicker(
            "Save to:", "— same folder as light frame —",
            pick_dir=False, optional=True
        )
        # Wire browse to save-as dialog
        out_browse_btn = self.out_picker.findChild(QPushButton)
        if out_browse_btn:
            out_browse_btn.clicked.disconnect()
            out_browse_btn.clicked.connect(self._pick_output)
        out_layout.addWidget(self.out_picker)
        left_layout.addWidget(out_group)

        left_layout.addStretch()

        # Run button
        self.run_btn = QPushButton("  ▶  Extract Monochrome")
        self.run_btn.setObjectName("run_btn")
        self.run_btn.clicked.connect(self._run)
        left_layout.addWidget(self.run_btn)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        left_layout.addWidget(self.progress)

        splitter.addWidget(left)

        # ── Right panel (preview + log) ───────────────────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(6, 0, 0, 0)
        right_layout.setSpacing(8)

        # Preview
        prev_group = QGroupBox("Preview")
        prev_layout = QVBoxLayout(prev_group)
        self.preview = PreviewPanel()
        prev_layout.addWidget(self.preview)
        right_layout.addWidget(prev_group, stretch=3)

        # Log
        log_group = QGroupBox("Processing Log")
        log_layout = QVBoxLayout(log_group)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(120)
        log_layout.addWidget(self.log)
        right_layout.addWidget(log_group, stretch=1)

        splitter.addWidget(right)
        splitter.setSizes([370, 730])

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready — select a RAW light frame to begin.")

    # ── Slots ─────────────────────────────────────────────────────────────

    def _on_format_changed(self, text: str):
        self.jpeg_quality_row.setVisible(text == "JPEG")

    def _on_gamma_changed(self, text: str):
        self.film_curve_row.setVisible(text.startswith("Film curve"))
        self.gamma_custom_row.setVisible(text.startswith("Custom gamma"))

    def _gamma_value(self) -> float:
        text = self.gamma_combo.currentText()
        if text.startswith("2.2"):   return 2.2
        if text.startswith("1.8"):   return 1.8
        if text.startswith("1.0"):   return 1.0
        if text.startswith("Film"):  return 2.2   # unused when curve is active
        return float(self.gamma_spinbox.value())

    def _curve_value(self) -> str:
        """Return the film curve key if film mode is active, else empty string."""
        if self.gamma_combo.currentText().startswith("Film"):
            return self.film_curve_combo.currentData()
        return ""

    def _pick_light(self):
        raw_filter = (
            "RAW files "
            "(*.CR3 *.cr3 *.CR2 *.cr2 *.NEF *.nef *.ARW *.arw "
            "*.RAF *.raf *.DNG *.dng *.RW2 *.rw2 *.ORF *.orf)"
        )
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Select light frame", "", raw_filter
        )
        if path_str:
            self._load_light(Path(path_str))

    def _load_light(self, path: Path):
        """Shared handler — called by both the browse button and drag-and-drop."""
        self.light_picker._path = str(path)
        self.light_picker.path_label.setText(path.name)
        self.light_picker.path_label.setToolTip(str(path))
        self.light_picker.path_label.setStyleSheet(
            "color: #e0e0e0; background: #1e1e1e; border: 1px solid #3a3a3a; "
            "border-radius: 4px; padding: 3px 6px;"
        )
        self.status.showMessage(f"Loading info for {path.name} …")
        self._is_xtrans = False
        try:
            info = get_raw_info(path)
            self.info_panel.update_info(info)
            self._is_xtrans = info.get("xtrans", False)
            self.xtrans_warning.setVisible(self._is_xtrans)
            self.run_btn.setEnabled(not self._is_xtrans)
            if self._is_xtrans:
                self.status.showMessage(
                    "X-Trans sensor detected — not supported. Please use a Bayer RAW file."
                )
            else:
                self.status.showMessage(
                    f"Loaded: {info['visible_size']} px  |  "
                    f"Pattern: {info['pattern']}  |  ISO {info['iso']}"
                )
        except Exception as exc:
            import traceback
            full = traceback.format_exc()
            print(f"[get_raw_info error]\n{full}", file=sys.stderr)
            self.status.showMessage(f"Could not read info: {type(exc).__name__}: {exc}")
            QMessageBox.warning(self, "File info error",
                                f"Could not read metadata from this file:\n\n"
                                f"{type(exc).__name__}: {exc}\n\n"
                                f"See terminal for full traceback.")

    def _pick_output(self):
        fmt_ext = {"TIFF (16-bit)": "tiff", "PNG": "png", "JPEG": "jpg", "NumPy (.npy)": "npy"}
        ext = fmt_ext.get(self.fmt_combo.currentText(), "tiff")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save output as", "",
            f"{ext.upper()} files (*.{ext})"
        )
        if path:
            self.out_picker._path = path
            self.out_picker.path_label.setText(Path(path).name)
            self.out_picker.path_label.setToolTip(path)
            self.out_picker.path_label.setStyleSheet(
                "color: #e0e0e0; background: #1e1e1e; border: 1px solid #3a3a3a; "
                "border-radius: 4px; padding: 3px 6px;"
            )

    def _run(self):
        light = self.light_picker.path()
        if not light:
            QMessageBox.warning(self, "No input", "Please select a light frame first.")
            return

        if getattr(self, "_is_xtrans", False):
            QMessageBox.warning(
                self, "X-Trans not supported",
                "The selected file uses a Fujifilm X-Trans sensor (6×6 CFA).\n\n"
                "This tool only supports standard Bayer sensors (2×2 RGGB/BGGR/etc.).\n"
                "Please select a Canon, Nikon, Sony, or other Bayer-sensor RAW file."
            )
            return

        # Determine output path
        fmt_map = {"TIFF (16-bit)": "tiff", "PNG": "png", "JPEG": "jpg", "NumPy (.npy)": "npy"}
        fmt = fmt_map.get(self.fmt_combo.currentText(), "tiff")
        ext = {"tiff": ".tiff", "png": ".png", "jpg": ".jpg", "npy": ".npy"}[fmt]

        # Mode mapping
        mode_map = {
            "green-avg  (best luminance)":  "green-avg",
            "average    (all channels)":    "average",
            "aerochrome (IR false colour)": "aerochrome",
            "R": "R", "G / G1": "G", "G2": "G2", "B": "B",
        }
        mode = mode_map.get(self.mode_combo.currentText(), "green-avg")
        bit_depth = 16 if self.depth_combo.currentText() == "16-bit" else 8

        suffix = "_aerochrome" if mode == "aerochrome" else "_bayer"
        out = self.out_picker.path()
        if not out:
            out = str(Path(light).parent / (Path(light).stem + suffix + ext))

        params = {
            "light":        light,
            "bias":         self.bias_picker.path() or None,
            "dark":         self.dark_picker.path() or None,
            "flat":         self.flat_picker.path() or None,
            "dark_flat":    self.darkflat_picker.path() or None,
            "mode":         mode,
            "normalize":    self.normalize_chk.isChecked(),
            "black_clip":   self.black_clip_spin.value(),
            "white_clip":   self.white_clip_spin.value(),
            "bit_depth":    bit_depth,
            "gamma":        self._gamma_value(),
            "curve":        self._curve_value(),
            "format":       fmt,
            "jpeg_quality": self.jpeg_quality_slider.value(),
            "output":       out,
        }

        self.run_btn.setEnabled(False)
        self.progress.setValue(0)
        self.log.clear()
        self.status.showMessage("Processing …")

        worker = ProcessWorker(params)
        worker.signals.log.connect(self._on_log)
        worker.signals.progress.connect(self.progress.setValue)
        worker.signals.preview.connect(self._on_preview)
        worker.signals.finished.connect(self._on_finished)
        worker.signals.error.connect(self._on_error)
        self._thread_pool.start(worker)

    @Slot(str)
    def _on_log(self, msg: str):
        self.log.append(msg)

    @Slot(np.ndarray)
    def _on_preview(self, mono: np.ndarray):
        self._last_mono = mono
        self.preview.set_mono(mono)

    @Slot(str)
    def _on_finished(self, out_path: str):
        self.run_btn.setEnabled(True)
        self.status.showMessage(f"Done — saved to {out_path}")

    @Slot(str)
    def _on_error(self, msg: str):
        self.run_btn.setEnabled(True)
        self.progress.setValue(0)
        self.status.showMessage("Error — see log for details.")
        self.log.append(f"\n⚠  ERROR:\n{msg}")
        QMessageBox.critical(self, "Processing error", msg[:400])


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Bayer Monochrome Extractor")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
