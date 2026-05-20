# Bayer Monochrome Extractor

## GUI

```bash
pip install PySide6 rawpy numpy pillow tifffile
```

```bash
python bayer_extractor_gui.py
```

## Command Line

```bash
pip install rawpy numpy pillow tifffile
```

```bash
python cr3_bayer_extract.py light.CR3
```

## Recommended Filters

### Monochrome (green-avg / average modes)
No filter required. A **circular polariser** can reduce glare and improve
contrast. A **UV/IR cut filter** (e.g. Hoya UV) is optional but keeps the
green channel clean on unmodified sensors.

### Near-IR / Red channel extraction
| Filter | Cut-on | Character |
|---|---|---|
| Hoya R72 | 720 nm | Classic IR look — white foliage, dark skies. Long exposures on unmodified cameras (~30s–several minutes in bright sun). |
| Hoya RM90 | 900 nm | Deeper IR, more extreme rendering, even longer exposures. |
| Kolari IR Chrome | 665 nm | Shorter exposures, retains some visible red — less extreme but more workable handheld. |
| Zomei 850nm | 850 nm | Budget option, similar character to RM90. |

### Aerochrome mode
Use the same filters as near-IR above. The **Hoya R72 (720 nm)** is the
classic choice — it was the closest approximation to Kodak Aerochrome's
sensitivity range and gives the most authentic false-colour rendering.
Mount on a tripod. Focus before attaching the filter as live-view
autofocus will struggle through an opaque IR filter.
