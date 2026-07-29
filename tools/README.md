# Tools

## extract_hair.py — make a hair asset from a photo

Turns any portrait into a transparent hair PNG the overlay can use, so you can
generate test assets without hunting stock sites.

```bash
# from the worker directory, with requirements installed
python tools/extract_hair.py my_portrait.jpg --name wavy --preview
```

Writes `hair_assets/wavy.png` (and `wavy_preview.png` on a checkerboard so you
can eyeball the alpha edge). Then set it live from the app, or:

```bash
# quick check it loads
python -c "import hair; h=hair.HairOverlay('/models'); print(h.list_styles())"
```

### Getting good results

- **Input matters most.** Front-facing, well-lit, hair clearly separated from
  the background. Busy backgrounds confuse the segmentation.
- **Check the preview.** If the edge is ragged or grabs background, raise
  `--feather` (e.g. 8) or pick a cleaner photo.
- **It's a front view.** Looks right facing the camera, degrades on big turns —
  inherent to any flat cutout.

### Flags

- `--name` (required): output style name
- `--out`: output folder (default ../hair_assets)
- `--feather`: edge softness px (default 5; raise for softer)
- `--pad`: crop padding px (default 20)
- `--preview`: also write a checkerboard preview to inspect the alpha
