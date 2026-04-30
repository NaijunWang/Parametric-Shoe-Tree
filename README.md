# Custom Parametric Shoe Trees

Foot scan to printable, custom-fit shoe tree.

By Alex, James, and Naijun for FABRIC-581.

## Demo

<video src="docs/demo/Front-End-Demo-Shoe-Tree.mov" controls width="100%"></video>

If the embedded video does not render, open the
[demo video](docs/demo/Front-End-Demo-Shoe-Tree.mov) directly. The final presentation deck is
also included at
[docs/presentation/Custom-Parametric-Shoe-Trees-Final-Presentation.pdf](docs/presentation/Custom-Parametric-Shoe-Trees-Final-Presentation.pdf).

## Problem

Traditional shoe trees are sold in "universal" sizes and expand at fixed regions such as
the toe box or length. That mismatch can push a shoe away from the owner's actual foot
shape, deforming the upper and accelerating structural wear.

Commercial shoe trees also cost about $20-$30. A scan-driven, 3D-printed version can be
produced for a few dollars, with a shape that reflects the user's own foot anatomy.
Athletic shoes are a strong use case because they deform after training and often need a
controlled break-in shape.

## Design Motivation

The project uses a parametric pipeline rather than a fixed-size catalog. Each shoe tree is
driven by measurements from a foot scan, so length, ball width, toe height, arch position,
heel center, and upper-surface allowance can change per person. This leaves room for future
design features such as breathable lattice bodies, split print layouts, and customized
snap-fit connections.

## Pipeline

The Python package is `custom-shoe-tree`, the import module is `custom_shoe_tree`, and the
CLI entry point is `custom-shoe-tree`.

| Phase | Module | What it does | Main outputs |
|-------|--------|--------------|--------------|
| 1. Input normalization and measurement | `align.py`, `measure.py` | Aligns the scan, detects units, fixes semantic orientation, snaps heel/sole coordinates, and extracts anatomical measurements. | `out/phase1/<scan_id>/mesh_aligned.obj`, `measurements.json`, `audit.json`, `annotated.png` |
| 2. Template registration | `template.py` | Prepares the canonical template shoe tree, decimates it, measures it, and builds template landmarks. | `out/phase2/base_shoe_tree_decimated.obj`, `base_shoe_tree_measurements.json`, `template_landmarks.json`, `template_landmarks.png` |
| 3. Parametric warping | `warp.py` | Samples 60 template sections, applies 7 width and 8 height adaptive control points, smooths the deformation, and trims collar fins. | `out/phase3/<scan_id>/shoe_tree_warp.obj`, `warp_report.json`, `render.png` |
| 4. Geometry refinement | `refine.py` | Runs non-rigid ICP with sole protection and toe sock-wrap cleanup, then selects the safest refined mesh. | `out/phase4/<scan_id>/shoe_tree_refined.obj`, `refine_report.json`, pass renders, `comparison.png` |
| 5. Final export | `finalize.py` | Repairs topology, exports fabrication-ready OBJ/STL files, and reports print-volume stats. | `out/phase5/<scan_id>/shoe_tree_<scan_id>.obj`, `shoe_tree_<scan_id>.stl`, `pipeline_report.json`, `review.png` |

The template input file is intentionally still named `cc_base_last.obj` because that is the
third-party source asset name. It is ignored by Git and expected at `template/cc_base_last.obj`.

## Measured Parameters

The measurement stage extracts:

- Foot length
- Ball width and ball perimeter
- Toe box width, toe box height, and toe angle
- Arch peak, arch height, arch length, and arch type
- Heel center and heel width
- Max foot width and max height
- Adaptive ball position and longitudinal height profile

## Fabrication Notes

The presentation prototype explored splitting the model along the Y-axis midpoint so larger
shoe trees fit common printer beds. The split design uses parametric tabs and sockets so the
halves can clip together after printing. The fabrication tests used BambuLab and Creality
printers; the largest size 13 print took about 6.5 hours at 5% infill, while smaller split
prints took about 3 hours.

The final slides show the printed shoe tree fitted into a Nike trainer. See the included
presentation PDF for the result photos and fabrication context.

## Repository Layout

```text
.
|-- README.md
|-- pyproject.toml
|-- uv.lock
|-- src/custom_shoe_tree/
|   |-- cli.py
|   |-- align.py
|   |-- measure.py
|   |-- template.py
|   |-- warp.py
|   |-- refine.py
|   |-- finalize.py
|   |-- viz.py
|   `-- io.py
|-- tests/
|-- scripts/smoke_all_scans.py
|-- sample-foot-scans/
|-- template/
|   `-- cc_base_last.obj       # local ignored asset
|-- out/
|-- webapp/
|   |-- backend/
|   `-- frontend/
`-- docs/
    |-- demo/
    `-- presentation/
```

`*.obj` and `*.stl` files are ignored, so cloned checkouts need local mesh assets before the
full pipeline can run. Keep the template asset at `template/cc_base_last.obj`; local sample
scans can be placed under `sample-foot-scans/`.

## Quick Start: CLI

```bash
uv sync
uv run custom-shoe-tree --help
uv run custom-shoe-tree pipeline "sample-foot-scans/0014-B.obj"
uv run python scripts/smoke_all_scans.py
uv run pytest
```

The pipeline command requires both a foot-scan OBJ and the ignored template asset at
`template/cc_base_last.obj`. If the sample scans are not present in your checkout, replace
`sample-foot-scans/0014-B.obj` with the path to your own scan.

## Quick Start: Web App

Start the backend:

```bash
cd webapp/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -e ../..
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Start the frontend in a second terminal:

```bash
cd webapp/frontend
npm install
npm run dev
```

Open `http://localhost:5173`, upload a `.obj` foot scan, review the measurements, adjust
the shoe-tree allowance if needed, generate the model, and download the STL.

## Tech Stack

- Python 3.12
- `trimesh`, `scipy`, `numpy`, `shapely`, `networkx`, `manifold3d`, `fast-simplification`
- FastAPI backend
- Vite + React frontend
- Fabrication experiments described in the deck also reference `pymeshfix` for mesh repair

## Datasets And Related Work

The deck references FOOT3D / FIND as related 3D foot-scan work:
[https://www.ollieboyne.com/FIND](https://www.ollieboyne.com/FIND).

This repository is structured for the FABRIC sample scans used during the project. Because
mesh files are ignored, local scan files are expected under `sample-foot-scans/` when running
the smoke script.

## Authors And Acknowledgments

Built by Alex, James, and Naijun as a FABRIC-581 partner project.

## Status

Project submission for FABRIC-581. No standalone license file is included in this nested
repository.
