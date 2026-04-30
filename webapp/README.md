# Custom Shoe Tree Generator - Web UI

A browser-based frontend that drives the FABRIC-581 custom shoe tree pipeline end to end.
Upload a foot scan, review the extracted measurements, generate a printable STL, and preview
the result in the browser.

## Prerequisites

| Tool | Min version |
|------|-------------|
| Python | 3.12 |
| Node.js | 18 |
| npm | 9 |

The backend imports the local `custom_shoe_tree` package from `../../src`, so the project
source does not need to be installed separately. The Python dependencies still need to be
installed in the backend environment.

## 1. Install Backend Dependencies

```bash
cd webapp/backend

python3 -m venv .venv
source .venv/bin/activate

pip install -e ../..
pip install -r requirements.txt
```

If you are already using the repository's `uv` environment, this is also fine:

```bash
cd webapp/backend
uv pip install -r requirements.txt
```

## 2. Start The Backend

```bash
cd webapp/backend
source .venv/bin/activate
uvicorn main:app --reload --port 8000
```

The API is available at `http://localhost:8000`, with interactive docs at
`http://localhost:8000/docs`.

## 3. Install And Start The Frontend

```bash
cd webapp/frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

## Demo Flow

1. **Upload** - Drop a foot scan `.obj` file.
   The backend aligns it and extracts anatomical measurements.

2. **Review** - Inspect and optionally adjust the extracted measurements.
   The UI also shows an estimated EU / US / UK shoe size. Drag the *Shoe Tree Allowance*
   slider to add more or less upper-surface offset.

3. **Generate** - Click **Generate Shoe Tree**.
   The pipeline runs all phases in the background:
   - Phase 2: template preparation, cached after the first run
   - Phase 3: measurement-driven warp
   - Phase 4: NRICP refinement
   - Phase 5: topology repair and STL export

4. **Download** - Download `shoe_tree_<scan_id>.stl` and import it into BambuLab Studio.
   A 3D preview renders directly in the browser (drag to rotate, scroll to zoom).

## Output Files

Pipeline artifacts are written to the same `out/` directory tree used by the CLI
(`custom-shoe-tree pipeline <scan.obj>`).

```
out/
|-- phase1/<scan_id>/   aligned mesh and measurements
|-- phase2/             cached template artifacts
|-- phase3/<scan_id>/   shoe_tree_warp.obj
|-- phase4/<scan_id>/   shoe_tree_refined.obj
`-- phase5/<scan_id>/   shoe_tree_<scan_id>.obj and .stl
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: custom_shoe_tree` | Start the backend from `webapp/backend`, or install the package with `pip install -e ../..` |
| Upload fails with 400 | File must be `.obj` (not `.stl`) |
| NRICP phase diverges | Try a smaller allowance (1–2 mm) or check that the scan has clean geometry |
| Frontend shows blank page | Run `npm install` in `webapp/frontend` first |
