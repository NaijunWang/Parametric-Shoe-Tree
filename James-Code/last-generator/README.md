# Last Generator

Phase 0 scaffold for the FABRIC-581 foot-scan to shoe-last pipeline.

## Commands

```bash
uv sync
uv run last-gen --help
uv run last-gen pipeline "../Sample Foot Scans/0014-B.obj"
uv run python scripts/smoke_all_scans.py
uv run pytest
```

## Current state

- `last-gen` is registered and exposes the planned phase commands.
- Phase commands are stubs that log `not implemented yet` and write JSON placeholders.
- `scripts/smoke_all_scans.py` exercises the stub pipeline on all 5 sample scans.
