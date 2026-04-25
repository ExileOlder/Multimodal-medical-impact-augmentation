# Stage A 1K Final Checkpoint

This directory stores the selected final adapter and its evaluation artifacts.

| File | Description |
| --- | --- |
| `adapter.pth` | Final Stage A colorfix adapter at step `0001000` |
| `structural_metrics.json` | Canonical structural/color metrics for the selected checkpoint |
| `triptych_sheet_canonical.png` | Evaluation sheet used in the frontend and thesis |
| `args.json` | Training arguments from the selected run |
| `run_manifest.json` | Run manifest for traceability |

The adapter is not a standalone full model. It must be loaded together with the RetinaLogos base checkpoint, normally placed at:

```text
checkpoints/consolidated.00-of-01.pth
```

Later continuation checkpoints from `0001500` to `0005000` were evaluated but rejected by the structure/color guard, so this `0001000` adapter is the final default.
