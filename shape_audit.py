from pathlib import Path
from collections import Counter
from datetime import datetime
import numpy as np


ROOT = Path(__file__).resolve().parent
REPORT_PATH = ROOT / f"shape_audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
CHECKS = [
    ("humanml_new_joint_vecs", ROOT / "humanml" / "new_joint_vecs", False),
    ("finetune_dataset_direct", ROOT / "humanml" / "arlm_new_joint_vecs", False),
    ("finetune_dataset_converted", ROOT / "humanml" / "arlm_train_pred_joint_vecs", False),
    ("samples_motion_files", ROOT / "samples", True),
]


def iter_files(directory: Path, sample_only: bool):
    if not directory.exists():
        return []
    files = []
    for path in directory.glob("*.npy"):
        if sample_only and not (
            path.name.endswith("_motion.npy") or path.name.endswith("_ar_motion.npy")
        ):
            continue
        files.append(path)
    return sorted(files)


with REPORT_PATH.open("w", encoding="utf-8") as report:
    for label, directory, sample_only in CHECKS:
        files = iter_files(directory, sample_only)
        shapes = Counter()
        dims = Counter()
        lengths = []
        failures = []

        for path in files:
            try:
                arr = np.load(path)
            except Exception as exc:
                failures.append((path.name, str(exc)))
                continue

            shapes[tuple(arr.shape)] += 1
            if arr.ndim == 2:
                dims[arr.shape[1]] += 1
                lengths.append(arr.shape[0])

        lines = [
            label,
            f"  dir={directory}",
            f"  files={len(files)} load_failures={len(failures)}",
            f"  shapes={shapes.most_common(8)}",
            f"  feature_dims={dict(dims) if dims else 'n/a'}",
        ]
        if lengths:
            lines.append(f"  length_range=({min(lengths)}, {max(lengths)})")
        lines.append("")

        block = "\n".join(lines)
        print(block, flush=True)
        report.write(block + "\n")
        report.flush()
