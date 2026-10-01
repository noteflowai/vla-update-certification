"""Verify immutable public development artifacts; does not execute native models."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "release-manifest.json").read_text())
for name, expected in manifest["files"].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
for cohort in ("admission-002", "joint-comparison-001", "coarse-comparison-001"):
    folder = root / "runs" / cohort
    record = json.loads((folder / "manifest.json").read_text())
    for name, expected in record.get("sources", {}).items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == expected, name
pdf = json.loads((root / "paper/verification.json").read_text())
assert hashlib.sha256((root / "paper/anonymous-development-review.pdf").read_bytes()).hexdigest() == pdf["pdf_sha256"]
assert hashlib.sha256((root / "paper/statewise-regression-budgets-stage1-sources.tar.gz").read_bytes()).hexdigest() == pdf["source_archive_sha256"]
print(json.dumps({"verified_files": len(manifest["files"]), "scope": manifest["scope"]}))
