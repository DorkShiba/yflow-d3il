"""Resume identity for paired comparisons; no machine-specific defaults."""

import hashlib
import json
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_output_signature(output, settings, files):
    """Reject resuming results produced with different inputs or settings."""
    output = Path(output)
    identity = {"settings": settings, "input_sha256": [file_sha256(p) for p in files]}
    target = output / "RUN_SIGNATURE.json"
    if target.exists():
        if json.loads(target.read_text()) != identity:
            raise ValueError(
                "Output contains a different comparison; choose a new output directory"
            )
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Nonempty comparison output has no input signature")
        output.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(identity, indent=2, allow_nan=False))
    return identity
