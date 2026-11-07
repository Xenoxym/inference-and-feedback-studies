"""Run the preserved sampling audit in isolation and keep existing results intact."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    destination = arguments.output.resolve()
    if destination.exists():
        raise FileExistsError(f"Choose a fresh output path: {destination}")

    project = Path(__file__).resolve().parents[1]
    audit = project / "experiments/inference/sampling_audit.py"
    utility = project / "sources/Adaptix/aded/model/utils.py"
    with tempfile.TemporaryDirectory(prefix="sampling-verifier-") as temporary:
        isolated = Path(temporary)
        for source in (audit, utility):
            target = isolated / source.relative_to(project)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        (isolated / "results/inference").mkdir(parents=True)
        subprocess.run(
            [sys.executable, str(isolated / audit.relative_to(project))],
            cwd=isolated,
            capture_output=True,
            text=True,
            check=True,
        )
        result = json.loads((isolated / "results/inference/sampling_audit.json").read_text(encoding="utf-8"))

    result["wrapper_provenance"] = {
        "original_audit_sha256": hashlib.sha256(audit.read_bytes()).hexdigest(),
        "upstream_utility_sha256": hashlib.sha256(utility.read_bytes()).hexdigest(),
        "execution": "Unchanged audit and utility executed in an isolated temporary directory.",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
