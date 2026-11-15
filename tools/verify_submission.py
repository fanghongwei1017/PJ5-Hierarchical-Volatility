"""Read-only SHA-256 verification of the submission and frozen experiment inputs."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys


def local_path(root, relative):
    """Accept only canonical, portable paths contained in the chosen root."""
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise ValueError(f"Invalid relative path: {relative!r}")
    parts = PurePosixPath(relative)
    if parts.is_absolute() or ".." in parts.parts or parts.as_posix() != relative:
        raise ValueError(f"Invalid relative path: {relative!r}")
    result = root.joinpath(*parts.parts).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f"Path leaves the package: {relative}")
    return result


def read_json(path):
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")
    return value


def check_files(root, records, label, errors):
    if not isinstance(records, dict) or not records:
        errors.append(f"{label}: missing or empty file inventory")
        return 0
    checked = 0
    for relative, expected in records.items():
        try:
            path = local_path(root, relative)
            if not isinstance(expected, dict):
                raise ValueError("invalid file record")
            size = expected.get("bytes")
            digest = expected.get("sha256")
            if type(size) is not int or size < 0:
                raise ValueError("invalid expected byte count")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("invalid expected SHA-256")
            if not path.is_file():
                raise ValueError("file missing")
            actual_size = path.stat().st_size
            if actual_size != size:
                raise ValueError(f"size mismatch ({actual_size} bytes; expected {size})")
            with path.open("rb") as stream:
                actual_digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual_digest != digest:
                raise ValueError("SHA-256 mismatch")
            checked += 1
        except (OSError, ValueError) as exc:
            errors.append(f"{label} / {relative}: {exc}")
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="submission directory (default: the parent of tools/)")
    args = parser.parse_args()
    root = args.root.resolve()
    errors = []
    counts = {}
    checks = (
        ("submission", root, "submission_manifest.json", "files"),
        ("static_experiment", root / "research",
         "依据/811重划实验_20261114/delivery_manifest.json", "files"),
        ("conditional_experiment", root / "research",
         "依据/条件波动拓展_20261114/delivery_manifest.json", "payload"),
    )
    for label, base, relative, key in checks:
        try:
            document = read_json(local_path(base, relative))
            records = document.get(key)
            if key == "payload":
                if not isinstance(records, list) or not records:
                    raise ValueError("missing or empty payload inventory")
                indexed = {}
                for record in records:
                    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                        raise ValueError("invalid payload entry")
                    name = record["path"]
                    if name in indexed:
                        raise ValueError(f"duplicate payload path: {name}")
                    indexed[name] = record
                records = indexed
            counts[label] = check_files(base, records, label, errors)
        except (OSError, ValueError) as exc:
            errors.append(f"{label}: {exc}")
    print(json.dumps({"status": "failed" if errors else "passed",
                      "verified_files_by_manifest": counts,
                      "errors": errors,
                      "scope": "File sizes and SHA-256 only; no fitting or scientific revalidation."},
                     ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
