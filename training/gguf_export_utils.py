"""Fail-fast, dependency-free helpers for transactional GGUF exports."""
from __future__ import annotations

import hashlib
import os
import shutil
import struct
import uuid
from pathlib import Path
from typing import BinaryIO, Iterable


class ExportValidationError(RuntimeError):
    """The staged export is incomplete, ambiguous, or cannot be verified."""


_FIXED_VALUE_SIZES = {
    0: 1,   # uint8
    1: 1,   # int8
    2: 2,   # uint16
    3: 2,   # int16
    4: 4,   # uint32
    5: 4,   # int32
    6: 4,   # float32
    7: 1,   # bool
    10: 8,  # uint64
    11: 8,  # int64
    12: 8,  # float64
}


def _read_exact(fh: BinaryIO, size: int) -> bytes:
    data = fh.read(size)
    if len(data) != size:
        raise ExportValidationError("truncated GGUF metadata")
    return data


def _u32(fh: BinaryIO) -> int:
    return struct.unpack("<I", _read_exact(fh, 4))[0]


def _u64(fh: BinaryIO) -> int:
    return struct.unpack("<Q", _read_exact(fh, 8))[0]


def _string(fh: BinaryIO) -> str:
    size = _u64(fh)
    if size > 128 * 1024 * 1024:
        raise ExportValidationError(f"unreasonable GGUF string length: {size}")
    return _read_exact(fh, size).decode("utf-8", "strict")


def _skip_value(fh: BinaryIO, value_type: int) -> None:
    if value_type in _FIXED_VALUE_SIZES:
        fh.seek(_FIXED_VALUE_SIZES[value_type], os.SEEK_CUR)
        return
    if value_type == 8:  # string
        fh.seek(_u64(fh), os.SEEK_CUR)
        return
    if value_type == 9:  # array
        element_type = _u32(fh)
        count = _u64(fh)
        if element_type in _FIXED_VALUE_SIZES:
            fh.seek(_FIXED_VALUE_SIZES[element_type] * count, os.SEEK_CUR)
            return
        if element_type == 8:
            for _ in range(count):
                fh.seek(_u64(fh), os.SEEK_CUR)
            return
        raise ExportValidationError(f"unsupported GGUF array element type: {element_type}")
    raise ExportValidationError(f"unsupported GGUF value type: {value_type}")


def gguf_general_name(path: Path) -> str:
    """Parse ``general.name`` from the complete GGUF metadata table.

    Missing, malformed, or unreadable metadata is a hard validation failure.
    """
    try:
        with path.open("rb") as fh:
            if _read_exact(fh, 4) != b"GGUF":
                raise ExportValidationError(f"not a GGUF file: {path}")
            version = _u32(fh)
            if version not in {2, 3}:
                raise ExportValidationError(f"unsupported GGUF version {version}: {path}")
            _u64(fh)  # tensor count
            metadata_count = _u64(fh)
            for _ in range(metadata_count):
                key = _string(fh)
                value_type = _u32(fh)
                if key == "general.name":
                    if value_type != 8:
                        raise ExportValidationError("GGUF general.name is not a string")
                    value = _string(fh).strip()
                    if not value:
                        raise ExportValidationError("GGUF general.name is empty")
                    return value
                _skip_value(fh, value_type)
    except ExportValidationError:
        raise
    except (OSError, UnicodeError, struct.error) as exc:
        raise ExportValidationError(f"cannot read GGUF metadata from {path}: {exc}") from exc
    raise ExportValidationError(f"GGUF general.name is missing: {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, object]:
    return {
        "path": str(path.relative_to(relative_to) if relative_to else path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def adapter_records(root: Path) -> list[dict[str, object]]:
    config = root / "adapter_config.json"
    weights = sorted(root.glob("adapter_model*.safetensors"))
    weights.extend(sorted(root.glob("adapter_model*.bin")))
    if not config.is_file():
        raise ExportValidationError(f"adapter config is missing: {config}")
    if not weights:
        raise ExportValidationError(f"adapter weights are missing: {root}")
    return [
        file_record(path, relative_to=root)
        for path in [config, *weights]
    ]


def validate_adapter_plan(
    *, stt_exists: bool, qa_exists: bool, skip_stt: bool, skip_qa: bool
) -> None:
    """Require every non-skipped adapter and at least one active adapter."""
    errors: list[str] = []
    if not skip_stt and not stt_exists:
        errors.append("STT adapter is required but missing (use MERGE_SKIP_STT=1 only intentionally)")
    if not skip_qa and not qa_exists:
        errors.append("QA adapter is required but missing (use MERGE_SKIP_QA=1 only intentionally)")
    if skip_stt and skip_qa:
        errors.append("both adapters are skipped; refusing to export the base model")
    if errors:
        raise ExportValidationError("; ".join(errors))


def validate_quantization_methods(methods: Iterable[str]) -> list[str]:
    values = list(methods)
    if not values:
        raise ExportValidationError("at least one explicit GGUF quantization is required")
    if len(values) != len(set(values)):
        raise ExportValidationError("duplicate GGUF quantization methods are not allowed")
    invalid = [value for value in values if not value or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in value)]
    if invalid:
        raise ExportValidationError(f"invalid GGUF quantization method(s): {invalid}")
    return values


def make_staging_dir(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = target.parent / f".{target.name}.staging-{uuid.uuid4().hex}"
    stage.mkdir()
    return stage


def validate_staged_ggufs(
    gguf_dir: Path,
    expected_model_files: Iterable[str],
    mmproj_name: str,
    expected_general_name: str,
) -> list[dict[str, object]]:
    """Validate exactly the GGUFs expected from the current staged run."""
    expected = set(expected_model_files) | {mmproj_name}
    actual = {path.name for path in gguf_dir.glob("*.gguf") if path.is_file()}
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        parts = []
        if missing:
            parts.append(f"missing GGUFs: {', '.join(missing)}")
        if unexpected:
            parts.append(f"unexpected GGUFs: {', '.join(unexpected)}")
        raise ExportValidationError("; ".join(parts))

    records: list[dict[str, object]] = []
    model_names: set[str] = set()
    for filename in sorted(expected):
        path = gguf_dir / filename
        name = gguf_general_name(path)
        record = file_record(path, relative_to=gguf_dir)
        record["general_name"] = name
        record["kind"] = "mmproj" if filename == mmproj_name else "model"
        records.append(record)
        if name != expected_general_name:
            raise ExportValidationError(
                f"{filename} has general.name={name!r}; expected {expected_general_name!r}"
            )
        if filename != mmproj_name:
            model_names.add(name)
    if len(model_names) != 1:
        raise ExportValidationError(
            f"quantized model GGUFs disagree on general.name: {sorted(model_names)}"
        )
    return records


def promote_directories(staged_targets: Iterable[tuple[Path, Path]]) -> None:
    """Promote staged directories as one rollback-capable transaction.

    Every staging directory must be a sibling of its target. Existing targets
    are renamed to unique backups and restored if any promotion fails.
    """
    pairs = list(staged_targets)
    if not pairs:
        raise ExportValidationError("no staged directories to promote")
    for stage, target in pairs:
        if not stage.is_dir():
            raise ExportValidationError(f"staging directory is missing: {stage}")
        if stage.parent.resolve() != target.parent.resolve():
            raise ExportValidationError(f"stage must be a sibling of target: {stage} -> {target}")
        if not stage.name.startswith(f".{target.name}.staging-"):
            raise ExportValidationError(f"unrecognized staging directory: {stage}")

    backups: dict[Path, Path] = {}
    promoted: list[Path] = []
    transaction_id = uuid.uuid4().hex
    try:
        for _, target in pairs:
            if target.exists():
                backup = target.parent / f".{target.name}.backup-{transaction_id}"
                target.replace(backup)
                backups[target] = backup
        for stage, target in pairs:
            stage.replace(target)
            promoted.append(target)
    except Exception:
        for target in reversed(promoted):
            if target.exists():
                failed_stage = target.parent / f".{target.name}.failed-{transaction_id}"
                target.replace(failed_stage)
                shutil.rmtree(failed_stage)
        for target, backup in backups.items():
            if backup.exists():
                backup.replace(target)
        raise
    for backup in backups.values():
        shutil.rmtree(backup)
