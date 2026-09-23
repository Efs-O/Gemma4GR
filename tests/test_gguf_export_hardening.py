from __future__ import annotations

import json
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from training.gguf_export_utils import (
    ExportValidationError,
    adapter_records,
    gguf_general_name,
    make_staging_dir,
    promote_directories,
    validate_adapter_plan,
    validate_quantization_methods,
    validate_staged_ggufs,
)


def _gguf_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def write_test_gguf(path: Path, name: str | None, *, padding: int = 0) -> None:
    metadata: list[tuple[str, int, bytes]] = []
    if padding:
        metadata.append(("test.padding", 8, _gguf_string("x" * padding)))
    if name is not None:
        metadata.append(("general.name", 8, _gguf_string(name)))
    with path.open("wb") as fh:
        fh.write(b"GGUF")
        fh.write(struct.pack("<IQQ", 3, 0, len(metadata)))
        for key, value_type, value in metadata:
            fh.write(_gguf_string(key))
            fh.write(struct.pack("<I", value_type))
            fh.write(value)


class GGUFMetadataTests(unittest.TestCase):
    def test_reads_general_name_beyond_first_megabyte(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "large-metadata.gguf"
            write_test_gguf(path, "Merged Model", padding=1_100_000)
            self.assertEqual(gguf_general_name(path), "Merged Model")

    def test_missing_or_invalid_metadata_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            missing = Path(temp) / "missing.gguf"
            invalid = Path(temp) / "invalid.gguf"
            write_test_gguf(missing, None)
            invalid.write_bytes(b"not a gguf")
            with self.assertRaises(ExportValidationError):
                gguf_general_name(missing)
            with self.assertRaises(ExportValidationError):
                gguf_general_name(invalid)


class StrictValidationTests(unittest.TestCase):
    def test_adapter_provenance_requires_config_and_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "adapter_config.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ExportValidationError, "weights"):
                adapter_records(root)
            (root / "adapter_model.safetensors").write_bytes(b"weights")
            records = adapter_records(root)
            self.assertEqual(
                {record["path"] for record in records},
                {"adapter_config.json", "adapter_model.safetensors"},
            )

    def test_explicit_model_path_never_silently_falls_back(self) -> None:
        from env_bootstrap import resolve_training_model_source

        with tempfile.TemporaryDirectory() as temp:
            model_dir = Path(temp) / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            self.assertEqual(
                resolve_training_model_source("example/model", str(model_dir)),
                str(model_dir),
            )
            with self.assertRaises(FileNotFoundError):
                resolve_training_model_source(
                    "example/model", str(Path(temp) / "missing-model")
                )

    def test_adapter_plan_requires_all_non_skipped_adapters(self) -> None:
        with self.assertRaisesRegex(ExportValidationError, "STT adapter"):
            validate_adapter_plan(
                stt_exists=False, qa_exists=True, skip_stt=False, skip_qa=False
            )
        with self.assertRaisesRegex(ExportValidationError, "base model"):
            validate_adapter_plan(
                stt_exists=True, qa_exists=True, skip_stt=True, skip_qa=True
            )
        validate_adapter_plan(
            stt_exists=False, qa_exists=True, skip_stt=True, skip_qa=False
        )

    def test_quantizations_must_be_explicit_unique_and_safe(self) -> None:
        self.assertEqual(validate_quantization_methods(["q4_k_m"]), ["q4_k_m"])
        for methods in ([], ["q4_k_m", "q4_k_m"], ["../q4"]):
            with self.subTest(methods=methods), self.assertRaises(ExportValidationError):
                validate_quantization_methods(methods)

    def test_staged_validation_rejects_missing_stale_and_wrong_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_test_gguf(root / "model-q4_k_m.gguf", "Merged Model")
            write_test_gguf(root / "model-mmproj.gguf", "Merged Model")
            records = validate_staged_ggufs(
                root, ["model-q4_k_m.gguf"], "model-mmproj.gguf", "Merged Model"
            )
            self.assertEqual(len(records), 2)

            (root / "model-mmproj.gguf").unlink()
            with self.assertRaisesRegex(ExportValidationError, "model-mmproj.gguf"):
                validate_staged_ggufs(
                    root, ["model-q4_k_m.gguf"], "model-mmproj.gguf", "Merged Model"
                )
            write_test_gguf(root / "model-mmproj.gguf", "Merged Model")
            write_test_gguf(root / "stale-q8_0.gguf", "Merged Model")
            with self.assertRaisesRegex(ExportValidationError, "unexpected GGUF"):
                validate_staged_ggufs(
                    root, ["model-q4_k_m.gguf"], "model-mmproj.gguf", "Merged Model"
                )
            (root / "stale-q8_0.gguf").unlink()
            write_test_gguf(root / "model-q4_k_m.gguf", "Base Model")
            with self.assertRaisesRegex(ExportValidationError, "expected 'Merged Model'"):
                validate_staged_ggufs(
                    root, ["model-q4_k_m.gguf"], "model-mmproj.gguf", "Merged Model"
                )


class PromotionTests(unittest.TestCase):
    def test_promotion_replaces_stale_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            targets = [root / "hf", root / "gguf", root / "release"]
            stages = []
            for target in targets:
                target.mkdir()
                (target / "stale.txt").write_text("old", encoding="utf-8")
                stage = make_staging_dir(target)
                (stage / "fresh.txt").write_text("new", encoding="utf-8")
                stages.append(stage)
            promote_directories(list(zip(stages, targets)))
            for target in targets:
                self.assertEqual((target / "fresh.txt").read_text(encoding="utf-8"), "new")
                self.assertFalse((target / "stale.txt").exists())
            self.assertFalse(list(root.glob(".*.backup-*")))

    def test_failed_multi_directory_promotion_rolls_back(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first, second = root / "first", root / "second"
            for target in (first, second):
                target.mkdir()
                (target / "value.txt").write_text("old", encoding="utf-8")
            first_stage = make_staging_dir(first)
            second_stage = make_staging_dir(second)
            (first_stage / "value.txt").write_text("new", encoding="utf-8")
            (second_stage / "value.txt").write_text("new", encoding="utf-8")

            original_replace = Path.replace

            def fail_second_stage(path: Path, target: Path) -> Path:
                if path == second_stage:
                    raise OSError("injected promotion failure")
                return original_replace(path, target)

            with patch.object(Path, "replace", fail_second_stage):
                with self.assertRaisesRegex(OSError, "injected"):
                    promote_directories([(first_stage, first), (second_stage, second)])
            self.assertEqual((first / "value.txt").read_text(encoding="utf-8"), "old")
            self.assertEqual((second / "value.txt").read_text(encoding="utf-8"), "old")


class MergeSummaryTests(unittest.TestCase):
    def test_prerequisite_failure_returns_one_and_records_reason(self) -> None:
        import training.merge_adapters as merge_module

        captured: list[dict] = []
        with patch.object(
            merge_module,
            "check_prerequisites_final",
            side_effect=ExportValidationError("missing required adapter"),
        ), patch.object(
            merge_module, "save_merge_summary", side_effect=lambda value: captured.append(value.copy())
        ):
            self.assertEqual(merge_module.merge(), 1)
        self.assertEqual(len(captured), 1)
        self.assertFalse(captured[0]["publication_completed"])
        self.assertIn("missing required adapter", captured[0]["merge_error"])
        self.assertIn("missing required adapter", captured[0]["gguf_error"])

    def test_export_failure_returns_one_and_publishes_nothing(self) -> None:
        import training.merge_adapters as merge_module

        fake_unsloth = types.ModuleType("unsloth")
        fake_unsloth.FastModel = object
        fake_unsloth.FastVisionModel = object
        captured: list[dict] = []
        model = SimpleNamespace(config=SimpleNamespace(_name_or_path="base"))
        processor = SimpleNamespace(tokenizer=object())
        with tempfile.TemporaryDirectory() as temp:
            qa_adapter = Path(temp) / "qa"
            qa_adapter.mkdir()
            (qa_adapter / "adapter_config.json").write_text(
                '{"base_model_name_or_path":"example/base"}', encoding="utf-8"
            )
            with patch.dict(sys.modules, {"unsloth": fake_unsloth}), patch.multiple(
                merge_module,
                MERGE_SKIP_STT=True,
                MERGE_SKIP_QA=False,
                QA_ADAPTER=qa_adapter,
            ), patch.object(
                merge_module,
                "check_prerequisites_final",
                return_value={"stt_adapter": False, "qa_adapter": True},
            ), patch.object(
                merge_module, "load_with_unsloth", return_value=(model, processor)
            ), patch.object(
                merge_module, "apply_qa_adapter", return_value=model
            ), patch.object(
                merge_module,
                "export_gguf",
                return_value=(False, [], "quantizer exploded", None),
            ), patch.object(
                merge_module,
                "save_merge_summary",
                side_effect=lambda value: captured.append(value.copy()),
            ):
                self.assertEqual(merge_module.merge(), 1)
        self.assertEqual(captured[0]["gguf_error"], "quantizer exploded")
        self.assertTrue(captured[0]["merge_completed"])
        self.assertFalse(captured[0]["publication_completed"])


class ExportIntegrationTests(unittest.TestCase):
    def test_successful_export_commits_only_current_verified_artifacts(self) -> None:
        import training.merge_adapters as merge_module

        class SavedObject:
            def __init__(self, kind: str) -> None:
                self.kind = kind

            def save_pretrained(self, directory: str, **_: object) -> None:
                root = Path(directory)
                root.mkdir(parents=True, exist_ok=True)
                if self.kind == "model":
                    (root / "model.safetensors").write_bytes(b"merged weights")
                    (root / "config.json").write_text("{}", encoding="utf-8")
                else:
                    (root / f"{self.kind}_config.json").write_text("{}", encoding="utf-8")

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            hf_target, gguf_target, release_target = (
                root / "merged-hf", root / "gguf", root / "release"
            )
            for target in (hf_target, gguf_target, release_target):
                target.mkdir()
                (target / "stale.txt").write_text("old", encoding="utf-8")
            qa_adapter = root / "qa-adapter"
            qa_adapter.mkdir()
            (qa_adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            (qa_adapter / "adapter_model.safetensors").write_bytes(b"adapter")
            converter, quantizer = root / "convert.py", root / "quantize.exe"
            converter.write_text("# converter", encoding="utf-8")
            quantizer.write_bytes(b"quantizer")

            def fake_run_step(cmd: list[object], label: str) -> None:
                if label.startswith("convert"):
                    output = Path(cmd[cmd.index("--outfile") + 1])
                else:
                    output = Path(cmd[2])
                write_test_gguf(output, "Test Merged Model")

            model = SavedObject("model")
            model.config = SimpleNamespace(_name_or_path="example/base")
            tokenizer = SavedObject("tokenizer")
            processor = SavedObject("processor")
            with patch.multiple(
                merge_module,
                EXPORT_GGUF=True,
                MERGED_HF_DIR=hf_target,
                GGUF_DIR=gguf_target,
                FINAL_DIR=release_target,
                LLAMA_CONVERT=converter,
                LLAMA_QUANTIZE=quantizer,
                GGUF_MODEL_NAME="Test Merged Model",
                GGUF_QUANT_METHODS=["q4_k_m", "q8_0"],
                MERGE_SKIP_STT=True,
                MERGE_SKIP_QA=False,
                QA_ADAPTER=qa_adapter,
                _MODEL="e4b",
                _output_suffix="-test",
            ), patch.object(merge_module, "run_step", side_effect=fake_run_step), patch.object(
                merge_module, "validate_tensor_outputs", return_value=None
            ):
                ok, files, error, manifest = merge_module.export_gguf(
                    model, processor, tokenizer
                )

            self.assertTrue(ok, error)
            self.assertIsNone(error)
            self.assertEqual(
                set(files),
                {
                    "gemma4gr-e4b-test-q4_k_m.gguf",
                    "gemma4gr-e4b-test-q8_0.gguf",
                    "gemma4gr-e4b-test-mmproj.gguf",
                },
            )
            self.assertIsNotNone(manifest)
            for target in (hf_target, gguf_target, release_target):
                self.assertFalse((target / "stale.txt").exists())
            release_manifest = json.loads(
                (release_target / "export_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(release_manifest["general_name"], "Test Merged Model")
            self.assertEqual(len(release_manifest["gguf_artifacts"]), 3)


class FailFastPolicyTests(unittest.TestCase):
    def test_merge_export_cannot_reintroduce_unsloth_gguf_fallback(self) -> None:
        source = (Path(__file__).resolve().parent.parent / "training" / "merge_adapters.py").read_text(encoding="utf-8")
        self.assertNotIn(".save_pretrained_gguf(", source)
        self.assertNotIn("save_pretrained_merged(", source)
        self.assertNotIn("base_snapshot_root", source)
        self.assertNotIn("rglob(\"*.gguf\")", source)

    def test_active_trainers_have_no_model_load_retry_or_direct_gguf_export(self) -> None:
        root = Path(__file__).resolve().parent.parent
        trainers = [
            "train_qa_local.py",
            "train_stt_qa_local.py",
            "train_stt_final_local.py",
        ]
        for filename in trainers:
            source = (root / "training" / filename).read_text(encoding="utf-8")
            with self.subTest(filename=filename):
                self.assertNotIn(".save_pretrained_gguf(", source)
                self.assertNotIn("local_files_only=True", source)
                self.assertNotIn("Retrying model load", source)

if __name__ == "__main__":
    unittest.main()
