"""
Run Piper training with an explicit CSVLogger (metrics.csv loss curves on disk).

PyTorch Lightning's piper_train CLI only allows --logger True/False; True picks
TensorBoard when installed, which is easy to miss in logs. This wrapper injects
CSVLogger under {default_root_dir}/csv_metrics/vits/version_*/metrics.csv.

Mounted in Docker as /app/piper_csv_launcher.py (see train_piper.py).
"""
from __future__ import annotations

from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger


def _apply_csv_logger_patch() -> None:
    if getattr(Trainer, "_gemma4gr_csv_patched", False):
        return

    _orig = Trainer.from_argparse_args.__func__

    @classmethod
    def _from_argparse_args(cls, args, **kwargs):
        use_logger = getattr(args, "logger", True)
        if use_logger is not False and "logger" not in kwargs:
            root = Path(
                str(
                    getattr(args, "default_root_dir", None)
                    or getattr(args, "dataset_dir", ".")
                )
            )
            root.mkdir(parents=True, exist_ok=True)
            # First logger sets where ModelCheckpoint writes .ckpt files (must stay lightning_logs).
            kwargs["logger"] = [
                TensorBoardLogger(save_dir=str(root), name="lightning_logs"),
                CSVLogger(save_dir=str(root), name="csv_metrics"),
            ]
        return _orig(cls, args, **kwargs)

    Trainer.from_argparse_args = _from_argparse_args
    Trainer._gemma4gr_csv_patched = True


def main() -> None:
    _apply_csv_logger_patch()
    from piper_train.__main__ import main as piper_main

    piper_main()


if __name__ == "__main__":
    main()
