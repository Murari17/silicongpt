from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def ensure_lfs_assets() -> None:
    """Restore required Git LFS files before preprocessing."""
    required_patterns = [
        "data/raw/*.pdf",
        "tokenizer.model",
        "model.pth",
        "model_best.pth",
    ]
    try:
        result = subprocess.run(
            ["git", "lfs", "pull", "--include=" + ",".join(required_patterns)],
            cwd=str(BASE_DIR),
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print("\n>> Warning: git lfs pull did not complete successfully. Continuing with available files.")
            if result.stdout:
                print(result.stdout.strip())
            if result.stderr:
                print(result.stderr.strip())
    except FileNotFoundError:
        print("\n>> Warning: git lfs is not installed; continuing without auto-restoring LFS files.")


def get_pipeline_python() -> str:
    # Use the project venv if it exists so runs are consistent across machines.
    candidates = [
        BASE_DIR / ".venv" / "bin" / "python",  # macOS/Linux path
        BASE_DIR / ".venv" / "Scripts" / "python.exe",  # Windows path
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def run_step(script_name: str, extra_env: dict[str, str] | None = None) -> None:
    script_path = BASE_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    python_exec = get_pipeline_python()
    print(f"\n{'='*60}")
    print(f"  Running: {script_name}")
    print(f"  Python:  {python_exec}")
    print(f"{'='*60}")
    result = subprocess.run([python_exec, str(script_path)], cwd=str(BASE_DIR), env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Step failed ({script_name}) with exit code {result.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full text-to-model pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --epochs 15           Full pipeline with GPU auto-detect
  python run_pipeline.py --epochs 20 --lr 5e-4 Custom learning rate
  python run_pipeline.py --skip-train           Preprocess only (no training)
  python run_pipeline.py --launch-ui            Train then launch web UI
        """,
    )
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs (default: 15)")
    parser.add_argument("--lr", type=str, default="3e-4", help="Max learning rate (default: 3e-4)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length (default: 128)")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training")
    parser.add_argument("--skip-tokenizer", action="store_true", help="Skip tokenizer training")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip PDF extraction and text cleaning")
    parser.add_argument("--launch-ui", action="store_true", help="Launch Gradio web UI after training")
    args = parser.parse_args()

    ensure_lfs_assets()

    # ---- Preprocessing ----
    if not args.skip_preprocess:
        run_step("textextraction.py")
        run_step("clean_text.py")
        run_step("sentence_split.py")
        run_step("create_dataset.py")
    else:
        print("\n>> Skipping preprocessing (--skip-preprocess)")

    # ---- Tokenizer ----
    if not args.skip_tokenizer and not args.skip_preprocess:
        run_step("train_tokenizer.py")
    elif args.skip_tokenizer:
        print("\n>> Skipping tokenizer training (--skip-tokenizer)")

    # ---- Training ----
    if not args.skip_train:
        run_step("train.py", extra_env={
            "EPOCHS": str(args.epochs),
            "LR": str(args.lr),
            "BATCH_SIZE": str(args.batch_size),
            "SEQ_LEN": str(args.seq_len),
            "GRAD_ACCUM": str(args.grad_accum),
        })
    else:
        print("\n>> Skipping training (--skip-train)")

    print(f"\n{'='*60}")
    print("  [OK] Pipeline complete!")
    print(f"{'='*60}")
    print("\nNext steps:")
    print("  CLI:  python generate.py")
    print("  Web:  python app.py")

    # ---- Optional UI launch ----
    if args.launch_ui:
        print("\nLaunching web UI...")
        run_step("app.py")


if __name__ == "__main__":
    main()
