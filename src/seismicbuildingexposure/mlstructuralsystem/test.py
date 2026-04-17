import subprocess
import sys
import os
from pathlib import Path
import time
from datetime import datetime, timedelta
import json

# ============================================================
# Helper Functions
# ============================================================

def print_header(text):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def print_section(text):
    print("\n" + "-" * 70)
    print(f"  {text}")
    print("-" * 70 + "\n")


def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))


def run_script(script_path, env):
    print(f"▶ Running: {script_path.name}")
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=script_path.parent,
            capture_output=True,
            text=True,
            check=True,
            env=env
        )

        elapsed = time.time() - start_time
        print(f"✓ Completed in {format_time(elapsed)}")

        log_path = script_path.parent / f"{script_path.stem}_output.log"
        with open(log_path, 'w') as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n" + result.stderr)

        return True, elapsed, None

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"✗ Failed after {format_time(elapsed)}")

        error_log_path = script_path.parent / f"{script_path.stem}_error.log"
        with open(error_log_path, 'w') as f:
            f.write(e.stdout or "")
            f.write("\n")
            f.write(e.stderr or "")

        return False, elapsed, str(e)


# ============================================================
# MODEL TEST RUNNER
# ============================================================

def run_model_test(model_name, model_idx_dir, cfg, env):
    print_section(f"TESTING MODEL: {model_name}")

    results = {
        "model": model_name,
        "status": "running",
        "start_time": datetime.now().isoformat(),
        "script": "4-test.py",
        "time": 0
    }

    script_path = model_idx_dir / "4-test.py"

    if not script_path.exists():
        print(f"⚠ Missing test script: {script_path}")
        results["status"] = "failed"
        results["error"] = "missing script"
        return results

    success, elapsed, error = run_script(script_path, env)

    results["time"] = elapsed
    results["status"] = "success" if success else "failed"
    results["error"] = error
    results["end_time"] = datetime.now().isoformat()

    print(f"\n✓ {model_name} test completed in {format_time(elapsed)}")

    return results


# ============================================================
# MAIN RUNNER
# ============================================================

def run(cfg):
    """
    Entry point for TEST pipeline (NOT training)
    cfg = experiment config
    """

    print_header("AUTOMATED MODEL TESTING PIPELINE")

    models_dir = cfg.PROJECT_ROOT / "models"

    # Environment setup
    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = cfg.MLFLOW_TRACKING_URI
    env["PROJECT_MODELS_DIR"] = str(cfg.PROJECT_ROOT / "models")

    print(f"Models directory: {models_dir}")
    print(f"Models: {', '.join(cfg.MODELS)}")

    overall_start = time.time()
    all_results = []

    for idx, model_name in enumerate(cfg.MODELS, start=1):
        print_header(f"Model {idx}/{len(cfg.MODELS)}: {model_name}")

        model_idx_dir = models_dir / model_name

        if not model_idx_dir.exists():
            print(f"⚠ Missing model directory: {model_idx_dir}")
            continue

        result = run_model_test(model_name, model_idx_dir, cfg, env)
        all_results.append(result)

        if result["status"] == "failed":
            print("⚠ Model failed. Continue? (y/n)")
            if input().lower() != "y":
                break

    overall_elapsed = time.time() - overall_start

    print_header("TEST EXECUTION SUMMARY")
    print(f"Total time: {format_time(overall_elapsed)}")

    results_file = models_dir / "test_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "execution_time": overall_elapsed,
            "models": all_results
        }, f, indent=2)

    print(f"\n📄 Results saved to: {results_file}")