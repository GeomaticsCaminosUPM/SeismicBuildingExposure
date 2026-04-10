"""
Master Script to Train and Evaluate All Models
Runs all model training pipelines in sequence
"""

import subprocess
import sys
import os
from pathlib import Path
import time
from datetime import datetime, timedelta
import json
import shutil

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


def check_model_completed(model_dir):
    if model_dir.name == "AutoGluon":
        model_path = model_dir / f"model_final_{model_dir.name}"
        if not model_path.exists():
            return False
    else:
        model_path = model_dir / f"model_final_{model_dir.name}.pkl"
        if model_dir.name == "RRAE":
            model_path = model_dir / f"model_final_{model_dir.name}.pth"

        if not model_path.exists():
            return False

    report_path = model_dir / "classification_report_test.txt"
    if not report_path.exists():
        return False

    return True


def run_script(script_path, model_name, env):
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

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"✗ Unexpected error after {format_time(elapsed)}")
        return False, elapsed, str(e)


def run_model(model_name, model_dir, config, env):
    print_section(f"Model: {model_name}")

    results = {
        'model': model_name,
        'status': 'running',
        'start_time': datetime.now().isoformat(),
        'scripts': [],
        'total_time': 0
    }

    if config.SKIP_COMPLETED and check_model_completed(model_dir):
        print(f"⏭ Model already completed, skipping...")
        results['status'] = 'skipped'
        return results

    model_cfg = config.MODEL_CONFIG.get(model_name, {
        "scripts": ["1-train_cv.py", "2-train_final.py", "3-test.py"],
        "estimated_time_minutes": 60
    })

    print(f"📊 Estimated time: ~{model_cfg['estimated_time_minutes']} minutes\n")

    for script_name in model_cfg["scripts"]:
        script_path = model_dir / script_name

        if not script_path.exists():
            print(f"⚠ Missing script: {script_name}")
            continue

        success, elapsed, error = run_script(script_path, model_name, env)

        results['scripts'].append({
            'name': script_name,
            'status': 'success' if success else 'failed',
            'time': elapsed,
            'error': error
        })

        results['total_time'] += elapsed

        if not success:
            results['status'] = 'failed'
            results['failed_script'] = script_name
            return results

    results['status'] = 'completed'
    results['end_time'] = datetime.now().isoformat()

    print(f"\n✓ {model_name} completed!")
    print(f"  Total time: {format_time(results['total_time'])}")

    return results


# ============================================================
# Main Runner (CONFIG-INJECTED)
# ============================================================

def run(cfg):
    """
    Entry point called from main.py
    cfg = experiment-specific config module
    """

    train_file_dir = Path(__file__).resolve().parent
    source_models_dir = train_file_dir / "models"

    print_header("AUTOMATED MODEL TRAINING AND EVALUATION")

    project_dir = Path.cwd()
    models_dir = cfg.MODELS_DIR

    # Pass config to subprocess via environment
    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = cfg.MLFLOW_TRACKING_URI
    env["PROJECT_MODELS_DIR"] = cfg.MODELS_DIR

    print(f"Working directory: {project_dir}")
    print(f"Models directory: {models_dir}")
    print(f"Models: {', '.join(cfg.MODELS)}")
    print(f"Skip completed: {cfg.SKIP_COMPLETED}")

    # Ensure models_dir exists
    models_dir.mkdir(parents=True, exist_ok=True)

    # Estimate total time
    total_estimated = sum(
        cfg.MODEL_CONFIG.get(m, {}).get('estimated_time_minutes', 60)
        for m in cfg.MODELS
    )

    print(f"\n📊 Total estimated time: ~{total_estimated} minutes")
    print()

    response = input("Start training all models? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return

    overall_start = time.time()
    all_results = []

    for idx, model_name in enumerate(cfg.MODELS, start=1):
        print_header(f"Model {idx}/{len(cfg.MODELS)}: {model_name}")

        model_dir = models_dir / model_name

        if not model_dir.exists():
            print(f"⚠ Missing model directory: {model_dir}")
            
            source_model_path = source_models_dir / model_name

            if not source_model_path.exists():
                raise FileNotFoundError(f"Source model not found in {source_models_dir}: {model_name}")

            print(f"→ Copying '{model_name}' from default models folder and using default model params...")

            shutil.copytree(source_model_path, model_dir)
            print(f"✓ Copied '{model_name}' successfully")
            

        result = run_model(model_name, model_dir, cfg, env)
        all_results.append(result)

        if result['status'] == 'failed':
            print("⚠ Model failed. Continue? (y/n)")
            if input().lower() != 'y':
                break

    overall_elapsed = time.time() - overall_start

    print_header("EXECUTION SUMMARY")

    print(f"Total time: {format_time(overall_elapsed)}")

    results_file = models_dir / "training_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'execution_time': overall_elapsed,
            'models': all_results
        }, f, indent=2)

    print(f"\n📄 Results saved to: {results_file}")