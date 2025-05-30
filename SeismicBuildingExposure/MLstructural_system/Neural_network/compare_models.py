import json
from pathlib import Path
from omegaconf import OmegaConf

def load_results():
    results = []

    # Recorremos tanto "outputs" como "multirun"
    for root in ["outputs", "multirun"]:
        for log_dir in Path(root).rglob("logs"):
            report_file = log_dir / "classification_report.json"
            config_file = log_dir / "config.yaml"
            if report_file.exists() and config_file.exists():
                try:
                    with open(report_file) as f:
                        report = json.load(f)
                    config = OmegaConf.load(config_file)

                    relative_dir = log_dir.relative_to(root)

                    results.append({
                        "run_path": f"{root}/{relative_dir.parent}",  # ejemplo: multirun/2025-04-17/20-18-47/1
                        "model_name": config.model_config.model_name,
                        "num_epochs": config.model_config.num_epochs,
                        "lr": config.model_config.lr,
                        "accuracy": report["accuracy"],
                        "f1_macro": report["macro avg"]["f1-score"],
                        "f1_weighted": report["weighted avg"]["f1-score"],
                    })
                except Exception as e:
                    print(f"❌ Error loading {log_dir}: {e}")
    return results

def print_sorted(results, key="f1_macro"):
    sorted_results = sorted(results, key=lambda x: x[key], reverse=True)
    print(f"\n{'Model':<15} {'Epochs':<6} {'LR':<8} {'F1 Macro':<10} {'F1 Weighted':<13} {'Accuracy':<9} Run Path")
    print("-" * 80)
    for r in sorted_results:
        print(f"{r['model_name']:<15} {r['num_epochs']:<6} {r['lr']:<8} {r['f1_macro']:<10.4f} {r['f1_weighted']:<13.4f} {r['accuracy']:<9.4f} {r['run_path']}")

if __name__ == "__main__":
    results = load_results()
    print_sorted(results)
