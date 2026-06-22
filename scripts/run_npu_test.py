#!/usr/bin/env python3
"""cd /home/airlab_compression/Orin_Compression
source .env/bin/activate
export MBLT_MODEL_ZOO_VERBOSE=true
python3 scripts/run_npu_test.py
"""


import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from typing import List, Optional

try:
    import mblt_model_zoo.vision as vision
except Exception as e:
    print(f"Failed to import mblt_model_zoo.vision: {e}")
    sys.exit(1)


def run_one(
    model_name: str,
    image_path: str,
    local_path: str,
    core_mode: str,
    runs: int,
    warmups: int,
) -> Optional[dict]:
    print(f"\n=== Model: {model_name} | local_path={local_path or '(auto)'} | infer_mode={core_mode} ===")

    if not hasattr(vision, model_name):
        available = [x for x in dir(vision) if not x.startswith("_") and x[0].isupper()][:10]
        print(f"Model '{model_name}' not found. Available models: {available}...")
        return None

    model_cls = getattr(vision, model_name)
    result = {
        "model": model_name,
        "core_mode": core_mode,
        "runs": runs,
        "warmups": warmups,
        "image": image_path,
        "status": "error",
        "total_s": None,
        "avg_ms": None,
        "fps": None,
        "error": None,
    }

    model = None
    try:
        model = model_cls(infer_mode=core_mode, local_path=local_path if local_path else None)
        input_img = model.preprocess(image_path)

        for _ in range(warmups):
            _ = model(input_img)

        t0 = time.perf_counter()
        for _ in range(runs):
            _ = model(input_img)
        t1 = time.perf_counter()

        total = t1 - t0
        avg = total / runs if runs > 0 else float("nan")
        fps = runs / total if total > 0 else float("nan")

        print(f"Runs: {runs}, Total: {total:.4f}s, Avg: {avg*1000:.2f}ms, FPS: {fps:.2f}")

        result.update({"status": "ok", "total_s": round(total, 4), "avg_ms": round(avg * 1000, 2), "fps": round(fps, 2)})

    except Exception as e:
        result["error"] = str(e)
        print(f"Error running model: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if model is not None:
            try:
                model.dispose()
            except Exception:
                pass

    return result


def save_results(results: list, results_dir: str, core_mode: str):
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(results_dir, f"benchmark_{core_mode}_{ts}")

    # JSON
    with open(f"{base}.json", "w") as f:
        json.dump(results, f, indent=2)

    # CSV
    fieldnames = ["model", "core_mode", "runs", "warmups", "status", "total_s", "avg_ms", "fps", "error"]
    with open(f"{base}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to:\n  {base}.json\n  {base}.csv")


def main(argv: List[str] = None):
    p = argparse.ArgumentParser(description="Run NPU benchmark for models using mblt_model_zoo.vision")
    p.add_argument("--models", nargs="*", default=[
        "DeiT_Tiny_Patch16_224",
        "ViT_Tiny_Patch16_224",
        "ViT_Small_Patch16_224",
        "DeiT_Small_Patch16_224",
    ], help="Model class names to test")
    p.add_argument("--image", default="data/imagenet_val/00001/0108333423881763.jpg", help="Image path for inference")
    p.add_argument("--local-path", default="", help="Optional local .mxq file path to use for all models")
    p.add_argument("--core-mode", default="global8", help="NPU core mode: single/multi/global4/global8")
    p.add_argument("--runs", type=int, default=5, help="Number of timed runs")
    p.add_argument("--warmups", type=int, default=1, help="Number of warmup runs")
    p.add_argument("--results-dir", default="data/results", help="Directory to save benchmark results")
    p.add_argument("--no-save", action="store_true", help="Skip saving results to disk")

    args = p.parse_args(argv)

    if not os.path.exists(args.image):
        print(f"Warning: image '{args.image}' not found. Update --image to a valid image path.")

    results = []
    for m in args.models:
        r = run_one(m, args.image, args.local_path, args.core_mode, args.runs, args.warmups)
        if r is not None:
            results.append(r)

    if results and not args.no_save:
        save_results(results, args.results_dir, args.core_mode)


if __name__ == "__main__":
    main()
