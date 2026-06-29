#!/usr/bin/env python3
"""
NPU 상세 벤치마크: Top-1/Top-5 정확도 + 추론 속도(latency/FPS) + 메모리 사용량.

정확도 레이블 구조
  data/imagenet_val/<CLASSIDX>/<image>.jpg
  폴더명(00065) = ImageNet 클래스 인덱스 65  ← Mobilint_samples/resnet50/resnet50.py 기준 동일

실행 전 준비
  cd /home/airlab_compression/Orin_Compression
  source .env/bin/activate

기본 실행 (200 클래스 × 5장 = 1000장, 4개 모델, 결과 data/results/ 에 자동 저장)
  python3 scripts/run_detailed_benchmark.py

tegrastats 병렬 실행 (시스템 RAM·전력·온도 동시 기록, 권장)
  sudo tegrastats --interval 500 &> data/results/tegrastats_$(date +%Y%m%d_%H%M%S).log &
  python3 scripts/run_detailed_benchmark.py
  pkill -f tegrastats

주요 옵션
  --models          DeiT_Tiny_Patch16_224 ViT_Small_Patch16_224   테스트할 모델 선택
  --classes         200                                            샘플링할 클래스 수 (최대 1000)
  --images-per-class  5                                           클래스당 이미지 수
  --timed-runs      10                                             이미지 1장당 타이밍 반복 횟수
  --warmup-runs     3                                              첫 이미지 추론 전 워밍업 횟수
  --infer-mode      single | multi | global4 | global8             NPU 코어 모드 (기본 global8)
  --seed            42                                             샘플링 재현성 시드
  --no-save                                                        결과 파일 저장 안 함

예시
  # 빠른 테스트 (20장)
  python3 scripts/run_detailed_benchmark.py --classes 10 --images-per-class 2 --timed-runs 3

  # 충분한 샘플 (1000장)
  python3 scripts/run_detailed_benchmark.py --classes 200 --images-per-class 5 --timed-runs 10

  # 전체 val set (50,000장, 오래 걸림)
  python3 scripts/run_detailed_benchmark.py --classes 1000 --images-per-class 50

  # 특정 모델만, NPU 단일 코어 모드
  python3 scripts/run_detailed_benchmark.py --models DeiT_Tiny_Patch16_224 --infer-mode single
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from datetime import datetime
from typing import List, Optional

import psutil
import torch

try:
    import mblt_model_zoo.vision as vision
except Exception as e:
    print(f"Failed to import mblt_model_zoo.vision: {e}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def collect_samples(val_dir: str, images_per_class: int, num_classes: int, seed: int):
    """Return list of (image_path, class_idx) sampled from val_dir."""
    class_dirs = sorted(d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d)))
    if num_classes < len(class_dirs):
        random.seed(seed)
        class_dirs = random.sample(class_dirs, num_classes)
        class_dirs.sort()

    samples = []
    for cls_dir in class_dirs:
        cls_idx = int(cls_dir)
        cls_path = os.path.join(val_dir, cls_dir)
        imgs = [f for f in os.listdir(cls_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if not imgs:
            continue
        random.seed(seed + cls_idx)
        chosen = random.sample(imgs, min(images_per_class, len(imgs)))
        for img in chosen:
            samples.append((os.path.join(cls_path, img), cls_idx))

    return samples


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2


# ---------------------------------------------------------------------------
# Per-model benchmark
# ---------------------------------------------------------------------------

def benchmark_model(
    model_name: str,
    samples: list,
    infer_mode: str,
    local_path: str,
    warmup_runs: int,
    timed_runs: int,
) -> Optional[dict]:
    print(f"\n{'='*60}")
    print(f"Model : {model_name}  |  infer_mode={infer_mode}  |  samples={len(samples)}")
    print(f"{'='*60}")

    if not hasattr(vision, model_name):
        print(f"[SKIP] '{model_name}' not found in mblt_model_zoo.vision")
        return None

    model_cls = getattr(vision, model_name)

    # --- Memory before load ---
    mem_before_mb = rss_mb()

    try:
        model = model_cls(infer_mode=infer_mode, local_path=local_path or None)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return None

    mem_after_load_mb = rss_mb()
    model_mem_mb = mem_after_load_mb - mem_before_mb
    print(f"Memory  before load : {mem_before_mb:.1f} MB")
    print(f"Memory  after load  : {mem_after_load_mb:.1f} MB  (model footprint: {model_mem_mb:+.1f} MB)")

    # --- Accuracy + latency measurement ---
    top1_correct = 0
    top5_correct = 0
    latencies_ms = []
    errors = []
    peak_mem_mb = mem_after_load_mb

    for img_path, gt_label in samples:
        try:
            inp = model.preprocess(img_path)

            # Warmup (only on first image)
            if not latencies_ms:
                for _ in range(warmup_runs):
                    _ = model(inp)

            # Timed inference
            t0 = time.perf_counter()
            for _ in range(timed_runs):
                raw = model(inp)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) / timed_runs * 1000)

            # Accuracy
            result = model.postprocess(raw)
            probs = result.output  # tensor [1, 1000]
            top5_idx = torch.topk(probs, 5).indices[0].tolist()
            if top5_idx[0] == gt_label:
                top1_correct += 1
            if gt_label in top5_idx:
                top5_correct += 1

            cur_mem = rss_mb()
            if cur_mem > peak_mem_mb:
                peak_mem_mb = cur_mem

        except Exception as e:
            errors.append(str(e))

    model.dispose()
    mem_after_dispose_mb = rss_mb()

    n = len(latencies_ms)
    if n == 0:
        print("[ERROR] No successful inferences.")
        return None

    import statistics
    avg_ms  = statistics.mean(latencies_ms)
    std_ms  = statistics.stdev(latencies_ms) if n > 1 else 0.0
    min_ms  = min(latencies_ms)
    max_ms  = max(latencies_ms)
    fps     = 1000.0 / avg_ms

    top1_acc = top1_correct / n * 100
    top5_acc = top5_correct / n * 100

    print(f"\n--- Latency ({n} images × {timed_runs} runs) ---")
    print(f"  Avg : {avg_ms:.2f} ms  |  Std : {std_ms:.2f} ms")
    print(f"  Min : {min_ms:.2f} ms  |  Max : {max_ms:.2f} ms")
    print(f"  FPS : {fps:.1f}")
    print(f"\n--- Accuracy ({n} images) ---")
    print(f"  Top-1 : {top1_correct}/{n}  =  {top1_acc:.2f}%")
    print(f"  Top-5 : {top5_correct}/{n}  =  {top5_acc:.2f}%")
    print(f"\n--- Memory ---")
    print(f"  Model footprint : {model_mem_mb:+.1f} MB")
    print(f"  Peak during run : {peak_mem_mb:.1f} MB")
    print(f"  After dispose   : {mem_after_dispose_mb:.1f} MB")
    if errors:
        print(f"\n  Errors ({len(errors)}): {errors[:3]}")

    return {
        "model":           model_name,
        "infer_mode":      infer_mode,
        "num_samples":     n,
        "timed_runs":      timed_runs,
        "warmup_runs":     warmup_runs,
        "top1_acc_pct":    round(top1_acc, 2),
        "top5_acc_pct":    round(top5_acc, 2),
        "top1_correct":    top1_correct,
        "top5_correct":    top5_correct,
        "avg_ms":          round(avg_ms, 2),
        "std_ms":          round(std_ms, 2),
        "min_ms":          round(min_ms, 2),
        "max_ms":          round(max_ms, 2),
        "fps":             round(fps, 1),
        "mem_model_mb":    round(model_mem_mb, 1),
        "mem_peak_mb":     round(peak_mem_mb, 1),
        "mem_after_dispose_mb": round(mem_after_dispose_mb, 1),
        "errors":          len(errors),
    }


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(results: list, results_dir: str, tag: str):
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(results_dir, f"detailed_{tag}_{ts}")

    with open(f"{base}.json", "w") as f:
        json.dump(results, f, indent=2)

    fieldnames = [
        "model", "infer_mode", "num_samples",
        "top1_acc_pct", "top5_acc_pct",
        "avg_ms", "std_ms", "min_ms", "max_ms", "fps",
        "mem_model_mb", "mem_peak_mb", "mem_after_dispose_mb", "errors",
    ]
    with open(f"{base}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved:\n  {base}.json\n  {base}.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: List[str] = None):
    p = argparse.ArgumentParser(description="Detailed NPU benchmark: accuracy + latency + memory")
    p.add_argument("--models", nargs="*", default=[
        "DeiT_Tiny_Patch16_224",
        "ViT_Tiny_Patch16_224",
        "ViT_Small_Patch16_224",
        "DeiT_Small_Patch16_224",
    ])
    p.add_argument("--val-dir",          default="data/imagenet_val")
    p.add_argument("--images-per-class", type=int, default=5,   help="Images sampled per class")
    p.add_argument("--classes",          type=int, default=200,  help="Number of classes to sample (max 1000)")
    p.add_argument("--timed-runs",       type=int, default=10,  help="Timed inference runs per image")
    p.add_argument("--warmup-runs",      type=int, default=3,   help="Warmup runs before timing")
    p.add_argument("--infer-mode",       default="global8",     help="single/multi/global4/global8")
    p.add_argument("--local-path",       default="",            help="Optional local .mxq path")
    p.add_argument("--results-dir",      default="data/results")
    p.add_argument("--seed",             type=int, default=42)
    p.add_argument("--no-save",          action="store_true")
    args = p.parse_args(argv)

    if not os.path.isdir(args.val_dir):
        print(f"[ERROR] val-dir not found: {args.val_dir}")
        sys.exit(1)

    print(f"Collecting samples from {args.val_dir} ...")
    samples = collect_samples(args.val_dir, args.images_per_class, args.classes, args.seed)
    print(f"  → {len(samples)} images across {args.classes} classes (seed={args.seed})")

    all_results = []
    for model_name in args.models:
        r = benchmark_model(
            model_name, samples,
            args.infer_mode, args.local_path,
            args.warmup_runs, args.timed_runs,
        )
        if r:
            all_results.append(r)

    if all_results:
        print(f"\n{'='*60}")
        print(f"{'Model':<28} {'Top-1':>7} {'Top-5':>7} {'Avg ms':>8} {'FPS':>7} {'MemMB':>7}")
        print(f"{'-'*60}")
        for r in all_results:
            print(f"{r['model']:<28} {r['top1_acc_pct']:>6.1f}% {r['top5_acc_pct']:>6.1f}% "
                  f"{r['avg_ms']:>7.2f} {r['fps']:>7.1f} {r['mem_model_mb']:>+7.1f}")

        if not args.no_save:
            save_results(all_results, args.results_dir, args.infer_mode)


if __name__ == "__main__":
    main()
