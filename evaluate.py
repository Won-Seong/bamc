import os
import argparse
import glob
import csv
import torch
from PIL import Image
import statistics
from dotenv import load_dotenv

from helpers.evaluators import LPIPSEvaluator, ArcFaceEvaluator
from helpers.utils import load_config

# Initialize environment
load_dotenv()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate_folder(config_path, output_dir_suffix="", input_dir_override=None):
    config = load_config(config_path)
    if input_dir_override is not None:
        config.experiments.input_dir = input_dir_override
        if not config.experiments.input_dir.endswith("/"):
            config.experiments.input_dir += "/"
    
    # Resolve output directory
    base_out = config.experiments.output_dir
    if base_out.endswith("/"):
        base_out = base_out[:-1]
    
    out_dir = f"{base_out}{output_dir_suffix}/best/"
    task_base_dir = f"{base_out}{output_dir_suffix}/"
    
    if not os.path.exists(out_dir):
        print(f"Directory {out_dir} does not exist. Skipping evaluation.")
        return

    # Gather generated images
    gen_images_paths = glob.glob(os.path.join(out_dir, "best_image_*.png"))
    if not gen_images_paths:
        print(f"No generated images found in {out_dir}.")
        return

    # Load Evaluators (VLM Removed)
    print(f"\n[{out_dir}] Loading Evaluators (ArcFace, LPIPS)...")
    lpips_eval = LPIPSEvaluator(device=DEVICE)
    arcface_eval = ArcFaceEvaluator(device=DEVICE)
    
    results = []

    for gen_path in sorted(gen_images_paths):
        filename = os.path.basename(gen_path)
        parts = filename.split("_")
        task_id = parts[2]
        
        iteration = "Unknown"
        if len(parts) >= 5 and parts[3] == "iter":
            iteration = parts[4]
        
        target_path = None
        for ext in [".png", ".jpg", ".jpeg", ".webp"]:
            cand = os.path.join(config.experiments.input_dir, "images", f"{task_id}_target{ext}")
            if os.path.exists(cand):
                target_path = cand
                break
        if target_path is None:
            print(f"Target for task {task_id} not found. Skipping.")
            continue

        # Find the initial guess image
        initial_guess_paths = glob.glob(os.path.join(task_base_dir, f"task_{task_id}", "**", "best_initial_guess_*.png"), recursive=True)
        if not initial_guess_paths:
            # Fallback for Random Baseline (best_so_far_0000)
            initial_guess_paths = glob.glob(os.path.join(task_base_dir, f"task_{task_id}", "**", "best_so_far_0000_*.png"), recursive=True)
            
        target_img = Image.open(target_path).convert("RGB")
        best_img = Image.open(gen_path).convert("RGB")
        
        has_init = len(initial_guess_paths) > 0
        if has_init:
            init_img = Image.open(initial_guess_paths[0]).convert("RGB")
        else:
            init_img = best_img # Fallback so metrics don't break

        # Evaluate LPIPS
        lpips_eval.set_target(target_img)
        best_lpips = lpips_eval.calculate_distance(target_img, best_img)
        init_lpips = lpips_eval.calculate_distance(target_img, init_img)

        # Evaluate ArcFace
        arcface_eval.set_target(target_img)
        best_arcface = arcface_eval.calculate_distance(target_img, best_img)
        init_arcface = arcface_eval.calculate_distance(target_img, init_img)

        # Calculate Diff and Ratio (per image)
        diff_lpips = init_lpips - best_lpips
        ratio_lpips = (init_lpips - best_lpips) / (init_lpips + 1e-8) * 100
        
        diff_arcface = init_arcface - best_arcface
        ratio_arcface = (init_arcface - best_arcface) / (init_arcface + 1e-8) * 100
        
        results.append({
            "task_id": task_id,
            "best_iter": iteration,
            "init_lpips": init_lpips,
            "best_lpips": best_lpips,
            "diff_lpips": diff_lpips,
            "ratio_lpips": ratio_lpips,
            
            "init_arcface": init_arcface,
            "best_arcface": best_arcface,
            "diff_arcface": diff_arcface,
            "ratio_arcface": ratio_arcface
        })
        
        print(f"Task {task_id} (Best Iter: {iteration}) -> LPIPS Diff: {diff_lpips:+.4f} ({ratio_lpips:+.4f}%), ArcFace Diff: {diff_arcface:+.4f} ({ratio_arcface:+.4f}%)")

    if not results:
        return

    # ---------------------------------------------------------
    # Calculate dataset-level averages for Paper Reporting
    # (Ratio of Averages instead of Average of Ratios)
    # ---------------------------------------------------------
    avg_init_lpips = statistics.mean([r["init_lpips"] for r in results])
    avg_best_lpips = statistics.mean([r["best_lpips"] for r in results])
    dataset_diff_lpips = avg_init_lpips - avg_best_lpips
    dataset_ratio_lpips = (dataset_diff_lpips) / (avg_init_lpips + 1e-8) * 100
    
    avg_init_arcface = statistics.mean([r["init_arcface"] for r in results])
    avg_best_arcface = statistics.mean([r["best_arcface"] for r in results])
    dataset_diff_arcface = avg_init_arcface - avg_best_arcface
    dataset_ratio_arcface = (dataset_diff_arcface) / (avg_init_arcface + 1e-8) * 100

    print("\n" + "="*60)
    print(f"Final Evaluation Summary for: {out_dir}")
    print("="*60)
    print(f"Avg Init LPIPS   : {avg_init_lpips:.4f} | Avg Best LPIPS   : {avg_best_lpips:.4f}")
    print(f"LPIPS Dataset Improvement: Diff = {dataset_diff_lpips:+.4f}, Ratio = {dataset_ratio_lpips:+.4f}%")
    print("-" * 60)
    print(f"Avg Init ArcFace : {avg_init_arcface:.4f} | Avg Best ArcFace : {avg_best_arcface:.4f}")
    print(f"ArcFace Dataset Improvement: Diff = {dataset_diff_arcface:+.4f}, Ratio = {dataset_ratio_arcface:+.4f}%")
    print("="*60 + "\n")

    # Save to CSV
    csv_path = os.path.join(task_base_dir, "final_evaluation.csv")
    fieldnames = [
        "task_id", "best_iter",
        "init_lpips", "best_lpips", "diff_lpips", "ratio_lpips",
        "init_arcface", "best_arcface", "diff_arcface", "ratio_arcface"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
        
        # Write average row (using dataset-level statistics directly)
        writer.writerow({
            "task_id": "AVERAGE",
            "best_iter": "",
            "init_lpips": avg_init_lpips,
            "best_lpips": avg_best_lpips,
            "diff_lpips": dataset_diff_lpips,
            "ratio_lpips": dataset_ratio_lpips,
            
            "init_arcface": avg_init_arcface,
            "best_arcface": avg_best_arcface,
            "diff_arcface": dataset_diff_arcface,
            "ratio_arcface": dataset_ratio_arcface
        })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir_suffix", type=str, default="")
    parser.add_argument("--input_dir", type=str, default=None, help="Override input directory")
    args = parser.parse_args()
    
    # We load config inside evaluate_folder, but we need to pass input_dir override.
    evaluate_folder(args.config, args.output_dir_suffix, args.input_dir)
