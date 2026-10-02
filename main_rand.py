import os
import warnings
from datetime import datetime
import argparse
import json
import csv
import torch
from PIL import Image
from diffusers import ZImageImg2ImgPipeline
from google import genai
from dotenv import load_dotenv

from helpers.evaluators import LPIPSEvaluator, Evaluator, ArcFaceEvaluator, VLMEvaluator
from helpers.utils import load_config, decode_latents, set_seed
from helpers.llm import call_vlm, call_image_generator
from helpers.models import SampleState

# Configuration
load_dotenv()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GENERATOR = torch.Generator(DEVICE)
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# ==========================================
# Task Execution Logic
# ==========================================
def run_single_task(
    task_id: int,
    pipeline: ZImageImg2ImgPipeline,
    evaluator: Evaluator,
    target: Image.Image,
    config
):
    """Handles the Random Sampling loop for a single task and saves it to its own folder."""
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    task_dir = os.path.join(
        config.experiments.output_dir,
        f"task_{task_id}",
        f"evaluator_{evaluator.name}",
        f"{timestamp}"
    )
    os.makedirs(task_dir, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"Starting Random Sampling [Task {task_id}] - Saving to: {task_dir}")
    print(f"{'='*50}")

    evaluator.set_target(target)

    # ---------------------------------------------------------
    # Step 1: VLM Prompt Extraction
    # ---------------------------------------------------------
    vlm_prompt = call_vlm(
        client=client,
        imgs=[target],
        model=config.model.vlm_id,
        prompt=config.prompt.vlm_describe_prompt,
        temperature=0.0
    )
    if config.experiments.type == "faces":
        vlm_prompt = f"A strictly front-facing portrait, looking directly at camera; {vlm_prompt}"

    prompt_path = os.path.join(task_dir, "vlm_prompt.txt")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(vlm_prompt)

    # ---------------------------------------------------------
    # Step 2: Random Sampling Loop
    # ---------------------------------------------------------
    rand_history = []
    
    global_best_state = None

    import math
    batch_size = config.process.num_samples_init
    num_iterations = math.ceil(config.process.num_samples_total / batch_size)

    for iteration in range(num_iterations):
        # Generate independent sample using strength=1.0 and pure noise
        sample_latents = pipeline(
            prompt=vlm_prompt,
            negative_prompt="",
            image=torch.randn(1, 3, 512, 512, generator=GENERATOR, device=DEVICE, dtype=DTYPE),
            num_inference_steps=config.process.num_inference_steps,
            guidance_scale=config.process.guidance_scale,
            height=512,
            width=512,
            num_images_per_prompt=config.process.num_samples_init,
            cfg_normalization=False,
            strength=1.0,  # Full generation, no dependence on input
            generator=GENERATOR,
            output_type="latent"
        ).images.to(DTYPE)

        sample_imgs = decode_latents(pipeline, sample_latents)

        distances = evaluator.calculate_distances_batch(sample_imgs)
        best_idx = distances.index(min(distances))
        cur_best_distance = distances[best_idx]

        is_new_best = False
        if global_best_state is None or cur_best_distance < global_best_state.distance:
            is_new_best = True
            global_best_state = SampleState(
                distance=cur_best_distance,
                latent=sample_latents[best_idx].clone(),
                iteration=iteration+1
            )
            # Save the new best image
            sample_imgs[best_idx].save(os.path.join(task_dir, f"best_so_far_{iteration:04d}_{cur_best_distance:.4f}.png"))

        if is_new_best:
            print(
                f"🌟 NEW BEST (Iter: {iteration+1}/{num_iterations}) \t | "
                f"Distance: {cur_best_distance:.4f}"
            )
        else:
            print(
                f"🔄 (Iter: {iteration+1}/{num_iterations}) \t | "
                f"Distance: {cur_best_distance:.4f} \t | "
                f"Best so far: {global_best_state.distance:.4f}"
            )

        # Log iteration record
        record = {
            "iteration": iteration + 1,
            "sample_distance": float(cur_best_distance),
            "candidate_distance": float(cur_best_distance),  # backward compatibility
            "best_distance": float(global_best_state.distance),
            "is_new_best": is_new_best
        }
        rand_history.append(record)

        # Incrementally save to CSV
        csv_path = os.path.join(task_dir, "rand_history.csv")
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)

    # Save tracking history to JSON
    history_path = os.path.join(task_dir, "rand_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(rand_history, f, indent=4)
        
    return global_best_state

# ==========================================
# Main Execution
# ==========================================
@torch.no_grad()
def main(config):
    # Set seed for reproducibility
    set_seed(config.experiments.seed, GENERATOR)

    # Directories
    os.makedirs(os.path.join(config.experiments.input_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(config.experiments.input_dir, "prompts"), exist_ok=True)
    os.makedirs(config.experiments.output_dir, exist_ok=True)
    os.makedirs(os.path.join(config.experiments.output_dir, "best"), exist_ok=True)
    os.makedirs(os.path.join(config.experiments.output_dir, "best-pair"), exist_ok=True)

    # Load Evaluator
    evaluator = None
    match config.evaluator.name:
        case "ArcFace":
            evaluator = ArcFaceEvaluator(device=DEVICE)
        case "LPIPS":
            evaluator = LPIPSEvaluator(device=DEVICE)
        case "VLM":
            evaluator = VLMEvaluator(
                device=DEVICE,
                client=client,
                model=config.model.vlm_id,
                prompt=config.prompt.vlm_evaluate_prompt
            )
        case _:
            raise ValueError(f"Unknown evaluator: {config.evaluator.name}")

    # Load Diffusion Model
    pipeline = ZImageImg2ImgPipeline.from_pretrained(
        config.model.model_id, torch_dtype=DTYPE).to(DEVICE)
    pipeline.set_progress_bar_config(disable=True)
    pipeline.vae.to(dtype=DTYPE)

    # Generate targets if needed
    num_files = len(os.listdir(os.path.join(config.experiments.input_dir, "images/")))
    if num_files < config.experiments.num_targets:
        print(f"Found {num_files} images, but {config.experiments.num_targets} tasks are requested.")
        print(f"Generating {config.experiments.num_targets - num_files} more images...")
        for i in range(num_files, config.experiments.num_targets):
            prompt = call_vlm(
                client=client,
                imgs=[],
                model=config.model.vlm_id,
                prompt=config.prompt.vlm_generate_prompt,
                temperature=2.0
            )
            if config.model.image_generator_id.startswith("gpt"):
                target = call_image_generator(
                    client=client,
                    model=config.model.image_generator_id,
                    prompt=prompt
                )
            else:
                target_latents = pipeline(
                    prompt=prompt,
                    negative_prompt="",
                    image=torch.randn(1, 3, 512, 512, generator=GENERATOR, device=DEVICE, dtype=DTYPE),
                    num_inference_steps=config.process.num_inference_steps,
                    guidance_scale=config.process.guidance_scale,
                    height=512,
                    width=512,
                    cfg_normalization=False,
                    num_images_per_prompt=1,
                    strength=1,
                    generator=GENERATOR,
                    output_type="latent"
                ).images.to(DTYPE)
                target = decode_latents(pipeline, target_latents)[0]

            target.save(os.path.join(config.experiments.input_dir, "images", f"{i}_target.png"))
            with open(os.path.join(config.experiments.input_dir, "prompts", f"{i}_prompt.txt"), "w", encoding="utf-8") as f:
                f.write(prompt)

    extensions = [".png", ".jpg", ".jpeg"]
    for task_id in range(config.experiments.num_targets):
        baseline_path = os.path.join(config.experiments.input_dir, "images", f"{task_id}_target")
        try:
            target = None
            for ext in extensions:
                file_path = baseline_path + ext
                if os.path.exists(file_path):
                    target = Image.open(file_path).convert("RGB")
                    break
            if target is None:
                raise FileNotFoundError(f"No target image found for task {task_id}")

            best_state = run_single_task(
                task_id=task_id,
                target=target,
                pipeline=pipeline,
                evaluator=evaluator,
                config=config
            )

            # Save the best latents and image with distance
            if best_state is not None:
                torch.save(
                    best_state.latent.cpu(),
                    os.path.join(
                        config.experiments.output_dir,
                        "best/",
                        f"best_latent_{task_id}_iter_{best_state.iteration}_{best_state.distance:.4f}.pt"
                    )
                )
                best_img = decode_latents(
                    pipeline,
                    best_state.latent.to(device=DEVICE, dtype=DTYPE)
                )[0]
                
                best_img.save(
                    os.path.join(
                        config.experiments.output_dir,
                        "best/",
                        f"best_image_{task_id}_iter_{best_state.iteration}_{best_state.distance:.4f}.png"
                    )
                )

                # Create a side-by-side pair (Target | Best Generated)
                target_resized = target.resize(best_img.size)
                pair_img = Image.new("RGB", (best_img.width * 2, best_img.height))
                pair_img.paste(target_resized, (0, 0))
                pair_img.paste(best_img, (best_img.width, 0))
                pair_img.save(
                    os.path.join(
                        config.experiments.output_dir,
                        "best-pair/",
                        f"pair_{task_id}_iter_{best_state.iteration}_{best_state.distance:.4f}.png"
                    )
                )

                print(f"Best image, pair, and latent saved for Task {task_id}")

        except Exception as e:
            print(f"❌ Error occurred in Task {task_id}: {e}")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Random Sampling Image Generation")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_z_target_lpips.yaml",
        help="Path to the config file"
    )
    parser.add_argument(
        "--num_samples_total",
        type=int,
        default=None,
        help="Override total number of samples for Random Sampling"
    )
    parser.add_argument(
        "--output_dir_suffix",
        type=str,
        default="_rand",
        help="Suffix to add to output directory to separate from MCMC results"
    )
    args = parser.parse_args()
    config = load_config(args.config)

    # Allow using MCMC configs seamlessly by providing fallbacks
    if args.num_samples_total is not None:
        config.process.num_samples_total = args.num_samples_total
    elif "num_samples_total" not in config.process:
        config.process.num_samples_total = config.process.get("num_loop", 5000) + config.process.get("num_samples_init", 32)
        
    if "guidance_scale" not in config.process:
        config.process.guidance_scale = config.process.get("guidance_scale_in_loop", 5.0)
        
    if "batch_size" not in config.process:
        config.process.batch_size = 1

    if args.output_dir_suffix:
        out_dir = config.experiments.output_dir
        if out_dir.endswith("/"):
            out_dir = out_dir[:-1]
        config.experiments.output_dir = f"{out_dir}{args.output_dir_suffix}/"

    if config.model.model_id == "Tongyi-MAI/Z-Image-Turbo":
        config.process.guidance_scale = 0.0

    warnings.filterwarnings(
        "ignore",
        message="Passing `image` as torch tensor.*",
        category=FutureWarning
    )

    main(config=config)
