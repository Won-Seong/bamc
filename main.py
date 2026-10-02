import os
import warnings
from datetime import datetime
import random
import argparse
import json
import csv
import torch
from PIL import Image
from diffusers import ZImageImg2ImgPipeline
from typing import Callable
from google import genai
from dotenv import load_dotenv

from helpers.evaluators import LPIPSEvaluator, Evaluator, VLMEvaluator, ArcFaceEvaluator
from helpers.utils import load_config, barker_acceptance_prob, metropolis_acceptance_prob, hill_climb_acceptance_prob, decode_latents, set_seed
from helpers.llm import call_vlm, call_image_generator
from helpers.models import MCMCTracker, SampleState

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
    tracker: MCMCTracker,
    target: Image.Image,
    strength_func: Callable,
    config
):
    """Handles the MCMC loop for a single task and saves it to its own folder."""
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    task_dir = os.path.join(
        config.experiments.output_dir,
        f"task_{task_id}",
        f"evaluator_{evaluator.name}",
        f"{timestamp}"
    )
    os.makedirs(task_dir, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"Starting [Task {task_id}] - Saving to: {task_dir}")
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
    # If the task is "faces", add the strictly front-facing portrait constraint
    if config.experiments.type == "faces":
        vlm_prompt = f"A strictly front-facing portrait, looking directly at camera; {vlm_prompt}"

    prompt_path = os.path.join(task_dir, "vlm_prompt.txt")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(vlm_prompt)

    # ---------------------------------------------------------
    # Step 2: Initial Anchor Generation (x^(0))
    # ---------------------------------------------------------
    init_latents = pipeline(
        prompt=vlm_prompt,
        negative_prompt="",
        image=torch.randn(1, 3, 512, 512, generator=GENERATOR, device=DEVICE, dtype=DTYPE),
        num_inference_steps=config.process.num_inference_steps,
        guidance_scale=config.process.guidance_scale_init,
        height=512,
        width=512,
        cfg_normalization=False,
        num_images_per_prompt=config.process.num_samples_init,
        strength=1,
        generator=GENERATOR,
        output_type="latent"
    ).images.to(DTYPE)

    init_imgs = decode_latents(pipeline, init_latents)

    # Evaluate initial candidate batch to anchor the Markov chain (x^(0))
    distances = evaluator.calculate_distances_batch(init_imgs)
    cur_idx = distances.index(min(distances))

    tracker.init = SampleState(
        distance=distances[cur_idx],
        latent=init_latents[cur_idx].clone(),
        iteration=0
    )

    # Save the initial anchor guess (x^(0))
    cur_img = init_imgs[cur_idx]
    cur_img.save(os.path.join(task_dir, f"best_initial_guess_{tracker.init.distance:.4f}.png"))

    torch.save(tracker.init.latent.cpu(), os.path.join(task_dir, f"best_initial_latent_{tracker.init.distance:.4f}.pt"))
    print(f"Initial anchor selected: {tracker.init.distance:.4f}\n")
    
    # Free memory from the large initial batch
    del init_latents
    del init_imgs
    torch.cuda.empty_cache()

    tracker.cur = tracker.init
    tracker.cur_best = tracker.init
    tracker.best = tracker.init

    # ---------------------------------------------------------
    # Step 3: BAMC Markov Chain Iteration Loop (x^(i) -> proposal x' -> x^(i+1))
    # ---------------------------------------------------------
    mcmc_history = []

    for iteration in range(config.process.num_loop):
        # Generate diffusion proposal x' from current chain state x^(i)
        proposal_latents = pipeline(
            prompt=vlm_prompt,
            negative_prompt="",
            image=tracker.cur.latent,
            num_inference_steps=config.process.num_inference_steps,
            guidance_scale=config.process.guidance_scale_in_loop,
            height=512,
            width=512,
            num_images_per_prompt=config.process.num_samples_in_loop,
            cfg_normalization=False,
            strength=max(0.3, min(1.0, strength_func(tracker.cur.distance))),
            generator=GENERATOR,
            output_type="latent"
        ).images.to(DTYPE)

        proposal_imgs = decode_latents(pipeline, proposal_latents)

        distances = evaluator.calculate_distances_batch(proposal_imgs)
        best_idx = distances.index(min(distances))
        proposal_distance = distances[best_idx]

        # delta_D = D(x') - D(x)
        delta_D = proposal_distance - tracker.cur.distance
        # Calculate acceptance probability
        acceptance_prob_func = {
            "barker": barker_acceptance_prob,
            "metropolis": metropolis_acceptance_prob,
            "hill": hill_climb_acceptance_prob
        }[config.process.acceptance_prob_func]
        p_accept = acceptance_prob_func(delta_D, config.process.temperature)
        rand_val = random.random()

        status = "REJECTED"
        if rand_val < p_accept:
            if delta_D < 0:
                status = "ACCEPTED"
                print(
                    f"✅ ACCEPTED (Iter: {iteration+1}/{config.process.num_loop}) \t | "
                    f"Prob: {p_accept:.3f} \t | "
                    f"Improved: {tracker.cur.distance:.4f} -> {proposal_distance:.4f} \t | "
                    f"Best so far: {tracker.best.distance:.4f}"
                )
            else:
                status = "ACCEPTED WORSE"
                print(
                    f"🔄 ACCEPTED WORSE (Iter: {iteration+1}/{config.process.num_loop}) \t | "
                    f"Prob: {p_accept:.3f} \t | "
                    f"Exploring: {tracker.cur.distance:.4f} -> {proposal_distance:.4f} \t | "
                    f"Best so far: {tracker.best.distance:.4f}"
                )

            # Update Markov Chain state: x^(i+1) = x'
            tracker.cur = SampleState(
                distance=proposal_distance,
                latent=proposal_latents[best_idx].clone(),
                iteration=iteration+1
            )

            if proposal_distance < tracker.best.distance:
                tracker.best = tracker.cur

            proposal_imgs[best_idx].save(os.path.join(task_dir, f"iter_{iteration:02d}_{proposal_distance:.4f}.png"))

        # Log iteration record
        record = {
            "iteration": iteration + 1,
            "status": status,
            "p_accept": float(p_accept),
            "rand_val": float(rand_val),
            "delta_D": float(delta_D),
            "proposal_distance": float(proposal_distance),
            "candidate_distance": float(proposal_distance),  # backward compatibility
            "current_distance": float(tracker.cur.distance),
            "best_distance": float(tracker.best.distance)
        }
        mcmc_history.append(record)

        # Incrementally save to CSV
        csv_path = os.path.join(task_dir, "mcmc_history.csv")
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)

    # Save tracking history to JSON
    history_path = os.path.join(task_dir, "mcmc_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(mcmc_history, f, indent=4)

# ==========================================
# Main Execution
# ==========================================
@torch.no_grad()
def main(config):
    """
    Create a folder for each task, and run the MCMC loop for each task.
    """
    # Set seed for reproducibility
    set_seed(config.experiments.seed, GENERATOR)

    # ========================
    # Directories
    # ========================
    # Create the input directory if it doesn't exist.
    os.makedirs(os.path.join(config.experiments.input_dir, "images"), exist_ok=True)

    # Create the input prompt directory if it doesn't exist.
    os.makedirs(os.path.join(config.experiments.input_dir, "prompts"), exist_ok=True)

    # Create the base output directory
    os.makedirs(config.experiments.output_dir, exist_ok=True)

    # Create the directory for saving best results
    os.makedirs(os.path.join(config.experiments.output_dir, "best"), exist_ok=True)
    os.makedirs(os.path.join(config.experiments.output_dir, "best-pair"), exist_ok=True)

    # Define strength function (Identity)
    strength_func = lambda d: d

    # Load Pipeline & Evaluator ONCE (Saves VRAM and Time)
    evaluator = None
    match config.evaluator.name:
        case "ArcFace":
            evaluator = ArcFaceEvaluator(
                device=DEVICE
            )
        case "LPIPS":
            evaluator = LPIPSEvaluator(
                device=DEVICE
            )
        case "VLM":
            evaluator = VLMEvaluator(
                device=DEVICE,
                client=client,
                model=config.model.vlm_id,
                prompt=config.prompt.vlm_evaluate_prompt
            )
            strength_func = lambda d: 0.6
        case _:
            raise ValueError(f"Unknown evaluator: {config.evaluator.name}")

    if hasattr(config.process, "strength") and config.process.strength is not None:
        strength_val = float(config.process.strength)
        strength_func = lambda d: strength_val

    # Load Diffusion Model
    pipeline = ZImageImg2ImgPipeline.from_pretrained(
        config.model.model_id, torch_dtype=DTYPE).to(DEVICE)
    pipeline.set_progress_bar_config(disable=True) # No progress bar
    pipeline.vae.to(dtype=DTYPE)

    # If the number of files is less than the number of tasks,
    # Generate images with OpenAI API.
    num_files = len(os.listdir(os.path.join(config.experiments.input_dir, "images/")))
    if num_files < config.experiments.num_targets:
        print(f"Found {num_files} images, but {config.experiments.num_targets} tasks are requested.")
        print(f"Generating {config.experiments.num_targets - num_files} more images...")
        for i in range(num_files, config.experiments.num_targets):
            prompt = None
            current_prompt_path = os.path.join(config.experiments.input_dir, "prompts", f"{i}_prompt.txt")
            
            # Check current prompt directory
            if os.path.exists(current_prompt_path):
                with open(current_prompt_path, "r", encoding="utf-8") as f:
                    prompt = f.read().strip()
                    
            # Check fallback directory (e.g., z-image)
            if not prompt:
                fallback_path = current_prompt_path.replace("gemini", "z-image").replace("gpt", "z-image")
                if os.path.exists(fallback_path):
                    with open(fallback_path, "r", encoding="utf-8") as f:
                        prompt = f.read().strip()

            if not prompt:
                print(f"Generating new prompt for target {i}...")
                prompt = call_vlm(
                    client=client,
                    imgs=[],
                    model=config.model.vlm_id,
                    prompt=config.prompt.vlm_generate_prompt,
                    temperature=2.0
                )
            else:
                print(f"Using existing prompt for target {i}...")
            if config.model.image_generator_id.startswith("gpt") or config.model.image_generator_id.startswith("imagen"):
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
                    guidance_scale=config.process.guidance_scale_init,
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

    # Loop through the predefined number of tasks
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

            tracker = MCMCTracker()

            run_single_task(
                task_id=task_id,
                target=target,
                tracker=tracker,
                pipeline=pipeline,
                evaluator=evaluator,
                strength_func=strength_func,
                config=config
            )

            # Save the best latents and image with distance
            torch.save(
                tracker.best.latent.cpu(),
                os.path.join(
                    config.experiments.output_dir,
                    "best/",
                    f"best_latent_{task_id}_iter_{tracker.best.iteration}_{tracker.best.distance:.4f}.pt"
                )
            )
            best_img = decode_latents(
                pipeline,
                tracker.best.latent.to(device=DEVICE, dtype=DTYPE)
            )[0]
            
            best_img.save(
                os.path.join(
                    config.experiments.output_dir,
                    "best/",
                    f"best_image_{task_id}_iter_{tracker.best.iteration}_{tracker.best.distance:.4f}.png"
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
                    f"pair_{task_id}_iter_{tracker.best.iteration}_{tracker.best.distance:.4f}.png"
                )
            )

            print(f"Best image, pair, and latent saved for Task {task_id}")

        except Exception as e:
            print(f"❌ Error occurred in Task {task_id}: {e}")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BAMC (Barker-Aligned Monte Carlo) Image Generation")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_z_target_lpips.yaml",
        help="Path to the config file"
    )
    parser.add_argument(
        "--acceptance_prob_func",
        type=str,
        default=None,
        help="Override acceptance probability function (e.g. barker, metropolis, hill)"
    )
    parser.add_argument(
        "--num_loop",
        type=int,
        default=None,
        help="Override number of loops"
    )
    parser.add_argument(
        "--num_samples_init",
        type=int,
        default=None,
        help="Override number of initial samples"
    )
    parser.add_argument(
        "--output_dir_suffix",
        type=str,
        default=None,
        help="Suffix to append to the output directory"
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=None,
        help="Override the generation strength in loop"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    
    # Apply CLI overrides if provided
    if args.acceptance_prob_func is not None:
        config.process.acceptance_prob_func = args.acceptance_prob_func
    if args.num_loop is not None:
        config.process.num_loop = args.num_loop
    if args.num_samples_init is not None:
        config.process.num_samples_init = args.num_samples_init
    if args.strength is not None:
        config.process.strength = args.strength

    if args.output_dir_suffix:
        out_dir = config.experiments.output_dir
        if out_dir.endswith("/"):
            out_dir = out_dir[:-1]
        config.experiments.output_dir = f"{out_dir}{args.output_dir_suffix}/"

    if config.model.model_id == "Tongyi-MAI/Z-Image-Turbo":
        # Turbo version doesn't support guidance scale
        config.process.guidance_scale_init = 0.0
        config.process.guidance_scale_in_loop = 0.0

    # Hide FutureWarnings from Diffusers library
    warnings.filterwarnings(
        "ignore",
        message="Passing `image` as torch tensor.*",
        category=FutureWarning
    )

    main(config=config)
