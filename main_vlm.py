from typing import Dict, Callable
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

from helpers.evaluators import LPIPSEvaluator, Evaluator, VLMEvaluator, ArcFaceEvaluator
from helpers.utils import load_config, decode_latents, set_seed
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
    evaluators: Dict[str, Evaluator],
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
        f"{timestamp}"
    )
    os.makedirs(task_dir, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"Starting [Task {task_id}] - Saving to: {task_dir}")
    print(f"{'='*50}")

    for evaluator in evaluators.values():
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

    # Evaluate initial candidate batch using LPIPS to anchor the Markov chain (x^(0))
    lpips_distances = evaluators["lpips"].calculate_distances_batch(init_imgs)
    cur_idx = lpips_distances.index(min(lpips_distances))

    # Set the selected initial anchor as current chain state x^(0)
    cur_img = init_imgs[cur_idx]

    # Calculate VLM distance for the initial anchor
    reasoning, distance = evaluators["vlm"].calculate_distance(cur_img)

    tracker.init = SampleState(
        distance=distance,
        latent=init_latents[cur_idx].clone(),
        iteration=0
    )

    # Save the initial anchor guess (x^(0))
    cur_img.save(os.path.join(task_dir, f"best_initial_guess_{tracker.init.distance:.4f}.png"))

    torch.save(tracker.init.latent.cpu(), os.path.join(task_dir, f"best_initial_latent_{tracker.init.distance:.4f}.pt"))
    print(f"Initial anchor selected by LPIPS: {min(lpips_distances):.4f} (VLM score: {tracker.init.distance:.4f})\n")
    
    # Free memory from the large initial batch
    del init_latents
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

        # Pairwise preference comparison: current state x^(i) vs proposal x'
        reasoning, batch_score = evaluators["vlm"].calculate_distances_batch(cur_img, proposal_imgs[0])
        arcface_distance = evaluators["arcface"].calculate_distances_batch(proposal_imgs)[0]
        lpips_distance = evaluators["lpips"].calculate_distances_batch(proposal_imgs)[0]

        status = "REJECTED"
        absolute_distance = tracker.cur.distance
        delta_D = 0.0

        if batch_score > 0:
            status = "ACCEPTED"
            
            # If accepted, calculate the absolute score to update tracker
            reasoning_abs, absolute_distance = evaluators["vlm"].calculate_distance(proposal_imgs[0])
            delta_D = absolute_distance - tracker.cur.distance
            
            print(
                f"✅ ACCEPTED (Iter: {iteration+1}/{config.process.num_loop}) \t | "
                f"Score: {absolute_distance:.4f} (was {tracker.cur.distance:.4f})"
            )

            # Update current image for the next comparison
            cur_img = proposal_imgs[0]

            # Update Markov Chain state: x^(i+1) = x'
            tracker.cur = SampleState(
                distance=absolute_distance,
                latent=proposal_latents[0].clone(),
                iteration=iteration+1
            )

            # In greedy VLM acceptance, preferred state becomes the new benchmark best
            tracker.best = tracker.cur

            proposal_imgs[0].save(os.path.join(task_dir, f"iter_{iteration+1:02d}_{absolute_distance:.4f}.png"))

        # Log iteration record
        record = {
            "iteration": iteration + 1,
            "status": status,
            "delta_D": float(delta_D),
            "proposal_distance": float(absolute_distance) if status == "ACCEPTED" else -1.0,
            "candidate_distance": float(absolute_distance) if status == "ACCEPTED" else -1.0,  # backward compatibility
            "current_distance": float(tracker.cur.distance),
            "best_distance": float(tracker.best.distance),
            "arcface_distance": float(arcface_distance),
            "lpips_distance": float(lpips_distance),
            "reasoning": reasoning,
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

    # Define strength function for VLM evaluator
    if getattr(config.process, "strength", None) is not None:
        strength_func = lambda d: config.process.strength
    else:
        strength_func = lambda d: d

    # Load Pipeline & Evaluator ONCE (Saves VRAM and Time)
    evaluators = {
        "vlm": VLMEvaluator(
            device=DEVICE,
            client=client,
            model=config.model.vlm_id,
            prompt=[
                config.prompt.vlm_calculate_distance_prompt,
                config.prompt.vlm_calculate_distances_batch_prompt
            ]
        ),
        "arcface": ArcFaceEvaluator(
            device=DEVICE
        ),
        "lpips": LPIPSEvaluator(
            device=DEVICE
        )
    }

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
                evaluators=evaluators,
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
    parser = argparse.ArgumentParser(description="Run BAMC VLM Preference-Guided Image Generation")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_vlm.yaml",
        help="Path to the config file"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Override input directory"
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
    
    if args.input_dir is not None:
        config.experiments.input_dir = args.input_dir
        if not config.experiments.input_dir.endswith("/"):
            config.experiments.input_dir += "/"
            
    if args.output_dir_suffix is not None:
        base_out = config.experiments.output_dir
        if base_out.endswith("/"):
            base_out = base_out[:-1]
        config.experiments.output_dir = f"{base_out}{args.output_dir_suffix}/"

    if args.strength is not None:
        config.process.strength = args.strength

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
