import sys
import warnings
import logging
warnings.filterwarnings("ignore")

import transformers
transformers.logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)

# Pre-populate dummy __path__ for fast processor alias modules
# to prevent Streamlit file watcher from triggering transformers logger warnings.
for _name, _mod in list(sys.modules.items()):
    if _name.startswith("transformers.models.") and _name.endswith("_fast"):
        _mod.__path__ = []
import os
import io
import csv
import json
from datetime import datetime
import torch
from PIL import Image
import random
import streamlit as st
from dotenv import load_dotenv

from diffusers import ZImageImg2ImgPipeline
from helpers.utils import load_config, decode_latents
from helpers.models import MCMCTracker, SampleState
from helpers.evaluators import LPIPSEvaluator, ArcFaceEvaluator

# ==========================================
# 1. Setup & Environment
# ==========================================
st.set_page_config(
    page_title="BAMC: Preference-Guided Composite Studio",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for clean, professional studio UI
st.markdown("""
<style>
    .main-title {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
    }
    .sub-title {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        font-size: 0.95rem;
        color: #555e68;
        margin-bottom: 1.5rem;
        line-height: 1.5;
    }
    .metric-box {
        background-color: #f8f9fa;
        border-radius: 6px;
        padding: 12px;
        border: 1px solid #dee2e6;
        text-align: center;
    }
    .step-badge {
        display: inline-block;
        background-color: #1e293b;
        color: #f8fafc;
        padding: 3px 10px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.78rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .stButton>button {
        border-radius: 4px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

load_dotenv()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

@st.cache_resource
def load_models(config_path="configs/config_app.yaml"):
    config = load_config(config_path)
    if config.model.model_id == "Tongyi-MAI/Z-Image-Turbo":
        config.process.guidance_scale_init = 0.0
        config.process.guidance_scale_in_loop = 0.0

    pipeline = ZImageImg2ImgPipeline.from_pretrained(
        config.model.model_id, torch_dtype=DTYPE).to(DEVICE)
    pipeline.set_progress_bar_config(disable=True)
    pipeline.vae.to(dtype=DTYPE)
    generator = torch.Generator(DEVICE).manual_seed(config.experiments.seed)
    return config, pipeline, generator

config, pipeline, GENERATOR = load_models()

# ==========================================
# 2. Session State Initialization
# ==========================================
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "montage"  # "montage" (Real-world) or "benchmark" (Evaluation)
if "step" not in st.session_state:
    st.session_state.step = "init_prompt"
if "task_dir" not in st.session_state:
    st.session_state.task_dir = None
if "target_img" not in st.session_state:
    st.session_state.target_img = None
if "target_prompt" not in st.session_state:
    st.session_state.target_prompt = ""
if "human_prompt" not in st.session_state:
    st.session_state.human_prompt = ""
if "tracker" not in st.session_state:
    st.session_state.tracker = MCMCTracker()
if "iteration" not in st.session_state:
    st.session_state.iteration = 0
if "mcmc_history" not in st.session_state:
    st.session_state.mcmc_history = []
if "accepted_timeline" not in st.session_state:
    st.session_state.accepted_timeline = []  # List of dicts: {"iter": int, "img": Image, "label": str}
if "current_strength" not in st.session_state:
    st.session_state.current_strength = 0.35
if "show_target_in_loop" not in st.session_state:
    st.session_state.show_target_in_loop = True

# ==========================================
# 3. Helper Functions
# ==========================================
def image_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def init_task_directory(mode_name: str):
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    task_dir = os.path.join(config.experiments.output_dir, mode_name, timestamp)
    os.makedirs(task_dir, exist_ok=True)
    st.session_state.task_dir = task_dir
    return task_dir

def handle_decision(idx: int, status: str):
    rec = {
        "iteration": st.session_state.iteration + 1,
        "status": status,
        "strength_used": float(st.session_state.current_strength),
        "selected_candidate_idx": idx,
        "prompt": st.session_state.human_prompt
    }
    st.session_state.mcmc_history.append(rec)

    # Save to CSV
    csv_path = os.path.join(st.session_state.task_dir, "mcmc_history.csv")
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rec.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(rec)

    # Save to JSON
    json_path = os.path.join(st.session_state.task_dir, "mcmc_history.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(st.session_state.mcmc_history, f, indent=4)

    if status == "ACCEPTED":
        cand_latent = st.session_state.candidate_latents[idx]
        st.session_state.tracker.cur = SampleState(
            distance=1.0,
            latent=cand_latent.clone(),
            iteration=st.session_state.iteration + 1
        )

        accepted_img = st.session_state.candidate_imgs[idx]
        st.session_state.accepted_timeline.append({
            "iter": st.session_state.iteration + 1,
            "img": accepted_img,
            "label": f"Step {st.session_state.iteration + 1}"
        })

        # Save accepted image
        if st.session_state.app_mode == "benchmark" and "lpips_eval" in st.session_state and st.session_state.lpips_eval is not None:
            lpips_dists = st.session_state.lpips_eval.calculate_distances_batch([accepted_img])
            lpips = lpips_dists[0]
            
            # In benchmark mode, track true best-so-far by objective metric (LPIPS)
            if st.session_state.tracker.best is None or lpips < st.session_state.tracker.best.distance:
                st.session_state.tracker.best = SampleState(
                    distance=lpips,
                    latent=cand_latent.clone(),
                    iteration=st.session_state.iteration + 1
                )
            filename = f"iter_{st.session_state.iteration+1:02d}_LPIPS_{lpips:.3f}.png"
        else:
            filename = f"iter_{st.session_state.iteration+1:02d}_accepted.png"
            
        accepted_img.save(os.path.join(st.session_state.task_dir, filename))

    st.session_state.iteration += 1

    if st.session_state.iteration >= config.process.num_loop:
        st.session_state.step = "finish"
    else:
        st.session_state.step = "loop"
    st.rerun()

# ==========================================
# 4. Sidebar Controls
# ==========================================
with st.sidebar:
    st.markdown("### Session Controls")
    
    # Mode selection
    mode_options = {
        "montage": "Facial Composite from Memory (Mental Image)",
        "benchmark": "Benchmark Evaluation (Reference Target Photo)"
    }
    
    selected_mode = st.radio(
        "Application Mode",
        options=list(mode_options.keys()),
        format_func=lambda x: mode_options[x],
        index=0 if st.session_state.app_mode == "montage" else 1,
        help="Select whether you are generating a composite from user preference (mental target) or running an evaluation against a known reference target image."
    )

    if selected_mode != st.session_state.app_mode:
        st.session_state.app_mode = selected_mode
        st.session_state.step = "init_prompt" if selected_mode == "montage" else "target_gen"
        st.session_state.target_img = None
        st.session_state.task_dir = None
        st.session_state.iteration = 0
        st.session_state.mcmc_history = []
        st.session_state.accepted_timeline = []
        st.rerun()

    if st.session_state.step in ["loop", "loop_eval", "finish"]:
        st.markdown("---")
        st.markdown("### Progress Tracker")
        st.markdown(f"- **Current Step**: `{st.session_state.iteration} / {config.process.num_loop}`")
        num_accepted = sum(1 for h in st.session_state.mcmc_history if h["status"] == "ACCEPTED")
        st.markdown(f"- **Accepted Improvements**: `{num_accepted}`")
        st.markdown(f"- **Variation Level**: `{st.session_state.current_strength:.2f}`")

    st.markdown("---")
    if st.button("Start New Session", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    with st.expander("⚙️ System & Technical Specs", expanded=False):
        st.markdown(f"- **Device**: `{DEVICE.upper()}`")
        st.markdown(f"- **Precision**: `{str(DTYPE).split('.')[-1]}`")
        st.markdown(f"- **Diffusion Model**: `{config.model.model_id.split('/')[-1]}`")
        st.markdown(f"- **Max Steps**: `{config.process.num_loop}`")
        st.markdown("- **Search Algorithm**: `BAMC (Barker-Aligned MCMC)`")

    st.markdown("---")
    st.caption("⚖️ **Disclaimer**: Research demonstration only. Not intended for biometric identification or real-world forensic/investigative decision-making.")

# ==========================================
# 5. UI Header
# ==========================================
st.markdown('<div class="main-title">AI Facial Composite Studio</div>', unsafe_allow_html=True)
st.caption("Powered by **BAMC (Barker-Aligned Monte Carlo)** | Official Research Demo")
if st.session_state.app_mode == "montage":
    st.markdown('<div class="sub-title">Reconstruct facial composites from memory through interactive visual preference feedback.</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="sub-title">Interactive facial fidelity evaluation against a reference target photograph.</div>', unsafe_allow_html=True)

# Collapsible Guides: General User Guide & Academic Methodology
with st.expander("💡 How to Use This App (Quick Guide)", expanded=False):
    st.markdown("""
    1. **Describe**: Enter what you remember about the person's face to create initial options.
    2. **Choose Starting Face**: Select the face that best matches the general facial structure.
    3. **Refine with Variations**: In each step, the AI generates a subtle variation. Compare it against the current face and choose whichever looks closer to your memory.
    4. **Finalize**: When you are satisfied with the composite, click **Finish** to view and download the result.
    """)

with st.expander("🔬 Academic & Research Methodology (BAMC Algorithm)", expanded=False):
    st.markdown(r"""
    * **Algorithm Overview**:  
      BAMC (Barker-Aligned Monte Carlo) formulates preference-guided generation as a
      Markov Chain Monte Carlo (MCMC) search process using a pretrained diffusion or
      flow model as the proposal mechanism.

    * **State & Proposal**:  
      Starting from the current state $x^{(i)}$, BAMC perturbs it and then
      generates a proposal $x'^{(i+1)}$ through the pretrained generative model:

      $$
      x^{(i)}
      \rightarrow
      x_{t_i}^{(i)}
      \rightarrow
      x'^{(i+1)}.
      $$

      In this interactive demo, the user can adjust the **Variation Level**
      at each iteration to control the practical perturbation strength used
      to generate the next proposal.

    * **Barker--Bradley--Terry Connection**:  
      Under the proposal construction analyzed in the paper and an idealized
      exact-reverse assumption, the Barker acceptance probability reduces to the
      Bradley--Terry pairwise preference probability:

      $$
      \alpha_{\mathrm{BAMC}}(y \mid x)
      =
      \frac{\exp(r(y))}
      {\exp(r(y)) + \exp(r(x))}
      =
      P_{\mathrm{BT}}(y \succ x).
      $$

      Therefore, under the Bradley--Terry preference model, a stochastic pairwise
      choice between the current state and the proposal directly realizes the BAMC
      acceptance decision.

    * **Theoretical Guarantee**:  
      The paper proves that, under a **fixed non-degenerate perturbation level** and
      the **exact-reverse assumption**, the resulting Markov chain is aperiodic,
      positive Harris recurrent, and converges in total variation to the desired
      reward-reweighted stationary distribution.
    """)

# Preset witness scenarios
WITNESS_PRESETS = [
    "A front-facing portrait of a man in his mid-30s with sharp jawline, short black cropped hair, piercing dark eyes, clean shaven",
    "A front-facing portrait of a woman in her late 20s with wavy brown hair, warm brown eyes, high cheekbones, gentle expression",
    "A front-facing portrait of an East Asian male in his early 40s with narrow rectangular glasses, short black hair, serious expression",
    "A front-facing portrait of a woman in her 30s with curly black hair, dark skin tone, full lips, direct gaze at camera",
    "A front-facing portrait of a man in his 50s with receding gray hair, slight stubble beard, tired eyes with slight wrinkles"
]

# ==========================================
# 6. Mode 1: Real-World Montage Flow (No Target)
# ==========================================
if st.session_state.app_mode == "montage":
    
    # ---------------------------------------------------------
    # Step 1: Subject Description Input
    # ---------------------------------------------------------
    if st.session_state.step == "init_prompt":
        st.markdown('<span class="step-badge">Step 1: Initial Description</span>', unsafe_allow_html=True)
        st.subheader("Describe the Person from Memory")
        st.info("Enter what you remember about the person's face. This initial description guides the AI to generate the first set of candidate faces.")

        with st.expander("Facial Attribute Builder (Optional Helper)", expanded=False):
            col_a1, col_a2, col_a3 = st.columns(3)
            with col_a1:
                attr_gender = st.selectbox("Gender", ["Man", "Woman", "Individual"], index=0)
                attr_age = st.selectbox("Approximate Age", ["Early 20s", "Late 20s", "Mid 30s", "Early 40s", "Late 40s", "50s or older"], index=2)
            with col_a2:
                attr_ethnicity = st.selectbox("Ethnicity / Complexion", ["East Asian", "Caucasian", "Black / Dark-skinned", "South Asian", "Hispanic / Latino", "Middle Eastern"], index=0)
                attr_hair = st.selectbox("Hair Style & Color", ["Short black hair", "Wavy brown hair", "Bald / shaved head", "Long blonde hair", "Curly black hair", "Short graying hair"], index=0)
            with col_a3:
                attr_jaw = st.selectbox("Face Shape / Jaw", ["Square jawline", "Oval face", "Angular sharp cheekbones", "Round soft face"], index=0)
                attr_feature = st.selectbox("Notable Features", ["Clean shaven", "Short stubble beard", "Wire-rim eyeglasses", "Noticeable dark circles", "Thick eyebrows"], index=0)

            if st.button("Apply Attribute Summary to Prompt"):
                st.session_state.human_prompt = (
                    f"A front-facing portrait of a {attr_ethnicity} {attr_gender.lower()} in their {attr_age.lower()}, "
                    f"with {attr_hair.lower()}, {attr_jaw.lower()}, {attr_feature.lower()}, looking directly at camera, photorealistic portrait"
                )

        col_pr, col_rnd = st.columns([5, 1])
        with col_pr:
            st.session_state.human_prompt = st.text_area(
                "Description of the Person (English)",
                value=st.session_state.human_prompt if st.session_state.human_prompt else WITNESS_PRESETS[0],
                height=110,
                help="Describe facial structure, age, hair, eyes, and distinctive traits to start the search."
            )
        with col_rnd:
            st.write("")
            st.write("")
            if st.button("Sample Preset"):
                st.session_state.human_prompt = random.choice(WITNESS_PRESETS)
                st.rerun()

        if st.button("Generate Initial Faces 🚀", type="primary", use_container_width=True):
            if not st.session_state.human_prompt.strip():
                st.error("Please provide a subject description prompt.")
            else:
                task_dir = init_task_directory("montage_sessions")
                with open(os.path.join(task_dir, "initial_subject_prompt.txt"), "w", encoding="utf-8") as f:
                    f.write(st.session_state.human_prompt)

                with st.spinner("Generating initial face options..."):
                    init_latents = pipeline(
                        prompt=st.session_state.human_prompt,
                        negative_prompt="",
                        image=torch.randn(1, 3, 512, 512).to(DTYPE),
                        num_inference_steps=config.process.num_inference_steps,
                        guidance_scale=config.process.guidance_scale_init,
                        height=512,
                        width=512,
                        cfg_normalization=False,
                        num_images_per_prompt=config.process.num_samples_init,
                        strength=1.0,
                        generator=GENERATOR,
                        output_type="latent"
                    ).images.to(DTYPE)

                    st.session_state.init_latents = init_latents
                    st.session_state.init_imgs = decode_latents(pipeline, init_latents)
                    st.session_state.step = "init_select"
                    st.rerun()

    # ---------------------------------------------------------
    # Step 2: Select Starting Face
    # ---------------------------------------------------------
    elif st.session_state.step == "init_select":
        st.markdown('<span class="step-badge">Step 2: Starting Face</span>', unsafe_allow_html=True)
        st.subheader("Select Starting Face")
        st.info("Pick the face that looks closest to the person you remember to begin refining.")

        num_imgs = len(st.session_state.init_imgs)
        cols_per_row = 4
        for row_start in range(0, num_imgs, cols_per_row):
            cols = st.columns(cols_per_row)
            for offset in range(cols_per_row):
                idx = row_start + offset
                if idx < num_imgs:
                    img = st.session_state.init_imgs[idx]
                    with cols[offset]:
                        st.image(img, caption=f"Option #{idx + 1}", use_container_width=True)
                        if st.button(f"Select #{idx + 1}", key=f"sel_init_{idx}", type="primary", use_container_width=True):
                            selected_latent = st.session_state.init_latents[idx]
                            st.session_state.tracker.init = SampleState(
                                distance=1.0,
                                latent=selected_latent.clone(),
                                iteration=0
                            )
                            st.session_state.tracker.cur = st.session_state.tracker.init

                            # Record to timeline
                            st.session_state.accepted_timeline.append({
                                "iter": 0,
                                "img": img,
                                "label": "Starting Face"
                            })

                            img.save(os.path.join(st.session_state.task_dir, "initial_selected_face.png"))
                            torch.save(selected_latent.cpu(), os.path.join(st.session_state.task_dir, "initial_latent.pt"))

                            st.session_state.step = "loop"
                            st.rerun()

    # ---------------------------------------------------------
    # Step 3: Interactive Refinement Loop
    # ---------------------------------------------------------
    elif st.session_state.step in ["loop", "loop_eval"]:
        st.markdown(f'<span class="step-badge">Step 3: Refine Face | Iteration {st.session_state.iteration + 1} / {config.process.num_loop}</span>', unsafe_allow_html=True)
        
        # Action bar
        col_hdr, col_fin = st.columns([7, 3])
        with col_hdr:
            st.subheader("Refine Facial Features")
        with col_fin:
            if st.button("Finish & View Final Composite 🏁", type="secondary", use_container_width=True):
                st.session_state.step = "finish"
                st.rerun()

        st.markdown("---")

        if st.session_state.step == "loop":
            col_curr, col_ctrl = st.columns([1, 1])

            with col_curr:
                st.markdown("#### Current Face")
                cur_img = decode_latents(pipeline, st.session_state.tracker.cur.latent)[0]
                st.image(cur_img, caption="Current best match", use_container_width=True)

            with col_ctrl:
                st.markdown("#### Variation Settings")
                st.info("Adjust how much the face should change, then generate a new variation to compare.")

                strength = st.slider(
                    "Variation Level",
                    min_value=0.10,
                    max_value=0.90,
                    value=float(st.session_state.current_strength),
                    step=0.05,
                    help="Controls how much the face changes. Lower values (0.15–0.35) subtly tweak fine details. Higher values (0.45–0.75) make broader adjustments to facial structure."
                )

                st.caption(f"🔒 **Base Description**: *\"{st.session_state.human_prompt}\"*")

                with st.expander("🔧 Advanced Technical Parameters", expanded=False):
                    st.markdown(f"- **Diffusion Noise Strength ($\\sigma$)**: `{strength:.2f}`")
                    st.markdown(f"- **Sampling Steps**: `{config.process.num_inference_steps}`")
                    st.markdown(f"- **Guidance Scale**: `{config.process.guidance_scale_in_loop}`")

                if st.button("Generate New Variation 🎲", type="primary", use_container_width=True):
                    st.session_state.current_strength = strength

                    with st.spinner("Generating facial variation..."):
                        cur_latent = st.session_state.tracker.cur.latent
                        if cur_latent.ndim == 3:
                            cur_latent = cur_latent.unsqueeze(0)

                        cand_latents = pipeline(
                            prompt=st.session_state.human_prompt,
                            negative_prompt="",
                            image=cur_latent,
                            num_inference_steps=config.process.num_inference_steps,
                            guidance_scale=config.process.guidance_scale_in_loop,
                            height=512,
                            width=512,
                            num_images_per_prompt=1,
                            cfg_normalization=False,
                            strength=strength,
                            generator=GENERATOR,
                            output_type="latent"
                        ).images.to(DTYPE)

                        st.session_state.candidate_latents = cand_latents
                        st.session_state.candidate_imgs = decode_latents(pipeline, cand_latents)
                        st.session_state.step = "loop_eval"
                        st.rerun()

        elif st.session_state.step == "loop_eval":
            st.info("**Compare & Choose**: Compare the new variation with your current face against your memory. Keep the new variation if it looks closer, or keep the current face to try another variation.")

            col_curr, col_cand = st.columns(2)
            cur_img = decode_latents(pipeline, st.session_state.tracker.cur.latent)[0]
            cand_img = st.session_state.candidate_imgs[0]

            with col_curr:
                st.markdown("#### Current Face")
                st.image(cur_img, caption="Current face", use_container_width=True)

            with col_cand:
                st.markdown("#### New Variation")
                st.image(cand_img, caption="New variation", use_container_width=True)

            st.markdown("---")
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                if st.button("👍 Keep New Variation", type="primary", use_container_width=True):
                    handle_decision(0, "ACCEPTED")
            with col_b2:
                if st.button("👎 Keep Current Face", type="secondary", use_container_width=True):
                    handle_decision(0, "REJECTED")

            with st.expander("🔬 MCMC Transition Details", expanded=False):
                st.markdown(f"- **Current State**: $x^{{({st.session_state.iteration})}}$")
                st.markdown(f"- **Proposal State**: $x'$ (generated with $\\sigma={st.session_state.current_strength:.2f}$)")
                st.markdown("- **Barker Decision**: Pairwise human choice realizes the acceptance probability $\\alpha_{\\mathrm{BAMC}}(x' \\mid x^{(i)})$.")

    # ---------------------------------------------------------
    # Step 4: Final Composite Results
    # ---------------------------------------------------------
    elif st.session_state.step == "finish":
        st.markdown('<span class="step-badge">Final Result: Facial Composite</span>', unsafe_allow_html=True)
        st.subheader("Composite Generation Completed")
        st.success("Your composite face has been generated based on your visual preferences.")

        final_img = decode_latents(pipeline, st.session_state.tracker.cur.latent)[0]
        init_img = decode_latents(pipeline, st.session_state.tracker.init.latent)[0]

        # Save final montage image
        final_img.save(os.path.join(st.session_state.task_dir, "final_composite.png"))

        col_main, col_info = st.columns([1, 1])
        with col_main:
            st.image(final_img, caption="Final Generated Composite", use_container_width=True)
            st.download_button(
                label="Download Final Composite (PNG)",
                data=image_to_bytes(final_img),
                file_name=f"composite_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png",
                type="primary",
                use_container_width=True
            )

        with col_info:
            st.markdown("### Session Summary")
            st.markdown(f"- **Description Used**: *\"{st.session_state.human_prompt}\"*")
            st.markdown(f"- **Total Variations Tested**: `{st.session_state.iteration}`")
            num_accepted = sum(1 for h in st.session_state.mcmc_history if h["status"] == "ACCEPTED")
            st.markdown(f"- **Accepted Improvements**: `{num_accepted}`")

            # CSV Download
            csv_path = os.path.join(st.session_state.task_dir, "mcmc_history.csv")
            if os.path.exists(csv_path):
                with open(csv_path, "r", encoding="utf-8") as f:
                    csv_data = f.read()
                st.download_button(
                    label="Download Session History (CSV)",
                    data=csv_data,
                    file_name="bamc_session_history.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with st.expander("📁 Technical Session Details", expanded=False):
                st.markdown(f"- **Session Directory**: `{st.session_state.task_dir}`")
                st.markdown(f"- **Acceptance Rate**: `{num_accepted / max(1, st.session_state.iteration):.1%}`")
                st.markdown("- **Checkpoints Saved**: `final_composite.png`, `initial_latent.pt`")

        st.markdown("---")
        st.markdown("### Face Evolution History")
        st.write("Visual progression of accepted facial improvements during the session:")

        timeline = st.session_state.accepted_timeline
        if timeline:
            cols = st.columns(min(len(timeline), 6))
            for i, item in enumerate(timeline[-6:]):
                with cols[i]:
                    st.image(item["img"], caption=item["label"], use_container_width=True)

        st.markdown("---")
        with st.expander("🔍 Compare with Actual Photo (Optional Verification)", expanded=False):
            st.write("If you have an actual reference photo of the subject, you can upload it here to measure objective similarity metrics (ArcFace identity distance & LPIPS perceptual distance).")
            verif_file = st.file_uploader("Upload Reference Photo", type=["png", "jpg", "jpeg"], key="verif_upload")
            if verif_file is not None:
                real_img = Image.open(verif_file).convert("RGB").resize((512, 512))
                arc_eval = ArcFaceEvaluator(device=DEVICE)
                lpips_eval = LPIPSEvaluator(device=DEVICE)
                arc_eval.set_target(real_img)
                lpips_eval.set_target(real_img)

                init_arc = arc_eval.calculate_distances_batch([init_img])[0]
                final_arc = arc_eval.calculate_distances_batch([final_img])[0]
                init_lpips = lpips_eval.calculate_distances_batch([init_img])[0]
                final_lpips = lpips_eval.calculate_distances_batch([final_img])[0]

                c_v1, c_v2, c_v3 = st.columns(3)
                with c_v1:
                    st.image(real_img, caption="Reference Photo", use_container_width=True)
                with c_v2:
                    st.image(init_img, caption="Starting Face", use_container_width=True)
                    st.metric("Identity Distance (ArcFace)", f"{init_arc:.4f}", help="ArcFace facial identity distance. Lower is closer.")
                    st.metric("Visual Distance (LPIPS)", f"{init_lpips:.4f}", help="LPIPS perceptual deep feature distance. Lower is closer.")
                with c_v3:
                    st.image(final_img, caption="Final Composite Face", use_container_width=True)
                    st.metric("Identity Distance (ArcFace)", f"{final_arc:.4f}", delta=f"{final_arc - init_arc:.4f}", delta_color="inverse", help="ArcFace facial identity distance. Lower is closer.")
                    st.metric("Visual Distance (LPIPS)", f"{final_lpips:.4f}", delta=f"{final_lpips - init_lpips:.4f}", delta_color="inverse", help="LPIPS perceptual deep feature distance. Lower is closer.")

# ==========================================
# 7. Mode 2: Interactive Evaluation with Known Target
# ==========================================
elif st.session_state.app_mode == "benchmark":
    
    # ---------------------------------------------------------
    # Step 1: Set Reference Target Photo
    # ---------------------------------------------------------
    if st.session_state.step == "target_gen":
        st.markdown('<span class="step-badge">Step 1: Reference Photo</span>', unsafe_allow_html=True)
        st.subheader("Set Reference Target Photo")
        st.info(
            "In benchmark mode, choose or upload a reference target photo to evaluate "
            "how closely the preference-guided search can reconstruct it.\n\n"
            "⏱️ **Note on Evaluation Time**: Benchmark mode additionally runs external "
            "evaluation models such as LPIPS and ArcFace to measure objective similarity. "
            "These computations add evaluation overhead to the observed runtime and are "
            "not part of the BAMC proposal-generation procedure itself."
        )

        if st.session_state.target_img is None:
            tab_prompt, tab_upload = st.tabs(["Generate with Local Model", "Upload Reference Photo"])

            with tab_prompt:
                col_sel, col_btn = st.columns([4, 1])
                with col_sel:
                    selected_preset = st.selectbox("Preset Target Descriptions", WITNESS_PRESETS, index=0)
                with col_btn:
                    st.write("")
                    st.write("")
                    if st.button("Randomize"):
                        st.session_state.custom_target_prompt = random.choice(WITNESS_PRESETS)
                        st.rerun()

                target_prompt = st.text_area(
                    "Target Prompt",
                    value=st.session_state.get("custom_target_prompt", selected_preset),
                    height=100
                )

                if st.button("Generate Target Image", type="primary", use_container_width=True):
                    if not target_prompt.strip():
                        st.error("Please enter a target prompt!")
                    else:
                        with st.spinner("Generating target image with local diffusion pipeline..."):
                            target_latents = pipeline(
                                prompt=target_prompt,
                                negative_prompt="",
                                image=torch.randn(1, 3, 512, 512).to(DTYPE),
                                num_inference_steps=config.process.num_inference_steps,
                                guidance_scale=config.process.guidance_scale_init,
                                height=512,
                                width=512,
                                cfg_normalization=False,
                                num_images_per_prompt=1,
                                strength=1.0,
                                generator=GENERATOR,
                                output_type="latent"
                            ).images.to(DTYPE)
                            target_img = decode_latents(pipeline, target_latents)[0]

                            task_dir = init_task_directory("benchmark_eval")
                            st.session_state.target_img = target_img
                            st.session_state.target_prompt = target_prompt
                            st.session_state.human_prompt = target_prompt

                            target_img.save(os.path.join(task_dir, "target.png"))
                            with open(os.path.join(task_dir, "target_prompt.txt"), "w", encoding="utf-8") as f:
                                f.write(target_prompt)

                            # Setup evaluators (LPIPS for fast in-loop tracking; ArcFace deferred to final step)
                            st.session_state.lpips_eval = LPIPSEvaluator(device=DEVICE)
                            st.session_state.lpips_eval.set_target(target_img)
                            st.session_state.arcface_eval = None
                            if "benchmark_metrics" in st.session_state:
                                del st.session_state["benchmark_metrics"]

                            st.rerun()

            with tab_upload:
                uploaded_file = st.file_uploader("Upload Reference Photo (PNG / JPG)", type=["png", "jpg", "jpeg"])
                if uploaded_file is not None:
                    preview_img = Image.open(uploaded_file).convert("RGB")
                    st.image(preview_img, caption="Uploaded Preview", width=256)
                    if st.button("Use This as Reference Photo", type="primary"):
                        target_img = preview_img.resize((512, 512))
                        task_dir = init_task_directory("benchmark_eval")
                        st.session_state.target_img = target_img
                        st.session_state.target_prompt = "User uploaded ground truth"
                        st.session_state.human_prompt = "A front-facing portrait of a person"

                        target_img.save(os.path.join(task_dir, "target.png"))
                        # Setup evaluators (LPIPS for fast in-loop tracking; ArcFace deferred to final step)
                        st.session_state.lpips_eval = LPIPSEvaluator(device=DEVICE)
                        st.session_state.lpips_eval.set_target(target_img)
                        st.session_state.arcface_eval = None
                        if "benchmark_metrics" in st.session_state:
                            del st.session_state["benchmark_metrics"]

                        st.rerun()
        else:
            st.success("Reference Target Photo Loaded!")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(st.session_state.target_img, caption="Reference Target Photo", width=256)
            with col2:
                st.markdown(f"**Target Prompt:** {st.session_state.target_prompt}")
                if st.button("Proceed to Generate Starting Faces", type="primary"):
                    st.session_state.step = "init_gen"
                    st.rerun()
                if st.button("Reset Target"):
                    st.session_state.target_img = None
                    st.rerun()

    # ---------------------------------------------------------
    # Step 2: Generate Starting Faces
    # ---------------------------------------------------------
    elif st.session_state.step == "init_gen":
        st.markdown('<span class="step-badge">Step 2: Starting Faces</span>', unsafe_allow_html=True)
        st.subheader("Generate Starting Faces")
        st.info("Enter an initial description to generate candidate starting faces.")

        st.session_state.human_prompt = st.text_area("Initial Prompt", value=st.session_state.human_prompt)

        if st.button("Generate Starting Faces 🚀", type="primary", use_container_width=True):
            if not st.session_state.human_prompt.strip():
                st.error("Please enter a prompt first.")
            else:
                with st.spinner("Generating starting faces..."):
                    init_latents = pipeline(
                        prompt=st.session_state.human_prompt,
                        negative_prompt="",
                        image=torch.randn(1, 3, 512, 512).to(DTYPE),
                        num_inference_steps=config.process.num_inference_steps,
                        guidance_scale=config.process.guidance_scale_init,
                        height=512,
                        width=512,
                        cfg_normalization=False,
                        num_images_per_prompt=config.process.num_samples_init,
                        strength=1.0,
                        generator=GENERATOR,
                        output_type="latent"
                    ).images.to(DTYPE)

                    st.session_state.init_latents = init_latents
                    st.session_state.init_imgs = decode_latents(pipeline, init_latents)
                    st.session_state.step = "settings"
                    st.rerun()

    # ---------------------------------------------------------
    # Step 3: Test Visibility Settings
    # ---------------------------------------------------------
    elif st.session_state.step == "settings":
        st.markdown('<span class="step-badge">Step 3: Visibility Settings</span>', unsafe_allow_html=True)
        st.subheader("Test Visibility Configuration")

        col_t, col_s = st.columns([1, 2])
        with col_t:
            st.image(st.session_state.target_img, caption="Reference Target Photo", width=256)
        with col_s:
            st.session_state.show_target_in_loop = st.radio(
                "Target Photo Visibility during Test",
                [True, False],
                format_func=lambda x: "Show Target Photo (Side-by-Side Comparison)" if x else "Hide Target Photo (Memory Recall Test)"
            )
            if st.button("Confirm & Select Starting Face", type="primary"):
                st.session_state.step = "init_select"
                st.rerun()

    # ---------------------------------------------------------
    # Step 4: Select Starting Face
    # ---------------------------------------------------------
    elif st.session_state.step == "init_select":
        st.markdown('<span class="step-badge">Step 4: Starting Face</span>', unsafe_allow_html=True)
        st.subheader("Select Starting Face")

        if st.session_state.show_target_in_loop:
            st.image(st.session_state.target_img, caption="Reference Target Photo", width=200)

        num_imgs = len(st.session_state.init_imgs)
        cols_per_row = 4
        for row_start in range(0, num_imgs, cols_per_row):
            cols = st.columns(cols_per_row)
            for offset in range(cols_per_row):
                idx = row_start + offset
                if idx < num_imgs:
                    img = st.session_state.init_imgs[idx]
                    with cols[offset]:
                        st.image(img, caption=f"Option #{idx + 1}", use_container_width=True)
                        if st.button(f"Select #{idx + 1}", key=f"sel_init_bm_{idx}", type="primary", use_container_width=True):
                            selected_latent = st.session_state.init_latents[idx]
                            
                            lpips_dists = st.session_state.lpips_eval.calculate_distances_batch([img])
                            lp = lpips_dists[0]

                            st.session_state.tracker.init = SampleState(
                                distance=lp,
                                latent=selected_latent.clone(),
                                iteration=0
                            )
                            st.session_state.tracker.cur = st.session_state.tracker.init
                            st.session_state.tracker.best = st.session_state.tracker.init

                            filename = f"initial_LPIPS_{lp:.3f}.png"
                            img.save(os.path.join(st.session_state.task_dir, filename))

                            st.session_state.accepted_timeline.append({
                                "iter": 0,
                                "img": img,
                                "label": "Starting Face"
                            })

                            st.session_state.step = "loop"
                            st.rerun()

    # ---------------------------------------------------------
    # Step 5: Interactive Comparison Loop
    # ---------------------------------------------------------
    elif st.session_state.step in ["loop", "loop_eval"]:
        st.markdown(f'<span class="step-badge">Step 5: Face Comparison | Iteration {st.session_state.iteration + 1} / {config.process.num_loop}</span>', unsafe_allow_html=True)

        col_hdr, col_fin = st.columns([7, 3])
        with col_hdr:
            st.subheader("Interactive Face Comparison")
        with col_fin:
            if st.button("Finish & View Results 🏁", use_container_width=True):
                st.session_state.step = "finish"
                st.rerun()

        st.markdown("---")

        if st.session_state.step == "loop":
            col_preview, col_params = st.columns(2)
            with col_preview:
                if st.session_state.show_target_in_loop:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Target Reference Photo**")
                        st.image(st.session_state.target_img, use_container_width=True)
                    with c2:
                        st.markdown("**Current Face**")
                        cur_img = decode_latents(pipeline, st.session_state.tracker.cur.latent)[0]
                        st.image(cur_img, use_container_width=True)
                else:
                    st.markdown("**Current Face**")
                    cur_img = decode_latents(pipeline, st.session_state.tracker.cur.latent)[0]
                    st.image(cur_img, width=320)

            with col_params:
                st.markdown("#### Variation Settings")
                strength = st.slider(
                    "Variation Level",
                    min_value=0.05,
                    max_value=0.95,
                    value=float(st.session_state.current_strength),
                    step=0.05,
                    help="Controls how much the face changes during variation generation. Lower values make fine tweaks; higher values make broader adjustments."
                )
                st.caption(f"🔒 **Base Description**: *\"{st.session_state.human_prompt}\"*")

                with st.expander("🔧 Advanced Technical Parameters", expanded=False):
                    st.markdown(f"- **Diffusion Noise Strength ($\\sigma$)**: `{strength:.2f}`")
                    st.markdown(f"- **Sampling Steps**: `{config.process.num_inference_steps}`")
                    st.markdown(f"- **Guidance Scale**: `{config.process.guidance_scale_in_loop}`")

                if st.button("Generate New Variation 🎲", type="primary", use_container_width=True):
                    st.session_state.current_strength = strength
                    with st.spinner("Generating facial variation..."):
                        cur_latent = st.session_state.tracker.cur.latent
                        if cur_latent.ndim == 3:
                            cur_latent = cur_latent.unsqueeze(0)

                        cand_latents = pipeline(
                            prompt=st.session_state.human_prompt,
                            negative_prompt="",
                            image=cur_latent,
                            num_inference_steps=config.process.num_inference_steps,
                            guidance_scale=config.process.guidance_scale_in_loop,
                            height=512,
                            width=512,
                            num_images_per_prompt=1,
                            cfg_normalization=False,
                            strength=strength,
                            generator=GENERATOR,
                            output_type="latent"
                        ).images.to(DTYPE)

                        st.session_state.candidate_latents = cand_latents
                        st.session_state.candidate_imgs = decode_latents(pipeline, cand_latents)
                        st.session_state.step = "loop_eval"
                        st.rerun()

        elif st.session_state.step == "loop_eval":
            st.info("**Compare & Choose**: Compare the new variation with the current face against the reference target. Choose whichever face looks closer to the target photo.")
            cur_img = decode_latents(pipeline, st.session_state.tracker.cur.latent)[0]
            cand_img = st.session_state.candidate_imgs[0]

            if st.session_state.show_target_in_loop:
                c_tgt, c_cur, c_cnd = st.columns(3)
                with c_tgt:
                    st.markdown("**Target Reference Photo**")
                    st.image(st.session_state.target_img, use_container_width=True)
                with c_cur:
                    st.markdown("**Current Face**")
                    st.image(cur_img, use_container_width=True)
                with c_cnd:
                    st.markdown("**New Variation**")
                    st.image(cand_img, use_container_width=True)
            else:
                c_cur, c_cnd = st.columns(2)
                with c_cur:
                    st.markdown("**Current Face**")
                    st.image(cur_img, use_container_width=True)
                with c_cnd:
                    st.markdown("**New Variation**")
                    st.image(cand_img, use_container_width=True)

            st.markdown("---")
            b1, b2 = st.columns(2)
            with b1:
                if st.button("👍 Keep New Variation", type="primary", use_container_width=True):
                    handle_decision(0, "ACCEPTED")
            with b2:
                if st.button("👎 Keep Current Face", type="secondary", use_container_width=True):
                    handle_decision(0, "REJECTED")

            with st.expander("🔬 MCMC Transition Details", expanded=False):
                st.markdown(f"- **Current State**: $x^{{({st.session_state.iteration})}}$")
                st.markdown(f"- **Proposal State**: $x'$ (generated with $\\sigma={st.session_state.current_strength:.2f}$)")
                st.markdown("- **Barker Decision**: Pairwise human choice directly samples the acceptance probability $\\alpha_{\\mathrm{BAMC}}(x' \\mid x^{(i)})$.")

    # ---------------------------------------------------------
    # Step 6: Final Quantitative Comparison
    # ---------------------------------------------------------
    elif st.session_state.step == "finish":
        st.markdown('<span class="step-badge">Final Evaluation Results</span>', unsafe_allow_html=True)
        st.subheader("Evaluation Results & Accuracy Metrics")

        init_img = decode_latents(pipeline, st.session_state.tracker.init.latent)[0]
        final_img = decode_latents(pipeline, st.session_state.tracker.cur.latent)[0]
        target_img = st.session_state.target_img

        if "benchmark_metrics" not in st.session_state:
            with st.spinner("Computing final evaluation metrics (ArcFace identity distance, LPIPS perceptual distance)..."):
                if "arcface_eval" not in st.session_state or st.session_state.arcface_eval is None:
                    arc_eval = ArcFaceEvaluator(device=DEVICE)
                    arc_eval.set_target(target_img)
                    st.session_state.arcface_eval = arc_eval
                else:
                    arc_eval = st.session_state.arcface_eval
                lpips_eval = st.session_state.lpips_eval

                arc_dists = arc_eval.calculate_distances_batch([init_img, final_img])
                lpips_dists = lpips_eval.calculate_distances_batch([init_img, final_img])

                st.session_state.benchmark_metrics = {
                    "init_arc": arc_dists[0],
                    "final_arc": arc_dists[1],
                    "init_lp": lpips_dists[0],
                    "final_lp": lpips_dists[1]
                }

        metrics = st.session_state.benchmark_metrics
        init_arc, final_arc = metrics["init_arc"], metrics["final_arc"]
        init_lp, final_lp = metrics["init_lp"], metrics["final_lp"]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### Target Reference Photo")
            st.image(target_img, use_container_width=True)

        with col2:
            st.markdown("#### Starting Face")
            st.image(init_img, use_container_width=True)
            st.metric("Identity Distance (ArcFace)", f"{init_arc:.4f}", help="ArcFace facial identity distance. Lower is closer.")
            st.metric("Visual Distance (LPIPS)", f"{init_lp:.4f}", help="LPIPS perceptual deep feature distance. Lower is closer.")

        with col3:
            st.markdown("#### Final Generated Face")
            st.image(final_img, use_container_width=True)
            st.metric("Identity Distance (ArcFace)", f"{final_arc:.4f}", delta=f"{final_arc - init_arc:.4f}", delta_color="inverse", help="ArcFace facial identity distance. Lower is closer.")
            st.metric("Visual Distance (LPIPS)", f"{final_lp:.4f}", delta=f"{final_lp - init_lp:.4f}", delta_color="inverse", help="LPIPS perceptual deep feature distance. Lower is closer.")

        st.markdown("---")
        with st.expander("🔬 Academic Evaluation & Metric Explanations", expanded=False):
            st.markdown("""
            * **ArcFace Identity Distance**: Cosine distance in ArcFace embedding space ($1 - \\cos(e_1, e_2)$). Quantifies facial identity similarity. A negative delta indicates identity convergence toward the reference target.
            * **LPIPS Perceptual Distance**: Learned Perceptual Image Patch Similarity. Quantifies deep feature perceptual resemblance. A negative delta indicates visual feature convergence.
            * **MCMC Convergence**: The BAMC chain iteratively navigates the generative prior manifold conditioned on pairwise preference judgments, demonstrably improving objective distance metrics over iterations.
            """)

        st.download_button(
            label="Download Final Composite (PNG)",
            data=image_to_bytes(final_img),
            file_name=f"benchmark_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            mime="image/png",
            type="primary"
        )
