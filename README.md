# BAMC: Barker-Aligned Monte Carlo for Preference-Guided Search

[![ACCV 2026](https://img.shields.io/badge/ACCV%202026-Accepted-success.svg)](https://accv2026.org/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Diffusers](https://img.shields.io/badge/Diffusers-0.37%2B-yellow.svg)](https://github.com/huggingface/diffusers)
[![Managed by uv](https://img.shields.io/badge/Managed%20by-uv-blueviolet.svg)](https://github.com/astral-sh/uv)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

Official PyTorch implementation of

**BAMC: Barker-Aligned Monte Carlo for Preference-Guided Search**

accepted to the **18th Asian Conference on Computer Vision (ACCV 2026)**.

---

## Overview

**BAMC (Barker-Aligned Monte Carlo)** is a preference-guided search framework that uses a pretrained diffusion or flow model as a proposal mechanism within Markov Chain Monte Carlo (MCMC).

We consider the reward-reweighted target distribution

$$
p^*(x) \propto p_D(x)\exp(r(x)),
$$

where $p_D$ denotes the generative data distribution and $r(x)$ represents user preference.

Under the proposal construction studied in the paper and an idealized exact-reverse assumption, the Barker acceptance probability reduces to the Bradley--Terry pairwise preference probability:

$$
\alpha_{\mathrm{BAMC}}(y \mid x) = \frac{\exp(r(y))}{\exp(r(y))+\exp(r(x))} = P_{\mathrm{BT}}(y \succ x).
$$

This establishes a direct connection between **stochastic binary pairwise preferences** and **MCMC acceptance decisions**.

The repository contains:

- reproduction scripts for the main paper experiments,
- LPIPS-based oracle-reward evaluation,
- VLM-based pairwise preference evaluation,
- Metropolis--Hastings, Hill Climbing, and Random Sampling baselines,
- perturbation-strength ablations,
- LPIPS and ArcFace evaluation tools,
- analysis and visualization scripts, and
- an interactive Streamlit application for human-in-the-loop BAMC search.

---

## Method at a Glance

At each BAMC iteration:

1. Start from the current chain state $x^{(i)}$.
2. Perturb the current state through the diffusion/flow probability path.
3. Generate a proposal $x'^{(i+1)}$ using the pretrained generative model.
4. Compare the current state and proposal using pairwise preference feedback.
5. Accept or reject the proposal according to the Barker-aligned decision.
6. Continue the Markov chain from the accepted state.

Conceptually,

```text
Current State
    x(i)
      │
      ▼
Diffusion / Flow Perturbation
      │
      ▼
Proposal x'(i+1)
      │
      ▼
Pairwise Preference
      │
      ▼
Barker-Aligned Accept / Reject
      │
      ▼
Next State x(i+1)
```

The paper provides the theoretical analysis for a fixed non-degenerate perturbation level. Practical experiments additionally consider fixed or adaptive perturbation strengths.

---

## Interactive BAMC Demo

We provide a Streamlit application for directly interacting with BAMC.

```bash
uv run streamlit run app.py
```

The application supports two modes.

### 1. Interactive Facial Composite Demo

A human-in-the-loop demonstration for preference-guided facial composite generation when no ground-truth target image is available.

Workflow:

1. Provide an initial textual description.
2. Generate an initial set of candidate faces.
3. Select the candidate closest to the intended mental image.
4. Adjust the perturbation strength.
5. Generate a diffusion-based proposal.
6. Compare the **Current State** and **Proposal**.
7. Accept or reject the proposal.
8. Repeat until the desired composite is obtained.

The application records the accepted-state trajectory and session history for later inspection.

### 2. Evaluation Mode with a Known Target

An interactive evaluation interface using a known reference image.

The interface can display:

- the target image,
- the current BAMC state,
- the proposal,
- LPIPS distance, and
- ArcFace identity distance.

This mode is intended for interactive inspection and demonstration.

> **Note**
>
> The Streamlit evaluation interface is not the exact automated protocol used to produce the paper tables.  
> For reproduction of the paper experiments, use the scripts described in [Reproducing the Paper Experiments](#reproducing-the-paper-experiments).

### Research-use notice

The interactive facial composite application is provided as a **research demonstration** of preference-guided generation. It has not been validated for real-world forensic identification, biometric identification, or investigative decision-making.

---

## Installation

### Requirements

- Linux
- Python 3.11+
- NVIDIA GPU with CUDA support
- PyTorch 2.x
- Hugging Face Diffusers
- `libgl1` for OpenCV in headless Linux environments

Install the required system library if necessary:

```bash
sudo apt-get update
sudo apt-get install -y libgl1
```

### Clone the repository

```bash
git clone https://github.com/Won-Seong/bamc.git
cd bamc
```

### Install dependencies

We recommend [`uv`](https://github.com/astral-sh/uv) for fast, reproducible dependency and environment management:

```bash
uv sync
```

Alternatively, you can install dependencies using `pip`:

```bash
pip install -r requirements.txt
```

---

## API Configuration

VLM-based experiments require a Google Gemini API key.

Create a local environment file:

```bash
cp .env.example .env
```

Then add:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

The oracle-reward experiments based on local LPIPS and ArcFace evaluation do not require a Gemini API key.

Never commit `.env` or private API credentials to the repository.

---

## Reproducing the Paper Experiments

The experiment scripts in `scripts/` reproduce the major experimental settings used in the paper.

### Main Comparison

Runs the oracle-reward experiments comparing:

- BAMC / Barker,
- Metropolis--Hastings,
- Hill Climbing, and
- Random Sampling.

```bash
bash ./scripts/main.sh
```

The experiments cover targets from:

- FFHQ,
- Z-Image, and
- GPT-generated images.

---

### VLM-Based Pairwise Preference Evaluation

Runs BAMC using a Vision-Language Model as the pairwise preference proxy.

```bash
bash ./scripts/vlm.sh
```

This experiment follows the pairwise-feedback setting described in the paper and supplementary material.

---

### Perturbation-Strength Ablation

Runs the perturbation-strength ablation experiments.

```bash
bash ./scripts/ablation_strength.sh
```

The paper compares an adaptive setting with several fixed perturbation strengths.

---

## Evaluation

### Quantitative Evaluation

Evaluation is automatically executed by the experiment scripts and can also be run separately.

Example:

```bash
uv run python evaluate.py \
    --config configs/config_z_target_lpips.yaml \
    --output_dir_suffix _barker
```

The evaluation reports:

- **LPIPS distance**  
  Perceptual distance between the generated image and the target.  
  Lower is better.

- **ArcFace identity distance**  
  Identity distance computed from ArcFace embeddings using $1 - \text{cosine similarity}$.  
  Lower is better.

Aggregated results are written to:

```text
final_evaluation.csv
```

---

## Analysis and Paper Figures

Use `analyze.py` to generate analysis outputs such as:

- acceptance-rate comparisons,
- best-so-far LPIPS trajectories,
- convergence plots, and
- experiment visualizations.

```bash
uv run python analyze.py \
    --base_dir images/results \
    --out_dir images/analysis
```

To generate plots including the random sampling baseline and perturbation-strength ablations:

```bash
# Include both the random sampling baseline and perturbation-strength ablations
uv run python analyze.py \
    --base_dir images/results \
    --out_dir images/analysis \
    --include_rand \
    --include_ablations
```

- `--include_rand`: Includes the random sampling baseline trajectory in convergence and comparison plots.
- `--include_ablations`: Includes fixed perturbation-strength ablation runs (e.g., `str0.3`, `str0.6`, `str0.9`).

---

## Project Structure

```text
.
├── app.py
│   └── Interactive Streamlit BAMC demo
│
├── main.py
│   └── Oracle-reward search experiments
│       (Barker / Metropolis-Hastings / Hill Climbing)
│
├── main_vlm.py
│   └── VLM-based pairwise preference experiments
│
├── main_rand.py
│   └── Random Sampling baseline
│
├── evaluate.py
│   └── LPIPS and ArcFace quantitative evaluation
│
├── analyze.py
│   └── Analysis and paper visualization utilities
│
├── configs/
│   ├── config_app.yaml
│   ├── config_z_target_lpips.yaml
│   ├── config_gpt_target_lpips.yaml
│   ├── config_ffhq_target_lpips.yaml
│   └── config_vlm.yaml
│
├── helpers/
│   ├── evaluators.py
│   │   └── LPIPS, ArcFace, and VLM evaluators
│   ├── llm.py
│   │   └── VLM / API utilities
│   ├── models.py
│   │   └── Experiment state and MCMC tracking structures
│   └── utils.py
│       └── Acceptance rules, latent decoding, and utilities
│
├── images/
│   └── targets/
│       └── 27 target facial images (FFHQ, Z-Image, GPT)
│
├── scripts/
│   ├── main.sh
│   ├── vlm.sh
│   └── ablation_strength.sh
│
├── pyproject.toml
├── requirements.txt
├── uv.lock
└── .env.example
```

---

## Experimental Settings

The main paper evaluates BAMC under two complementary settings.

### Oracle-Reward Setting

LPIPS distance to a known target is used to define the reward

$$
r(x) = -\frac{d_{\mathrm{LPIPS}}(x, x_{\mathrm{target}})}{\tau},
$$

which induces

$$
p^*(x) \propto p_D(x) \exp\left(-\frac{d_{\mathrm{LPIPS}}(x, x_{\mathrm{target}})}{\tau}\right).
$$

For the main quantitative experiments,

```text
τ = 0.002
```

is used.

### VLM Preference Setting

The VLM directly performs pairwise preference comparisons between the current state and proposal relative to the target identity.

This setting does not require an explicit differentiable scalar reward during the interactive search.

For additional prompts and experimental details, please refer to the supplementary material.

---

## Reproducibility Notes

The paper uses fixed random seeds and configuration files for controlled evaluation.

For reproducibility:

- keep the provided YAML configurations unchanged,
- record the software and CUDA environment,
- keep model revisions fixed where possible,
- use the provided experiment scripts rather than the interactive application for paper reproduction, and
- be aware that external VLM APIs may exhibit backend-dependent variation even when temperature is set to zero.

---

## Results

The main paper evaluates BAMC on 27 target facial images consisting of:

- 9 FFHQ targets,
- 9 Z-Image-generated targets, and
- 9 GPT-generated targets.

BAMC is evaluated using LPIPS and ArcFace identity distance and compared against Random Sampling, Hill Climbing, and Metropolis--Hastings.

For complete quantitative results, ablations, and qualitative comparisons, please refer to the ACCV 2026 paper and supplementary material.

---

## Paper

**BAMC: Barker-Aligned Monte Carlo for Preference-Guided Search**  
Sungwon Kim  
Asian Conference on Computer Vision (**ACCV 2026**)

Paper and supplementary links will be added upon public release.

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{kim2026bamc,
  title     = {BAMC: Barker-Aligned Monte Carlo for Preference-Guided Search},
  author    = {Kim, Sungwon},
  booktitle = {Asian Conference on Computer Vision (ACCV)},
  year      = {2026}
}
```

The citation entry will be updated when the final proceedings metadata becomes available.

---

## License

This project is released under the [Apache License 2.0](LICENSE).

---

## Acknowledgements

This research was supported by the Institute of Information & Communication Technology Planning & Evaluation (IITP) under the Leading Generative AI Human Resources Development (IITP-2026-RS-2026-25544188) grant funded by the Korea government (MSIT).
