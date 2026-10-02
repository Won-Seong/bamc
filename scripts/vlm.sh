#!/bin/bash

# ==============================================================================
# Configuration for VLM MCMC
# ==============================================================================
GENERATION="${GENERATION:-true}"  # Set to false (e.g. GENERATION=false bash scripts/vlm.sh) to skip generation and only run evaluate.py

CONFIG="configs/config_vlm.yaml"

# The input target directories to test against
INPUT_DIRS=(
    "images/targets/gpt/faces/"
    "images/targets/z-image/faces/"
    "images/targets/ffhq/faces/"
)

# Optional labels for clean output suffix
LABELS=(
    "gpt"
    "z-image"
    "ffhq"
)

STRENGTH=0.6

# ==============================================================================
# Experiment Loop
# ==============================================================================
echo "Starting VLM MCMC Experiments..."

for i in "${!INPUT_DIRS[@]}"; do
    input_dir="${INPUT_DIRS[$i]}"
    label="${LABELS[$i]}"
    
    out_suffix="_${label}"

    echo "==========================================================="
    echo "Running VLM MCMC"
    echo "Input Dir: ${input_dir}"
    echo "Label/Suffix: ${label}"
    echo "==========================================================="
    
    if [ "$GENERATION" == "true" ]; then
        uv run main_vlm.py \
            --config "${CONFIG}" \
            --input_dir "${input_dir}" \
            --output_dir_suffix "${out_suffix}" \
            --strength "${STRENGTH}"
            
        echo "Finished VLM MCMC generation for ${label}"
    else
        echo "GENERATION=false: Skipping VLM MCMC generation. Proceeding directly to evaluation."
    fi
    
    # Run Evaluation
    echo "Running Evaluation..."
    uv run evaluate.py \
        --config "${CONFIG}" \
        --input_dir "${input_dir}" \
        --output_dir_suffix "${out_suffix}"
    
    echo "-----------------------------------------------------------"
done

echo "All VLM MCMC experiments completed!"
