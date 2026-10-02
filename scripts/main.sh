#!/bin/bash

# ==============================================================================
# Configuration
# ==============================================================================
# Constants that you want to keep the same across all experiments
GENERATION="${GENERATION:-true}"  # Set to false (e.g. GENERATION=false bash scripts/main.sh) to skip generation and only run evaluate.py
NUM_SAMPLES_INIT="${NUM_SAMPLES_INIT:-16}"
NUM_LOOP="${NUM_LOOP:-1024}"

# The targets (configs) you want to test
CONFIGS=(
    "configs/config_gpt_target_lpips.yaml"
    "configs/config_z_target_lpips.yaml"
    "configs/config_ffhq_target_lpips.yaml"
)

# The MCMC acceptance functions you want to compare
ACCEPT_FUNCS=(
    "barker"
    "metropolis"
    "hill"
)

# The strengths to test
STRENGTHS=(
    "null"
)

# ==============================================================================
# Experiment Loop
# ==============================================================================
echo "Starting Experiments..."

for config in "${CONFIGS[@]}"; do
    for accept_func in "${ACCEPT_FUNCS[@]}"; do
        for strength in "${STRENGTHS[@]}"; do
            
            if [ "$strength" == "null" ]; then
                strength_arg=""
                out_suffix="_${accept_func}"
            else
                strength_arg="--strength ${strength}"
                out_suffix="_${accept_func}_str${strength}"
            fi

            echo "==========================================================="
            echo "Running Config: ${config}"
            echo "Acceptance Func: ${accept_func}"
            echo "Strength: ${strength}"
            echo "==========================================================="
            
            # Execute the uv script with overrides
            if [ "$GENERATION" == "true" ]; then
                uv run main.py \
                    --config "${config}" \
                    --acceptance_prob_func "${accept_func}" \
                    --num_samples_init "${NUM_SAMPLES_INIT}" \
                    --num_loop "${NUM_LOOP}" \
                    --output_dir_suffix "${out_suffix}" \
                    ${strength_arg}
                    
                echo "Finished MCMC generation for ${config} with ${accept_func} (str=${strength})"
            else
                echo "GENERATION=false: Skipping MCMC generation. Proceeding directly to evaluation."
            fi
            
            # Run Evaluation
            echo "Running Evaluation..."
            uv run evaluate.py \
                --config "${config}" \
                --output_dir_suffix "${out_suffix}"
            
            echo "-----------------------------------------------------------"
        done
    done
done

echo "All MCMC experiments completed!"

# ==============================================================================
# Random Sampling Baseline Loop
# ==============================================================================
echo ""
echo "Starting Random Sampling Baselines..."
NUM_SAMPLES_TOTAL=$((NUM_SAMPLES_INIT + NUM_LOOP))

for config in "${CONFIGS[@]}"; do
    echo "==========================================================="
    echo "Running Random Baseline for Config: ${config}"
    echo "Total Samples (N): ${NUM_SAMPLES_TOTAL}"
    echo "==========================================================="
    
    if [ "$GENERATION" == "true" ]; then
        uv run main_rand.py \
            --config "${config}" \
            --num_samples_total "${NUM_SAMPLES_TOTAL}" \
            --output_dir_suffix "_rand"
            
        echo "Finished Random Baseline generation for ${config}"
    else
        echo "GENERATION=false: Skipping Random Baseline generation. Proceeding directly to evaluation."
    fi
    
    # Run Evaluation
    echo "Running Evaluation..."
    uv run evaluate.py \
        --config "${config}" \
        --output_dir_suffix "_rand"
        
    echo "-----------------------------------------------------------"
done

echo "All experiments (MCMC + Random Baseline) completed!"
