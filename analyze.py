import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

METHOD_NAME_MAP = {
    'barker': 'BAMC (Ours)',
    'hill': 'Hill',
    'metropolis': 'MH',
    'rand': 'Random',
}

METHOD_COLORS = {
    'BAMC (Ours)': '#1f77b4',
    'Ours': '#1f77b4',
    'MH': '#ff7f0e',
    'Hill': '#2ca02c',
    'Random': '#d62728',
}

def get_dataset_and_method(dir_name):
    if dir_name.startswith("faces_"):
        parts = dir_name.replace("faces_", "").split("_")
        dataset = parts[0]
        method = "_".join(parts[1:])
    elif dir_name.startswith("vlm_"):
        dataset = dir_name.replace("vlm_", "")
        method = "vlm"
    else:
        dataset, method = "unknown", "unknown"
    return dataset, method

def main():
    parser = argparse.ArgumentParser(description="Analyze MCMC and baseline results.")
    parser.add_argument("--base_dir", type=str, default="images/results", help="Directory containing experiment results.")
    parser.add_argument("--out_dir", type=str, default="images/analysis", help="Directory to save analysis plots.")
    parser.add_argument("--include_rand", action="store_true", help="Include random baseline in convergence plots.")
    parser.add_argument("--include_ablations", action="store_true", help="Include ablation runs (e.g. str0.3) in plots.")
    args = parser.parse_args()

    base_dir = args.base_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    
    # Collect all CSV files
    mcmc_files = glob.glob(f"{base_dir}/**/mcmc_history.csv", recursive=True)
    rand_files = glob.glob(f"{base_dir}/**/rand_history.csv", recursive=True)
    
    all_data = []
    
    # Process MCMC
    for f in mcmc_files:
        parts = f.split(os.sep)
        if "results" not in parts:
            continue
        idx = parts.index("results")
        if len(parts) <= idx + 2:
            continue
        dir_name = parts[idx + 1]
        task_name = parts[idx + 2]
        
        dataset, method = get_dataset_and_method(dir_name)
        
        try:
            df = pd.read_csv(f)
            df['dataset'] = dataset
            df['method'] = method
            df['task'] = task_name
            all_data.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
        
    # Process Rand
    if args.include_rand:
        for f in rand_files:
            parts = f.split(os.sep)
            if "results" not in parts:
                continue
            idx = parts.index("results")
            if len(parts) <= idx + 2:
                continue
            dir_name = parts[idx + 1]
            task_name = parts[idx + 2]
            
            dataset, method = get_dataset_and_method(dir_name)
            
            try:
                df = pd.read_csv(f)
                df['dataset'] = dataset
                df['method'] = method
                df['task'] = task_name
                df['status'] = 'SAMPLED'
                if 'current_distance' not in df.columns:
                    df['current_distance'] = df['candidate_distance']
                all_data.append(df)
            except Exception as e:
                print(f"Error reading {f}: {e}")
        
    if not all_data:
        print(f"No result CSV files found under '{base_dir}'.")
        return
        
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Exclude VLM from standard comparison plots
    combined_df = combined_df[combined_df['method'] != 'vlm']
    
    # Exclude ablation runs (e.g. str0.3) unless requested
    if not args.include_ablations:
        combined_df = combined_df[~combined_df['method'].str.contains('str')]
    
    if not args.include_rand:
        combined_df = combined_df[combined_df['method'] != 'rand']
        
    if combined_df.empty:
        print("No matching data remaining after filtering.")
        return

    # Calculate initial distance for normalization
    def get_init_dist(group):
        first_row = group.iloc[0]
        if 'delta_D' in first_row and pd.notna(first_row['delta_D']):
            return first_row['candidate_distance'] - first_row['delta_D']
        return first_row['best_distance']
        
    mcmc_only = combined_df[~combined_df['method'].isin(['rand', 'Random'])]
    if mcmc_only.empty:
        print("No MCMC runs found to calculate initial distances.")
        return
        
    init_dists = mcmc_only.groupby(['dataset', 'method', 'task']).apply(get_init_dist, include_groups=False).reset_index(name='init_distance')
    
    # Init distance is the same for a task across all methods, average it to merge
    task_init_dists = init_dists.groupby(['dataset', 'task'])['init_distance'].mean().reset_index()
    combined_df = combined_df.merge(task_init_dists, on=['dataset', 'task'])
    
    # Calculate percentage improvement
    combined_df['norm_current'] = (combined_df['init_distance'] - combined_df['current_distance']) / (combined_df['init_distance'] + 1e-8) * 100
    combined_df['norm_best'] = (combined_df['init_distance'] - combined_df['best_distance']) / (combined_df['init_distance'] + 1e-8) * 100

    # Replace method column values in batch
    combined_df['method'] = combined_df['method'].replace(METHOD_NAME_MAP)
    
    # ---------------------------------------------------------
    # 1. Acceptance Rate Analysis
    # ---------------------------------------------------------
    print("Generating Acceptance Rate Plots...")
    mcmc_df = combined_df[~combined_df['method'].isin(['rand', 'Random'])]
    
    datasets = mcmc_df['dataset'].unique()
    for ds in datasets:
        ds_data = mcmc_df[mcmc_df['dataset'] == ds]
        
        # Calculate counts per status and method
        counts = ds_data.groupby(['method', 'status']).size().unstack(fill_value=0)
        
        # Ensure columns exist
        for col in ['ACCEPTED', 'ACCEPTED WORSE', 'REJECTED']:
            if col not in counts.columns:
                counts[col] = 0
                
        # Reorder columns
        counts = counts[['ACCEPTED', 'ACCEPTED WORSE', 'REJECTED']]
        
        # Calculate percentages relative to total attempts
        row_sums = counts.sum(axis=1).replace(0, 1)
        percentages = counts.div(row_sums, axis=0) * 100
        
        # Exclude REJECTED from the plot to scale properly
        percentages = percentages[['ACCEPTED', 'ACCEPTED WORSE']]
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        percentages.plot(kind='bar', stacked=True, color=['#2ca02c', '#ff7f0e'], ax=ax)
        
        plt.title(f'Acceptance Rate ({ds.upper()})')
        plt.xlabel('Method')
        plt.ylabel('Acceptance Rate (%)')
        plt.xticks(rotation=0)
        plt.legend(title='Status', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'acceptance_rate_{ds}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # ---------------------------------------------------------
    # 2. Convergence Analysis (Average Best Distance)
    # ---------------------------------------------------------
    print("Generating Convergence Plots...")
    for ds in combined_df['dataset'].unique():
        ds_data = combined_df[combined_df['dataset'] == ds]
        
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=ds_data, x='iteration', y='norm_best', hue='method', palette=METHOD_COLORS, errorbar='sd')
        
        plt.title(f'Average Convergence Curve ({ds.upper()})')
        plt.xlabel('Iteration')
        plt.ylabel('Best Improvement (%)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'convergence_avg_{ds}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # ---------------------------------------------------------
    # 3. Example Trajectory (Current State only, all methods)
    # ---------------------------------------------------------
    print("Generating Example Trajectory Plots...")
    for ds in combined_df['dataset'].unique():
        plt.figure(figsize=(10, 5))
        for method in combined_df[combined_df['dataset'] == ds]['method'].unique():
            if method in ('rand', 'Random'):
                continue
                
            task0_data = combined_df[(combined_df['dataset'] == ds) & (combined_df['method'] == method) & (combined_df['task'] == 'task_0')]
            if task0_data.empty:
                continue
                
            plt.plot(task0_data['iteration'], task0_data['norm_current'], alpha=0.7, label=method, linewidth=1.5, color=METHOD_COLORS.get(method, 'gray'))
            
        plt.title(f'MCMC Current State Trajectory Example ({ds.upper()})')
        plt.xlabel('Iteration')
        plt.ylabel('Current Improvement (%)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'trajectory_example_{ds}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Analysis complete! Plots saved to {out_dir}")

if __name__ == "__main__":
    main()