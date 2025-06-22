import os
import subprocess
from multiprocessing.pool import ThreadPool

def find_directories_needing_eval():
    """Find directories that have train_metrics.json but not eval_metrics_16.json"""
    directories_to_eval = []
    
    # Search in both cifar and cifar100 directories
    for base_dir in ['cifar', 'cifar100']:
        if not os.path.exists(base_dir):
            print(f"Warning: {base_dir} directory not found, skipping...")
            continue
            
        for root, dirs, files in os.walk(base_dir):
            # Check if this directory has train_metrics.json
            if 'train_metrics.json' in files:
                # Check if it does NOT have eval_metrics_16.json
                if 'eval_metrics_16.json' not in files:
                    directories_to_eval.append(root)
                    print(f"Found directory needing evaluation: {root}")
    
    return directories_to_eval

# GPUs to use
gpus = [1, 2, 3, 4]

# Find all directories that need evaluation
directories_to_eval = find_directories_needing_eval()

if not directories_to_eval:
    print("No directories found that need evaluation.")
    exit(0)

print(f"\nFound {len(directories_to_eval)} directories needing evaluation")

# Build evaluation commands
commands = []
for idx, ckpt_dir in enumerate(directories_to_eval):
    gpu_id = gpus[idx % len(gpus)]
    log_file = f"logs/eval.log"
    
    # Determine which eval script to use based on directory path
    if ckpt_dir.startswith('cifar100/'):
        eval_script = "cifar100/eval.py"
    elif ckpt_dir.startswith('cifar/'):
        eval_script = "cifar/eval.py"
    else:
        print(f"Warning: Unknown directory structure for {ckpt_dir}, skipping...")
        continue
    
    cmd = (
        f"CUDA_VISIBLE_DEVICES={gpu_id} "
        f"python3 {eval_script} --dir='{ckpt_dir}' >> {log_file} 2>&1"
    )
    commands.append(cmd)

print(f"\nPrepared {len(commands)} evaluation commands")

# Function to run one command
def run_command(cmd):
    print(f"Launching: {cmd}")
    subprocess.run(cmd, shell=True)

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Run using thread pool, one thread per GPU
if commands:
    with ThreadPool(processes=len(gpus)) as pool:
        pool.map(run_command, commands)
    print("All evaluation jobs submitted!")
else:
    print("No commands to run.")