import itertools
import subprocess
import os
from multiprocessing.pool import ThreadPool

# Define the grid
lr_inits = [0.1]
alphas = [0.1, 0.25, 0.5, 1., 5., 50.,]


# GPUs to use
gpus = [0,1,2,3,4,5]

# Make output directory if not exists
os.makedirs("logs", exist_ok=True)

# Generate all combinations
grid = list(itertools.product(lr_inits, alphas))

# Prepare commands
commands = []
for idx, (lr_init, alpha) in enumerate(grid):
    gpu_id = gpus[idx % len(gpus)]
    log_path = f"logs/sgula_{alpha}.log"
    cmd = (
        f"CUDA_VISIBLE_DEVICES={gpu_id} "
        f"python3 cifar/sgula.py "
        f"--sampler_type='sa-sgula' --lr_init={lr_init} --alpha={alpha}"
        f"> {log_path} 2>&1"
    )
    commands.append(cmd)

def run_command(cmd):
    print(f"Launching: {cmd}")
    subprocess.run(cmd, shell=True)

# Use a thread pool (1 thread per GPU)
with ThreadPool(processes=len(gpus)) as pool:
    pool.map(run_command, commands)