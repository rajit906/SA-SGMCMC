import itertools
import subprocess
import os
import time
import signal
import atexit
from concurrent.futures import ThreadPoolExecutor, as_completed

# Grid search parameters - updated to match your actual run
lr_inits = ["0.01", "0.02", "0.03", "0.04", "0.05", "0.06", "0.07", "0.08", "0.09", "0.1"]
seeds = [1,2,3,4,5]
gpus = [3, 5, 6, 7]  # Updated to match your actual GPU usage

os.makedirs("logs", exist_ok=True)
grid = list(itertools.product(lr_inits, seeds))

def clear_gpu_memory(gpu_id):
    """Clear GPU memory for a specific GPU"""
    try:
        # Fixed indentation issue in the Python command string
        clear_cmd = f'''python3 -c "
import torch
import gc
import sys
try:
    if torch.cuda.is_available():
        torch.cuda.set_device({gpu_id})
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        print(f'GPU {gpu_id} memory cleared successfully')
    else:
        print('CUDA not available')
except Exception as e:
    print(f'Error clearing GPU {gpu_id}: {{e}}')
    sys.exit(1)
"'''
        
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        result = subprocess.run(clear_cmd, shell=True, env=env, 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print(f"✓ GPU {gpu_id} memory cleared successfully")
        else:
            print(f"⚠ Warning: Failed to clear GPU {gpu_id}: {result.stderr.strip()}")
            
    except subprocess.TimeoutExpired:
        print(f"⚠ Warning: Timeout clearing GPU {gpu_id}")
    except Exception as e:
        print(f"⚠ Warning: Exception clearing GPU {gpu_id}: {e}")

def clear_all_gpus():
    """Clear memory on all GPUs"""
    print("🧹 Clearing memory on all GPUs...")
    for gpu_id in gpus:
        clear_gpu_memory(gpu_id)

def cleanup_handler():
    """Global cleanup function"""
    print("\n🛑 Performing cleanup...")
    clear_all_gpus()
    print("✅ Cleanup complete")

def signal_handler(signum, frame):
    """Handle interrupt signals (Ctrl+C, termination)"""
    print(f"\n🚨 Received signal {signum}, cleaning up...")
    cleanup_handler()
    print("🏃 Exiting...")
    exit(0)

# Register cleanup functions
atexit.register(cleanup_handler)
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Termination

def run_command(cmd_tuple):
    cmd, gpu_id = cmd_tuple
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    print(f"🚀 Starting on GPU {gpu_id}: {cmd}")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, shell=True, env=env)
        end_time = time.time()
        
        if result.returncode == 0:
            print(f"✅ GPU {gpu_id} completed successfully in {end_time - start_time:.2f}s")
            return True
        else:
            print(f"❌ GPU {gpu_id} failed with return code {result.returncode} after {end_time - start_time:.2f}s")
            return False
        
    except KeyboardInterrupt:
        print(f"🛑 GPU {gpu_id} job interrupted by user")
        raise
    except Exception as e:
        end_time = time.time()
        print(f"💥 GPU {gpu_id} job crashed after {end_time - start_time:.2f}s: {e}")
        return False
    finally:
        # Always clear GPU memory after job completion/termination
        print(f"🧹 Cleaning GPU {gpu_id} memory...")
        clear_gpu_memory(gpu_id)

# Prepare commands with correct parameter names and formatting
commands = []
for lr_init, seed in grid:
    # Create log filename matching your pattern
    log_path = f"logs/sgld_lr{lr_init}_seed{seed}.log"
    
    # FIXED: Added space between parameters
    cmd = (
        f"python3 cifar/sgld.py "
        f"--sampler_type='sgld' "
        f"--lr_init={float(lr_init)} "
        f"--experiment_dir={lr_init} "  # Added space here
        f"--seed={seed} "               # Added space here for consistency
        f"> {log_path} 2>&1"
    )
    
    # Distribute jobs across GPUs
    gpu_id = gpus[len(commands) % len(gpus)]
    commands.append((cmd, gpu_id))

print(f"🎯 Starting grid search with {len(commands)} jobs across {len(gpus)} GPUs (3 jobs per GPU)")
print(f"📊 Estimated runtime: ~{len(commands) / (len(gpus) * 3) * 3:.1f} hours")

# Use ThreadPoolExecutor with 3 jobs per GPU
max_workers = len(gpus) * 3  # 3 jobs per GPU as requested
completed_jobs = 0
failed_jobs = 0

try:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_cmd = {executor.submit(run_command, cmd_tuple): cmd_tuple 
                        for cmd_tuple in commands}
        
        # Process completed jobs
        for future in as_completed(future_to_cmd):
            cmd_tuple = future_to_cmd[future]
            try:
                success = future.result()
                if success:
                    completed_jobs += 1
                else:
                    failed_jobs += 1
                    
                print(f"📈 Progress: {completed_jobs + failed_jobs}/{len(commands)} "
                      f"(✅ {completed_jobs} success, ❌ {failed_jobs} failed)")
                      
            except KeyboardInterrupt:
                print("🛑 Grid search interrupted by user")
                break
            except Exception as e:
                failed_jobs += 1
                print(f"💥 Job failed with error: {e}")
                print(f"📈 Progress: {completed_jobs + failed_jobs}/{len(commands)} "
                      f"(✅ {completed_jobs} success, ❌ {failed_jobs} failed)")

except KeyboardInterrupt:
    print("🛑 Grid search interrupted during setup")

finally:
    # Final cleanup and results
    print(f"\n📊 Final Results:")
    print(f"  ✅ Completed: {completed_jobs}")
    print(f"  ❌ Failed: {failed_jobs}")
    print(f"  📝 Total: {len(commands)}")
    
    print("\n🧹 Performing final cleanup...")
    clear_all_gpus()
    print("🎉 Grid search finished!")