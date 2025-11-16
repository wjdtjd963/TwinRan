import os
import time
import subprocess
from datetime import datetime
import multiprocessing as mp

def get_gpu_utilization(gpu_id):
    try:
        result = subprocess.run(
            ['nvidia-smi', f'--id={gpu_id}', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        return int(result.stdout.strip())
    except (subprocess.SubprocessError, ValueError):
        print(f"Error getting GPU {gpu_id} utilization")
        return -1

def monitor_and_restart_workers(monitoring_active, processes, worker_args, gpu_ids, utilization_threshold=5, check_interval=60, output_dir=None):
    monitor_log_path = os.path.join(output_dir, "log", "gpu_monitor_log.txt")
    def log_message(message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_message = f"[{timestamp}] {message}"
        print(full_message)
        try:
            with open(monitor_log_path, 'a') as log_file:
                log_file.write(full_message + "\n")
                log_file.flush()
        except Exception as e:
            print(f"Error writing to monitor log: {e}")
    log_message(f"GPU monitoring started with threshold {utilization_threshold}%, check interval {check_interval}s")
    log_message(f"Monitoring GPUs: {gpu_ids}")
    last_log_times = {}
    completed_workers = set()
    for i, args in enumerate(worker_args):
        json_file = args[-1]
        log_file = os.path.join(output_dir, "log", f"{os.path.splitext(json_file)[0]}_log.txt")
        if os.path.exists(log_file):
            last_log_times[i] = os.path.getmtime(log_file)
        else:
            last_log_times[i] = 0
    while monitoring_active:
        time.sleep(check_interval)
        for i, (process, args, gpu_id) in enumerate(zip(processes, worker_args, gpu_ids)):
            if i in completed_workers:
                continue
            utilization = get_gpu_utilization(gpu_id)
            log_message(f"GPU {gpu_id} utilization: {utilization}%")
            json_file = args[-1]
            log_file = os.path.join(output_dir, "log", f"{os.path.splitext(json_file)[0]}_log.txt")
            log_updated = False
            if os.path.exists(log_file):
                current_mtime = os.path.getmtime(log_file)
                if i in last_log_times and current_mtime > last_log_times[i]:
                    log_updated = True
                    last_log_times[i] = current_mtime
            worker_completed = False
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r') as f:
                        log_content = f.read()
                        if "All UEs processed" in log_content:
                            worker_completed = True
                            completed_workers.add(i)
                            log_message(f"Worker for GPU {gpu_id} (processing {json_file}) has completed all UEs")
                except Exception as e:
                    log_message(f"Error reading log file for worker {i}: {e}")
            if worker_completed:
                continue
            if utilization < utilization_threshold:
                log_message(f"GPU {gpu_id} utilization is below threshold of {utilization_threshold}%")
                if process.is_alive():
                    if log_updated:
                        log_message(f"Process for GPU {gpu_id} (PID: {process.pid}) has low GPU usage but log was recently updated. Assuming it's working normally.")
                        continue
                    log_message(f"Process for GPU {gpu_id} (PID: {process.pid}) has low GPU usage and log hasn't been updated. Terminating...")
                    try:
                        process.terminate()
                        process.join(5)
                        if process.is_alive():
                            log_message(f"Process didn't terminate gracefully. Killing...")
                            process.kill()
                    except Exception as e:
                        log_message(f"Error terminating process: {e}")
                else:
                    log_message(f"Process for GPU {gpu_id} is not alive and job is not complete")
                log_message(f"Starting new worker for GPU {gpu_id} with {args[-1]}")
                new_process = mp.Process(
                    target=worker_args[0][0],
                    args=args,
                    name=f"worker_{args[-1]}_GPU{gpu_id}_restarted_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                new_process.start()
                processes[i] = new_process
                log_message(f"Worker restarted on GPU {gpu_id} with PID {new_process.pid}")
            else:
                if not process.is_alive():
                    log_message(f"Process for GPU {gpu_id} is not alive but GPU utilization is {utilization}%. Starting new process...")
                    new_process = mp.Process(
                        target=worker_args[0][0],
                        args=args,
                        name=f"worker_{args[-1]}_GPU{gpu_id}_restarted_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                    new_process.start()
                    processes[i] = new_process
                    log_message(f"Worker restarted on GPU {gpu_id} with PID {new_process.pid}")
                else:
                    log_message(f"Process for GPU {gpu_id} (PID: {process.pid}) is running normally with {utilization}% GPU utilization")
        if len(completed_workers) == len(processes) or all(not p.is_alive() for p in processes):
            log_message("All workers have finished. Stopping monitoring.")
            break
