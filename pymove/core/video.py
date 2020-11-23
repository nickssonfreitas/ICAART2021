import subprocess

def nvidia_gpu_count():
    try:
        return str(subprocess.check_output(["nvidia-smi", "-L"])).count('UUID')
    except:
        return 0