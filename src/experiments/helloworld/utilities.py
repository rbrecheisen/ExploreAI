import subprocess


def get_gpu_info():
    command = 'nvidia-smi --query-gpu=fan.speed --format=csv,noheader,nounits'
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
    fan_speeds = [int(x) for x in result.stdout.strip().split('\n')]
    return {0: fan_speeds[0], 1: fan_speeds[2]}


def select_idle_gpu_device():
    gpu_info = get_gpu_info()
    device_nr = min(gpu_info, key=lambda k: gpu_info[k])
    return f'cuda:{device_nr}'
