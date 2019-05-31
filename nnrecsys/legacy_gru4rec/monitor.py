import os
import time

import GPUtil
import multiprocessing as mp
import pickle


def record_gpu_usage(path: str):
    running_record = []
    while True:
        for gpu in GPUtil.getGPUs():
            running_record.append({'id': gpu.id,
                                   'memoryUtil': gpu.memoryUtil,
                                   'load': gpu.load,
                                   'time': int(time.time())})

        with open(os.path.join(path, 'gpu_util.pkl'), 'wb') as f:
            pickle.dump(running_record, f)
        time.sleep(0.1)


def monitor_gpu(path):
    worker = mp.Process(target=record_gpu_usage, args=(path,))
    worker.daemon = True
    worker.start()
