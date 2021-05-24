import os
import multiprocessing
import subprocess
import numpy as np
import shlex
import time
import math
import socket
from time import localtime, strftime
import hashlib
import itertools
import random

import run_utils

from constants.config_sets.ale import CONFIG_SETS as CONFIG_SETS_ALE
from constants.config_sets.box2d import CONFIG_SETS as CONFIG_SETS_BOX2D
from constants.config_sets.mape import CONFIG_SETS as CONFIG_SETS_MAPE
from constants.config_sets.minatar import CONFIG_SETS as CONFIG_SETS_MINATAR

lock = multiprocessing.Lock()

def work(cmd):
    lock.acquire()
    time.sleep(1)
    lock.release()
    return subprocess.call(shlex.split(cmd), shell=False)  # return subprocess.call(cmd, shell=False)


if __name__ == '__main__':

    machine_name = 'UNDEFINED'
    machine_id = 0
    n_processors = 1
    n_seeds = 1

    seeds_all = list(range((machine_id + 1) * 100 + 1, (machine_id + 1) * 100 + 21))

    # ------------------------------------------------------------------------------------------------------------------

    env_keys = ['BOX2D-LunarLander']
    run_config_idx = [1000]

    # ------------------------------------------------------------------------------------------------------------------

    i_parameter_set = 0
    i_command = 0
    commands = []

    seeds = [302]

    print('Seeds: {}\n'.format(seeds))

    for env_key in env_keys:
        for run_config_id in run_config_idx:

            run_config = None
            if 'ALE' in env_key:
                run_config = CONFIG_SETS_ALE[run_config_id]
            elif 'BOX2D' in env_key:
                run_config = CONFIG_SETS_BOX2D[run_config_id]
            elif 'MAPE' in env_key:
                run_config = CONFIG_SETS_MAPE[run_config_id]
            elif 'MinAtar' in env_key:
                run_config = CONFIG_SETS_MINATAR[run_config_id]

            run_config[0]['env-key'] = env_key

            run_id = run_utils.generate_run_id(run_config[0]['env-key'], run_config[0]['dqn-type'], run_config_id)

            run_config[0]['machine-name'] = str(machine_name)
            run_config[0]['process-index'] = str(i_command % n_processors)
            run_config[0]['run-id'] = str(run_id)

            for seed in seeds:
                seed_run_config = run_config.copy()
                seed_run_config[0]['seed'] = str(seed)
                commands.append(run_utils.config_to_command(seed_run_config))
                i_command += 1
            i_parameter_set += 1

    # ------------------------------------------------------------------------------------------------------------------

    print('There are {} commands.'.format(len(commands)))

    n_cycles = int(math.ceil(len(commands) / n_processors))

    print('There are {} cycles.'.format(n_cycles))

    for i_cycle in range(n_cycles):
        pool = multiprocessing.Pool(processes=n_processors)

        start = (n_processors * i_cycle)
        end = start + n_processors

        print('start and end:', start, end)

        if end > len(commands):
            end = len(commands)

        print('start and end:', start, end)

        print(pool.map(work, commands[(n_processors * i_cycle):(n_processors * i_cycle) + n_processors]))
