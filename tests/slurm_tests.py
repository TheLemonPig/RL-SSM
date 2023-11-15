import numpy as np
import random
import os
import argparse
from tests.test_rl_calibration import test_rl_calibration

parser = argparse.ArgumentParser("parser")
parser.add_argument('--slurm_id', type=int, default=0, help='number of slurm array')
args = parser.parse_args()
rep = args.slurm_id

# Tests
test_rl_calibration(rep)

# for test_script in os.listdir('tests/'):
#     exec(test_script)
