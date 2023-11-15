#!/bin/sh

#SBATCH -J CalibrationStudy
#SBATCH --account=carney-brainstorm-condo
#SBATCH --time=10:00:00
#SBATCH --array=0-29
#SBATCH --mem=4GB
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

# Print key runtime properties for records
echo Master process running on `hostname`
echo Directory is `pwd`
echo Starting execution at `date`
echo Current PATH is $PATH

module load graphviz/2.40.1
module load python/3.9.0
module load git/2.29.2
source ~/RL-SSM/venv/bin/activate
cd /users/jhewson/RL-SSM/tests

# Run job
python slurm_tests.py --slurm_id $SLURM_ARRAY_TASK_ID