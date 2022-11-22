#!/bin/bash

#SBATCH --job-name=cp
#SBATCH --partition=gpu
#SBATCH --time=100
#SBATCH --mem=60g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1


date

source /opt/easybuild/software/Anaconda3/2019.07/etc/profile.d/conda.sh
conda init bash
conda activate tasks-pip

rm -r tmp_out
python run_example_cellpose.py

date
