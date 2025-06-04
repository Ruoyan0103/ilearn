#!/bin/bash

#SBATCH --time=00:10:00
#SBATCH --partition=sumo
#SBATCH --account=sumo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=2G
#SBATCH --job-name=tde

#SBATCH --array=0-20  # Change this if you have a different count

module load gcc openmpi fftw openblas eigen ffmpeg zstd
FOLDER=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" folder_list.txt)

cd "$FOLDER"
srun lmp -in in.tde
