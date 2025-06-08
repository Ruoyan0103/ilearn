#!/bin/bash

#SBATCH --time=00:10:00
#SBATCH --partition=sumo
#SBATCH --account=sumo
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --mem=5G
#SBATCH --job-name=thermalize


module load gcc openmpi fftw openblas eigen ffmpeg zstd
srun lmp -in in.thermalize

