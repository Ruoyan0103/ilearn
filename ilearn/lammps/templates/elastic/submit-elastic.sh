#!/bin/bash

#SBATCH --time=00:08:00
#SBATCH --partition=sumo
#SBATCH --account=sumo
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --mem=5G
#SBATCH --job-name=elastic


module load gcc openmpi fftw openblas eigen ffmpeg zstd
srun lmp -in in.elastic

