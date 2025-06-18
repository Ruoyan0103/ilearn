#!/bin/bash

#SBATCH --time=00:15:00
#SBATCH --partition=sumo
#SBATCH --account=sumo
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=2G
#SBATCH --job-name=in.dispersion


module load gcc openmpi fftw openblas eigen ffmpeg zstd
srun lmp < in.dispersion
