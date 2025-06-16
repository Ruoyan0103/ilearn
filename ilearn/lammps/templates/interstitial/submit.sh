#!/bin/bash

#SBATCH --time=00:03:00
#SBATCH --partition=sumo
#SBATCH --account=sumo
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --mem=5G
#SBATCH --job-name={in_file}


module load gcc openmpi fftw openblas eigen ffmpeg zstd
srun lmp -in {in_file}

