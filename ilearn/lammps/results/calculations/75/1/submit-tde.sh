#!/bin/bash

#SBATCH --time=00:10:00
#SBATCH --partition=sumo
#SBATCH --account=sumo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=2G
#SBATCH --job-name=tde


module load gcc openmpi fftw openblas eigen ffmpeg zstd
srun lmp -in in.tde 

