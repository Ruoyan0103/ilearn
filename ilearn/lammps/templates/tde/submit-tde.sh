#!/bin/bash

#SBATCH --time=06:00:00
#SBATCH --partition=batch
##SBATCH --account=sumo
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --mem=20G
#SBATCH --job-name=tde


module load gcc openmpi fftw openblas eigen ffmpeg zstd
srun lmp -in in.tde 

