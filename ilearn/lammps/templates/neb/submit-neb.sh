#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH --partition=sumo
#SBATCH --account=sumo
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=40
#SBATCH --mem=5G
#SBATCH --job-name=validate
##SBATCH	--output=job.out
##SBATCH	--error=job.err


module load gcc openmpi fftw openblas eigen ffmpeg zstd scicomp-python-env
#python defect.py
#srun lmp_mpi -in write.in
srun --ntasks={ntasks} lmp_mpi -partition {size}x10 -in in.neb

