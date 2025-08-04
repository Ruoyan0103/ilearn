from ase import Atoms
from ase.io import write, read
import numpy as np
from ase.build import bulk
from ase.calculators.vasp import Vasp
import os, shutil
import sys
import argparse
import copy 

def create_distortedBulk(nstruc, outfile, lattice_constant=4.2749):

    calc = Vasp(               
        command='srun vasp_std',
        setups={'Ge':'_d'},

        # Initialisation: 
        istart = 0,         # (Default = 0; from scratch) 

        # Ionic:
        ibrion = -1,        # (Static calculation, default for nsw=0)
        nsw    = 0,         # (Max ionic steps) 

        # Electronic:
        ediff  = 1E-07,     # (SCF energy convergence; eV) 
        ismear = 0,         # (Electronic temperature, Gaussian smearing)
        sigma  = 0.05,      # (Smearing value in eV)
        nelm   = 100,       # (Max SCF steps)   
        gga    = 'PE',      # (Exchange-correlation functional)

        # Plane wave basis set:
        encut  = 500,         # (Default = largest ENMAX in the POTCAR file)
        prec   = 'Accurate',  # (Accurate forces are required)
        lasph  = True,        # (Non-spherical elements included)

        # Reciprocal space
        kspacing = 0.15,    # (Smallest spacing between k points in 1/A)
        kgamma   = True,    # (Default = Gamma centered)

        # Parallelisation: 
        ncore  = 4,         # ()

        # Output features:
        lwave  = False,     # (WAVECAR is NOT written out)
        lcharg = False      # (CHGCAR is NOT written out)

        )
    
    bulk_list = []

    Ge_bulk = bulk('Ge', 'fcc', a=lattice_constant)

    # perfect cell matrix 
    perfect_cell = Ge_bulk.get_cell()
    maxVal = np.max(perfect_cell)

    # perfect volume
    perfect_vol = Ge_bulk.get_volume()
    natoms = len(Ge_bulk)

    # random cell matrix
    dist_max = 0.1                                             # distortion rate
    for n in range(int(nstruc)):
        rands = np.random.uniform(-dist_max, dist_max, 9)         # 9 random numbers once a time
        rands = rands.reshape(3, 3)
        distortions = maxVal * rands
        new_cell = perfect_cell + distortions 
        Ge_bulk.set_cell(new_cell, scale_atoms=True)
        
        # calculate
        Ge_bulk.set_calculator(calc)
        e = Ge_bulk.get_potential_energy(force_consistent=True) # TOTEN

        # record volume
        v = Ge_bulk.get_volume()
        with open('database/volume', 'a') as vfile:
            vfile.write(str(n+1) + ' , ' + str(v/natoms) + '\n')
        
        # xyz info
        stress = Ge_bulk.get_stress(voigt=False)
        Ge_bulk.info['virial'] = v * -stress
        del Ge_bulk.calc.results['dipole']
        del Ge_bulk.calc.results['magmom']
        del Ge_bulk.calc.results['magmoms']
        Ge_bulk.info["config_type"] = "fcc_bulk"

        # deepcopy bulk matrix, forces and energy
        Ge_add = copy.deepcopy(Ge_bulk)
        bulk_list.append(Ge_add)
        write("mydatabase/random_%i.xyz" % n, Ge_add, format='extxyz')

    # write whole distorted bulk list
    write(outfile, bulk_list, format='extxyz')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--nstruc", dest = "nstruc")
    parser.add_argument("-l", "--lattice", dest = "lattice")
    parser.add_argument("-f", "--outfile", dest = "outfile")
    args = parser.parse_args()
    create_distortedBulk(int(args.nstruc), str(args.outfile))
    






  
