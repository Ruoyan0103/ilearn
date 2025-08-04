from ase import Atoms
from ase.io import write, read
import numpy as np
from ase.calculators.vasp import Vasp
import os, shutil
import sys
import argparse
import copy 

'''
    Usage: static calculations calculater
    Parameters: 
    Return: calculater
'''
def set_cal():
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
    return calc


'''
    Usage: create distorted bulks
    Parameters: nstruc: number of structures, outfile: output bulk list, a: optimized a0
    Return: 
'''
def crt_bulks(nstruc, outfile, a=4.0561, c_a_ratio=1.6496):
    bulk_list = []
    a_per = a

    dist_max = 0.1   # distortion rate                                          
    for n in range(int(nstruc)):
        rand = np.random.uniform(1-dist_max, 1+dist_max, 1)
        a = a_per*rand[0]
        c = a*c_a_ratio
        # create bulk
        z = 0.06281
    
        lattice = np.array([[0.5*a, -0.5*np.sqrt(3)*a, 0], [0.5*a, 0.5*np.sqrt(3)*a, 0], [0, 0, c]])
        basis   = np.array([[0.5*a, a/np.sqrt(12), z*c], [0.5*a, -a/np.sqrt(12), (0.5+z)*c], 
                       [0.5*a, a/np.sqrt(12), (0.5-z)*c], [0.5*a, -a/np.sqrt(12), (-z)*c]])
        pbc = [1, 1, 1]
        symbols = 'Ge4'                                                     
        Ge_bulk = Atoms(symbols, positions=basis, cell=lattice, pbc=pbc)

        # record volume
        v = Ge_bulk.get_volume()
        natoms = len(Ge_bulk)
        with open('database/volume', 'a') as vfile:
            vfile.write(str(v/natoms) + '\n')
        
        # calculate
        Ge_bulk.set_calculator(set_cal())
        e = Ge_bulk.get_potential_energy(force_consistent=True) # TOTEN

        # xyz info
        stress = Ge_bulk.get_stress(voigt=False)
        Ge_bulk.info['virial'] = v * -stress
        del Ge_bulk.calc.results['dipole']
        del Ge_bulk.calc.results['magmom']
        del Ge_bulk.calc.results['magmoms']
        Ge_bulk.info["config_type"] = "hd"

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
    crt_bulks(int(args.nstruc), str(args.outfile))
    






  
