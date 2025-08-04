from ase.build import bulk
from ase.calculators.vasp import Vasp
from ase.io import write
import os, shutil
import sys
import argparse

'''
    static calculations for ground state lattice constant
'''
def calLattice_constant(lattice_constant):
    atoms = bulk('Ge', 'diamond', a=lattice_constant, cubic=False)

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

    atoms.set_calculator(calc)

    folder = 'Final/' + str(lattice_constant)
    if not os.path.exists(folder): os.mkdir(folder)
### Save POSCAR
    write(folder + '/bulk-' + str(lattice_constant) + '.vasp', atoms, format='vasp') 
    write(folder + '/bulk-' + str(lattice_constant) + '.data', atoms, format='lammps-data') 

    n = len(atoms)
    e = atoms.get_potential_energy(force_consistent=True)/n         # TOTEN/number of atoms
    v = atoms.get_volume()/n                                        # volume/number of atoms
    # f = dimer.calc.results['forces']
    # fz = -f[1][2]

### Save lattice_constant-energy-force as results 
    with open('Final/lattice-vol-energy', 'a') as file:
        file.write(str(lattice_constant) + ' , ' + str(v) + ' , ' + str(e) + '\n')

### Save OUTCAR, OSZICAR, vasprun.xml
    shutil.copy('OUTCAR', folder + '/OUTCAR')
    shutil.copy('OSZICAR', folder + '/OSZICAR')
    shutil.copy('vasprun.xml', folder + '/vasprun.xml')


if __name__ == "__main__":
    '''
    optlat = 5.7620
    rate = 0.1
    calLattice_constant(optlat)
    calLattice_constant(round(optlat * (1-rate), 4))
    calLattice_constant(round(optlat * (1+rate), 4))
    '''
    assert len(sys.argv) > 3
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start", dest = "start")
    parser.add_argument("-e", "--end", dest = "end")
    parser.add_argument("-i", "--interval", dest = "interval")
    args = parser.parse_args()
    start = float(args.start) 
    end = float(args.end)
    interval = float(args.interval)

    for l in range(int(start/interval), int(end/interval+1), 1):
        calLattice_constant(round(l*interval,4))
