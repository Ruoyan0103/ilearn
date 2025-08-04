from ase.io import read, write
import os, shutil, random
import numpy as np
from ase.build import bulk

#def displace(in_poscar, out_poscar, lattice_rate=0.01, atom_disp=0.1):
def displace(in_poscar, out_poscar, lattice_rate=0, atom_disp=0.15):
    '''
    lattice_rate: the strain rate
    atom_disp (Å): the stdev of rattle
    '''
    matrix = np.random.uniform(-lattice_rate, lattice_rate, (3, 3))
    Ge_bulk = read(in_poscar, format='vasp')
    new_cell = Ge_bulk.get_cell() + Ge_bulk.get_cell() * matrix
    Ge_bulk.set_cell(new_cell, scale_atoms=True)
    seed = random.randint(0, 2**32 - 1)
    Ge_bulk.rattle(stdev=atom_disp, seed=seed)
    write(out_poscar, Ge_bulk, format='vasp')

# def displace2(in_poscar, out_poscar, lattice_rate=0.01, atom_disp=0.1):
#     '''
#     lattice_rate: the strain rate
#     atom_disp (Å): the stdev of rattle
#     '''
#     matrix = np.random.uniform(-lattice_rate, lattice_rate, (3, 3))
#     Ge_bulk = read(in_poscar, format='vasp')
#     seed = int(time.time())
#     Ge_bulk.rattle(stdev=atom_disp, seed=seed)
#     new_cell = Ge_bulk.get_cell() + Ge_bulk.get_cell() * matrix
#     Ge_bulk.set_cell(new_cell, scale_atoms=True)
#     write(out_poscar, Ge_bulk, format='vasp')

def crt_rattled_copies(fn, tn):
    '''
    n: number of rattled copies of each POSCAR
    '''
    directory_path = '..'
    in_poscar = os.path.join(directory_path, '03-opt/s2/CONTCAR')
    rattled_path = os.path.join(directory_path, '01-rattled')
    print(rattled_path)
    for i in range(fn, tn):
        out_poscar = os.path.join(rattled_path, f'POSCAR-{i}')
        displace(in_poscar, out_poscar)

n = 10
fn = 20
tn = fn + n
crt_rattled_copies(fn, tn)




