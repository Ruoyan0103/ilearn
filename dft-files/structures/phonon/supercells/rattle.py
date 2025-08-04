from ase.io import read, write
import time, os, shutil, random
import numpy as np

def displace(in_poscar, out_poscar, lattice_rate=0.01, atom_disp=0.1):
    '''
    lattice_rate: the strain rate
    atom_disp (Å): the stdev of rattle
    '''
    matrix = np.random.uniform(-lattice_rate, lattice_rate, (3, 3))
    Ge_bulk = read(in_poscar, format='vasp')
    new_cell = Ge_bulk.get_cell() + Ge_bulk.get_cell() * matrix
    Ge_bulk.set_cell(new_cell, scale_atoms=True)
    #seed = int(time.time())                 # previous same seed every time
    seed = random.randint(0, 2**32 - 1)      # different seed every time
    Ge_bulk.rattle(stdev=atom_disp, seed=seed)
    write(out_poscar, Ge_bulk, format='vasp')

def displace2(in_poscar, out_poscar, lattice_rate=0.01, atom_disp=0.1):
    '''
    lattice_rate: the strain rate
    atom_disp (Å): the stdev of rattle
    '''
    matrix = np.random.uniform(-lattice_rate, lattice_rate, (3, 3))
    Ge_bulk = read(in_poscar, format='vasp')
    seed = int(time.time())
    Ge_bulk.rattle(stdev=atom_disp, seed=seed)
    new_cell = Ge_bulk.get_cell() + Ge_bulk.get_cell() * matrix
    Ge_bulk.set_cell(new_cell, scale_atoms=True)
    write(out_poscar, Ge_bulk, format='vasp')

def crt_rattled_copies(n):
    '''
    n: number of rattled copies of each POSCAR
    '''
    for i in range(1, 11):
        directory_path = f'sup-{i}'
        in_poscar = os.path.join(directory_path, 'POSCAR')
        rattled_path = os.path.join(directory_path, '01-rattled')
        if os.path.exists(rattled_path):
            shutil.rmtree(rattled_path)
        os.makedirs(rattled_path, exist_ok=True)
        for j in range(n):
            out_poscar = os.path.join(rattled_path, f'POSCAR-{j}')
            displace(in_poscar, out_poscar)

n = 50
crt_rattled_copies(n)




