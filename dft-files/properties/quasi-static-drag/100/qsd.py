from ase.build import bulk, make_supercell
from ase.io import read, write
import numpy as np
from ase import Atoms
import os, copy, random
        
def drag_atom(size):
    a0 = 5.762
    Ge_cubic = bulk('Ge', 'diamond', a=a0, cubic=True)
    Ge_sup = Ge_cubic * size
    pos4 = Ge_sup[3].position   # position of atom 4
    pos36 = Ge_sup[35].position # position of atom 36
    dist = np.linalg.norm(pos4 - pos36)
    # int(dist) = 5
    # 10 calculation points
    # (x-10)/x = 1.5/5 >> x = 14
    del Ge_sup[[atom.index for atom in Ge_sup if all(atom.position == pos4)]]
    for i in range(0, 11):
        inter_pos = i*(pos36 - pos4)/14 + pos4
        atom = Atoms('Ge', positions=[inter_pos])
        #write(f'POSCAR-{i}', Ge_sup+atom, format='vasp')
        Ge_sup += atom
    write('POSCAR', Ge_sup, format='vasp')

size = 2
drag_atom(size)






