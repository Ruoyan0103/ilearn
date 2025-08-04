from ase.build import bulk, make_supercell
from ase.io import read, write
import numpy as np
from ase import Atoms
import os, copy, random
        
def drag_atom(size):
    a0 = 5.762
    Ge_cubic = bulk('Ge', 'diamond', a=a0, cubic=True)
    Ge_sup = Ge_cubic * size
    pos1 = Ge_sup[0].position
    pos19 = Ge_sup[18].position
    # dist = np.linalg.norm(pos19 - pos2)
    # int(dist) = 9
    # 10 calculation points
    # (x-20)/x = 1.5/9 >> x = 24
    del Ge_sup[[atom.index for atom in Ge_sup if all(atom.position == pos1)]]
    for i in range(0, 21):
        inter_pos = i*(pos19 - pos1)/24 + pos1
        atom = Atoms('Ge', positions=[inter_pos])
        #write(f'POSCAR-{i}', Ge_sup+atom, format='vasp')
        Ge_sup += atom
    write('POSCAR', Ge_sup, format='vasp')
    
size = 2
drag_atom(size)






