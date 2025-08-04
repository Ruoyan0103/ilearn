from ase.build import bulk, make_supercell
from ase.io import read, write
import numpy as np
from ase import Atoms
import os, copy, random
        
def drag_atom(size):
    a0 = 5.762
    Ge_cubic = bulk('Ge', 'diamond', a=a0, cubic=True)
    Ge_sup = Ge_cubic * size
    pos2 = Ge_sup[1].position
    pos57 = Ge_sup[56].position
    # dist = np.linalg.norm(pos57 - pos2)
    # int(dist) = 7
    # 10 calculation points
    # (x-10)/x = 1.5/7 >> x = 13
    del Ge_sup[[atom.index for atom in Ge_sup if all(atom.position == pos2)]]
    for i in range(0, 11):
        inter_pos = i*(pos57 - pos2)/13 + pos2
        atom = Atoms('Ge', positions=[inter_pos])
        write(f'POSCAR-{i}', Ge_sup+atom, format='vasp')
        #Ge_sup += atom
    #write('POSCAR', Ge_sup, format='vasp')
    
size = 2
drag_atom(size)






