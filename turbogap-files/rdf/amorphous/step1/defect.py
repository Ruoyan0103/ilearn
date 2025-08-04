from ase.build import bulk, make_supercell
from ase.io import read, write
import numpy as np
from ase import Atoms

a0 = 5.760
Ge_cubic = bulk('Ge', 'diamond', a=a0, cubic=True)
size = 3
matrix = [[size, 0, 0], [0, size, 0], [0, 0, size]]
Ge_sup = make_supercell(Ge_cubic, matrix)

write('Ge.xyz', Ge_sup, format='extxyz')

