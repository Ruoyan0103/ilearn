from ase.build import bulk, make_supercell
from ase.io import read, write
import numpy as np
from ase import Atoms

a0 = 5.6524
Ge_pri = bulk('Ge', 'diamond', a=a0)

write('POSCAR', Ge_pri, format='vasp')

