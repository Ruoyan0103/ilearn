from ase.build import bulk, make_supercell
from ase.io import read, write
import numpy as np
from ase import Atoms

a0 = 5.6524
Ge_cell = bulk('Ge', 'diamond', a=a0, cubic=False)
#size = 2 
#matrix = [[size, 0, 0], [0, size, 0], [0, 0, size]]
#Ge_sup = make_supercell(Ge_cell, matrix)
write('Ge-lammps', Ge_cell, format='lammps-data')
#Atom Type Labels

#1 Ge

