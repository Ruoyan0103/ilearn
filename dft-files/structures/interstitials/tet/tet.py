from ase.build import bulk, make_supercell
from ase.io import read, write
import numpy as np
from ase import Atoms

a0 = 5.7620
Ge_cubic = bulk('Ge', 'diamond', a=a0, cubic=True)
matrix = [[3, 0, 0], [0, 3, 0], [0, 0, 3]]
Ge_sup = make_supercell(Ge_cubic, matrix)
interstitial_position = np.array([0.5*a0*3, 0.5*a0*3, 0.5*a0*3])
interstitial_atom = Atoms('Ge', positions=[interstitial_position])
combined_structure = Ge_sup + interstitial_atom
# del Ge_sup[5]
# alpha = (2.495/2*np.sqrt(2)/2/a0)
# interstitial_position1 = np.array([(0.75-alpha)*a0, (0.25+alpha)*a0, 0.75*a0])  
# interstitial_atom1 = Atoms('Ge', positions=[interstitial_position1])
# interstitial_position2 = np.array([(0.75+alpha)*a0, (0.25-alpha)*a0, 0.75*a0])  
# interstitial_atom2 = Atoms('Ge', positions=[interstitial_position2])
# combined_structure = Ge_sup + interstitial_atom1 + interstitial_atom2
write('Ge.per', Ge_sup, format='lammps-data')
write('Ge.inter', combined_structure, format='lammps-data')




