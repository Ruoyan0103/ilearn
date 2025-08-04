from ase.build import bulk, make_supercell
from ase.io import write, read
import os

def crt_supercell(matrix, folder_path):
    Ge_prim = bulk('Ge', 'diamond', a=5.762)
    supercell = make_supercell(Ge_prim, matrix)
    write(f'{folder_path}/POSCAR', supercell, format='vasp')
    return supercell

matrices = []
m1 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
m2 = [[3, -1, -1], [0, 1, 0], [0, 0, 1]]
m3 = [[1, 0, 0], [0, 0, 1], [1, -3, 1]] # interexchange
m4 = [[1, 1, 1], [1, 0, -1], [0, 1, -1]]
m5 = [[1, 2, -1], [0, 0, 1], [1, -1, 0]] # interexchange
m6 = [[1, 0, 0], [1, -1, -2], [0, 1, -1]]
m7 = [[1, 1, -1], [0, 0, 1], [1, -2, 0]] # interexchange
m8 = [[1, 0, 0], [1, -1, -1], [0, 1, -2]]
m9 = [[1, 1, -1], [0, 1, -2], [-1, 1, 0]] # interexchange
m10 = [[1, 0, 1], [1, -1, -1], [0, 1, -1]]

matrices.append(m1)
matrices.append(m2)
matrices.append(m3)
matrices.append(m4)
matrices.append(m5)
matrices.append(m6)
matrices.append(m7)
matrices.append(m8)
matrices.append(m9)
matrices.append(m10)

for i in range(10):
    folder_path = os.path.join('supercells', f'sup-{i+1}')
    os.mkdir(folder_path)
    crt_supercell(matrices[i], folder_path)



    












