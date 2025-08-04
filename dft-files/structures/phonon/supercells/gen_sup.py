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
#m3 = [[1, 0, 0], [0, 0, 1], [1, -3, 1]] # interexchange second_third rows 
#m4 = [[1, 1, 1], [1, 0, -1], [0, 1, -1]]

# interchange rows, org: [1,2,-1  1,-1,0  0,0,1]
m5_1 = [[1, 2, -1], [0, 0, 1], [1, -1, 0]]
m5_2 = [[0, 0, 1], [1, -1, 0], [1, 2, -1]]
m5_3 = [[1, -1, 0], [1, 2, -1], [0, 0, 1]]
# multiplying one row by -1
m5_4 = [[-1, -2, 1], [1, -1, 0], [0, 0, 1]]
m5_5 = [[1, 2, -1], [-1, 1, 0], [0, 0, 1]]
m5_6 = [[1, 2, -1], [1, -1, 0], [0, 0, -1]]
# multiplying three rows by -1
m5_7 = [[-1, -2, 1], [-1, 1, 0], [0, 0, -1]]

#m6 = [[1, 0, 0], [1, -1, -2], [0, 1, -1]]

# interchange rows: org: [1,1,-1  1,-2,0  0,0,1]
m7_1 = [[1, 1, -1], [0, 0, 1], [1, -2, 0]]
m7_2 = [[0, 0, 1], [1, -2, 0], [1, 1, -1]]
m7_3 = [[1, -2, 0], [1, 1, -1], [0, 0, 1]]
# multiplying one row by -1
m7_4 = [[-1, -1, 1], [1, -2, 0], [0, 0, 1]] 
m7_5 = [[1, 1, -1], [-1, 2, 0], [0, 0, 1]] 
m7_6 = [[1, 1, -1], [1, -2, 0], [0, 0, -1]] 
# multiplying three rows by -1
m7_7 = [[-1, -1, 1], [-1, 2, 0], [0, 0, -1]]

#m8 = [[1, 0, 0], [1, -1, -1], [0, 1, -2]]
#m9 = [[1, 1, -1], [0, 1, -2], [-1, 1, 0]] # interexchange second_third rows
#m10 = [[1, 0, 1], [1, -1, -1], [0, 1, -1]]

matrices.append(m1)
matrices.append(m2)
#matrices.append(m3)
#matrices.append(m4)

matrices.append(m5_1)
matrices.append(m5_2)
matrices.append(m5_3)
#matrices.append(m5_4)
#matrices.append(m5_5)
#matrices.append(m5_6)
matrices.append(m5_7)

#matrices.append(m6)

matrices.append(m7_1)
matrices.append(m7_2)
matrices.append(m7_3)
#matrices.append(m7_4)
#matrices.append(m7_5)
#matrices.append(m7_6)
matrices.append(m7_7)

#matrices.append(m8)
#matrices.append(m9)
#matrices.append(m10)

for i in range(10):
    folder_path = os.path.join('supercells', f'sup-{i+1}')
    os.mkdir(folder_path)
    crt_supercell(matrices[i], folder_path)



    












