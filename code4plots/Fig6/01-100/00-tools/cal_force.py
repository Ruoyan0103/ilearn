from ase.io import read
import os
import numpy as np

initial_positon = [1.4405, 4.3215, 4.3215]
x = []
y = []
z = []
d = []
for i in range(1, 11):
    dir_path = f'../02-cal/cal_{i}'
    outcar = os.path.join(dir_path, 'OUTCAR')
    atom = read(outcar)
    positions = atom.get_positions()
    position64 = positions[63]
    distance = np.linalg.norm(position64 - initial_positon)
    d.append(distance)

    forces = atom.get_forces()
    force64 = forces[63]
    x.append(force64[0])
    y.append(force64[1])
    z.append(force64[2])

for dv, xv, yv, zv in zip(d, x, y, z):
    with open('dist-force', 'a') as file:
        file.write(f'{dv}, {xv}, {yv}, {zv}\n')


