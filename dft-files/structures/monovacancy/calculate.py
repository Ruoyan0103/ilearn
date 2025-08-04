from ase.io import write, read
from ase.calculators.vasp import Vasp
import numpy as np
import os, shutil



def calculate():
    directory_path = '../03-inter/05-split2'
    #directory_path = '../02-vacs'
    rattled_path = os.path.join(directory_path, '01-rattled_v2')
    for j in range(0, 5):
        poscar_path = os.path.join(rattled_path, f'POSCAR-{j}')
        incar_path = 'INCAR-2vacs'
        potcar_path = 'POTCAR'
        cal_path = os.path.join(directory_path, f'02-cal_v2/cal_{j}') #01-vac/02-cal/cal_1
        os.makedirs(cal_path, exist_ok=True)
        poscar_copy_path = os.path.join(cal_path, 'POSCAR')
        shutil.copy(poscar_path, poscar_copy_path)
        incar_copy_path = os.path.join(cal_path, 'INCAR')
        shutil.copy(incar_path, incar_copy_path)
        shutil.copy(potcar_path, cal_path)

def writexyz():
    directory_path = '../03-inter/05-split2'
    #directory_path = '../02-vacs'
    for i in range(5):
        cal_path = os.path.join(directory_path, f'02-cal_v2/cal_{i}')
        outcar_path = os.path.join(cal_path, 'OUTCAR')
        if not os.path.isfile(outcar_path):
            print(i)
            continue
        try:
            cell = read(outcar_path)
            volume = cell.get_volume()
            stress = cell.get_stress(voigt=False)
            virial = volume * -stress
            cell.info['virial'] = virial
            cell.info['config_type'] = 'split2'
            cell.info['sub_config_type'] = f'split2_{i}'
            write(f'{directory_path}/02-cal_v2/split2_0.18.xyz', cell, append=True)
        except RuntimeError as e:
            print(f"RuntimeError occurred while reading {outcar_path}: {e}")


#calculate()
writexyz()

