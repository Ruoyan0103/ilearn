from ase.io import write, read
from ase.calculators.vasp import Vasp
import numpy as np
import os, shutil



def calculate():
    for i in range(1, 11):
        directory_path = f'../05-final/supercells/sup-{i}'
        rattled_path = os.path.join(directory_path, 'rattled_std_0.1')
        for j in range(50):
            poscar_path = os.path.join(rattled_path, f'POSCAR-{j}')
            incar_path = 'INCAR'
            potcar_path = 'POTCAR'
            cal_path = os.path.join(directory_path, f'02-cal/cal_{j}') #sup-1/cal/cal_1
            os.makedirs(cal_path, exist_ok=True)
            poscar_copy_path = os.path.join(cal_path, 'POSCAR')
            shutil.copy(poscar_path, poscar_copy_path)
            shutil.copy(incar_path, cal_path)
            shutil.copy(potcar_path, cal_path)

def writexyz():
    for i in range(1, 11):
        directory_path = f'../05-final/supercells/sup-{i}'
        for j in range(50):
            cal_path = os.path.join(directory_path, f'02-cal/cal_{j}')
            outcar_path = os.path.join(cal_path, 'OUTCAR')
            cell = read(outcar_path)
            volume = cell.get_volume()
            stress = cell.get_stress(voigt=False)
            virial = volume * -stress
            cell.info['virial'] = virial
            cell.info['config_type'] = f'phonon'
            write(f'{directory_path}/02-cal/phonon.xyz', cell, append=True)
            
                       
#calculate()
writexyz()

