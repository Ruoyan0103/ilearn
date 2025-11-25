import os
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read


class DFT:
    def __init__(self, initial_position, cal_folder, output_file, diff_output):
        self.cal_folder = cal_folder
        self.initial_position = initial_position
        self.output_file = output_file
        self.diff_output = diff_output
        self.dist = []
        self.energy = []
        self.fx = []
        self.fy = []
        self.fz = []

    def _createFORCE(self):
        for i in range(0, 11):
            outcar = os.path.join(self.cal_folder, f'../02-cal/cal_{i}/OUTCAR')
            if not os.path.isfile(outcar):
                print(i)
                continue
            atom = read(outcar)
            positions = atom.get_positions()
            position64 = positions[63]
            distance = np.linalg.norm(position64 - self.initial_position)
            self.dist.append(distance)

            forces = atom.get_forces()
            force64 = forces[63]
            self.fx.append(force64[0])
            self.fy.append(force64[1])
            self.fz.append(force64[2])
            
    def _createTOTEN(self):
        for i in range(0, 11):
            outcar = os.path.join(self.cal_folder, f'../02-cal/cal_{i}/OUTCAR')
            if not os.path.isfile(outcar):
                print(i)
                continue
            atom = read(outcar)
            eng = atom.get_potential_energy(force_consistent=True)
            self.energy.append(eng)

    def writeFile(self):
        self._createFORCE()
        self._createTOTEN()
        if os.path.isfile(self.output_file):
            os.remove(self.output_file)
        with open(self.output_file, 'a') as file:
            for d, e, fx, fy, fz in zip(self.dist, self.energy, self.fx, self.fy, self.fz):
                file.write(f'{d}, {e}, {fx}, {fy}, {fz}\n')
        if os.path.isfile(self.diff_output):
            os.remove(self.diff_output)
        with open(self.diff_output, 'a') as file:
            for d, e, fx, fy, fz in zip(self.dist, self.energy, self.fx, self.fy, self.fz):
                file.write(f'{d}, {e-self.energy[0]}, {fx-self.fx[0]}, {fy-self.fy[0]}, {fz-self.fz[0]}\n')


initial_position = [1.4404999999999999, 1.4404999999999999, 1.4404999999999999]
cal_folder = '../02-cal'
output_file = 'dist-energy-force-DFT'
diff_output = 'dist-diffenergy-force-DFT'
dft = DFT(initial_position, cal_folder, output_file, diff_output)
dft.writeFile()
            


    

       





    

    



    
    



    
   

    


