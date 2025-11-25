import numpy as np
import matplotlib.pyplot as plt
import os

class ZBL:
    p = 0.23
    inner = 1.5
    outer = 2.3
    # angstrom
    a0 = 0.46848
    # tungsten paper
    #c = [0.32825, 0.09219, 0.58110]
    #d = [2.54931, 0.29182, 0.59231]
    # original ZBL
    # c = [0.02817, 0.28022, 0.50986, 0.18175]
    # d = [0.20162, 0.40290, 0.94229, 3.19980]

    # fitted value from checkForce/fit_mol.m
    c = [0.45526, 0.49210, 0.05284]
    d = [1.43007, 0.45871, 4.50619]
        
    def __init__(self, zi, zj):
        self.zi = zi
        self.zj = zj
        
        # e * e / (4 * pi * epsilon_0) / electron_volt / angstrom
        self.zzeij = 14.399645478425668 * self.zi * self.zj  # eV.angstrom
        self.a = self.a0 / (zi ** self.p + zj ** self.p)
        
        
    def _phi(self, x):
        phi = np.sum([(cv) * np.exp(-(dv) * x) for cv, dv in zip(self.c, self.d)])
        return phi
    
    def _phi_prime(self, x):
        return np.sum([cv*(-dv) * np.exp(-dv * x) for cv, dv in zip(self.c, self.d)])
    
    def _cutoff(self, d):
        if d > self.outer:     # x > 1
            return 0.
        elif d <= self.inner:   # x < 0
            return 1
        else: 
            #return 0.5 * (np.cos((d - inner)/(outer - inner)*np.pi) + 1)
            x = (d-self.inner)/(self.outer-self.inner)
            ret = 1-x**3*(6*x**2-15*x+10)
            #ret = 0.5 * (np.cos((d - inner)/(outer - inner)*np.pi) + 1)
            #ret = 1+3*x**5-6*x**4+2*x**3
            return ret
        
    def _cutoff_prime(self, d):
        x = (d-self.inner)/(self.outer-self.inner)
        return -30*x**4+60*x**3-30*x**2

        
    def e_zbl(self, r):
        phi = self._phi(r/self.a)
        ret = self.zzeij / r * phi
        return ret
    
    def e_zbl2(self, r):
        phi = self._phi(r/self.a)
        ret = self._cutoff(r)*self.e_zbl(r)
        return ret
    
    def e_zbl_prime(self, r):
        return self.zzeij*(-r**(-2)*self._phi(r/self.a)+r**(-1)*self._phi_prime(r/self.a)/self.a)
    
    
    def force_zbl2(self, r):
        if r < self.inner:
            return self.e_zbl_prime(r)
        if r > self.outer:
            return 0
        else:
            return self.e_zbl_prime(r)*self._cutoff(r)+self.e_zbl(r)*self._cutoff_prime(r)

    def force_zbl(self, r):
        return self.e_zbl_prime(r)
 
    def force_zbl_delta(self, r, delta_r):
        energy_r_minus_delta = self.e_zbl2(r - delta_r)
        energy_r_plus_delta = self.e_zbl2(r + delta_r)
        energy_r = self.e_zbl2(r)
        force = (energy_r_plus_delta - energy_r_minus_delta) / (2*delta_r)
        return force
    

def cutGAP(zbl_file, check_force_folder):
    file1 = os.path.join(check_force_folder, 'zbl1')
    file2 = os.path.join(check_force_folder, 'zbl2')
    if os.path.isfile(file1):
        os.remove(file1)
    if os.path.isfile(file2):
        os.remove(file2)
    data = np.loadtxt(zbl_file, delimiter=' ', max_rows=345)
    np.savetxt(file1, data, fmt='%f')
    data = np.loadtxt(zbl_file, delimiter=' ', skiprows=342)
    np.savetxt(file2, data, fmt='%f')

# zbl = ZBL(32, 32)  
# if os.path.isfile('zbl.tab'):
#     os.remove('zbl.tab')
# if os.path.isfile('zbl_v2.tab'):
#     os.remove('zbl_v2.tab')

# rs = np.linspace(0.002, 3, num=600, endpoint=True)
# energies = [zbl.e_zbl(r) for r in rs]
# energies_v2 = [zbl.e_zbl2(r) for r in rs]
# forces = [zbl.force_zbl(r) for r in rs]   # remove the force at r = 0.002

# for i, e in zip(rs, energies):
#     with open('zbl.tab', 'a') as file:
#         file.write(f'{i} {e}\n')

# # zbl_v2 is used to plot energies with cutoff
# for i, e in zip(rs, energies_v2):
#     with open('zbl_v2.tab', 'a') as file:
#         file.write(f'{i} {e}\n')
# cutGAP('zbl_v2.tab', '01-test/reference')
# check_force_folder = '03-check_force'


# zbl_file = '03-check_force/zbl-force'
# if os.path.isfile(zbl_file):
#     os.remove(zbl_file)
# for r, f in zip(rs, forces):
#     with open(zbl_file, 'a') as file:
#         file.write(f'{r} {f}\n')
# cutGAP(zbl_file, check_force_folder)

