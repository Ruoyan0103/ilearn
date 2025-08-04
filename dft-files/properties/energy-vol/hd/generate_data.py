import numpy as np
from ase import Atoms
from ase.io import write
import os


def get_struct_volume(a = 4.0564, c = 6.6896):
    z = 0.06281
    lattice = np.array([[0.5*a, -0.5*np.sqrt(3)*a, 0], [0.5*a, 0.5*np.sqrt(3)*a, 0], [0, 0, c]])
    basis   = np.array([[0.5*a, a/np.sqrt(12), z*c], [0.5*a, -a/np.sqrt(12), (0.5+z)*c],
                       [0.5*a, a/np.sqrt(12), (0.5-z)*c], [0.5*a, -a/np.sqrt(12), (-z)*c]])

    pbc = [1, 1, 1]
    symbols = 'Ge4'
    Ge_hd = Atoms(symbols, positions=basis, cell=lattice, pbc=pbc)

    n = len(Ge_hd)                                                 
    v = Ge_hd.get_volume()/n                                       
    return Ge_hd, v

def generate_data(folder, vfile, a_sat, a_end, interval):
    a_list = np.arange(a_sat, a_end, interval)
    c_a_ratio=1.6496
    for a in a_list:
        Ge_hd, v = get_struct_volume(a, a*c_a_ratio)

        subfolder = os.path.join(folder, str(round(a, 3)))
        if not os.path.exists(subfolder): os.makedirs(subfolder)
        write(os.path.join(subfolder, 'bulk.data'), Ge_hd, format='lammps-data')
        with open(os.path.join(folder, vfile), 'a') as file:
            file.write(str(a) + ' , ' + str(v) + '\n')

'''
calculate energy of each structure, and added the calculate energy into file a-vol
'''
def createTOTEN(path, engfile):
    for folder in os.listdir(path):    # folder name is lattice constant a0
        try:
            fname = os.path.join(path, folder, 'log.lammps')
            if os.path.isfile(fname):
                filein = open(fname, 'r')
                Lines = filein.readlines()
                for line in Lines:
                    if 'Loop time' in line:
                        break
                    tmpline = line
                words = tmpline.split()
                energy = float(words[3])
                with open (os.path.join(path, engfile), 'a') as file:
                    file.write(str(folder) + ' , ' + str(energy) + '\n')
        except ValueError:
            print(folder)
        except OSError as e:
            print(folder)


def sortFile(vfile, folder, engfile, outfile):
    natoms = 4                              # modify manualy, e.g. for beta/hcp/dia=2, f/bcc=1, bc8=8, hd=4, st12=12
    volume = []
    with open(vfile, 'r') as f:
        for line in f:
            words = line.split(',')
            volume.append(float(words[1]))  # in DFT results, volume per atom, a0, energy per atom
    x = []
    y = []
    with open (os.path.join(folder, engfile), 'r') as f:
        for line in f:
            words = line.split(',')
            x.append(float(words[0]))   # in lammps results, a0, system energy
            y.append(float(words[1])/natoms)
        y_sorted = [y for x,y in sorted(zip(x, y))]
        x_sorted = [x for x,y in sorted(zip(x, y))]

        for x, v, y in zip(x_sorted, volume, y_sorted):
            with open(os.path.join(folder, outfile), 'a') as f:
                f.write(f'{x}, {v}, {y} \n')

folder = 'Final2'
vfile = 'a-vol'
a_sat, a_end, interval = 4.5, 4.55, 0.05
#generate_data(folder, vfile, a_sat, a_end, interval)
# run MD

engfile = 'lattice-eng'
if os.path.isfile(os.path.join(folder, engfile)):
    os.remove(os.path.join(folder, engfile))
createTOTEN(folder, engfile)
vfile = os.path.join(folder, vfile)
outfile = 'GAP'
if os.path.isfile(os.path.join(folder, outfile)):
    os.remove(os.path.join(folder, outfile))
sortFile(vfile, folder, engfile, outfile)


