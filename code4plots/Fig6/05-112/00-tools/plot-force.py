import matplotlib.pyplot as plt
import numpy as np

def read_files(files):
    for myfile in files:
        data = np.loadtxt(myfile, delimiter=",")
        d = data[:, 0]  
        x = data[:, 1]  
        plt.plot(d, x)
    plt.savefig('dist-force.png', dpi=300)

files = ['dist-force', 'dist-force-GAP']
read_files(files)

