import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import seaborn as sns

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(1, 2, figsize=(10, 6))
dataset_path = ['../00-train', '../01-test']
energy_path = 'energies'
x_labels = ['train', 'test']

for i, dataset in enumerate(dataset_path):
    _energy_path = os.path.join(dataset, energy_path)
    # Load data
    label_energies, predict_energies, number_atoms = np.loadtxt(_energy_path, delimiter=',', unpack=True)
    
    # Normalize
    eng_in = [a / np.sqrt(n) for a, n in zip(label_energies, number_atoms)]
    eng_out = [b / np.sqrt(n) for b, n in zip(predict_energies, number_atoms)]
    eng_ae = np.abs(np.array(eng_in) - np.array(eng_out))
    
    # Violin plot
    sns.violinplot(y=eng_ae, ax=ax[i])
    ax[i].set_title(x_labels[i])
    ax[i].set_ylabel('Absolute Error')
    ax[i].set_xlabel('')

fig.suptitle('Violin Plots of Absolute Errors (train vs. test)')
plt.tight_layout()
fig.savefig('../MAE.png')


