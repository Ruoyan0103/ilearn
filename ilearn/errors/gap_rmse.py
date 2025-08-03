from ilearn.errors.rmse import RMSECalculator
from ilearn.potentials import gap
import os 
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Serif'
class getRMSE:
    def __init__(self, dataset_path, potential, files_folder, energy_path, \
                force_true, force_pred, virial_true, virial_pred, \
                config_type, figure_path, rmse_path, ax1, ax2):
        self.calc = RMSECalculator(dataset_path, potential, files_folder, energy_path, \
                                   force_true, force_pred, virial_true, virial_pred, \
                                   config_type, figure_path, rmse_path)
        self.ax1 = ax1
        self.ax2 = ax2

    def rmse_all_type(self):
        self.calc.get_efv()
        self.calc.plot_energy(self.ax1)
        self.calc.plot_force(self.ax2)

    def rmse_config_types(self):
        self.calc.get_efv_config_type()

figure, ax = plt.subplots(2, 2, figsize=(7, 7))
ax = ax.flatten()
results_path = 'temp2'

dataset_path = 'datasets/t_pached.xyz'
potential = gap.GAPotential(
            potential_name='xml_label=GAP_2025_3_8_120_22_43_25_989',
            param_filename=os.path.join('potentials_folder', 'Ge-v10.xml'),
            calc_args='local_gap_variance')
files_folder = os.path.join(results_path, 'files_train')
if not os.path.exists(files_folder):
    os.makedirs(files_folder)
energy_path = 'energy'
force_true = 'force_true'
force_pred = 'force_pred'
virial_true = 'virial_true'
virial_pred = 'virial_pred'
config_type = os.path.join(results_path, 'config_type_train')
figure_path = os.path.join(results_path, 'train.png')
rmse_path = os.path.join(results_path, 'config_rmse_train.txt')
rmse_calculator = getRMSE(dataset_path, potential, files_folder, energy_path, 
                         force_true, force_pred, virial_true, virial_pred, 
                         config_type, figure_path, rmse_path, ax[0], ax[1])
rmse_calculator.rmse_all_type()
#rmse_calculator.rmse_config_types()

dataset_path = 'datasets/test_pached.xyz'
potential = gap.GAPotential(
            potential_name='xml_label=GAP_2025_3_8_120_22_43_25_989',
            param_filename=os.path.join('potentials_folder', 'Ge-v10.xml'),
            calc_args='local_gap_variance')
files_folder = os.path.join(results_path, 'files_test')
if not os.path.exists(files_folder):
    os.makedirs(files_folder)
energy_path = 'energy'
force_true = 'force_true'
force_pred = 'force_pred'
virial_true = 'virial_true'
virial_pred = 'virial_pred'
config_type = os.path.join(results_path, 'config_type_test')
figure_path = os.path.join(results_path, 'test.png')
rmse_path = os.path.join(results_path, 'config_rmse_test.txt')
rmse_calculator = getRMSE(dataset_path, potential, files_folder, energy_path, 
                         force_true, force_pred, virial_true, virial_pred, 
                         config_type, figure_path, rmse_path, ax[2], ax[3])
#rmse_calculator.rmse_all_type()
#rmse_calculator.rmse_config_types()

labels = ['(a)', '(b)', '(c)', '(d)']
for i, a in enumerate(ax):
    a.text(0.02, 0.95, labels[i], transform=a.transAxes,
           fontsize=12, 
           va='top', ha='left')

figure.tight_layout(rect=[0.02, 0.01, 0.98, 0.97])
plt.subplots_adjust(wspace=0.3, hspace=0.28)
figure.text(0.5, 0.96, "RMSE on the training dataset", ha='center', fontsize=12)
figure.text(0.5, 0.47, "RMSE on the test dataset", ha='center', fontsize=12)
figure.savefig(os.path.join(results_path, 'rmse.png'), dpi=300)



