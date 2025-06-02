import os
import numpy as np 
import matplotlib.pyplot as plt 

from quippy.potential import Potential
from ase.io import read, write
from sklearn.metrics import mean_squared_error

class RMSECalculator:
    def __init__(self, dataset_path, potential, \
                files_folder, \
                energy_true_predict_path, \
                force_path_true, force_path_predict, \
                virial_path_true, virial_path_predict, \
                configure_type_path, figure_path, rmse_path):
        self.dataset_path = dataset_path
        self.dataset = read(self.dataset_path, format='extxyz', index=':')
        self.potential = potential
        self.pot = Potential(self.potential.potential_name,
                             param_filename=self.potential.param_filename,
                             calc_args=self.potential.calc_args)
        self.files_folder = files_folder
        self.energy_path = energy_true_predict_path
        self.force_path_true = force_path_true
        self.force_path_predict = force_path_predict
        self.virial_path_true = virial_path_true
        self.virial_path_predict = virial_path_predict
        self.configure_type_path = configure_type_path
        self.figure_path = figure_path
        self.rmse_path = rmse_path 
        self.config_type = None
        self.num_config_type = None
        self._energy_path = None 
        self._force_path_true = None
        self._force_path_predict = None
        self._virial_path_true = None
        self._virial_path_predict = None
        self.number_atoms = []
        self.label_energies = []
        self.predict_energies = []
        self.label_forces = []
        self.predict_forces = []
        self.label_virials = []
        self.predict_virials = []


    def _write_energy(self, label_energy, predict_energy, number_atom, config_type=None):
        if config_type:
            energy_path = os.path.join(self.files_folder, f'{self.energy_path}_{config_type}')
        else:
            energy_path = os.path.join(self.files_folder, self.energy_path)
        # if os.path.exists(energy_path):
        #     os.remove(energy_path)
        with open(energy_path, 'a') as file:
            file.write(f'{label_energy}, {predict_energy}, {number_atom}\n')
        self._energy_path = energy_path
        return energy_path

    def _write_force(self, label_force, predict_force, is_true=True, config_type=None):
        if is_true:
            if config_type:
                force_path_true = os.path.join(self.files_folder, f'{self.force_path_true}_{config_type}')
            else:
                force_path_true = os.path.join(self.files_folder, self.force_path_true)
            # if os.path.exists(force_path_true):
            #     os.remove(force_path_true)
            with open(force_path_true, 'a') as file:
                file.write(' '.join(map(str, label_force)) + '\n')
            self._force_path_true = force_path_true
            return force_path_true
        else:
            if config_type:
                force_path_predict = os.path.join(self.files_folder, f'{self.force_path_predict}_{config_type}')
            else:
                force_path_predict = os.path.join(self.files_folder, self.force_path_predict)
            # if os.path.exists(force_path_predict):
            #     os.remove(force_path_predict)
            with open(force_path_predict, 'a') as file:
                file.write(' '.join(map(str, predict_force)) + '\n')
            self._force_path_predict = force_path_predict
            return force_path_predict
            

    def _write_virial(self, label_virial, predict_virial, is_true=True, config_type=None):
        if is_true:
            if config_type:
                virial_path_true = os.path.join(self.files_folder, f'{self.virial_path_true}_{config_type}')
            else:
                virial_path_true = os.path.join(self.files_folder, self.virial_path_true)
            if os.path.exists(virial_path_true):
                os.remove(virial_path_true)
            with open(virial_path_true, 'a') as file:
                file.write(' '.join(map(str, label_virial)) + '\n')
            self._virial_path_true = virial_path_true
            return virial_path_true
        else:
            if config_type:
                virial_path_predict = os.path.join(self.files_folder, f'{self.virial_path_predict}_{config_type}')
            else:
                virial_path_predict = os.path.join(self.files_folder, self.virial_path_predict)
            if os.path.exists(virial_path_predict):
                os.remove(virial_path_predict)
            with open(virial_path_predict, 'a') as file:
                file.write(' '.join(map(str, predict_virial)) + '\n')
            self._virial_path_predict = virial_path_predict
            return virial_path_predict


    def _extract_config_type(self):
        configure_type_set = set()
        dataset = read(self.dataset_path, format='extxyz', index=':')
        for struct in dataset:
            if 'config_type' in struct.info:
                config_type = struct.info['config_type']
                configure_type_set.add(config_type)
        for i in configure_type_set:
            with open(self.configure_type_path, 'a') as file:
                file.write(f'{i}\n')
        return configure_type_set


    def get_efv_config_type(self):
        configure_type_set = self._extract_config_type()
        data_config_type = {config: [] for config in configure_type_set}

        for struct in self.dataset:
            if 'config_type' in struct.info:
                config_type = struct.info['config_type']
                data_config_type[config_type].append(struct)
        
        for config_type, structs in data_config_type.items():
            # print(f'config type: {config_type}, data points: {len(structs)}')
            self.config_type = config_type
            self.num_config_type = len(data_config_type[config_type])
            dataset_config = data_config_type[config_type]
            self.get_efv(dataset=dataset_config)
            self.get_rmse()
            

    def get_efv(self, dataset=None):
        '''
        write label/predicted energies, label/predicted forces into two files
        '''
        if dataset is None:
            dataset = self.dataset
        self.number_atoms = []
        self.label_energies = []
        self.predict_energies = []
        self.label_forces = []
        self.predict_forces = []
        self.label_virials = []
        self.predict_virials = []
        cnt = 0
        for struct in dataset:
            print(f'data points: {cnt}')
            self.number_atoms.append(len(struct))
            self.label_energies.append(struct.get_potential_energy(force_consistent=True))
            if 'virial' in struct.info:
                self.label_virials.append(struct.info['virial'].flatten())
            else:
                self._virial_path_true = None 
                self._virial_path_predict = None
            for i in range(len(struct)):
                self.label_forces.append(struct.get_forces()[i])

            struct.calc = self.pot
            self.predict_energies.append(struct.get_potential_energy(force_consistent=True))
            self.predict_virials.append((struct.get_stress(voigt=False) * -struct.get_volume()).flatten())
            for i in range(len(struct)):
                self.predict_forces.append(struct.get_forces()[i])
            cnt += 1
        
        for label_energy, predict_energy, number_atom in zip(self.label_energies, self.predict_energies, self.number_atoms):
            self._write_energy(label_energy, predict_energy, number_atom, config_type=self.config_type)
        for label_virial, predict_virial in zip(self.label_virials, self.predict_virials):
            self._write_virial(label_virial, predict_virial, config_type=self.config_type)
            self._write_virial(label_virial, predict_virial, is_true=False, config_type=self.config_type)
        for label_force, predict_force in zip(self.label_forces, self.predict_forces):
            self._write_force(label_force, predict_force, config_type=self.config_type)
            self._write_force(label_force, predict_force, is_true=False, config_type=self.config_type)
            

    def plot_energy(self, ax, is_plot=True):
        '''
        plot predicted and label energies
        '''
        label_energies, predict_energies, number_atoms = np.loadtxt(self._energy_path, delimiter=',', unpack=True)
        label_energies = np.atleast_1d(label_energies)
        predict_energies = np.atleast_1d(predict_energies)
        number_atoms = np.atleast_1d(number_atoms)
        # using np.sqrt to normalize energies per atom,  J. Chem. Phys. 158, 121501 (2023), Figure 4.
        eng_in = [a / np.sqrt(n) for a, n in zip(label_energies, number_atoms)]
        eng_out = [b / np.sqrt(n) for b, n in zip(predict_energies, number_atoms)]
        if is_plot:
            ax.scatter(eng_in, eng_out)
            for_limits = np.array(eng_in + eng_out)
            elim = (for_limits.min() - 0.05, for_limits.max() + 0.05)
            ax.set_xlim(elim)
            ax.set_ylim(elim)
            ax.plot(elim, elim, c='k', linewidth=1)
            ax.set_ylabel('GAP (eV)', fontsize=9)
            ax.set_xlabel('DFT (eV)', fontsize=9)
            ax.tick_params(axis='both', which='major', labelsize=9, width=1, length=4, direction='in') 
            ax.spines['top'].set_linewidth(1)
            ax.spines['bottom'].set_linewidth(1)
            ax.spines['left'].set_linewidth(1)
            ax.spines['right'].set_linewidth(1)
            # J. Chem. Phys. 158, 121501 (2023), RMSE equation (1b) 
            _rms = np.sqrt(mean_squared_error(eng_in, eng_out))
            # _std = std(eng_in, eng_out)
            rmse_text = f'Energy RMSE: \n {_rms:.3e} eV/atom'
            ax.text(0.9, 0.1, rmse_text, transform=ax.transAxes, fontsize=9.5, horizontalalignment='right',
                    verticalalignment='bottom')
        else:
            _rms = np.sqrt(mean_squared_error(eng_in, eng_out))
            return _rms

    def plot_force(self, ax, is_plot=True):
        '''
        plot predicted and label forces
        '''
        force_in = []
        force_out = []
        number_atoms = []
        with open(self._force_path_true, 'r') as f:
            lines_true = f.readlines()
            for line in lines_true:
                numbers = list(map(float, line.strip().split()))
                force_in.append(numbers)
        with open(self._force_path_predict, 'r') as f:
            lines_predict = f.readlines()
            for line in lines_predict:
                numbers = list(map(float, line.strip().split()))
                force_out.append(numbers)
        if is_plot:
            ax.scatter(np.array(force_in), np.array(force_out))
            
            # get the appropriate limits for the plot
            for_limits = np.array(np.array(force_in) + np.array(force_out))
            flim = (for_limits.min() - 1, for_limits.max() + 1)
            ax.set_xlim(flim)
            ax.set_ylim(flim)
            # add line of
            ax.plot(flim, flim, c='k', linewidth=1)
            # set labels
            ax.set_ylabel('GAP (eV/Å)', fontsize=9)
            ax.set_xlabel('DFT (eV/Å)', fontsize=9)
            ax.tick_params(axis='both', which='major', labelsize=9, width=1, length=4, direction='in') 
            ax.spines['top'].set_linewidth(1)
            ax.spines['bottom'].set_linewidth(1)
            ax.spines['left'].set_linewidth(1)
            ax.spines['right'].set_linewidth(1)
            # J. Chem. Phys. 158, 121501 (2023), RMSE_component, equation (5)
            _rms = np.sqrt(mean_squared_error(force_in, force_out))
            # _std = std(force_in, force_out)
            rmse_text = f'Force RMSE: \n {_rms:.3e} eV/Å'
            ax.text(0.9, 0.1, rmse_text, transform=ax.transAxes, fontsize=9.5, horizontalalignment='right',
                    verticalalignment='bottom')
        else:
            _rms = np.sqrt(mean_squared_error(force_in, force_out))
            return _rms

    def plot_virial(self, ax, is_plot=True):
        '''
        plot predicted and label virials
        '''
        virial_in = []
        virial_out = []
        if self._virial_path_true is None or self._virial_path_predict is None:
            return -100000
        
        with open(self._virial_path_true, 'r') as f:
            lines_true = f.readlines()
            for line in lines_true:
                numbers = list(map(float, line.strip().split()))
                virial_in.append(numbers)
        with open(self._virial_path_predict, 'r') as f:
            lines_predict = f.readlines()
            for line in lines_predict:
                numbers = list(map(float, line.strip().split()))
                virial_out.append(numbers)
        if is_plot:
            ax.scatter(np.array(virial_in), np.array(virial_out))
            # get the appropriate limits for the plot
            for_limits = np.array(np.array(virial_in) + np.array(virial_out))
            flim = (for_limits.min() - 1, for_limits.max() + 1)
            ax.set_xlim(flim)
            ax.set_ylim(flim)
            # add line of
            ax.plot(flim, flim, c='k', linewidth=1)
            # set labels
            ax.set_ylabel('GAP (eV)', fontsize=9)
            ax.set_xlabel('DFT (eV)', fontsize=9)
            ax.tick_params(axis='both', which='major', labelsize=9, width=1, length=4, direction='in') 
            ax.spines['top'].set_linewidth(1)
            ax.spines['bottom'].set_linewidth(1)
            ax.spines['left'].set_linewidth(1)
            ax.spines['right'].set_linewidth(1)
            # add text about RMSE
            _rms = np.sqrt(mean_squared_error(virial_in, virial_out))
            rmse_text = f'Virial RMSE: \n {_rms:.3e} eV/atom'
            ax.text(0.9, 0.1, rmse_text, transform=ax.transAxes, fontsize=9.5, horizontalalignment='right',
                    verticalalignment='bottom')
        else:
            _rms = np.sqrt(mean_squared_error(virial_in, virial_out))
            return _rms

    def plot_rmse(self):
        '''
        plot predicted and label energies, forces, virials
        '''
        fig, ax = plt.subplots(1, 3, figsize=(9, 3))
        self.plot_energy(ax[0])
        self.plot_force(ax[1])
        self.plot_virial(ax[2])
        fig.tight_layout(pad=0.5)
        fig.savefig(self.figure_path, dpi=300)


    def get_rmse(self):
        '''
        calculate RMSE for energies, forces, virials
        '''
        energy_rmse = self.plot_energy(ax=None, is_plot=False)
        force_rmse = self.plot_force(ax=None, is_plot=False)
        virial_rmse = self.plot_virial(ax=None, is_plot=False)
        virial_str = f'{virial_rmse:14.3e}' if virial_rmse >= 0 else ' ' * 14
        # Write header if file is empty
        if not os.path.exists(self.rmse_path) or os.stat(self.rmse_path).st_size == 0:
            with open(self.rmse_path, 'a') as f:
                f.write(f'{"config":<14}{"number":>8}{"energy":>14}{"force":>14}{"virial":>14}\n\n')

        # Append data row
        with open(self.rmse_path, 'a') as f:
            f.write(
                f'{self.config_type:<10}'
                f'{self.num_config_type:>8}'
                f'{energy_rmse:>14.3e}'
                f'{force_rmse:>14.3e}'
                f'{virial_str}\n'
            )



