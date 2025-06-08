# # coding: utf-8
# class GAPotential:
#     def __init__(self, potential_name, param_filename, calc_args=None):
#         self.potential_name = potential_name     # 'xml_label=GAP_2025_3_8_120_22_43_25_989'
#         self.param_filename = param_filename     # 'Ge-v10.xml'
#         self.calc_args = calc_args               # 'local_gap_variance'


# coding: utf-8
# Distributed under the terms of the BSD License.

"""This module provides TurboSOAP-GAP interatomic potential class."""

import re
import os
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
from ase.io import read 
from quippy.potential import Potential
from sklearn.metrics import mean_squared_error
from ilearn.lammps.calcs import ThresholdDisplacementEnergy
from ilearn.potentials import IPotential

module_dir = os.path.dirname(__file__)
results_dir = os.path.join(module_dir, 'results')

class GAPotential(IPotential):
    """
    This class implements Turbo Smooth Overlap of Atomic Position potentials.
    """
    pair_style = 'pair_style        quip'
    pair_coeff = 'pair_coeff        * * {} {} {}'

    def __init__(self, name=None, param=None):
        """

        Args:
            name (str): Name of force field.
            param (dict): The parameter configuration of potentials.
        """
        self.name = name if name else "GAPotential"
        self.param = param if param else {}
        self.specie = None

    def train(self, dataset_filename, default_sigma=[0.0005, 0.1, 0.05, 0.01],
             use_energies=True, use_forces=True, use_stress=False, **kwargs):
        """
        Training data with gaussian process regression.

        Args:
            dataset_filename (string): File containing a list of ASE Structure objects.
            default_sigma (list): Error criteria in energies, forces, stress
                and hessian. Should have 4 numbers.
            use_energies (bool): Whether to use dft total energies for training.
                Default to True.
            use_forces (bool): Whether to use dft atomic forces for training.
                Default to True.
            use_stress (bool): Whether to use dft virial stress for training.
                Default to False.

            kwargs:
                l_max (int): Parameter to configure GAP. The band limit of
                    spherical harmonics basis function. Default to 12.
                n_max (int): Parameter to configure GAP. The number of radial basis
                    function. Default to 10.
                atom_sigma (float): Parameter to configure GAP. The width of gaussian
                    atomic density. Default to 0.5.
                zeta (float): Present when covariance function type is do product.
                    Default to 4.
                cutoff (float): Parameter to configure GAP. The cutoff radius.
                    Default to 4.0.
                cutoff_transition_width (float): Parameter to configure GAP.
                    The transition width of cutoff radial. Default to 0.5.
                delta (float): Parameter to configure Sparsification.
                    The signal variance of noise. Default to 1.
                f0 (float): Parameter to configure Sparsification.
                    The signal mean of noise. Default to 0.0.
                n_sparse (int): Parameter to configure Sparsification.
                    Number of sparse points.
                covariance_type (str): Parameter to configure Sparsification.
                    The type of convariance function. Default to dot_product.
                sparse_method (str): Method to perform clustering in sparsification.
                    Default to 'cur_points'.

                sparse_jitter (float): Intrisic error of atomic/bond energy,
                    used to regularise the sparse covariance matrix.
                    Default to 1e-8.
                e0 (float): Atomic energy value to be subtracted from energies
                    before fitting. Default to 0.0.
                e0_offset (float): Offset of baseline. If zero, the offset is
                    the average atomic energy of the input data or the e0
                    specified manually. Default to 0.0.
        """
        exe_command = ["gap_fit"]
        exe_command.append('at_file={}'.format(dataset_filename))
        gap_configure_params = ['l_max', 'n_max', 'atom_sigma', 'zeta', 'cutoff',
                                'cutoff_transition_width', 'delta', 'f0', 'n_sparse',
                                'covariance_type', 'sparse_method']

        preprocess_params = ['sparse_jitter', 'e0', 'e0_offset']
        if len(default_sigma) != 4:
            raise ValueError("The default sigma is supposed to have 4 numbers.")

        gap_command = ['soap']
        for param_name in gap_configure_params:
            param = kwargs.get(param_name) if kwargs.get(param_name) \
                else soap_params.get(param_name)
            gap_command.append(param_name + '=' + '{}'.format(param))
        exe_command.append("gap=" + "{" + "{}".format(' '.join(gap_command)) + "}")

        for param_name in preprocess_params:
            param = kwargs.get(param_name) if kwargs.get(param_name) \
                else soap_params.get(param_name)
            exe_command.append(param_name + '=' + '{}'.format(param))

        default_sigma = [str(f) for f in default_sigma]
        exe_command.append("default_sigma={%s}" % (' '.join(default_sigma)))

        if use_energies:
            exe_command.append('energy_parameter_name=dft_energy')
        if use_forces:
            exe_command.append('force_parameter_name=dft_force')
        if use_stress:
            exe_command.append('virial_parameter_name=dft_virial')
        exe_command.append('gp_file={}'.format(xml_filename))

        with ScratchDir('.'):
            p = subprocess.Popen(exe_command, stdout=subprocess.PIPE)
            stdout = p.communicate()[0]
            rc = p.returncode
            if rc != 0:
                error_msg = 'QUIP exited with return code %d' % rc
                msg = stdout.decode("utf-8").split('\n')[:-1]
                try:
                    error_line = [i for i, m in enumerate(msg)
                                  if m.startswith('ERROR')][0]
                    error_msg += ', '.join([e for e in msg[error_line:]])
                except Exception:
                    error_msg += msg[-1]
                raise RuntimeError(error_msg)

            def get_xml(xml_file):
                tree = ET.parse(xml_file)
                root = tree.getroot()
                potential_label = root.tag
                gpcoordinates = list(root.iter('gpCoordinates'))[0]
                param_file = gpcoordinates.get('sparseX_filename')
                param = np.loadtxt(param_file)
                return tree, param, potential_label

            tree, param, potential_label = get_xml(xml_filename)
            self.param['xml'] = tree
            self.param['param'] = param
            self.param['potential_label'] = potential_label
        return rc

    @staticmethod
    def from_config(filename):
        """
        Initialize potentials with parameters file.

        ARgs:
            filename (str): The file storing parameters of potentials.

        Returns:
            GAPotential.
        """
        if filename.endswith('.xml'):
            def get_xml(xml_file):
                tree = ET.parse(xml_file)
                root = tree.getroot()
                potential_label = root.tag
                pairpot = root.find('pairpot')
                glue_params = pairpot.find('Glue_params')
                per_type_data = glue_params.find('per_type_data')
                specie_z = per_type_data.get('atomic_num')
                return filename, potential_label, specie_z

            filename, potential_label, specie_z = get_xml(filename)
            parameters = dict(xml_file=filename, potential_label=potential_label, specie_z=specie_z)
            return GAPotential(param=parameters)


    def write_param(self, xml_filename=None):
        """
        Write potential parameters for lammps calculation.

        Args:
            xml_filename (str): Filename to store xml formatted parameters.
        """
        self.specie = self.param.get('specie_z', None)
        self.pair_coeff = self.pair_coeff.format(xml_filename, 
                                            '\"Potential xml_label={}\"'.format(self.param.get('potential_label', None)),
                                            self.param.get('specie_z', None))
        ff_settings = [self.pair_style, self.pair_coeff]
        return ff_settings


    def _evaluate_helper(self, test_structures_dataset,
                         predict_energies=True, predict_forces=True, predict_virials=False):
        """
        Evaluate energies, forces and stresses of structures with trained
        interatomic potentials.

        Args:
            test_structures_dataset (list): A list of ASE Structure objects, 
                                            with reference energies, forces and stresses.
            predict_energies (bool): Whether to predict energies of configurations.
            predict_forces (bool): Whether to predict forces of configurations.
            predict_stress (bool): Whether to predict virial stress of configurations.
        Returns:
            Dict{num_atoms, ref_energies, ref_forces, ref_virial, pred_energies, pred_forces, pred_virial}.
            RMSE of energies, forces and virials.
        """
        values_dict = { 'num_atoms': [],
                        'ref_energies': [],
                        'pred_energies': [],
                        'ref_forces': [],
                        'pred_forces': [],
                        'ref_virials': [],
                        'pred_virials': [],}
        for struct in test_structures_dataset:
            values_dict['num_atoms'].append(len(struct))
            if predict_energies:
                ref_energy = struct.get_potential_energy()
                values_dict['ref_energies'].append(ref_energy)
                struct.calc = Potential(self.param['potential_label'], self.param['xml_file'])
                pred_energy = struct.get_potential_energy()
                values_dict['pred_energies'].append(pred_energy)
            if predict_forces:
                for i in range(len(struct)):
                    values_dict['ref_forces'].append(struct.get_forces()[i])
                struct.calc = Potential(self.param['potential_label'], self.param['xml_file'])
                for i in range(len(struct)):
                    values_dict['pred_forces'].append(struct.get_forces()[i])
            if predict_virials:
                if 'virial' in struct.info:
                    values_dict['ref_virials'].append(struct.info['virial'].flatten())
                struct.calc = Potential(self.param['potential_label'], self.param['xml_file'])
                if 'virial' in struct.info:
                    values_dict['pred_virials'].append(struct.info['virial'].flatten())
        
        # calculate RMSE:
        rmse_dict = {'rmse_energy': -100, 'rmse_force': -100, 'rmse_virial': -100}
        if predict_energies:
            # using np.sqrt to normalize energies per atom,  J. Chem. Phys. 158, 121501 (2023), Figure 4.
            eng_ref = [a / np.sqrt(n) for a, n in zip(values_dict['ref_energies'], values_dict['num_atoms'])]
            eng_pred = [b / np.sqrt(n) for b, n in zip(values_dict['pred_energies'], values_dict['num_atoms'])]
            # J. Chem. Phys. 158, 121501 (2023), RMSE equation (1b) 
            rmse_dict['rmse_energy'] = np.sqrt(mean_squared_error(eng_ref, eng_pred))
        if predict_forces:
            # J. Chem. Phys. 158, 121501 (2023), RMSE_component, equation (5)
            rmse_dict['rmse_force'] = np.sqrt(mean_squared_error(values_dict['ref_forces'], values_dict['pred_forces']))
        if predict_virials:
            rmse_dict['rmse_virial'] = np.sqrt(mean_squared_error(values_dict['ref_virials'], values_dict['pred_virials']))
        return values_dict, rmse_dict


    # def _write_eval_results(values_dict, files_dict):
    #     if values_dict['ref_energies'] and values_dict['pred_energies']:
    #         efile = os.path.join(results_dir, files_dict['energies_file'])
    #         with open(efile, 'a') as file:
    #             file.write(f'{values_dict['ref_energies']}, {values_dict['pred_energies']}\n')
    #     if values_dict['ref_forces'] and values_dict['pred_forces']:
    #         ffile = os.path.join(results_dir, files_dict['ref_forces_file'])
    #         with open(ffile, 'a') as file:
    #             file.write(' '.join(map(str, {values_dict['ref_forces']})) + '\n')
    #         ffile = os.path.join(results_dir, files_dict['pred_forces_file'])
    #         with open(ffile, 'a') as file:
    #             file.write(' '.join(map(str, {files_dict['pred_forces_file']})) + '\n')
    #     if values_dict['ref_virials'] and values_dict['pred_virials']:
    #         vfile = os.path.join(results_dir, files_dict['ref_virials_file'])
    #         with open(vfile, 'a') as file:
    #             file.write(' '.join(map(str, {values_dict['ref_virials']})) + '\n')
    #         vfile = os.path.join(results_dir, files_dict['pred_virials_file'])
    #         with open(vfile, 'a') as file:
    #             file.write(' '.join(map(str, {values_dict['pred_virials']})) + '\n')


    def evaluate(self, test_structures_dataset, config_type, config_type_set=None,
                 predict_energies=True, predict_forces=True, predict_virials=False):
        """
        Evaluate energies, forces and stresses of structures with trained
        interatomic potentials.

        Args:
            test_structures_dataset (list): A list of ASE Structure objects, 
                                            with reference energies, forces and stresses.
            config_type (boll): Whether to evaluate configurations of a specific type.
            config_type_set (set): Set of configuration types to evaluate.
            predict_energies (bool): Whether to predict energies of configurations.
            predict_forces (bool): Whether to predict forces of configurations.
            predict_stress (bool): Whether to predict virial stress of configurations.
        Returns:
            Dict{True_energys, True_forces, True_stresses, Predicted_energies, Predicted_forces, Predicted_stresses}.
            RMSE of energies, forces and stresses.
        """
        # test_structures = read(test_structures_dataset, format='extxyz', index=':')
                    
        if config_type:
            if config_type_set is None:  # not recommended to use this option
                config_type_set = set()
                for struct in test_structures:
                    if 'config_type' in struct.info:
                        config_type = struct.info['config_type']
                        config_type_set.add(config_type)
            else:
                config_type_dict = {config: [] for config in config_type_set}
            for struct in test_structures:
                if 'config_type' in struct.info:
                    config_type = struct.info['config_type']
                    data_config_type[config_type].append(struct)
                    values_dict, rmse_dict = self._evaluate_helper(data_config_type[config_type], 
                                                                  predict_energies=True, predict_forces=True, predict_virials=False)
        else:
            values_dict, rmse_dict = self._evaluate_helper(test_structures_dataset, 
                                                          predict_energies=True, predict_forces=True, predict_virials=False)
            
    #def predict(self, test_structures_dataset):

# example usage
if __name__ == "__main__":
    gap_file = os.path.join(module_dir, 'params', 'GAP', 'Ge-v10.xml')
    gap = GAPotential.from_config(gap_file)
    ff_settings = gap.write_param(gap_file)
    alat = 5.76
    pka_id = 2766
    temp = 0
    element = 'Ge'
    mass = 72.56
    min_velocity = 30
    max_velocity = 35
    velocity_interval = 5
    kin_eng_threshold = 4
    simulation_size = 9
    thermal_time = 60     # in second
    tde_time = 8*3600     # in second

    tde = ThresholdDisplacementEnergy(ff_settings, element, mass, alat, temp,
                                      pka_id, min_velocity, max_velocity, 
                                      velocity_interval, kin_eng_threshold, simulation_size,
                                      thermal_time, tde_time)
    vector1 = [0., 0., 1.] / np.linalg.norm([0., 0., 1.])  # Normalize the vector
    vector2 = [1., 0., 1.] / np.linalg.norm([1., 0., 1.])  # Normalize the vector
    vector3 = [1., 1., 1.] / np.linalg.norm([1., 1., 1.])  # Normalize the vector
    vectors = np.array((vector1, vector2, vector3))
    tde.get_uniform_angles(vectors, 4)
    tde.set_hkl_from_angles()
    # tde.check_interval()
    tde.calculate(needed_thermalization=True)
    # tde.plot()
    # tde.plot_no_interplation()
    # tde.average_TDE()