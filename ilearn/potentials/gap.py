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
from ilearn.lammps.calcs import ThresholdDisplacementEnergy, LatticeConstant, ElasticConstant, \
                                VacancyDefectFormation,InterstitialDefectFormation, NudgedElasticBand
from ilearn.phonopy.calcs import PhononDispersion, Quasiharmonic
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
        self.element = None

    def train(self, dataset_filename, default_sigma=[0.0005, 0.1, 0.05, 0.01],
             use_energies=True, use_forces=True, use_stress=False, **kwargs):
        pass 

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
        self.element = self.param.get('specie_z', None)
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


    def evaluate(self, test_structures_dataset, use_config_type, config_type_set=None,
                 predict_energies=True, predict_forces=True, predict_virials=False):
        """
        Evaluate energies, forces and stresses of structures with the interatomic potential.

        Args:
            test_structures_dataset (list): A list of ASE Structure objects, 
                                            with reference energies, forces and stresses.
            use_config_type (bull): Whether to evaluate configurations of a specific type.
            config_type_set (set): Set of configuration types to evaluate.
            predict_energies (bool): Whether to predict energies of configurations.
            predict_forces (bool): Whether to predict forces of configurations.
            predict_stress (bool): Whether to predict virial stress of configurations.
        Returns:
            Dict{True_energys, True_forces, True_stresses, Predicted_energies, Predicted_forces, Predicted_stresses}.
            RMSE of energies, forces and stresses.
        """
        test_structures = read(test_structures_dataset, format='extxyz', index=':')   
        if use_config_type:
            if config_type_set is None:  
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
                    config_type_dict[config_type].append(struct)
            for config_type in config_type_set:
                values_dict, rmse_dict = self._evaluate_helper(config_type_dict[config_type], 
                                                            predict_energies=True, predict_forces=True, predict_virials=False)
        else:
            values_dict, rmse_dict = self._evaluate_helper(test_structures_dataset, 
                                                          predict_energies=True, predict_forces=True, predict_virials=False)
            

    def predict(self, test_structures_dataset):
        pass
    

# example usage
if __name__ == "__main__":
    gap_file = os.path.join(module_dir, 'params', 'GAP', 'Ge-v10.xml')
    gap = GAPotential.from_config(gap_file)
    ff_settings = gap.write_param(gap_file)
    alat = 5.76
    pka_id = 1202
    temp = 0
    element = 'Ge'
    mass = 72.64
    min_velocity = 39
    max_velocity = 45
    velocity_interval = 3
    kin_eng_threshold = 4
    simulation_size = 9
    thermal_time = 60       # in second
    tde_time = 1.8*3600       # in second
    lattice = 'diamond'

    # example usage
    tde = ThresholdDisplacementEnergy('GAP', ff_settings, element, mass, alat, temp,
                                       pka_id, min_velocity, max_velocity, 
                                       velocity_interval, kin_eng_threshold, simulation_size,
                                       thermal_time, tde_time)
    vector1 = [0., 0., 1.] / np.linalg.norm([0., 0., 1.])  # Normalize the vector
    vector2 = [1., 0., 1.] / np.linalg.norm([1., 0., 1.])  # Normalize the vector
    vector3 = [1., 1., 1.] / np.linalg.norm([1., 1., 1.])  # Normalize the vector
    vectors = np.array((vector1, vector2, vector3))
    # tde.get_uniform_angles(vectors, 4)
    # tde.set_hkl_from_angles()
    # # tde.check_interval()
    # tde.calculate(needed_thermalization=True)
    # tde.plot()
    # tde.plot_no_interplation()
    tde.average_TDE()

    # example usage 
    # lc = LatticeConstant(ff_settings, mass, element, lattice, alat=5.3, cubic=True)
    # lc.calculate()

    # example usage
    # elastic = ElasticConstant(ff_settings, mass, lattice, alat)
    # elastic.calculate()

    # example usage
    # vf = VacancyDefectFormation(ff_settings, mass, lattice, alat, size=3, del_id=0)
    # vf.calculate()

    # example usage
    # neb = NudgedElasticBand(ff_settings, mass, alat, size=2, element, lattice, num_images=5, path='2NN')
    # neb.calculate()

    # example usage
    # inter = InterstitialDefectFormation(ff_settings, mass, element, lattice, alat, size=2)
    # inter.calculate()
