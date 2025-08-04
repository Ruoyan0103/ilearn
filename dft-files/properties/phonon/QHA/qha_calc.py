import numpy as np
import yaml
from phonopy import PhonopyQHA

class QHACalc:
    def __init__(self, volumes, energies, thermal_yaml_folder_path, result_folder_path):
        """
        This function initializes the parameters.
        Parameters:
        - volumes: (list) The list of volumes.
        - energies: (list) The list of energies.
        - thermal_yaml_folder_path: (String) The path of thermal yaml files folder.
        - result_folder_path: (String) The written files are stored in this folder.
        Returns:

        """
        self.volumes = volumes
        self.energies = energies
        self.thermal_yaml_folder_path = thermal_yaml_folder_path
        self.result_folder_path = result_folder_path


    def qha(self, start_index, end_index):
        """
        This function reads data of volumes and energies lists, and thermal_yaml files, 
        and do qha calculation.
        Parameters:
        - start_index: (Int) Start index of yaml file.
        - end_index: (Int) End index of yaml file.
        Returns:
        - 
        """

        entropy = []
        cv = []
        fe = []
        for index in range(start_index, end_index):
            file_name = f'{self.thermal_yaml_folder_path}thermal_properties.yaml{index}'
            thermal_properties = yaml.load(open(file_name), Loader=yaml.CLoader)['thermal_properties']
            temperatures = [v['temperature'] for v in thermal_properties]
            cv.append([v['heat_capacity'] for v in thermal_properties])
            entropy.append([v['entropy'] for v in thermal_properties])
            fe.append([v['free_energy'] for v in thermal_properties])


        qha = PhonopyQHA(
            self.volumes,
            self.energies,
            temperatures=temperatures,
            free_energy=np.transpose(fe),
            cv=np.transpose(cv),
            entropy=np.transpose(entropy),
            t_max=1000,
            eos='birch_murnaghan',
            verbose=True
        )
        #qha.write_volume_temperature(f'{self.result_folder_path}vt.dat')
        #qha.plot_volume_temperature().savefig(f'{self.result_folder_path}v.png')

        qha.write_thermal_expansion(f'{self.result_folder_path}thermal_expansion.dat')
        #qha.plot_thermal_expansion().savefig(f'{self.result_folder_path}thermal.png')

        qha.write_heat_capacity_P_polyfit(f'{self.result_folder_path}cp_polyfit.dat')
        #qha.plot_heat_capacity_P_polyfit().savefig(f'{self.result_folder_path}cv.png')
        
        #qha.plot_helmholtz_volume().savefig(f'{self.result_folder_path}hv_final.png')
        qha.write_gruneisen_temperature(f'{self.result_folder_path}gruneisen.dat')
        #qha.plot_gruneisen_temperature().savefig(f'{self.result_folder_path}g.png')
        
        qha.write_bulk_modulus_temperature(f'{self.result_folder_path}bulk.dat')

evFile = 'e-v.dat'
volumes = []
energies = []
with open(evFile, 'r') as file:
    for line in file:
        volumes.append(float(line.split()[0]))
        energies.append(float(line.split()[1]))

thermal_folder_path = '03-thermals/'
result_folder_path = '04-result/'
qha_c = QHACalc(volumes, energies, thermal_folder_path, result_folder_path)

start_index = -10
end_index = 8
qha_c.qha(start_index, end_index)
