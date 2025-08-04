"""This class provides calculation using Phonopy."""
import os, subprocess, time
import numpy as np
from ilearn.loggers.logger import AppLogger
from abc import ABC, abstractmethod
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.interface.calculator import write_crystal_structure
from phonopy.file_IO import parse_FORCE_SETS
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from phonopy import PhonopyQHA
import yaml

module_dir = os.path.dirname(os.path.abspath(__file__))
result_dir = os.path.join(module_dir, 'results')
log_dir = os.path.join(module_dir, 'logs')


class PhonopyCalculator(ABC):
    def __init__(self, task_name, ff_settings, mass, alat, size=None, element=None, lattice=None):
        self.template_dir = os.path.join(module_dir, 'templates', task_name)
        self.calculation_dir = os.path.join(result_dir, task_name)
        self.log_file = os.path.join(log_dir, f'{task_name}_GAP.log')

        os.makedirs(self.template_dir, exist_ok=True)
        os.makedirs(self.calculation_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # if os.path.exists(log_file):
        #     os.remove(log_file) 
        # delete log file manually
        self.logger = AppLogger(__name__, self.log_file, overwrite=True).get_logger()
        self.ff_settings = ff_settings
        self.mass = mass
        self.element = element
        self.lattice = lattice
        self.alat = alat
        self.size = size

    @abstractmethod
    def _setup(self):
        """
        Setup the input file for the LAMMPS simulation.
        This method prepares the input file with the necessary parameters.
        """
        pass

    @abstractmethod
    def calculate(self):
        """
        Calculate the properties using LAMMPS.
        This method sets up the simulation and starts the calculation.
        """
        pass


class PhononDispersion(PhonopyCalculator):
    """
    Phonon dispersion calculator
    """
    def __init__(self, task_name, ff_settings, mass, alat, size, element, lattice):
        super().__init__(task_name, ff_settings, mass, alat, size=size, element=element, lattice=lattice)

        self.unitcell_volume = None
        self.unitcell_energy = None


    def _setup(self):
        with open(os.path.join(self.template_dir, 'in.dispersion'), 'r') as f:
            input_template = f.read()
        input_file = os.path.join(self.calculation_dir, 'in.dispersion')
        with open(input_file, 'w') as f:
            f.write(input_template.format(ff_settings='\n'.join(self.ff_settings), mass=self.mass))

        with open(os.path.join(self.template_dir, 'submit.sh'), 'r') as f:
            submit_template = f.read()
        submit_file = os.path.join(self.calculation_dir, 'submit.sh')
        with open(submit_file, 'w') as f:
            f.write(submit_template.format(in_file='in.dispersion'))

        # pre-process 
        if self.lattice == 'diamond':
            unitcell = PhonopyAtoms(symbols=[self.element] * 8,
                            cell=(np.eye(3) * self.alat),
                            scaled_positions=[[0, 0, 0],
                                            [0, 0.5, 0.5],
                                            [0.5, 0, 0.5],
                                            [0.5, 0.5, 0],
                                            [0.25, 0.25, 0.25],
                                            [0.25, 0.75, 0.75],
                                            [0.75, 0.25, 0.75],
                                            [0.75, 0.75, 0.25]])
            phonon = Phonopy(unitcell,
                    supercell_matrix=[[self.size, 0, 0], [0, self.size, 0], [0, 0, self.size]],
                    primitive_matrix=[[0, 0.5, 0.5],
                                      [0.5, 0, 0.5],
                                      [0.5, 0.5, 0]],
                    calculator='lammps')
            phonon.generate_displacements(distance=0.01)
            phonon.save(os.path.join(self.calculation_dir, 'phonopy_disp.yaml'))

            unitcell_file = os.path.join(self.calculation_dir, 'unitcell')
            self.unitcell_volume = unitcell.get_volume()
            write_crystal_structure(unitcell_file, unitcell, interface_mode='lammps')
            supercells = phonon.supercells_with_displacements
            supercell_file = os.path.join(self.calculation_dir, 'supercell-001')
            write_crystal_structure(supercell_file, supercells[0], interface_mode='lammps')

        elif self.lattice == 'fcc':
            pass

        # get force sets
        subprocess.run('sbatch submit.sh', shell=True, check=True, cwd=self.calculation_dir)
        time.sleep(10)
        
        file_path = os.path.join(self.calculation_dir, 'log.lammps')
        with open(file_path, 'r') as file:
            lines = file.readlines()
        eng_idx = 0
        for index, line in enumerate(lines):
            if 'Loop' in line:
                eng_idx = index - 1
        self.unitcell_energy = lines[eng_idx].split()[4]

        subprocess.run('phonopy -f force.0', shell=True, check=True, cwd=self.calculation_dir)
        time.sleep(5)

        # post-process
        force_sets = parse_FORCE_SETS(filename=os.path.join(self.calculation_dir, 'FORCE_SETS'))
        phonon.set_displacement_dataset(force_sets)
        phonon.produce_force_constants()

        return phonon 
    

    def calculate(self):
        phonon = self._setup()
        if self.lattice == 'diamond':
            path = [[[0, 0, 0], [0.5, 0, 0.5], [0.5, 0.5, 1]], 
                    [[0.375, 0.375, 0.75], [0, 0, 0], [0.5, 0.5, 0.5]]]
            labels = ['$\Gamma$', 'X', 'K', '$\Gamma$', 'L']
            qpoints, connections = get_band_qpoints_and_path_connections(path, npoints=11)
            phonon.run_band_structure(qpoints, path_connections=connections, labels=labels)
            phonon.write_yaml_band_structure(filename=os.path.join(self.calculation_dir, 'band.yaml')) 
            time.sleep(10)
            # subprocess.run('phonopy-bandplot --gnuplot band.yaml > raw-data.txt', shell=True, check=True, cwd=self.calculation_dir)
            # time.sleep(20)
            phonon.run_mesh([45, 45, 45])
            phonon.run_total_dos()
            phonon.run_thermal_properties(t_step=10,
                              t_max=1000,
                              t_min=0)
            phonon.write_yaml_thermal_properties(filename=os.path.join(self.calculation_dir, 'thermal_properties.yaml'))
            phonon.write_total_dos(filename=os.path.join(self.calculation_dir, 'total_dos.dat'))


class Quasiharmonic(PhonopyCalculator):
    """
    Quasiharmonic approximation calculator.
    """
    def __init__(self, ff_settings, mass, alat, size, element, lattice, start_rate, end_rate, num_points):
        """
        Initialize the Quasiharmonic calculator.
        Args:
            ff_settings (str): Force field settings for the simulation.
            mass (float): Atomic mass of the element.
            alat (float): Lattice constant in Angstrom.
            size (int): Supercell size.
            element (str): Chemical symbol of the element.
            lattice (str): Type of lattice structure ('diamond', 'fcc', etc.).
            start_rate (float): Starting rate for volume compression, negative.
            end_rate (float): Ending rate for volume expansion, positive. 
            step_rate (float): Step rate for volume change
        """
        super().__init__('quasiharmonic', ff_settings, mass, alat, size=size, element=element, lattice=lattice)
        self.start_rate = start_rate
        self.end_rate = end_rate
        self.num_points = num_points
        self.volumes = []
        self.energies = []


    def _setup(self):
        rates = np.linspace(self.start_rate, self.end_rate, self.num_points)
        for index, rate in enumerate(rates):
            l = self.alat * (1+round(rate, 3))
            ph_unitcell = PhononDispersion(f'quasiharmonic', self.ff_settings, self.mass, 
                                           l, size=1, element=self.element, lattice=self.lattice)
            ph_unitcell.calculate()
            e, v = ph_unitcell.unitcell_energy, ph_unitcell.unitcell_volume
            self.volumes.append(v)
            self.energies.append(e)

            ph_supercell = PhononDispersion(f'quasiharmonic', self.ff_settings, self.mass,
                                            l, size=self.size, element=self.element, lattice=self.lattice)
            ph_supercell.calculate()
            os.rename(os.path.join(ph_supercell.calculation_dir, 'thermal_properties.yaml'),
                      os.path.join(self.calculation_dir, f'thermal_properties{index}.yaml'))
        if os.path.exists(os.path.join(self.calculation_dir, 'e-v.dat')):
            os.remove(os.path.join(self.calculation_dir, 'e-v.dat'))
        for v, e in zip(self.volumes, self.energies):
            with open(os.path.join(self.calculation_dir, 'e-v.dat'), 'a') as f:
                f.write(f'{v} {e}\n')
            

    def calculate(self):
        self._setup()
        # post-process
        self.volumes = np.loadtxt(os.path.join(self.calculation_dir, 'e-v.dat'), usecols=0)
        self.energies = np.loadtxt(os.path.join(self.calculation_dir, 'e-v.dat'), usecols=1)
        entropy = []
        cv = []
        fe = []
        rates = np.linspace(self.start_rate, self.end_rate, self.num_points)
        for index, _ in enumerate(rates):
            file_name = os.path.join(self.calculation_dir, f'thermal_properties{index}.yaml')
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
            eos='vinet',
            verbose=True
        )
        qha.write_thermal_expansion(os.path.join(self.calculation_dir, 'thermal_expansion.dat'))
        qha.write_heat_capacity_P_polyfit(os.path.join(self.calculation_dir, 'cp_polyfit.dat'))
        qha.write_gruneisen_temperature(os.path.join(self.calculation_dir, 'gruneisen.dat'))
        qha.write_bulk_modulus_temperature(os.path.join(self.calculation_dir, 'bulk.dat'))


    

    

    
            

        
        

        

        








        
        


