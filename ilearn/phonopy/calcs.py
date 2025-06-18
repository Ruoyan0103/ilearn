import os, subprocess, time, shutil 
import numpy as np
from ilearn.loggers.logger import AppLogger
from abc import ABC, abstractmethod
from ase.build import bulk 
from ase.io import read, write
import phonopy 
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.interface.calculator import read_crystal_structure, write_crystal_structure
from phonopy.file_IO import parse_FORCE_SETS
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections


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
    def __init__(self, ff_settings, mass, alat, size, element, lattice):
        super().__init__('dispersion', ff_settings, mass, alat, size=size, element=element, lattice=lattice)


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
            write_crystal_structure(unitcell_file, unitcell, interface_mode='lammps')
            supercells = phonon.supercells_with_displacements
            supercell_file = os.path.join(self.calculation_dir, 'supercell-001')
            write_crystal_structure(supercell_file, supercells[0], interface_mode='lammps')

        elif self.lattice == 'fcc':
            pass

        # get force sets
        subprocess.run('sbatch submit.sh', shell=True, check=True, cwd=self.calculation_dir)
        time.sleep(10)
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
            time.sleep(20)
            # subprocess.run('phonopy-bandplot --gnuplot band.yaml > raw-data.txt', shell=True, check=True, cwd=self.calculation_dir)
            # time.sleep(20)
            phonon.run_mesh([45, 45, 45])
            phonon.run_total_dos()
            phonon.write_total_dos(filename=os.path.join(self.calculation_dir, 'total_dos.dat'))
            

        
        

        

        








        
        


