import os 
from ilearn.potentials import IPotential
from ilearn.lammps.calcs import ThresholdDisplacementEnergy
from ilearn.phonopy.calcs import PhononDispersion
import numpy as np

module_dir = os.path.dirname(__file__)
class MEAMPotential(IPotential):
    pair_style = 'pair_style        meam'
    pair_coeff = 'pair_coeff        * * {} {} {} {}'

    def __init__(self, name=None, param=None):
        """

        Args:
            name (str): Name of force field.
            param (dict): The parameter configuration of potentials.
        """
        self.name = name if name else "MEAMPotential"
        self.param = param if param else {}
        self.specie = None

    def write_param(self, library_filename, element_filename, element_symbol):
        """
        Write potential parameters for lammps calculation.

        Args:
            library_filename (str): Path to the MEAM library file.
            element_filename (str): Path to the MEAM element file.
            element_symbol (str): Symbol of the element (e.g., 'Ge').
        """
        self.pair_coeff = self.pair_coeff.format(library_filename, element_symbol,
                                                 element_filename, element_symbol)
        ff_settings = [self.pair_style, self.pair_coeff]
        return ff_settings

# example usage
if __name__ == "__main__":
    library_file = os.path.join(module_dir, 'params', 'MEAM', 'library.meam')
    element_file = os.path.join(module_dir, 'params', 'MEAM', 'Ge.meam')
    element_symbol = 'Ge'
    meam = MEAMPotential()
    ff_settings = meam.write_param(library_file, element_file, element_symbol)
    alat = 5.658
    pka_id = 2766
    temp = 0
    element = 'Ge'
    mass = 72.56
    # velocity in angstrom per picosecond
    # test range (min_velocity, max_velocity] 
    min_velocity = 70
    max_velocity = 75
    velocity_interval = 5
    kin_eng_threshold = 4
    simulation_size = 9
    thermal_time = 5  # in second
    tde_time = 50     # in second

    tde = ThresholdDisplacementEnergy(ff_settings, element, mass, alat, temp,
                                      pka_id, min_velocity, max_velocity, 
                                      velocity_interval, kin_eng_threshold, simulation_size,
                                      thermal_time, tde_time)
    vector1 = [0., 0., 1.] / np.linalg.norm([0., 0., 1.])  # Normalize the vector
    vector2 = [1., 0., 1.] / np.linalg.norm([1., 0., 1.])  # Normalize the vector
    vector3 = [1., 1., 1.] / np.linalg.norm([1., 1., 1.])  # Normalize the vector
    vectors = np.array((vector1, vector2, vector3))
    # tde.get_uniform_angles(vectors, 2)
    # tde.set_hkl_from_angles()
    # tde.check_interval()
    # tde.calculate(needed_thermalization=True)
    #tde.plot()
    #tde.plot_no_interplation()
    #tde.average_TDE()
    ph = PhononDispersion(ff_settings, mass, alat, size=2, element=element, lattice='diamond')
    ph.calculate()
    
    


        
