import os 
from ilearn.potentials import IPotential
from ilearn.lammps.calcs import ThresholdDisplacementEnergy
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
    pka_id = 450
    temp = 0
    element = 'Ge'
    mass = 72.56
    min_velocity = 110
    max_velocity = 115
    velocity_interval = 5
    kin_eng_threshold = 4
    simulation_size = 5

    tde = ThresholdDisplacementEnergy(ff_settings, element, mass, alat, temp,
                                      pka_id, min_velocity, max_velocity, 
                                      velocity_interval, kin_eng_threshold, simulation_size)
    vector1 = [0., 0., 1.] / np.linalg.norm([0., 0., 1.])  # Normalize the vector
    vector2 = [1., 0., 1.] / np.linalg.norm([1., 0., 1.])  # Normalize the vector
    vector3 = [1., 1., 1.] / np.linalg.norm([1., 1., 1.])  # Normalize the vector
    vectors = np.array((vector1, vector2, vector3))
    tde.get_uniform_angles(vectors, 4)
    tde.set_hkl_from_angles()
    # tde.check_interval()
    tde.calculate()
    tde.plot()
    tde.plot_no_interplation()
    
    


        
