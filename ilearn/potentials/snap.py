import re
import os
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
from ase.io import read 
from quippy.potential import Potential
from sklearn.metrics import mean_squared_error
from ilearn.lammps.calcs import ThresholdDisplacementEnergy, LatticeConstant, ElasticConstant, \
                                VacancyDefectFormation, NudgedElasticBand
from ilearn.potentials import IPotential

module_dir = os.path.dirname(__file__)
results_dir = os.path.join(module_dir, 'results')

class SNAPotential(IPotential):
    pair_style = 'pair_style        snap'
    pair_coeff = 'pair_coeff        * * {} {} {}'

    def __init__(self, name=None, param=None):
        """

        Args:
            name (str): Name of force field.
            param (dict): The parameter configuration of potentials.
        """
        self.name = name if name else "SNAPotential"
        self.param = param if param else {}
        self.specie = None