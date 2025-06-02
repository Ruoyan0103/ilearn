# coding: utf-8
# Distributed under the terms of the BSD License.

"""This package contains Potential classes representing Interatomic Potentials."""
from abc import ABC 

class Potential(ABC):
    """
    Abstract Base class for a Interatomic Potential.
    """

    @abstractmethod
    def train(self, dataset_filename, **kwargs):
        """
        Train interatomic potentials with energies, forces and
        stresses corresponding to structures.

        Args:
            dataset_filename (string): File containing a list of ASE Structure objects.
            energies (list): List of DFT-calculated total energies of each structure
                in structures list.
            forces (list): List of DFT-calculated (m, 3) forces of each structure
                with m atoms in structures list. m can be varied with each single
                structure case.
            stresses (list): List of DFT-calculated (6, ) virial stresses of each
                structure in structures list.
        """
        pass

    @abstractmethod
    def evaluate(self, test_structures, ref_energies, ref_forces, ref_stresses):
        """
        Evaluate energies, forces and stresses of structures with trained
        interatomic potentials.

        Args:
            test_structures (list): List of ASE Structure Objects.
            ref_energies (list): List of DFT-calculated total energies of each
                structure in structures list.
            ref_forces (list): List of DFT-calculated (m, 3) forces of each
                structure with m atoms in structures list. m can be varied with
                each single structure case.
            ref_stresses (list): List of DFT-calculated (6, ) viriral stresses of
                each structure in structures list.

        Returns:
            DataFrame of original data and DataFrame of predicted data.
        """
        pass

    @abstractmethod
    def predict(self, structure):
        """
        Predict energy, forces and stresses of the structure.

        Args:
            structure (Structure): ASE Structure object.

        Returns:
            energy, forces, stress
        """
        pass


