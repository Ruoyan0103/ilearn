import os
import unittest
from unittest import mock
import tempfile
import numpy as np
from matplotlib import pyplot as plt

from ilearn.errors.rmse import RMSECalculator
from ilearn.potentials import gap 


CWD = os.getcwd()
test_dataset = os.path.join(os.path.dirname(__file__), 'test.extxyz')
potential = gap.GAPotential(
    potential_name='xml_label=GAP_2025_3_8_120_22_43_25_989',
    param_filename=os.path.join(CWD, 'testpot/Ge-v10.xml'),
    calc_args='local_gap_variance')

class TestRMSECalculator(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory and dummy file paths
        self.tmpdir = tempfile.TemporaryDirectory()
        self.files_folder = self.tmpdir.name
        self.energy_path = 'energy'
        self.force_true = 'force_true'
        self.force_pred = 'force_pred'
        self.virial_true = 'virial_true'
        self.virial_pred = 'virial_pred'
        self.config_type = 'config_type'
        self.figure = os.path.join(self.tmpdir.name, 'figure.png')

        CWD = os.path.dirname(__file__)
        test_dataset = os.path.join(CWD, 'test.extxyz')
        potential = gap.GAPotential(
            potential_name='xml_label=GAP_2025_3_8_120_22_43_25_989',
            param_filename=os.path.join(CWD, 'testpot/Ge-v10.xml'),
            calc_args='local_gap_variance')
    
        self.dataset_path = test_dataset
        self.potential = potential

        # Instantiate RMSECalculator
        self.calc = RMSECalculator(
            self.dataset_path,
            self.potential,
            self.files_folder,
            self.energy_path,
            self.force_true,
            self.force_pred,
            self.virial_true,
            self.virial_pred,
            self.config_type,
            self.figure
        )

    def tearDown(self):
        # Cleanup temporary directory
        self.tmpdir.cleanup()


    def test_write_energy_and_read_back(self):
        energy_path = self.calc._write_energy(1.23, 1.11, 5)
        with open(energy_path) as f:
            line = f.readline().strip()
        self.assertEqual(line, "1.23, 1.11, 5")

    def test_write_force(self):
        true_force = [0.1, 0.2, 0.3]
        pred_force = [0.15, 0.25, 0.35]

        force_true = self.calc._write_force(true_force, pred_force, is_true=True)
        force_pred = self.calc._write_force(true_force, pred_force, is_true=False)

        with open(force_true) as f_true, open(force_pred) as f_pred:
            self.assertEqual(f_true.readline().strip(), "0.1 0.2 0.3")
            self.assertEqual(f_pred.readline().strip(), "0.15 0.25 0.35")

    def test_plot_energy(self):
        # Write dummy energy data
        with open(self.energy_path, 'w') as f:
            f.write("1.0, 0.95, 1\n2.0, 1.9, 4\n")

        fig, ax = plt.subplots()
        self.calc.plot_energy(ax)

        # Scatter plot should be present
        self.assertGreater(len(ax.collections), 0)

    def test_plot_force(self):
        with open(self.force_true, 'w') as f_true, open(self.force_pred, 'w') as f_pred:
            f_true.write("0.1 0.2 0.3\n")
            f_pred.write("0.15 0.25 0.35\n")

        fig, ax = plt.subplots()
        self.calc.plot_force(ax)

        self.assertGreater(len(ax.collections), 0)

    def test_plot_virial(self):
        with open(self.virial_true, 'w') as f_true, open(self.virial_pred, 'w') as f_pred:
            f_true.write("0.1 0.2 0.3\n")
            f_pred.write("0.15 0.25 0.35\n")

        fig, ax = plt.subplots()
        self.calc.plot_virial(ax)

        self.assertGreater(len(ax.collections), 0)

    def test_get_efv(self):
        self.calc.get_efv()

        self.assertTrue(self.calc.number_atoms[0] == 1)
        self.assertTrue(self.calc.label_energies[0] == -4.09401418)
        self.assertTrue(self.calc.predict_energies[0] - (-4.09401418) < 1e-3)
        self.assertTrue(self.calc.label_forces[0][0] == 10.012)
        self.assertTrue(self.calc.predict_forces[0][0] - 10.012 < 1e-3)

    def test_extract_config_type(self):
        config_types = self.calc._extract_config_type()
        self.assertEqual(next(iter(config_types)), 'fcc_bulk')

    def test_get_efv_config_types(self):
        self.calc.get_efv_config_type()

if __name__ == '__main__':
    unittest.main()
