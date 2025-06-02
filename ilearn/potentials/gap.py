# coding: utf-8
class GAPotential:
    def __init__(self, potential_name, param_filename, calc_args=None):
        self.potential_name = potential_name     # 'xml_label=GAP_2025_3_8_120_22_43_25_989'
        self.param_filename = param_filename     # 'Ge-v10.xml'
        self.calc_args = calc_args               # 'local_gap_variance'