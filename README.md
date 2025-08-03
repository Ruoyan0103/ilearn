# iLearn: Benchmark Suite for Interatomic Potentials

`ilearn` is a benchmark suite for interatomic potentials in materials science, with a focus on radiation damage research. It provides tools to evaluate the performance of interatomic potentials based on thermal, elastic, and vibrational properties, as well as threshold displacement energies (TDEs). It is compatible with other machine learning and empirical interatomic potentials, and its modular design allows easy extension to support new potential types.

## Structure

The `ilearn` package is organized as follows:

- **Main entry points:**
  - [`ilearn/potentials/gap.py`](ilearn/potentials/gap.py): Implements GAP-related functionality.
  - [`ilearn/potentials/meam.py`](ilearn/potentials/meam.py): Implements MEAM-related functionality.

- **Core modules:**
  - `ilearn.lammps`: Provides methods based on **LAMMPS** for computing properties such as energy, forces, and elastic constants.
  - `ilearn.phonopy`: Integrates with **PHONOPY** to compute vibrational and thermal properties.
    - `ilearn.errors`: Implements RMSE-based error metrics for comparing potential predictions with reference data.


## 📦 Required Dependencies

The following Python packages are required to run `ilearn`:

- `ase==3.25.0`  
- `Deprecated==1.2.18`  
- `matplotlib==3.8.4`  
- `numpy==2.3.1`  
- `ovito==3.13.0`  
- `PyYAML==6.0.2`  
- `quippy==0.3.3`  
- `scikit_learn==1.7.0`  
- `scipy==1.16.0`  
- `seaborn==0.13.2`  

> **Note**: Duplicate entries (e.g., `PyYAML`) have been removed.

---

### 📥 Installation

Install all dependencies with:

```bash
pip install -r requirements.txt


