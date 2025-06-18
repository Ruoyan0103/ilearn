import os, math, subprocess, time, shutil
import numpy as np
from deprecated import deprecated
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from ovito.io import import_file
from ovito.modifiers import WignerSeitzAnalysisModifier
from ilearn.loggers.logger import AppLogger
from ase.io import write
from ase import Atoms 
from ase.build import bulk
from abc import ABC, abstractmethod

plt.rcParams['font.family'] = 'DejaVu Serif'

AMU_TO_KG = 1.66053906660E-27 # Atomic mass unit to kg conversion factor
JOULE_TO_EV = 6.241509074E18  # Joule to eV conversion factor
ANGSTROM_TO_METER = 1E-10     # Angstroms/picosecond to meters/second conversion factor
PS_TO_S = 1E-12               # Picoseconds to seconds conversion factor

module_dir = os.path.dirname(os.path.abspath(__file__))
result_dir = os.path.join(module_dir, 'results')
log_dir = os.path.join(module_dir, 'logs')


class LMPStaticCalculator(ABC):
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


class ThresholdDisplacementEnergy(LMPStaticCalculator):
    """ 
    Threshold displacement energy calculator.
    """
    def __init__(self, ff_settings, element, mass, alat, temp, pka_id,
                 min_velocity, max_velocity, velocity_interval, kin_eng_threshold, size,
                 thermal_time, tde_time):
        '''
        Initialize the ThresholdDisplacementEnergy class.
        Parameters
        ----------
        ff_settings : list
            Force field settings, either as a Potential object or a list of strings.
        element : str
            Element symbol for the primary knock-on atom.
        mass : float
            Mass of the primary knock-on atom (in atomic mass units).
        alat : float    
            Lattice constant (in Angstroms).
        temp : float    
            Temperature for the simulation (in Kelvin).
        pka_id : int    
            Primary knock-on atom ID.
        min_velocity : float
            Minimum velocity for the primary knock-on atom (in m/s).
        max_velocity : float
            Maximum velocity for the primary knock-on atom (in m/s).
        velocity_interval : float
            Interval for velocity sampling (in m/s).
        kin_eng_threshold : float   
            Threshold for kinetic energy difference (in eV).
        '''
        super().__init__('tde', ff_settings, mass, alat, size=size, element=element)
        self.angle_set = set()
        self.angle_list = []
        self.listCoords = []
        self.hkl_list = []
        self.temp = temp
        self.pka_id = pka_id
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.velocity_interval = velocity_interval
        self.kin_eng_threshold = kin_eng_threshold  
        self.thermal_file = os.path.join(self.calculation_dir, 'data.thermalized')
        self.finished_hkl_file = os.path.join(self.calculation_dir, 'finished_hkl.txt')
        self.thermal_time = thermal_time  # Time for thermalization in seconds
        self.tde_time = tde_time          # Time for TDE calculation in seconds
        
        
    def _setup_helper(self, velocity, hkl, vel_hkl_dir):
        '''
        Prepare the input file for the TDE calculation. (in.tde)
        Parameters
        ----------
        velocity : float
            Velocity of the primary knock-on atom (in ang/ps).
        hkl : np.ndarray
            HKL direction vector (3D numpy array).
        vel_hkl_dir : str
            Directory path for the current velocity and HKL combination.
        '''
        # ---------------------- write in.tde -----------------------
        with open(os.path.join(self.template_dir, 'in.tde'), 'r') as f:
            input_template = f.read()
        ff_settings = self.ff_settings
        input_file = os.path.join(vel_hkl_dir, 'in.tde')
        with open(input_file, 'w') as f:
            Vx = velocity * hkl[0]
            Vy = velocity * hkl[1]
            Vz = velocity * hkl[2]
            f.write(input_template.format(ff_settings='\n'.join(ff_settings),
                                          mass=self.mass, alat=self.alat, pka_id=self.pka_id,
                                          temp=self.temp, element=self.element,
                                          V_x=Vx, V_y=Vy, V_z=Vz))
        # ---------------------- copy submit-tde.sh ------------------
        shutil.copy(os.path.join(self.template_dir, 'submit-tde.sh'), 
                    os.path.join(vel_hkl_dir, 'submit-tde.sh'))
        

    @deprecated(reason="Outside dir is hkl, inside dir is velocity, bad design, use _setup instead.")
    def _setup_old(self):
        '''
        Setup the input file for the LAMMPS simulation.
        This method prepares the input file with the necessary parameters.
        '''
        if not self.hkl_list:
            self.logger.error("HKL list is empty. Please use set_hkl_from_angles() first.")
            raise ValueError("HKL list is empty. Please generate HKL values first.")
        for hkl in self.hkl_list:
            hkl = np.array(hkl)
            v = self.min_velocity
            v_interval = self.velocity_interval
            kinetic_energy = lambda v: 0.5 * self.mass * AMU_TO_KG * np.sum(hkl**2) * (v*ANGSTROM_TO_METER/PS_TO_S)**2 * JOULE_TO_EV
            
            prev_kin_eng = kinetic_energy(v)
            while v <= self.max_velocity:
                next_v = v + v_interval
                next_kin_eng = kinetic_energy(next_v)
                energy_diff = next_kin_eng - prev_kin_eng
                # If energy difference too large, reduce step and try again
                while energy_diff > self.kin_eng_threshold:
                    v_interval /= 2
                    print('--------------------------- Interval reduced-- ----------------------')
                    print(f"Current energy diff : {energy_diff}. Reducing velocity interval to {v_interval}.")
                    if v_interval < 1:  # safeguard minimum step
                        raise ValueError("Too small velocity interval.")
                    next_v = v + v_interval
                    next_kin_eng = kinetic_energy(next_v)
                    energy_diff = next_kin_eng - prev_kin_eng
                    print(f"New energy diff : {energy_diff} with reduced interval {v_interval}.\n")
                
                print(f'--------------------------- Folder created {hkl}------------------------')
                print(f"Velocity: {next_v} ang/pic, Kinetic energy: {next_kin_eng:.2f} eV\n\n")
                self._setup_helper(hkl, next_v)
                v = next_v
                prev_kin_eng = next_kin_eng

                    
    def _setup(self):
        '''
        Set up the simulation environment for the TDE calculation.
        Create folders - dir: velocity 
                         -- dir: hkl 
                            -- files: in.tde, submit-tde.sh
        '''
        v = self.min_velocity
        while v <= self.max_velocity:
            velocity_dir = os.path.join(self.calculation_dir, str(v))
            for idx, hkl in enumerate(self.hkl_list):
                vel_hkl_dir = os.path.join(velocity_dir, str(idx))
                os.makedirs(vel_hkl_dir, exist_ok=True)
                self._setup_helper(v, hkl, vel_hkl_dir)
            v += self.velocity_interval
    

    def _check_vacancies_with_reference(self, velocity, hkl_idx, trajectory_file):
        '''
        Check for vacancies in the thermalized data by comparing with a reference file.
        Parameters
        ----------
        velocity : float
            Previous velocity used for the calculation.
        hkl_idx : int   
            Index of the HKL direction in the hkl_list.
        trajectory_file : str
            Path to the trajectory file containing particle data.
        Returns 
        -------
        bool
            True if vacancies are detected, False otherwise.
        Raises
        -------
        ValueError
            If the thermal file is not set or the trajectory file is not found.
        Notes
        -----
        This method uses the Wigner-Seitz Analysis Modifier to check for vacancies.
        It requires the thermal file to be set from a previous thermalization step.
        It raises an error if the trajectory file does not exist.
        
        '''
        vac_flag = False
        reference_file = self.thermal_file
        if not os.path.isfile(trajectory_file):
            self.logger.error(f"{trajectory_file} is not found. Please wait for TDE simulation to finish.")
            raise FileNotFoundError(f"{trajectory_file} is not found. Please wait for TDE simulation to finish.")
        reference_pipeline = import_file(reference_file)
        pipeline = import_file(trajectory_file)
        wsam = WignerSeitzAnalysisModifier(per_type_occupancies=True, output_displaced=False)
        wsam.reference = reference_pipeline.source
        pipeline.modifiers.append(wsam)

        data = pipeline.compute(0)
        if 'Occupancy' not in data.particles or 'Particle Identifier' not in data.particles or 'Position' not in data.particles:
            self.logger.error("Required data (Occupancy, Particle Identifier, or Position) not found in the trajectory file.")
            raise ValueError("Required data (Occupancy, Particle Identifier, or Position) not found in the trajectory file.")
        for particle_id, occupancy, position in zip(data.particles['Particle Identifier'],
                                                    data.particles['Occupancy'],
                                                    data.particles['Position']):
            if occupancy == 0:
                self.logger.info(f"{velocity}/{hkl_idx}/dump_out: Vacancy detected. Particle ID: {particle_id}, Position: {position}")
                vac_flag = True
                break 
        if not vac_flag:
            self.logger.info(f"{velocity}/{hkl_idx}/dump_out: No vacancies detected.")
        return vac_flag
    

    def _thermalize(self):
        '''
        Thermalize the system using the input template and force field settings.
        This method creates an input file for the thermalization process,
        and start the thermalization calculation.
        The calculation is done in 'module_dir/results/calculation'.
        '''
        # ------------------------------ write in.thermalize ------------------------------
        with open(os.path.join(self.template_dir, 'in.thermalize'), 'r') as f:
            input_template = f.read()
            ff_settings = self.ff_settings
        input_file = os.path.join(self.calculation_dir, 'in.thermalize')
        # self.thermal_file = os.path.join(calculation_dir, 'data.thermalized')
        with open(input_file, 'w') as f:
            f.write(input_template.format(ff_settings='\n'.join(ff_settings),
                                            mass=self.mass, alat=self.alat, size=self.size,
                                            output_thermalized=self.thermal_file)) 
        # ------------------------------ copy submit-thermal.sh ------------------------------
        shutil.copy(os.path.join(self.template_dir, 'submit-thermal.sh'), 
                    os.path.join(self.calculation_dir, 'submit-thermal.sh'))
        # ------------------------------------- submit job -----------------------------------
        subprocess.run('sbatch submit-thermal.sh', shell=True, check=True, cwd=self.calculation_dir)
        time.sleep(self.thermal_time)  
        # dummy trajectory file for the minimum velocity
        # all calculations will start with minumum velocity + velocity interval
        velocity_dir = os.path.join(self.calculation_dir, str(self.min_velocity))
        for idx, _ in enumerate(self.hkl_list):
            vel_hkl_dir = os.path.join(velocity_dir, str(idx))
            shutil.copy(os.path.join(self.calculation_dir, 'data.thermalized'),
                        os.path.join(vel_hkl_dir, 'dump_out'))


    def _write_TDE(self, hkl_idx, velocity):
        '''
        Write the threshold displacement energy (TDE) to a file.
        Parameters
        ----------
        hkl_idx : int
            Index of the HKL direction in the hkl_list.
        velocity : float
            Velocity of the primary knock-on atom (in m/s).
        Raises
        ------
        ValueError
            If the HKL list is empty or if the angle list is empty.
        Notes
        -----
        This method calculates the TDE based on the HKL direction and velocity,
        and writes the results to a file named 'TDE.txt' in the calculation directory.
        The TDE is calculated using the formula:
        TDE = 0.5 * mass * (sum(hkl^2)) * (velocity * ANGSTROM_TO_METER / PS_TO_S)^2 * JOULE_TO_EV
        where:
        - mass is the mass of the primary knock-on atom in atomic mass units (AMU).
        - hkl is the HKL direction vector.
        - velocity is the velocity of the primary knock-on atom in m/s.
        '''
        # Check if file exists to determine if we need to write header
        tde_file = os.path.join(self.calculation_dir, 'TDE.txt')
        write_header = not os.path.exists(tde_file)
        
        hkl = self.hkl_list[hkl_idx]
        angle = self.angle_list[hkl_idx]
        kinetic_energy = 0.5 * self.mass * AMU_TO_KG * np.sum(hkl**2) * (velocity*ANGSTROM_TO_METER/PS_TO_S)**2 * JOULE_TO_EV
        with open(tde_file, 'a') as f:
            if write_header:
                # Write header line
                f.write("# hkl_idx  h       k       l       phi[rad]    theta[rad]   velocity[m/s]  TDE[eV]   phi[deg]  theta[deg]\n")
                f.write("# ----------------------------------------------------------------------------------------------\n")
            
            # Write data line
            f.write(
                f"{hkl_idx:<8.2f}{hkl[0]:<8.2f}{hkl[1]:<8.2f}{hkl[2]:<8.2f}"
                f"{angle[0]} {angle[1]}   "
                f"{velocity * ANGSTROM_TO_METER / PS_TO_S:<14.1f}"
                f"{kinetic_energy:<9.2f}"
                f"{math.degrees(angle[0]):<10.2f}"
                f"{math.degrees(angle[1]):<10.2f}\n"
            )


    def get_random_angles(self, min_phi, max_phi, min_theta, max_theta, num_points):
        '''
        Generate random angles on a sphere within specified ranges.
        Parameters
        ----------
        min_phi : float
            Minimum azimuthal angle (φ) in degrees.
        max_phi : float
            Maximum azimuthal angle (φ) in degrees.
        min_theta : float
            Minimum polar angle (θ) in degrees.
        max_theta : float
            Maximum polar angle (θ) in degrees.
        num_points : int
            Number of random points to generate.
        '''
        min_phi = np.radians(min_phi)
        max_phi = np.radians(max_phi)
        min_theta = np.radians(min_theta)
        max_theta = np.radians(max_theta)
        phi = np.random.uniform(min_phi, max_phi, num_points)                          # azimuthal angle (φ)
        costheta = np.random.uniform(np.cos(min_theta), np.cos(max_theta), num_points) 
        theta = np.arccos(costheta)                                                    # polar angle (θ)
        self.angle_set = set(zip(phi, theta))                                          # Store unique angles


    def get_uniform_angles(self, vectors, degree):
        '''
        Generate uniform angles on a sphere using the Sierpinski triangle method.
        Chakraborty, Aritra & Eisenlohr, Philip. (2017).
        Consistent visualization and uniform sampling of crystallographic directions. 10.13140/RG.2.2.35880.67847.
        Parameters
        ----------
        vectors : np.ndarray
            Array of shape (3, 3) containing three normalized vectors
        degree : int
            Degree of recursion for the Sierpinski triangle method. Higher degree means more points.
        '''
        def append(vector):
            if np.linalg.norm(vector) != 0.0:
                vector = vector / np.linalg.norm(vector)        # Normalize the vector to calculate angles
            phi = math.atan2(vector[1],vector[0])               # azimuthal angle (φ)
            theta = math.acos(vector[2])                        # polar angle (θ)
            X = vector[0] /((1 + abs(vector[2])))               # stereographic projection
            Y = vector[1] /((1 + abs(vector[2])))
            if [X,Y] not in self.listCoords:
                self.listCoords.append([X,Y])
                self.angle_list.append(np.array((phi, theta)))  # Store angles in a list, listCoords helps to remove duplicates
            self.angle_set.add((phi, theta))                    # Store unique angles, no need of listCoords here

        def mid(vector1, vector2):
            mid1 = np.array(([(vector1[0] + vector2[0])/2, (vector1[1] + vector2[1])/2, (vector1[2] + vector2[2])/2]))
            append(mid1)
            return mid1

        def sierpenski(vectors, degree):
            # original code
            # if degree > 0:
            #     sierpenski([vectors[0],mid(vectors[0],vectors[1]),mid(vectors[0],vectors[2])], degree - 1)
            #     sierpenski([vectors[1],mid(vectors[0],vectors[1]),mid(vectors[1],vectors[2])], degree - 1)
            #     sierpenski([vectors[2],mid(vectors[2],vectors[1]),mid(vectors[0],vectors[2])], degree - 1)
            #     sierpenski([mid(vectors[0],vectors[1]), mid(vectors[0],vectors[2]), mid(vectors[1],vectors[2])], degree - 1)
            # return
            # modified code
            if degree > 0:
                vmid_01 = mid(vectors[0], vectors[1])
                vmid_02 = mid(vectors[0], vectors[2])
                vmid_12 = mid(vectors[1], vectors[2])
                sierpenski([vectors[0], vmid_01, vmid_02], degree - 1)
                sierpenski([vectors[1], vmid_01, vmid_12], degree - 1)
                sierpenski([vectors[2], vmid_12, vmid_02], degree - 1)
                sierpenski([vmid_01, vmid_02, vmid_12], degree - 1)
            return

        append(vectors[0])
        append(vectors[1])
        append(vectors[2])
        sierpenski(vectors, degree)
    

    def set_hkl_from_angles(self):
        self.angle_list.append(np.array((np.radians(45), np.radians(1))))    # (phi, theta) : (45°, 1°)
        self.angle_set.add((np.radians(45), np.radians(1)))                  # (phi, theta) : (45°, 1°)
        # self.angle_list.append(np.array((np.radians(0), np.radians(5))))   # (phi, theta) : (0°, 5°)
        # self.angle_set.add((np.radians(0), np.radians(5)))                 # (phi, theta) : (0°, 5°)
        for angle in self.angle_list:
            phi, theta = angle
            h = np.sin(theta) * np.cos(phi)
            k = np.sin(theta) * np.sin(phi)
            l = np.cos(theta)
            self.hkl_list.append(np.array((h, k, l)))
        hkl_file = os.path.join(self.calculation_dir, 'hkl_list.dat')
        with open(hkl_file, 'w') as f:  # 'a' for append, 'w' for overwrite
            np.savetxt(f, np.array(self.hkl_list), 
                       fmt='%.8f',      # 8 decimal places
                       delimiter=' ',   # Space separator
                       header='h     k     l')  # File header


    def plot(self):
        png_file = os.path.join(self.calculation_dir, 'tde_GAP.png')
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        # azimuthal = [angle[0] for angle in self.angle_set]
        # polar = [angle[1] for angle in self.angle_set]
        # energy = np.random.uniform(0, 1, len(azimuthal))  # Random energy values for demonstration

        TDE_txt_file = os.path.join(self.calculation_dir, 'TDE.txt')
        if not os.path.isfile(TDE_txt_file):
            self.logger.error(f"{TDE_txt_file} is not found. Cannot plot.")
            raise FileNotFoundError(f"{TDE_txt_file} is not found. Cannot plot.")
        txt = np.loadtxt(TDE_txt_file, skiprows=2, usecols=(0, 1, 2, 3, 4, 5, 6, 7))
        azimuthal = txt[:, 4]   # Azimuthal angles in radians
        polar = txt[:, 5]       # Polar angles in radians
        energy = txt[:, 7]      # Energy values in eV

        phi_grid = np.linspace(0, np.max(azimuthal), 1000)
        theta_grid = np.linspace(0, np.max(polar), 1000)
        phi_grid2, theta_grid2 = np.meshgrid(phi_grid, theta_grid)
        
        # Mask invalid points in the interpolation grid
        # rcos(theta) = l         (1)
        # rsin(theta)cos(phi) = h (2)
        # rsin(theta)sin(phi) = k (3)
        # The maximum polar angle is when h = l, so we can find the maximum polar angle
        # using (1) and (2), max_polar = np.arctan((h/l) / cos(phi)) = np.arctan(1 / cos(phi))
        max_polar_grid = np.arctan(1 / np.cos(phi_grid2))
        valid_mask = theta_grid2 <= max_polar_grid    # boolean
    
        # Interpolation with masking
        energy_grid = griddata(
            (azimuthal, polar), energy, (phi_grid2, theta_grid2), method='linear'
        )
        energy_grid[~valid_mask] = np.nan  # Set invalid regions to NaN
        
        ax.set_thetalim(0, np.pi/4)
        ax.set_rlim(0, np.deg2rad(54.7))
        ax.set_yticks(np.radians([0, 15, 30, 45, 54.7]))            # Set ticks in polar
        ax.tick_params(axis='y', labelsize=8)
        ax.set_yticklabels(['0°', '15°', '30°', '45°', '54.7°'])
        ax.grid(True, linewidth=0.3, color='#D3D3D3')

        ax.set_xticks(np.radians([0, 5, 10, 15, 20, 25, 30, 35, 40, 45]))  # Set ticks in radians
        ax.set_xticklabels(['0°', '5°', '10°', '15°', '20°', '25°', '30°', '35°', '40°', '45°'], fontsize=8)
        contourf = ax.contourf(phi_grid2, theta_grid2, energy_grid, levels=10, cmap='viridis', alpha=0.89)
        ax.contour(phi_grid2, theta_grid2, energy_grid, levels=10, colors='black', linewidths=0.18)
        ax.scatter(azimuthal, polar, c='#D3D3D3', s=0.8, edgecolors='#D3D3D3')
        
        # Text 1: Along the arc ("Azimuthal")
        angle = np.pi / 8                                # Position along the arc (midway between 0 and pi/4)
        radius = np.deg2rad(54.7)                        # Maximum polar angle
        text_radius = radius * 1.12                      # Slightly farther from the arc
        ax.text(
            angle, text_radius, 'Azimuthal angle, φ', 
            rotation=np.degrees(angle - np.pi / 2),      # Align with the arc's tangent
            ha='center', va='center', fontsize=9, color='black',
        )

        # Text 2: Below the center
        ax.text(
            0.2, 0, '[001]',  # Below the origin
            ha='right', va='bottom', fontsize=9, color='black'
        )

        # Text 3: Below the 54.7° polar angle
        ax.text(
            -0.1, np.deg2rad(45), '[101]',  # Below the arc
            ha='center', va='center', fontsize=9, color='black'
        )

        # Text 4: Above the 45° azimuthal angle
        ax.text(
            0.87, np.deg2rad(53.5), '[111]',  # Slightly above the 45° azimuthal angle
            ha='center', va='center', fontsize=9, color='black'
        )

        # Text 5: xlable
        ax.text(
            -0.2, np.deg2rad(27), 'Polar angle, θ',  # Adjust x and y to position the text
            ha='center', va='center', fontsize=9, color='black'
        )

        vmin, vmax = np.nanmin(energy_grid), np.nanmax(energy_grid)
        tick_step = 3
        ticks = np.arange(np.floor(vmin), np.ceil(vmax) + tick_step, tick_step)
        char = fig.colorbar(contourf, ax=ax, pad=0.15, shrink=0.85, ticks=ticks)
        char.set_label('Threshold displacement energy (eV)', fontsize=9)        
        char.ax.tick_params(
            labelsize=8,          # Smaller font size for numbers
            length=2,             # Shorter tick marks
            width=0.5             # Thinner tick marks
        )
        plt.savefig(png_file, dpi=300)


    def plot_no_interplation(self):
        '''
        Plot the angles without interpolation.
        Used to show how sampling points are distributed in the polar plot.
        '''
        png_file_no_interpolation = os.path.join(self.calculation_dir, 'tde_no_interpolation_GAP.png')
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        TDE_txt_file = os.path.join(self.calculation_dir, 'TDE.txt')
        if not os.path.isfile(TDE_txt_file):
            self.logger.error(f"{TDE_txt_file} is not found. Cannot plot.")
            raise FileNotFoundError(f"{TDE_txt_file} is not found. Cannot plot.")
        txt = np.loadtxt(TDE_txt_file, skiprows=2, usecols=(0, 1, 2, 3, 4, 5, 6, 7))
        azimuthal = txt[:, 4]  
        polar = txt[:, 5]      

        ax.set_thetalim(0, np.pi/4)
        ax.set_rlim(0, np.deg2rad(54.7))
        ax.set_yticks(np.radians([0, 15, 30, 45, 54.7]))            # Set ticks in polar
        ax.tick_params(axis='y', labelsize=8)
        ax.set_yticklabels(['0°', '15°', '30°', '45°', '54.7°'])
        ax.grid(True, linewidth=0.3, color='#D3D3D3')

        ax.set_xticks(np.radians([0, 5, 10, 15, 20, 25, 30, 35, 40, 45]))
        ax.set_xticklabels(['0°', '5°', '10°', '15°', '20°', '25°', '30°', '35°', '40°', '45°'], fontsize=8)
        ax.scatter(azimuthal, polar, c='red', s=2, edgecolors='red')

        plt.savefig(png_file_no_interpolation, dpi=300)


    def average_TDE(self):
        '''
        Average TDE with angles.
        Returns
        -------
        float
            Averaged TDE value.
        '''
        TDE_txt_file = os.path.join(self.calculation_dir, 'TDE.txt')
        if not os.path.isfile(TDE_txt_file):
            self.logger.error(f"{TDE_txt_file} is not found. Cannot calculate average energy.")
            raise FileNotFoundError(f"{TDE_txt_file} is not found. Cannot calculate average energy.")
        txt = np.loadtxt(TDE_txt_file, skiprows=2, usecols=(0, 1, 2, 3, 4, 5, 6, 7))
        if len(txt) < len(self.hkl_list):
            self.logger.error(f"{len(self.hkl_list)-len(txt)} HKL directions are not calculated. ")
        azimuthal = txt[:, 4]   # Azimuthal angles in radians
        polar = txt[:, 5]       # Polar angles in radians
        energy = txt[:, 7]      # Energy values in eV

        number_of_blocks = 100
        phi_grid = np.linspace(0, np.max(azimuthal), number_of_blocks)
        theta_grid = np.linspace(0, np.max(polar), number_of_blocks)

        phi_grid2, theta_grid2 = np.meshgrid(phi_grid, theta_grid)
        max_polar_grid = np.arctan(1 / np.cos(phi_grid2))
        valid_mask = theta_grid2 <= max_polar_grid    # boolean
    
        # Interpolation with masking
        energy_grid = griddata(
            (azimuthal, polar), energy, (phi_grid2, theta_grid2), method='linear'
        )
        energy_grid[~valid_mask] = np.nan  # Set invalid regions to NaN

        d_phi = (np.max(azimuthal)-0)/number_of_blocks
        d_theta = (np.max(polar)-0)/number_of_blocks
        
        numberator = 0
        denominator = 0
        for i in range(number_of_blocks):
            for j in range(number_of_blocks):
                if np.isnan(energy_grid[i][j]):
                    continue
                numberator += energy_grid[i][j] * np.sin(theta_grid2[i][j]) * d_theta * d_phi
                denominator += np.sin(theta_grid2[i][j]) * d_theta * d_phi
        ave_energy = numberator / denominator

        self.logger.info(f"Average TDE: {ave_energy:.2f} eV")
        return ave_energy
    

    def check_interval(self):
        '''
        Check if the velocity interval is safe to satisfy the kinetic energy threshold (threshold).
        '''
        safe_v_interval = True
        for hkl in self.hkl_list:
            hkl = np.array(hkl)
            v = self.min_velocity
            v_interval = self.velocity_interval
            kinetic_energy = lambda v: 0.5 * self.mass * AMU_TO_KG * np.sum(hkl**2) * (v*ANGSTROM_TO_METER/PS_TO_S)**2 * JOULE_TO_EV
            
            prev_kin_eng = kinetic_energy(v)
            while v <= self.max_velocity:
                next_v = v + v_interval
                next_kin_eng = kinetic_energy(next_v)
                energy_diff = next_kin_eng - prev_kin_eng
                if energy_diff > self.kin_eng_threshold:
                    print(f"Velocity {v} ang/pic, Kinetic energy: {prev_kin_eng:.2f} eV")
                    print(f"Energy difference {energy_diff} exceeds threshold {self.kin_eng_threshold}.")
                    safe_v_interval = False
                v = next_v
                prev_kin_eng = next_kin_eng
        return safe_v_interval
    

    def calculate(self, needed_thermalization):
        '''
        Calculate the threshold displacement energy (TDE) for each HKL direction.
        The calculated velocity range is [self.min_velocity+self.velocity_interval, self.max_velocity]
        
        Raises
        ------
        ValueError
            If the HKL list is empty or if the thermal file is not set.
        FileNotFoundError
            If the trajectory file is not found.
        TimeoutError
            If the trajectory file does not exist after the maximum wait time.
        Notes
        -----
        This method performs the following steps:
        1. Calls `_setup` to prepare the input files for the TDE calculation.
        2. Calls `thermalize` to thermalize the system.
        3. Creates a dummy trajectory file for the minimum velocity.
        4. Iterates through the velocity range, checking for existing trajectory files.
        5. If a trajectory file exists, checks for vacancies against the reference thermalized data.
        6. If vacancies are not detected, submits a job for the next velocity.
        Raises an error if the trajectory file is not found.
        '''
        self._setup()
        # if continue from previous calculation, comment thermalize, 
        # and minimum velocity is the last velocity used in the previous calculation,
        # otherwise, start from thermalization
        if needed_thermalization:
            self._thermalize()
            
        
        if os.path.exists(self.finished_hkl_file):
            loaded_ints = np.loadtxt(self.finished_hkl_file, dtype=int) 
            finished_hkl = loaded_ints.astype(bool).tolist()
        else:
            finished_hkl = [False] * len(self.hkl_list)   # initialize a flag list, all hkl are not finished

        pre_v = self.min_velocity
        while pre_v <= self.max_velocity:
            if all(finished_hkl):
                self.logger.info("----------------------------------------------All TDE values are written with velocity < max_velocity.------------------------------------------------")
                break
            higher_energy_needed = False
            for idx, _ in enumerate(self.hkl_list):
                if finished_hkl[idx]:                 # If this hkl is already finished, skip it
                    continue
                velocity_dir = os.path.join(self.calculation_dir, str(pre_v))
                vel_hkl_dir = os.path.join(velocity_dir, str(idx))
                trajectory_file = os.path.join(vel_hkl_dir, 'dump_out')
                if self._check_vacancies_with_reference(pre_v, idx, trajectory_file):
                    finished_hkl[idx] = True
                    self._write_TDE(idx, pre_v)
                else:
                    if pre_v + self.velocity_interval <= self.max_velocity:   # if next velocity is still in range, otherwise, it runs, but won't be checked
                        higher_energy_needed = True
                        next_v = pre_v + self.velocity_interval
                        velocity_dir = os.path.join(self.calculation_dir, str(next_v))
                        vel_hkl_dir = os.path.join(velocity_dir, str(idx))
                        subprocess.run('sbatch submit-tde.sh', shell=True, check=True, cwd=vel_hkl_dir)
            if higher_energy_needed:
                time.sleep(self.tde_time) 
            pre_v += self.velocity_interval

        if not all(finished_hkl):
            self.logger.info(f"--------------------------------The max_velocity is too low for these directions-------------------------------------------")
            for idx, finished_flag in enumerate(finished_hkl):
                if not finished_flag:
                    self.logger.info(f"{idx}: HKL: {self.hkl_list[idx]}, Angle: {math.degrees(self.angle_list[idx][0]):.2f}°/{math.degrees(self.angle_list[idx][1]):.2f}°")
        else:
            self.logger.info("------------------------------------------------All TDE values are written!------------------------------------------------")
        
        finished_hkl_int = np.array(finished_hkl, dtype=int)
        np.savetxt(self.finished_hkl_file, finished_hkl_int, fmt='%d')
                

class LatticeConstant(LMPStaticCalculator):
    """
    Lattice Constant Relaxation Calculator.
    """
    def __init__(self, ff_settings, mass, element, lattice, alat, cubic):
        """
        Initialize the Lattice Constant calculator.
        Parameters
        ----------
        ff_settings : String
            Force field settings.
        mass : float
            Mass of the atom in atomic mass units (AMU).
        element : str
            Element symbol (e.g., 'Ge').
        lattice : str
            Lattice type (e.g., 'diamond').
        alat : float
            Lattice constant in Angstroms.
        cubic : bool
            Whether the lattice is cubic.
        """
        super().__init__('latt', ff_settings, mass, alat, element=element, lattice=lattice)
        static_bulk = bulk(self.element, self.lattice, a=self.alat, cubic=cubic)
        write(os.path.join(self.calculation_dir, 'data.static'), static_bulk, format='lammps-data')


    def _setup(self):
        with open(os.path.join(self.template_dir, 'in.latt'), 'r') as f:
            input_template = f.read()
        input_file = os.path.join(self.calculation_dir, 'in.latt')
        with open(input_file, 'w') as f:
            f.write(input_template.format(ff_settings='\n'.join(self.ff_settings), mass=self.mass))
        shutil.copy(os.path.join(self.template_dir, 'submit-latt.sh'), 
                    os.path.join(self.calculation_dir, 'submit-latt.sh'))


    def calculate(self):
        """
        Calculate the lattice constant by running a LAMMPS simulation.
        """
        self._setup()
        subprocess.run('sbatch submit-latt.sh', shell=True, check=True, cwd=self.calculation_dir)
        time.sleep(10)
        a, b, c = np.loadtxt(os.path.join(self.calculation_dir, 'lattice.txt'))
        self.logger.info(f"Lattice constant: {a}, {b}, {c}")
        return a, b, c


class ElasticConstant(LMPStaticCalculator):
    """ 
    Elastic constant calculator.
    """
    def __init__(self, ff_settings, mass, lattice, alat,
                 deformation_size=1e-6, jiggle=1e-5):
        """
        Initialize the Elastic Constant calculator.
        Parameters
        ----------
        ff_settings : Potential or list
            Force field settings, either as a Potential class or a list of strings.
        lattice : str
            Lattice type (e.g., 'diamond').
        alat : float    
            Lattice constant in Angstroms.
        deformation_size : float, optional  
            Size of the deformation in Angstroms. Default is 1e-6.
        jiggle : float, optional    
            Jiggle size in Angstroms. Default is 1e-5.
        """
        super().__init__('tde', ff_settings, mass, alat, lattice=lattice)
        self.deformation_size = deformation_size
        self.jiggle = jiggle


    def _setup(self):
        template_dir = os.path.join(os.path.dirname(__file__), 'templates', 'elastic')


        with open(os.path.join(template_dir, 'init.mod'), 'r') as f:
            init_template = f.read()
        init_file = os.path.join(self.calculation_dir, 'init.mod')
        with open(init_file, 'w') as f:
            f.write(init_template.format(mass=self.mass, lattice=self.lattice, alat=self.alat,
                                         deformation_size=self.deformation_size, jiggle=self.jiggle))
        with open(os.path.join(template_dir, 'potential.mod'), 'r') as f:
            potential_template = f.read()
        potential_file = os.path.join(self.calculation_dir, 'potential.mod')
        with open(potential_file, 'w') as f:
            f.write(potential_template.format(ff_settings='\n'.join(self.ff_settings)))
        shutil.copy(os.path.join(template_dir, 'in.elastic'),
                    os.path.join(self.calculation_dir, 'in.elastic'))
        shutil.copy(os.path.join(template_dir, 'displace.mod'),
                    os.path.join(self.calculation_dir, 'displace.mod'))
        shutil.copy(os.path.join(template_dir, 'submit-elastic.sh'),
                    os.path.join(self.calculation_dir, 'submit-elastic.sh'))

        
    def calculate(self):
        self._setup()
        subprocess.run('sbatch submit-elastic.sh', shell=True, check=True, cwd=self.calculation_dir)
        time.sleep(30)
        self.logger.info('-------------------------------KRH Expression-------------------------------')
        self._KRH_expression()


    def _KRH_expression(self):
        """
        Calculate the KRH expression for the elastic constant.
        This file reads in the file log.lammps generated by the script in.elastic
        It prints out the 6x6 tensor of elastic constants Cij
        followed by the 6x6 tensor of compliance constants Sij
        It uses the same conventions as described in:
        Sprik, Impey and Klein PRB (1984).
        The units of Cij are whatever was used in log.lammps (usually GPa)
        The units of Sij are the inverse of that (usually 1/GPa)
        origin from lammps github

        Returns
        -------
        float
            KRH expression value.
        """
        nvals = 21
        valpos = 4
        valstr = '\nElastic Constant C'

        # define order of Cij in logfile
        cindices = [0]*nvals
        cindices[0] = (0,0)
        cindices[1] = (1,1)
        cindices[2] = (2,2)
        cindices[3] = (0,1)
        cindices[4] = (0,2)
        cindices[5] = (1,2)
        cindices[6] = (3,3)
        cindices[7] = (4,4)
        cindices[8] = (5,5)
        cindices[9] = (0,3)
        cindices[10] = (0,4)
        cindices[11] = (0,5)
        cindices[12] = (1,3)
        cindices[13] = (1,4)
        cindices[14] = (1,5)
        cindices[15] = (2,3)
        cindices[16] = (2,4)
        cindices[17] = (2,5)
        cindices[18] = (3,4)
        cindices[19] = (3,5)
        cindices[20] = (4,5)

        with open(os.path.join(self.calculation_dir, 'log.lammps'), 'r') as logfile:
            txt = logfile.read()

        # search for 21 elastic constants
        c = np.zeros((6,6))
        s2 = 0

        for ival in range(nvals):
            s1 = txt.find(valstr,s2)
            if (s1 == -1):
                self.logger.error("Failed to find elastic constants in log file")
                exit(1)
            s1 += 1
            s2 = txt.find("\n",s1)
            line = txt[s1:s2]
        # print line
            words = line.split()
            (i1,i2) = cindices[ival]
            c[i1,i2] = float(words[valpos])
            c[i2,i1] = c[i1,i2]
        self.logger.info("C tensor [GPa]")
        for i in range(6):
            row_str = " ".join(f"{c[i][j]:8.3f}" for j in range(6))
            self.logger.info(row_str)

        KV = 1/9*((c[0][0] + c[1][1] + c[2][2]) + \
            2 * (c[0][1] + c[1][2] + c[2][0]))
        GV = 1/15*((c[0][0] + c[1][1] + c[2][2]) - \
            (c[0][1] + c[1][2] + c[2][0]) + \
            3 * (c[3][3] + c[4][4] + c[5][5]))
        EV = 9*KV*GV/(3*KV+GV)

        # Ruoyan: factor parts are in original script, but the equations are conflict to Vaspkit
        # apply factor of 2 to columns of off-diagonal elements

        # for i in range(6):
        #     for j in range(3,6):
        #         c[i][j] *= 2.0

        s = np.linalg.inv(c)

        # apply factor of 1/2 to columns of off-diagonal elements

        # for i in range(6):
        #     for j in range(3,6):
        #         s[i][j] *= 0.5

        
        KR = 1 / ((s[0][0] + s[1][1] + s[2][2]) + 2 * (s[0][1] + s[1][2] + s[2][0]))
        # --------- or KR = KV

        GR = 15 / (4 * ((s[0][0] + s[1][1] + s[2][2]) - 4 * (s[0][1] + s[1][2] + s[2][0]) + 3 * (s[3][3] + s[4][4] + s[5][5])))
        GR = 5 / (4*(s[0][0]-s[0][1])+3*s[3][3]) # cubic
        # --------- or GR = 15/(12/(c[0][0]-c[0][1])+9/c[3][3])  Solid State Communications 324 (2021) 114136

        ER = 9*KR*GR/(3*KR+GR)

        KH = (KV+KR)/2
        GH = (GV+GR)/2
        EH = 9*KH*GH/(3*KH+GH)

        self.logger.info("")
        self.logger.info("KV       EV       GV")
        self.logger.info(f"{KV:8.3f} {EV:8.3f} {GV:8.3f}")
        self.logger.info("KR       ER       GR")
        self.logger.info(f"{KR:8.3f} {ER:8.3f} {GR:8.3f}")
        self.logger.info("KH       EH       GH")
        self.logger.info(f"{KH:8.3f} {EH:8.3f} {GH:8.3f}")


class VacancyDefectFormation(LMPStaticCalculator):
    """
    Defect Formation Energy Calculator.
    """
    def __init__(self, ff_settings, mass, lattice, alat, size, del_id):
        """
        Initialize the Defect Formation calculator.
        Parameters
        ----------
        ff_settings : Potential or list
            Force field settings, either as a Potential class or a list of strings.
        mass : float
            Mass of the atom in atomic mass units (AMU).
        lattice : str
            Lattice type (e.g., 'diamond').
        alat : float
            Lattice constant in Angstroms.
        size : int
            Size of the supercell.
        del_id : int
            ID of the defect to be created.
        """
        super().__init__('vacancy', ff_settings, mass, alat, size, lattice=lattice)
        self.del_id = del_id


    def _setup(self):
        with open(os.path.join(self.template_dir, 'in.vac'), 'r') as f:
            input_template = f.read()
        input_file = os.path.join(self.calculation_dir, 'in.vac')
        with open(input_file, 'w') as f:
            f.write(input_template.format(ff_settings='\n'.join(self.ff_settings), mass=self.mass, lattice=self.lattice, alat=self.alat,
                                         size=self.size, del_id=self.del_id))
        shutil.copy(os.path.join(self.template_dir, 'submit-vac.sh'),
                    os.path.join(self.calculation_dir, 'submit-vac.sh'))


    def calculate(self):
        """
        Calculate the defect formation energy by running a LAMMPS simulation.
        """
        self._setup()
        subprocess.run('sbatch submit-defect.sh', shell=True, check=True, cwd=self.calculation_dir)
        time.sleep(30)
        self.logger.info('-------------------------------Vacancy Formation Energy-------------------------------')
        self._vacancy_formation_energy()


    def _vacancy_formation_energy(self):
        with open(os.path.join(self.calculation_dir, 'log.lammps'), 'r') as logfile:
            txt = logfile.read()
        # find line contains "Vacancy formation energy"
        s1 = txt.find("Vacancy formation energy") 
        if s1 == -1:
            self.logger.error("Failed to find vacancy formation energy in log file")
            exit(1)
        
        next_line_start = txt.find('\n', s1) + 1  
        next_line_end = txt.find('\n', next_line_start)
        next_line = txt[next_line_start:next_line_end].strip()
        words = next_line.split('=')
        vacancy_energy = float(words[-1])
        self.logger.info(f"Vacancy formation energy: {vacancy_energy:.2f} eV")


class InterstitialDefectFormation(LMPStaticCalculator):
    """
    Interstitial Defect Formation Energy Calculator.
    """
    def __init__(self, ff_settings, mass, element, lattice, alat, size):
        """
        Initialize the Interstitial Defect Formation calculator.
        Parameters
        ----------
        ff_settings : Potential or list
            Force field settings, either as a Potential class or a list of strings.
        mass : float
            Mass of the atom in atomic mass units (AMU).
        lattice : str
            Lattice type (e.g., 'diamond').
        alat : float
            Lattice constant in Angstroms.
        size : int
            Size of the supercell.
        """
        super().__init__('interstitial', ff_settings, mass, alat, size=size, element=element, lattice=lattice)
        self.interstitials_types = ['split_110', 'hex', 'tet', 'bond']


    def _find_value(self, log_file, keyword):
        """
        Find the value associated with a keyword in the text.
        Parameters
        ----------
        log_file : str
            The text to search in.
        keyword : str
            The keyword to search for.
        Returns
        -------
        float
            The value associated with the keyword.
        """
        with open(log_file, 'r') as logfile:
            txt = logfile.read()
        s1 = txt.find(keyword)
        if s1 == -1:
            self.logger.error(f"Failed to find {keyword} in log file")
            exit(1)
        next_line_start = txt.find('\n', s1) + 1  
        next_line_end = txt.find('\n', next_line_start)
        next_line = txt[next_line_start:next_line_end].strip()
        words = next_line.split('=')
        return float(words[-1])
    

    def _setup_helper(self):
        """
        Helper function to set up the perfect structure for interstitial defect calculations.
        This function creates the perfect structure data file and input file for LAMMPS.
        Returns
        -------
        float
            The energy of the perfect system.
        """
        unit_cell = bulk(self.element, self.lattice, a=self.alat, cubic=True)
        super_cell = unit_cell * [self.size, self.size, self.size]
        write(os.path.join(self.calculation_dir, 'data.perfect'), super_cell, format='lammps-data')

        with open(os.path.join(self.template_dir, 'in.perfect'), 'r') as f:
            input_template = f.read()
        input_file = os.path.join(self.calculation_dir, 'in.perfect')
        with open(input_file, 'w') as f:
            f.write(input_template.format(ff_settings='\n'.join(self.ff_settings), mass=self.mass))
        shutil.copy(os.path.join(self.template_dir, 'submit-perfect.sh'),
                    os.path.join(self.calculation_dir, 'submit-perfect.sh'))
        subprocess.run('sbatch submit-perfect.sh', shell=True, check=True, cwd=self.calculation_dir)
        time.sleep(10)

        log_file = os.path.join(self.calculation_dir, 'log.lammps')
        perfect_energy = self._find_value(log_file, "Energy of perfect system")
        self.logger.info('-------------------------------Perfect system Energy-------------------------------')
        self.logger.info(f"Perfect system energy: {perfect_energy:.2f} eV")
        return perfect_energy


    def _setup(self):
        perfect_energy = self._setup_helper()
        with open(os.path.join(self.template_dir, 'in.interstitial'), 'r') as f:
            input_template = f.read()
        for inter_type in self.interstitials_types:
            inter_dir = os.path.join(self.calculation_dir, inter_type)
            os.makedirs(inter_dir, exist_ok=True)

            Ge_cubic = bulk(self.element, self.lattice, a=self.alat, cubic=True)
            Ge_sup = Ge_cubic * [self.size, self.size, self.size]
            if inter_type == 'split_110':
                del Ge_sup[1]
                alpha = (self.alat/2*np.sqrt(3)/2/self.alat)
                interstitial_position1 = np.array([(0.25-alpha)*self.alat, (0.25-alpha)*self.alat, 0.25*self.alat])
                interstitial_atom1 = Atoms(self.element, positions=[interstitial_position1])
                interstitial_position2 = np.array([(0.25+alpha)*self.alat, (0.25+alpha)*self.alat, 0.25*self.alat])
                interstitial_atom2 = Atoms(self.element, positions=[interstitial_position2])
                combined_structure = Ge_sup + interstitial_atom1 + interstitial_atom2
            elif inter_type == 'hex':
                interstitial_position = np.array([0.625*self.alat, 0.625*self.alat, 0.625*self.alat])
                interstitial_atom = Atoms(self.element, positions=[interstitial_position])
                combined_structure = Ge_sup + interstitial_atom          
            elif inter_type == 'tet':
                interstitial_position = np.array([0.5*self.alat, 0.5*self.alat, 0.5*self.alat])
                interstitial_atom = Atoms(self.element, positions=[interstitial_position])
                combined_structure = Ge_sup + interstitial_atom
            else:
                interstitial_position = np.array([0.125*self.alat, 0.125*self.alat, 0.125*self.alat]) 
                interstitial_atom = Atoms(self.element, positions=[interstitial_position])
                combined_structure = Ge_sup + interstitial_atom
            write(os.path.join(inter_dir, 'data.interstitial'), combined_structure, format='lammps-data')

            input_file = os.path.join(inter_dir, 'in.interstitial')
            with open(input_file, 'w') as f:
                f.write(input_template.format(ff_settings='\n'.join(self.ff_settings), mass=self.mass, perfect_energy=perfect_energy))
            shutil.copy(os.path.join(self.template_dir, 'submit-interstitial.sh'),
                        os.path.join(inter_dir, 'submit-interstitial.sh'))


    def calculate(self):
        """
        Calculate the interstitial defect formation energy by running a LAMMPS simulation.
        """
        self._setup()
        self.logger.info('-------------------------------Interstitial Formation Energy-------------------------------')
        self.logger.info(f'--------------------------------Supercell: [{self.size} {self.size} {self.size}]---------------------------------')
        for inter_type in self.interstitials_types:
            inter_dir = os.path.join(self.calculation_dir, inter_type)
            subprocess.run('sbatch submit-interstitial.sh', shell=True, check=True, cwd=inter_dir)
            time.sleep(30)
            log_file = os.path.join(inter_dir, 'log.lammps')
            inter_formation_energy = self._find_value(log_file, "Interstitial formation energy")
            self.logger.info(f"{inter_type}: {inter_formation_energy:.2f} eV")
        

class NudgedElasticBand(LMPStaticCalculator):
    """
    Nudged Elastic Band (NEB) migration barrier Calculator.
    """
    def __init__(self, ff_settings, mass, alat, size, element, lattice, num_images, path='1NN'):
        """
        Initialize the Nudged Elastic Band calculator.
        Parameters
        ----------
        ff_settings : Potential or list
            Force field settings, either as a Potential class or a list of strings.
        mass : float
            Mass of the atom in atomic mass units (AMU).
        alat : float
            Lattice constant in Angstroms.
        size : int
            Size of the supercell.
        lattice : str, optional
            Lattice type (default is 'diamond').
        element : str
            Element symbol (e.g., 'Ge').
        num_images : int
            Number of images for the NEB calculation.
        path : str 
            '1NN' or '2NN', indicating the path for the NEB calculation.
        """
        super().__init__('neb', ff_settings, mass, alat, size=size, element=element, lattice=lattice)
        self.num_images = num_images
        self.path = path
    

    def _setup(self):
        with open(os.path.join(self.template_dir, 'in.relax'), 'r') as f:
            relax_template = f.read()
        with open(os.path.join(self.template_dir, 'in.neb'), 'r') as f:
            neb_template = f.read()
        lattice_calculator = LatticeConstant(self.ff_settings, self.mass, self.element, self.lattice, self.alat, cubic=True)
        a, _, _ = lattice_calculator.calculate()
        unit_cell = bulk(self.element, self.lattice, a=a, cubic=True)
        scale_factor = [self.size, self.size, self.size]
        super_cell = unit_cell * scale_factor
        write(os.path.join(self.calculation_dir, 'data.supercell'), super_cell, format='lammps-data')

        start_idx = len(super_cell) - 1
        if self.path == '1NN':
            final_idx = len(super_cell) - 2
        elif self.path == '2NN':
            final_idx = len(super_cell) - 3
       
        relax_file = os.path.join(self.calculation_dir, 'in.relax')
        shutil.copy(os.path.join(self.template_dir, 'submit-relax.sh'),
                    os.path.join(self.calculation_dir, 'submit-relax.sh'))
        with open(relax_file, 'w') as f:
            f.write(relax_template.format(ff_settings='\n'.join(self.ff_settings),
                                          lattice=self.lattice, alat=a, element=self.element, mass=self.mass,
                                          del_id=start_idx + 1, relaxed_file='initial.relaxed'))
        subprocess.run('sbatch submit-relax.sh', shell=True, check=True, cwd=self.calculation_dir)
        time.sleep(30)
        with open(relax_file, 'w') as f:
            f.write(relax_template.format(ff_settings='\n'.join(self.ff_settings),
                                          lattice=self.lattice, alat=a, element=self.element, mass=self.mass,
                                          del_id=final_idx + 1, relaxed_file='final.relaxed'))
        subprocess.run('sbatch submit-relax.sh', shell=True, check=True, cwd=self.calculation_dir)
        time.sleep(30)

        def process_lammps_atoms(input_file, output_file, start_idx, final_idx):
            with open(input_file, 'r') as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                if 'Velocities' in line:
                    end = i
                    break
            lines = lines[15:end]   # file from 16th line 

            processed_lines = [f'{len(lines)-1}\n']

            for line in lines:
                tokens = line.strip().split()
                if len(tokens) >= 5:
                    new_line = f"{tokens[0]} {tokens[2]} {tokens[3]} {tokens[4]}\n"
                    if int(tokens[0]) == start_idx + 1:
                        new_line = f"{final_idx + 1} {tokens[2]} {tokens[3]} {tokens[4]}\n"
                    processed_lines.append(new_line)

            with open(output_file, 'w') as f:
                f.writelines(processed_lines)

        input_file = os.path.join(self.calculation_dir, 'final.relaxed')
        output_file = os.path.join(self.calculation_dir, 'data.final_replica')
        process_lammps_atoms(input_file, output_file, start_idx, final_idx)

        input_file = os.path.join(self.calculation_dir, 'in.neb')
        with open(input_file, 'w') as f:
            f.write(neb_template.format(ff_settings='\n'.join(self.ff_settings), mass=self.mass))
        with open(os.path.join(self.template_dir, 'submit-neb.sh'), 'r') as f:
            submit_template = f.read()
        submit_file = os.path.join(self.calculation_dir, 'submit-neb.sh')
        with open(submit_file, 'w') as f:
            ntasks = self.num_images * 10
            f.write(submit_template.format(ntasks=ntasks, size=self.num_images))


    def calculate(self):
        """
        Calculate the NEB barrier given Potential class.
        """
        self._setup()
        subprocess.run('sbatch submit-neb.sh', shell=True, check=True, cwd=self.calculation_dir)
        time.sleep(180)
        self.logger.info('-------------------------------NEB Barrier-------------------------------')
        migration_barrier = self._parse()
        self.logger.info(f"Migration barrier: {migration_barrier:.2f} eV")
        return migration_barrier


    def _parse(self):
        """
        Parse results from dump files.

        """
        with open(os.path.join(self.calculation_dir, 'log.lammps')) as f:
            lines = f.readlines()[-1:]
        migration_barrier = float(lines[0].split()[6]) # read EBF
        return migration_barrier


class Liquid:
    # Use TurboGAP MD, files in self.templates/liquid
    pass

class Amorphous:
    # Use TurboGAP MD, files in self.templates/amorphous
    pass




### EV 



        

        
   










