import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from itertools import product
import os, math
#from ilearn.potentials import Potential

plt.rcParams['font.family'] = 'DejaVu Serif'
module_dir = os.path.dirname(os.path.abspath(__file__))

class ThresholdDisplacementEnergy:
    def __init__(self, ff_settings, cell_file, alat, pka_id,
                 temp, output_file, element, mass, 
                 min_velocity, max_velocity, velocity_interval,
                 kin_eng_threshold=0.01):
        '''
        Initialize the ThresholdDisplacementEnergy class.
        Parameters
        ----------
        ff_settings : Potential or list
            Force field settings, either as a Potential object or a list of strings.
        cell_file : str 
            Path to the cell file containing lattice parameters.
        alat : float    
            Lattice constant (in Angstroms).
        pka_id : int    
            Primary knock-on atom ID.
        temp : float    
            Temperature for the simulation (in Kelvin).
        output_file : str   
            Name of the output file for the simulation results.
        element : str
            Element symbol for the primary knock-on atom.
        mass : float
            Mass of the primary knock-on atom (in atomic mass units).
        min_velocity : float
            Minimum velocity for the primary knock-on atom (in m/s).
        max_velocity : float
            Maximum velocity for the primary knock-on atom (in m/s).
        velocity_interval : float
            Interval for velocity sampling (in m/s).
        kin_eng_threshold : float   
            Threshold for kinetic energy difference (in eV).
        '''
        self.ff_settings = ff_settings
        self.angle_set = set()
        self.hkl_list = []
        self.cell_file = cell_file
        self.alat = alat
        self.pka_id = pka_id
        self.temp = temp
        self.output_file = output_file
        self.element = element
        self.mass = mass
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.velocity_interval = velocity_interval
        self.kin_eng_threshold = kin_eng_threshold  # Threshold for kinetic energy difference
        
    def get_random_angles(min_phi, max_phi, min_theta, max_theta, num_points):
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
                vector = vector / np.linalg.norm(vector) # Normalize the vector to calculate angles
            phi = math.atan2(vector[1],vector[0])        # azimuthal angle (φ)
            theta = math.acos(vector[2])                 # polar angle (θ)
            self.angle_set.add((phi, theta))             # Store unique angles

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
    

    def get_hkl_from_angles(self):
        added_theta = np.linspace(0, np.deg2rad(5), 2)           # Add 0 and 5 degrees polar angles for interpolation
        added_phi = np.linspace(0, np.pi / 4, 2)                 
        self.angle_set.update(product(added_phi, added_theta))
        
        for angle in self.angle_set:
            phi, theta = angle
            h = np.sin(theta) * np.cos(phi)
            k = np.sin(theta) * np.sin(phi)
            l = np.cos(theta)
            self.hkl_list.append(np.array((h, k, l)))

    def plot(self):
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        azimuthal = [angle[0] for angle in self.angle_set]
        polar = [angle[1] for angle in self.angle_set]
        energy = np.random.uniform(0, 1, len(azimuthal))  # Random energy values for demonstration

        phi_grid = np.linspace(0, np.pi/4, 1000)
        theta_grid = np.linspace(0, np.deg2rad(54.7), 1000)
        phi_grid2, theta_grid2 = np.meshgrid(phi_grid, theta_grid)
        
        # Mask invalid points in the interpolation grid
        # h*tan(theta) = k/cos(phi)
        # max_polar = np.arctan((k/h) / cos(phi))
        # k <= h, so the max polar is when k = h
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
        contour = ax.contour(phi_grid2, theta_grid2, energy_grid, levels=10, colors='black', linewidths=0.18)
        scatter = ax.scatter(azimuthal, polar, c='#D3D3D3', s=0.8, edgecolors='#D3D3D3')
        
        # Text 1: Along the arc ("Azimuthal")
        angle = np.pi / 8                                # Position along the arc (midway between 0 and pi/4)
        radius = np.deg2rad(54.7)                        # Maximum polar angle
        text_radius = radius * 1.12                       # Slightly farther from the arc
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
        tick_step = 1
        ticks = np.arange(np.floor(vmin), np.ceil(vmax) + tick_step, tick_step)
        char = fig.colorbar(contourf, ax=ax, pad=0.15, shrink=0.85, ticks=ticks)
        char.set_label('Threshold displacement energy (eV)', fontsize=9)        
        char.ax.tick_params(
            labelsize=8,          # Smaller font size for numbers
            length=2,            # Shorter tick marks
            width=0.5            # Thinner tick marks
        )
        plot_file = os.path.join(module_dir, 'results', 'tde.png')
        plt.savefig(plot_file, dpi=300)

    def _setup_helper(self, hkl, veloctiy):
        template_dir = os.path.join(os.path.dirname(__file__), 'templates', 'tde')
        with open(os.path.join(template_dir, 'in.tde'), 'r') as f:
            input_template = f.read()
        if isinstance(self.ff_settings, Potential):
            ff_settings = self.ff_settings.write_param()
        else:
            ff_settings = self.ff_settings
        hkl_str = '-'.join(f'{x:.3f}'.replace('.', '_') for x in hkl)
        calculation_dir = os.path.join(module_dir, 'results', 'calculation', hkl_str, str(velocity))   
        os.makedirs(calculation_dir, exist_ok=True)
        input_file = os.path.join(calculation_dir, 'in.tde')
        with open(input_file, 'w') as f:
            Vx = velocity * hkl[0]
            Vy = velocity * hkl[1]
            Vz = velocity * hkl[2]
            f.write(input_template.format(ff_settings='\n'.join(ff_settings),
                                          alat=self.alat, pka_id=self.pka_id,
                                          temp=self.temp, output_file=self.output_file,
                                          element=self.element, mass=self.mass,
                                          Vx=Vx, Vy=Vy, Vz=Vz))    
        return input_file

    def _setup(self):
        '''
        Setup the input file for the LAMMPS simulation.
        This method prepares the input file with the necessary parameters.
        '''
        if not self.hkl_list:
            raise ValueError("HKL list is empty. Please generate HKL values first.")
        for hkl in self.hkl_list:
            hkl = np.array(hkl)
            v = self.min_velocity
            v_interval = self.velocity_interval

            prev_kin_eng = 0.5 * self.mass * np.sum(hkl**2) * v**2

            while v <= self.max_velocity:
                next_v = v + v_interval
                next_kin_eng = 0.5 * self.mass * np.sum(hkl**2) * next_v**2
                energy_diff = next_kin_eng - prev_kin_eng

                # If energy difference too large, reduce step and try again
                while energy_diff > self.kin_eng_threshold:
                    v_interval /= 2
                    if v_interval < 1:  # safeguard minimum step
                        raise ValueError("Too small velocity interval. Cannot find suitable velocity.")
                    next_v = v + v_interval
                    next_kin_eng = 0.5 * self.mass * np.sum(hkl**2) * next_v**2
                    energy_diff = next_kin_eng - prev_kin_eng

                # Once acceptable, setup and move forward
                self._setup_helper(hkl, next_v)

                v = next_v
                prev_kin_eng = next_kin_eng


        #def calculate(self):


    


# example usage
tde = ThresholdDisplacementEnergy()
vector1 = [0., 0., 1.] / np.linalg.norm([0., 0., 1.])  # Normalize the vector
vector2 = [1., 0., 1.] / np.linalg.norm([1., 0., 1.])  # Normalize the vector
vector3 = [1., 1., 1.] / np.linalg.norm([1., 1., 1.])  # Normalize the vector
vectors = np.array((vector1, vector2, vector3))
tde.get_uniform_angles(vectors, 4)
tde.get_hkl_from_angles()
tde.plot()







