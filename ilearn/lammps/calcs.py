import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from itertools import product
import os 
from ilearn.potentials import Potential

plt.rcParams['font.family'] = 'DejaVu Serif'

class ThresholdDisplacementEnergy:
    def __init__(self, ff_settings):
        self.ff_settings = ff_settings
        self.angle_set = set()
        self.hkl_list = []
        
    def get_random_angles(min_phi, max_phi, min_theta, max_theta, num_points):
        min_phi = np.radians(min_phi)
        max_phi = np.radians(max_phi)
        min_theta = np.radians(min_theta)
        max_theta = np.radians(max_theta)
        phi = np.random.uniform(min_phi, max_phi, num_points)                          # azimuthal angle (φ)
        costheta = np.random.uniform(np.cos(min_theta), np.cos(max_theta), num_points) 
        theta = np.arccos(costheta)                                                    # polar angle (θ)
        self.angle_set = set(zip(phi, theta))                                          # Store unique angles


    def get_uniform_angles(self, vectors, degree):
        def append(vector):
            if np.linalg.norm(vector) != 0.0:
                vector = vector / np.linalg.norm(vector)
            phi = np.arctan2(vector[1], vector[0])     # azimuthal angle (φ)
            theta = np.arccos(vector[2])               # polar angle (θ)
            self.angle_set.add((phi, theta))           # Store unique angles

        def mid(vector1, vector2):
            mid1 = np.array(([(vector1[0] + vector2[0])/2, (vector1[1] + vector2[1])/2, (vector1[2] + vector2[2])/2]))
            append(mid1)
            return mid1

        def sierpenski(vectors, degree):
            if degree > 0:
                sierpenski([vectors[0],mid(vectors[0],vectors[1]),mid(vectors[0],vectors[2])], degree - 1)
                sierpenski([vectors[1],mid(vectors[0],vectors[1]),mid(vectors[1],vectors[2])], degree - 1)
                sierpenski([vectors[2],mid(vectors[2],vectors[1]),mid(vectors[0],vectors[2])], degree - 1)
                sierpenski([mid(vectors[0],vectors[1]), mid(vectors[0],vectors[2]), mid(vectors[1],vectors[2])], degree - 1)
            return

        append(vectors[0])
        append(vectors[1])
        append(vectors[2])
        sierpenski(vectors, degree)
    

    def get_hkl_from_angles(self):
        added_theta = np.linspace(0, np.deg2rad(5), 3)
        added_phi = np.linspace(0, np.pi / 4, 3)
        self.angle_set.update(product(added_phi, added_theta))
        
        for angle in self.angle_set:
            phi, theta = angle
            h = np.sin(theta) * np.cos(phi)
            k = np.sin(theta) * np.sin(phi)
            l = np.cos(theta)
            self.hkl_list.append(np.array((h, k, l)))

    def plot(self):
        print(f"Number of unique angles: {len(self.angle_set)}")
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

        plt.savefig('save4.png', dpi=300)

    def _setup(self):
        template_dir = os.path.join(os.path.dirname(__file__), 'templates', 'tde')
        with open(os.path.join(template_dir, 'in.tde'), 'r') as f:
            input_template = f.read()
        if isinstance(self.ff_settings, Potential):
            ff_settings = self.ff_settings.write_param()
        else:
            ff_settings = self.ff_settings
        input_file = 'in.tde'
        with open(input_file, 'w') as f:
            f.write(input_template.format(ff_settings='\n'.join(ff_settings)))
        return input_file



#tde = ThresholdDisplacementEnergy()
# vector1 = [0., 0., 1.]
# vector2 = [1., 0., 1.]
# vector3 = [1., 1., 1.]
# vectors = np.array((vector1, vector2, vector3))
# tde.get_uniform_angles(vectors, 4)
# tde.get_hkl_from_angles()
# tde.plot()




