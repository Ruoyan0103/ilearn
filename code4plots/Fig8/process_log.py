import numpy as np
import matplotlib.pyplot as plt

def process_file(filename: str, target: str):
    time, temp, poteng = [], [], []
    blocklength = 10

    with open(filename, "r") as file:
        lines = file.readlines()

    idx = 0
    filelength = len(lines)
    while idx < filelength:
        if target in lines[idx]:
            data_lines = [lines[idx + i + 2].strip() for i in range(blocklength)]
            
            time_block = [float(line.split()[2])*1000 for line in data_lines]
            temp_block = [float(line.split()[3]) for line in data_lines]
            poteng_block = [float(line.split()[5])/1000 for line in data_lines]

            time.extend(time_block)
            temp.extend(temp_block)
            poteng.extend(poteng_block)
            
            idx += blocklength + 1  
        else:
            idx += 1  
    # potential increase
    basepot = poteng[0]
    poteng = [pe - basepot for pe in poteng]
    return np.array(time), np.array(temp), np.array(poteng)

def plot(time, temp, poteng, ax1, ax2, color, label_suffix=''):
    plt.rcParams['font.family'] = 'DejaVu Serif'
    font_props = {'fontname': 'DejaVu Serif', 'fontsize': 14}

    # Top plot
    ax1.plot(time, temp, color=color, label=f'{label_suffix}')
    ax1.set_ylabel('Temperature (K)', **font_props)
    ax1.legend(fontsize=12)
    ax1.set_xscale('log')
    ax1.set_xlim(1, max(time))
    ax1.set_ylim(min(temp)*0.95, max(temp)*1.1)
    # no tick labels for x axis
    ax1.set_xticklabels([])
    # set font for x tick labels
    for tick in ax1.get_xticklabels():
        tick.set_fontname('DejaVu Serif')
        tick.set_fontsize(12)
    # set font for y tick labels
    for tick in ax1.get_yticklabels():
        tick.set_fontname('DejaVu Serif')
        tick.set_fontsize(12)
    # text
    ax1.text(0.05, 0.87, '(a)', transform=ax1.transAxes, fontsize=16, fontname='DejaVu Serif')

    # Bottom plot
    ax2.plot(time, poteng, color=color, label=f'{label_suffix}')
    ax2.set_xlabel('Time (fs)', **font_props)
    ax2.set_ylabel('Potential Energy (keV)', **font_props)
    ax2.set_xscale('log')
    ax2.set_xlim(min(time), max(time))
    ax2.set_ylim(min(poteng), max(poteng)*1.1)
    # set font for x tick labels
    for tick in ax2.get_xticklabels():
        tick.set_fontname('DejaVu Serif')
        tick.set_fontsize(12)
    # set font for y tick labels
    for tick in ax2.get_yticklabels():
        tick.set_fontname('DejaVu Serif')
        tick.set_fontsize(12)
    ax2.text(0.05, 0.87, '(b)', transform=ax2.transAxes, fontsize=16, fontname='DejaVu Serif')

#########################################################################
target = "   Step           Dt            Time           Temp          Press          PotEng         KinEng    "
filename1 = "GAP/2000/1/log.lammps"
filename2 = "MEAM/2000/1/log.lammps"

# Process both files
time1, temp1, poteng1 = process_file(filename1, target)
time2, temp2, poteng2 = process_file(filename2, target)

# Print max potential energy info
maxpot1 = max(poteng1)
maxpot_index1 = np.where(poteng1 == maxpot1)[0][0]
maxtime1 = time1[maxpot_index1]
print(f"{filename1}, Max potential energy:", maxpot1, "keV at time:", maxtime1, "fs")

maxpot2 = max(poteng2)
maxpot_index2 = np.where(poteng2 == maxpot2)[0][0]
maxtime2 = time2[maxpot_index2]
print(f"{filename2}, Max potential energy:", maxpot2, "keV at time:", maxtime2, "fs")

# Plot both files on the same axes
# ax1 and ax2 share x axis
fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

plot(time1, temp1, poteng1, ax[0], ax[1], 'red', label_suffix='GAP')
plot(time2, temp2, poteng2, ax[0], ax[1], 'skyblue', label_suffix='MEAM')

plt.tight_layout()
plt.savefig("temperature_potential_energy.png", dpi=300)
