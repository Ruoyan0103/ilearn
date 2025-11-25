import matplotlib.pyplot as plt
import numpy as np
import argparse
import matplotlib.cm as cm
plt.rcParams['font.family'] = 'DejaVu Serif'
'''
    Usage: plot energy vs volume 
    Parameters: list of energy files, e.g. ['DFT', 'GAP', 'MEAM'], outfile, e.g. all.png
    Return: 
'''
def plotAll(files, outfile):
    color_map = plt.get_cmap('plasma', len(files))
    colors = ['black', 'blueviolet', 'crimson',
              'hotpink', 'royalblue', 'saddlebrown',
              'darkorange', 'forestgreen']
    colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf',
          '#a65628', '#984ea3', '#999999', '#e41a1c']
    labels = ['dia','dia', 'hd','hd', 'bc8','bc8', 'st12','st12', r'$\beta$-Sn', r'$\beta$-Sn', 'hcp', 'hcp', 'fcc', 'fcc', 'bcc', 'bcc' ]
    i = 0
    plt.clf()
    x = []                                              # volume
    y = []                                              # energy
    for i, f in enumerate(files):
        c = color_map(i / len(files))
        with open (f, 'r') as file:
            for line in file:
                words = line.split(',')
                x.append(float(words[1]))
                #if 'GAP' in f or 'DFT' in f:            # for GAP and DFT, energy/atom-Eiso
                    #y.append(float(words[2])-(-0.78630447))
                #else:
                y.append(float(words[2]))
        
        names = f.split('/')
        #c = colors[int(i/2)]
        if 'DFT' in f:
            plt.plot(x, y, 'D', markersize=4, color=c) 
        if 'GAP' in f:
            plt.plot(x, y,  markersize=0.8, color=c, label=labels[i])

        #else:
            #plt.plot(x, y, '--', label=names[0], markersize=3)
        x = []
        y = [] 
        i += 1
    #labelfont = {'fontname':'Times New Roman', 'fontsize':14}
    #tickfont = {'fontname':'Times New Roman', 'fontsize':12}
    #plt.rcParams["font.family"] = "Times New Roman" 
    '''
    plt.xlim(left=15.5, right=32.5)
    plt.ylim(top=-3.95, bottom=-4.525)
    plt.xlabel('Volume (Å³/atom)', fontsize=12)
    plt.ylabel('Energy (eV/atom)', fontsize=12)                     
    plt.xticks()
    plt.yticks()
    plt.tight_layout()
    plt.grid()
    plt.legend(fontsize=10)
    plt.savefig(outfile, dpi=300) 
    '''

    plt.legend(fontsize=10, loc='lower left', frameon=True, facecolor='white', framealpha=0.8)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.xlabel('Volume (Å³/atom)', fontsize=14, labelpad=10)
    plt.ylabel('Energy (eV/atom)', fontsize=14, labelpad=10)
    plt.xlim(15, 32)
    plt.ylim(-4.55, -3.95)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--outfile", dest = "outfile")
    args = parser.parse_args()
    files = ['01-dia/DFT', '01-dia/GAP', '06-hd/DFT', '06-hd/GAP', 
             '04-bc8/DFT', '04-bc8/GAP','07-st12/DFT', '07-st12/GAP', 
             '08-beta/DFT', '08-beta/GAP', '02-fcc/DFT', '02-fcc/GAP', 
             '05-hcp/DFT', '05-hcp/GAP', '03-bcc/DFT', '03-bcc//GAP']
    #files = ['st12/DFT', 'st12/GAP', 'st12/MEAM', 'st12/Tersoff']
    #files = ['../08-beta/DFT', '../Final/GAP']
    plotAll(files, str(args.outfile))


    

    


    


