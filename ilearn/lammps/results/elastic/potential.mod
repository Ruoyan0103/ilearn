# NOTE: This script can be modified for different pair styles 
# See in.elastic for more info.

# Choose potential
pair_style        sw
pair_coeff        * * /scratch/phys/t30429_nume-dft-ml/04-Ruoyan/00-MLP/04-short_range/ilearn/ilearn/potentials/params/SW/Ge.sw Ge

# Setup neighbor style
neighbor 1.0 nsq
neigh_modify once no every 1 delay 0 check yes

# Setup minimization style
min_style	     cg
min_modify	     dmax ${dmax} line quadratic

# Setup output
thermo		1
thermo_style custom step temp pe press pxx pyy pzz pxy pxz pyz lx ly lz vol fmax fnorm
thermo_modify norm no
