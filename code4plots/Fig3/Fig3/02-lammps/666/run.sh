# python3 defect.py
# phonopy -c Ge-lammps --dim 6 6 6
# run lammps 

phonopy -f force.0
phonopy -c Ge-lammps band.conf
phonopy-bandplot --gnuplot band.yaml > raw-data.txt
phonopy -c Ge-lammps  mesh.conf -p -t
phonopy --dos mesh.conf
