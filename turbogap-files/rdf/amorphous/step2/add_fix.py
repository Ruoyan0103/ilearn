from ase.io import read, write
import numpy as np
import sys

input_database = "trajectory_out.xyz"
output_database = "Ge_melted.xyz"

print("Reading database...")
at = read(input_database, index=":")
print("")
print("... done.")


print("")
print("")

print("Writing to file...")
write(output_database, at[-1])
print("")
print("... done.")
