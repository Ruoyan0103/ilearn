import numpy as np
import matplotlib.pyplot as plt
import glob

# Get list of RDF files
rdf_files = sorted(glob.glob("rdf_ave.*"))  # Reads rdf_ave.0, rdf_ave.1, ...

# Initialize storage for averaging
rdf_data = None
num_files = len(rdf_files)

# Read and accumulate RDF data
for file in rdf_files:
    data = np.loadtxt(file, comments="#")  # Ignore header lines with '#'
    r_values = data[:, 0]  # Pair separation distance
    g_values = data[:, 1]  # g(r) values

    if rdf_data is None:
        rdf_data = np.zeros_like(g_values)  # Initialize array

    rdf_data += g_values  # Accumulate RDF values

# Compute the average RDF
rdf_data /= num_files 
print(num_files) 

# Save the averaged RDF to a file
output_file = "liquid.dat"
with open(output_file, "w") as f:
    f.write("# Averaged Radial Distribution Function\n")
    f.write("# r (Å)    g(r)\n")
    for r, g in zip(r_values, rdf_data):
        f.write(f"{r:.6f} {g:.6f}\n")

print(f"Averaged RDF saved to {output_file}")

# Plot the averaged RDF
plt.plot(r_values, rdf_data, label="GAP")
#plt.tight_layout()
plt.xlabel("r (Å)")
plt.ylabel("g(r)")
plt.legend()
plt.savefig('ave_rdf.png', dpi=300)

