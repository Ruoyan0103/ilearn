import matplotlib.pyplot as plt

# Define the file path
file_path = 'exp'  # Replace with your actual file path

# Initialize lists to store data
x_values = []
y_values = []

# Read data from the file
with open(file_path, 'r') as file:
    for line in file:
        if line.strip():  # Check if line is not empty
            x, y = line.strip().split(',')
            x_values.append(float(x.strip()))
            y_values.append(float(y.strip()))

# Plotting
plt.figure(figsize=(8, 6))  # Adjust figure size as needed
plt.scatter(x_values, y_values, color='b', label='Data')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Plot of X vs Y')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('exp.png', dpi=300)
