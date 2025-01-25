import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.interpolate import make_interp_spline  # For smoothing lines

# Load the CSV file, specifying that the delimiter is a semicolon
data = pd.read_csv('Book2.csv', encoding='utf-8', delimiter=';', header=0)

# Print the first few rows to see the data structure
print("First few rows of data:\n", data.head())

# Extract the necessary columns
x = data['Index']
y1 = data['Option3']  # First dataset for plotting
y2 = data['Option6']  # Second dataset for plotting

# Smooth the lines using cubic spline interpolation
x_smooth = np.linspace(x.min(), x.max(), 300)  # Create a smooth x-axis
y1_smooth = make_interp_spline(x, y1)(x_smooth)
y2_smooth = make_interp_spline(x, y2)(x_smooth)

# Create figure and axes
fig, ax = plt.subplots(figsize=(10, 8), dpi=120)

# Add a rounded box around the plot
box = FancyBboxPatch(
    (0, 0), 1, 1,
    boxstyle="round,pad=0.1,rounding_size=0.2",
    transform=ax.transAxes,
    facecolor="white",
    edgecolor="black",
    linewidth=1.5
)
ax.add_patch(box)

# Plot smoothed data for y1 and y2
ax.plot(x_smooth, y1_smooth, label='Exp4', color='blue', linewidth=2.5)
ax.plot(x_smooth, y2_smooth, label='SSE', color='red', linewidth=2.5)

# Adjust Y-axis and X-axis limits to start at 0.0
ax.set_xlim(left=0.0)  # Ensure X-axis starts at 0.0
ax.set_ylim(bottom=0.0)  # Ensure Y-axis starts at 0.0 if needed

# Add legend
ax.legend(loc='upper right', fontsize=14)

# Labels and title
ax.set_title('Accumulated Regret Plot', fontsize=18, weight='bold')
ax.set_xlabel('nth-Question', fontsize=14, weight='bold')
ax.set_ylabel('Accumulated Regret', fontsize=14, weight='bold')

# Customization for a scientific look
ax.grid(visible=True, linestyle='--', linewidth=0.7, alpha=0.7)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Set rounded corners and thicker spines
for spine in ax.spines.values():
    spine.set_linewidth(1.8)

# Tight layout for spacing
plt.tight_layout()

# Show plot
plt.show()
