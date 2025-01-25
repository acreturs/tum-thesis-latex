import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

# Load the CSV file, specifying that the delimiter is a semicolon
data = pd.read_csv('Book2.csv', encoding='utf-8', delimiter=';', header=0)

# Print the first few rows to see the data structure
print("First few rows of data:\n", data.head())

# Extract the necessary columns
x = data['Index']
y1 = data['Option3']  # First dataset for plotting
y2 = data['Option6']  # Second dataset for plotting

# Create the figure and axis
fig, ax1 = plt.subplots(figsize=(10, 5))

# Plot the first dataset on the primary y-axis
color1 = 'tab:blue'
ax1.set_xlabel("Time")
ax1.set_ylabel("Cumulative Reward", color=color1)
ax1.plot(x, y1, label="Option3", color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.legend(loc='upper left')

# Create a twin axis for the second dataset
ax2 = ax1.twinx()
color2 = 'tab:orange'
ax2.set_ylabel("Cumulative Regret", color=color2)
ax2.plot(x, y2, label="Option6", color=color2)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.legend(loc='upper right')

# Add rounded corners to the plot background
def add_rounded_background(ax, radius=15):
    box = ax.get_position()
    rounded_box = FancyBboxPatch(
        (box.x0, box.y0), box.width, box.height,
        boxstyle=f"round,pad=0.1,rounding_size={radius}",
        edgecolor='none', facecolor='white', zorder=-1
    )
    ax.figure.patches.append(rounded_box)

add_rounded_background(ax1)
add_rounded_background(ax2)

# Adjust layout and show the plot
fig.tight_layout()
plt.title("Comparison of Leader Reward and Stackelberg Regret", y=1.05)
plt.show()
