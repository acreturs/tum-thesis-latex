import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from scipy.interpolate import make_interp_spline 
path = "C:/Users/Alexander Cornett/Desktop/Bachelorthesis writing/tum-thesis-latex/graphs/TeamAnswers/"

path_regret = "C:/Users/Alexander Cornett/Desktop/Bachelorthesis writing/tum-thesis-latex/graphs/regretPlots/"

"""
Calculates percentages for each category in the team_answers folder
"""

def calc_percntages(number_category_questions, n, category):
    averages = {}
    files = os.listdir(path)
    for folders in files:
        print(path)
        print(folders)
        data = pd.read_csv(path + str(folders) +f"/{category}.csv", encoding='utf-8', delimiter=';')
        last_row = data.iloc[-1]
        average = 0
        for index, value in last_row.items():
            print(f"Column: {index}, Value: {value}")
            average += (number_category_questions- value) / n
            print(average)
        averages[folders] = (average / 3) * 100  # Convert to percentage

    return averages, category, n

def plot_boxplot():
    files = [["easy",10,10], ["medium",10,10], ["hard",30,30], ["alle",50,50]]
    for i in files:

        averages, category, n = calc_percntages(i[1], i[0])
        # Plotting the boxplot
        plt.figure(figsize=(10, 6))
        plt.bar(averages.keys(), averages.values(), color='skyblue')
        plt.ylim(0, 100)  # Set y-axis limits from 0 to 100
        plt.xlabel('Folders')
        plt.ylabel('Averages (%)')
        plt.title(f'Averages for {category} (n={n})')
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()  # Adjust layout to prevent label cutoff
        plt.show()

def different_colors():
    files = [["easy",10,50], ["medium",10,50], ["hard",30,50]]
    results_calc = {}
    for i in files:
        averages, category, n = calc_percntages(i[1],i[2] ,i[0])
        results_calc[category] = averages
    df = pd.DataFrame(results_calc)
    categories = ["easy", "medium", "hard"]
    colors = ["#66c2a5", "#fc8d62", "#8da0cb"]  # Colors for each category
    df.reset_index(inplace=True)
    df.rename(columns={"index": "model"}, inplace=True)
    print(df)
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.ylim(0, 100)
    bottom = [0] * len(df)  # Start with a baseline at 0 for stacking

    # Iterate over categories and plot each as a segment of the stack
    for category, color in zip(categories, colors):
        plt.bar(df["model"], df[category], bottom=bottom, label=category, color=color)
        bottom = [i + j for i, j in zip(bottom, df[category])]  # Update the baseline for stacking

    # Add labels, title, and legend
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Percentage", fontsize=12)
    plt.title("Questions answered correctly Percentages by Category", fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(title="Category")

    # Display the plot
    plt.tight_layout()
    plt.savefig(f"allModelsAlle.pdf", format="pdf", bbox_inches="tight")
    #plt.show()

"""
Funktion ist noch quatsch
"""

def different_colors_text():
    files = [["easy", 10, 50], ["medium", 10, 50], ["hard", 30, 50]]
    results_calc = {}

    # Calculate averages for each category and store them
    for i in files:
        averages, category, n = calc_percntages(i[1], i[2], i[0])
        results_calc[category] = averages

    # Create a DataFrame from the results
    df = pd.DataFrame(results_calc)
    categories = ["easy", "medium", "hard"]
    colors = ["#66c2a5", "#fc8d62", "#8da0cb"]  # Colors for each category

    # Reset index and rename for better plotting
    df.reset_index(inplace=True)
    df.rename(columns={"index": "model"}, inplace=True)

    # Create the stacked bar plot
    plt.figure(figsize=(12, 6))
    bottom = [0] * len(df)  # Start with a baseline at 0 for stacking

    # Iterate over categories and plot each as a segment of the stack
    for category, color in zip(categories, colors):
        plt.bar(df["model"], df[category], bottom=bottom, label=category, color=color)

        # Add percentage labels on top of each segment
        for idx, value in enumerate(df[category]):
            plt.text(
                x=idx, 
                y=bottom[idx] + value / 2,  # Position: half the height of the bar segment
                s=f"{value:.1f}%",  # Format as percentage
                ha="center", 
                va="center", 
                fontsize=10, 
                color="black",  # Text color
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="none", facecolor="white", alpha=0.7)  # Background box
            )

    # Update the baseline for the next category
    bottom = [i + j for i, j in zip(bottom, df[category])]

# Ensure y-axis ranges from 0 to 100
    plt.ylim(0, 100)

    # Add labels, title, and legend
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Percentage", fontsize=12)
    plt.title("Stacked Bar Plot of Percentages by Category", fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(title="Category")

    # Save the plot as a PDF
    plt.savefig("stacked_bar_plot_with_percentages.pdf", format="pdf", bbox_inches="tight")

    # Display the plot
    plt.show()


def plot_percentages_per_category():
    files = [["easy",10,10], ["medium",10,10], ["hard",30,50]]
    for i in files:
        averages, category, n = calc_percntages(i[1], i[2], i[0])
        # Plotting the boxplot
        plt.figure(figsize=(10, 6))
        plt.bar(averages.keys(), averages.values(), color='skyblue')
        plt.ylim(0, 100)  # Set y-axis limits from 0 to 100
        plt.xlabel('Folders')
        plt.ylabel('Averages (%)')
        plt.title(f'Averages for {category} (n={n})')
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()  # Adjust layout to prevent label cutoff
        plt.savefig(f"{i[0]}.pdf", format="pdf", bbox_inches="tight")
        #plt.show()

def generate_regret_lists(category):
    files = os.listdir(path_regret)
    dataframes = []
    
    for folders in files:
        liste = []
        print(path_regret)
        print(folders)
        # Load data
        data = pd.read_csv(path_regret + str(folders) + f"/{category}.csv", encoding='utf-8', delimiter=';')
        
        # Compute averages
        for _, row in data.iterrows():
            average = sum(row) / len(row)  # Calculate the row average
            liste.append(average)
        
        # Create a DataFrame for this folder
        df = pd.DataFrame({folders: liste})  # Create DataFrame with folder name as column
        df['nth-Question'] = range(1, len(df) + 1)  # Add 'nth-Question' as a column
        dataframes.append(df)
    
    # Merge all DataFrames on nth-Question
    result = dataframes[0]
    for df in dataframes[1:]:
        result = pd.merge(result, df, on='nth-Question', how='outer')
    print(result)
    return result

def plot_sse_majority():
    files = [["easy",10], ["medium",10], ["hard",30], ["alle",50]]
    #files = ["alle",50]
    for i in files:
        df = generate_regret_lists(i[0])
        x = df['nth-Question']
        y1 = df['majority']  # First dataset for plotting
        y2 = df['popularitySSE']  # Second dataset for plotting

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
        ax.set_xlim(left=0.0, right=50)  # Ensure X-axis starts at 0.0
        ax.set_ylim(bottom=0.0, top=i[1])  # Ensure Y-axis starts at 0.0 if needed

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

plot_sse_majority()
#plot_percentages_per_category()