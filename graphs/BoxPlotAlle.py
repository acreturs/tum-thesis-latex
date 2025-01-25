import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.interpolate import make_interp_spline  # For smoothing
path_team_answers = "C:/Users/Alexander Cornett/Desktop/Bachelorthesis writing/tum-thesis-latex/graphs/TeamAnswers/"
path_regret = "C:/Users/Alexander Cornett/Desktop/Bachelorthesis writing/tum-thesis-latex/graphs/regretPlots/"

def calc_percntages(number_category_questions, n, category):
    averages = {}
    files = os.listdir(path_team_answers)
    for folders in files:
        print(path_team_answers)
        print(folders)
        data = pd.read_csv(path_team_answers + str(folders) + f"/{category}.csv", encoding='utf-8', delimiter=';')
        last_row = data.iloc[-1]
        average = 0
        for index, value in last_row.items():
            print(f"Column: {index}, Value: {value}")
            average += (number_category_questions - value) / n
            print(folders)
            print(average)
        averages[folders] = (average / 3) * 100  # Convert to percentage

    return averages, category, n


def plot_boxplot():
    files = [["easy", 10, 10]]  # , ["medium",10,10], ["hard",30,30], ["alle",50,50]]
    plt.figure(figsize=(10, 6))
    for i in files:
        averages, category, n = calc_percntages(i[1], i[1], i[0])
        plt.bar(averages.keys(), averages.values(), color='skyblue')
        plt.ylim(0, 100)  # Set y-axis limits from 0 to 100
        plt.xlabel('Folders')
        plt.ylabel('Averages (%)')
        plt.title(f'Correct Answers for {category} questions')
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("easy_boxplot.pdf", format="pdf", bbox_inches="tight")
    plt.tight_layout()  # Adjust layout to prevent label cutoff
    plt.show()
    print("jetzt saven")


def different_colors():
    files = [["easy", 10, 50], ["medium", 10, 50], ["hard", 30, 50]]
    results_calc = {}
    for i in files:
        averages, category, n = calc_percntages(i[1], i[2], i[0])
        results_calc[category] = averages
        print(results_calc)
    df = pd.DataFrame(results_calc)
    df = df.reset_index()
    df.rename(columns={df.columns[0]: "model"}, inplace=True)
    print(df)

    categories = ["easy", "medium", "hard"]
    colors = ["#66c2a5", "#fc8d62", "#8da0cb"]  # Colors for each category

    # Create the plot
    plt.figure(figsize=(12, 6))
    bottom = [0] * len(df)  # Start with a baseline at 0 for stacking

    # Iterate over categories and plot each as a segment of the stack
    for category, color in zip(categories, colors):
        plt.bar(df["model"], df[category], bottom=bottom, label=category, color=color)
        bottom = [i + j for i, j in zip(bottom, df[category])]  # Update the baseline for stacking

    # Add labels, title, and legend
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Percentage", fontsize=12)
    plt.title("Correct Answers per category", fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(title="Category")
    plt.savefig("different_colors_overall_correct.pdf", format="pdf", bbox_inches="tight")
    # Display the plot
    plt.tight_layout()
    plt.show()

def average_regret(category):
    files = os.listdir(path_regret)
    averages = []# brfüllt mit "liste eine liste" pro file
    for folders in files:
        liste = []
        print(path_regret)
        print(folders)
        data = pd.read_csv(path_regret + str(folders) + f"/{category}.csv", encoding='utf-8', delimiter=';')
        for index, row in data.iterrows():
            average = 0 # Access the full row
            for value in row:
                average += value  # Access individual values in the row
            liste.append(average/3)
        averages.append(liste)
    df = pd.DataFrame(averages).T
    df['nth-Question'] = range(1, len(df) + 1)
    for i, column in enumerate(df.columns):
        try:
            new_name = f"{files[i]}"  # Example of renaming columns to Column_1, Column_2, etc.
            df.rename(columns={column: new_name}, inplace=True)
        except:
            break
    return df


def plot_regret():

    df = average_regret("alle")
    x = df['nth-Question']
    files = os.listdir(path_regret)
    y1 = df[files[0]]
    y2 = df[files[1]]
    y3 = df[files[2]]
    y4 = df[files[3]]
    # Smooth the lines using cubic spline interpolation
    x_smooth = np.linspace(x.min(), x.max(), 300)  # Create a smooth x-axis
    y1_smooth = make_interp_spline(x, y1)(x_smooth)
    y2_smooth = make_interp_spline(x, y2)(x_smooth)
    y3_smooth = make_interp_spline(x, y3)(x_smooth)
    y4_smooth = make_interp_spline(x, y4)(x_smooth)
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
    ax.plot(x_smooth, y3_smooth, label='SSE', color='red', linewidth=2.5)
    ax.plot(x_smooth, y4_smooth, label='SSE', color='green', linewidth=2.5)
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


plot_regret()