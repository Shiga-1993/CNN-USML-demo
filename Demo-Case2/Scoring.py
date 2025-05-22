import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Retrieve flags for plotting and score calculation from command-line arguments
plot_flag = sys.argv[1].lower() == 'yes' if len(sys.argv) > 1 else False
score_flag = sys.argv[2].lower() == 'yes' if len(sys.argv) > 2 else False

# Retrieve keywords from environment variables
NNtype = os.getenv('NNtype')
layer_type = os.getenv('layer_type')
type_of_DR = os.getenv('type_of_DR')
DRtype_for_dir = os.getenv('type_of_DR_for_dir')
h_param = os.getenv('h_param')
output_dir = os.getenv('output_dir')
hp_name = os.getenv('hp_name')

# Create the output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Function to load data
def load_data(file_path):
    data = pd.read_csv(file_path, delimiter=' ', header=None, usecols=[1, 2])
    return data.to_numpy()

# Function to extract label from a filename using a label mapping dictionary
def extract_label(file_name, label_mapping):
    parts = file_name.split('_')
    for part in parts:
        if part in label_mapping:
            return label_mapping[part]
    return file_name  # Return the full filename if no label is found

# Label mapping dictionary
rock_types = {
    "BanderaGray": "A",
    "Parker": "B",
    "Kirby": "C",
    "BanderaBrown": "D",
    "BSG": "E",
    "BUG": "F",
    "Berea": "G",
    "CastleGate": "H",
    "BB": "I",
    "Leopard": "J",
    "Bentheimer": "K"
}

# Generate colors for each label based on the sorted label list
labels_list = sorted(rock_types.values())
colors = plt.cm.jet(np.linspace(0, 1, len(labels_list)))
label_to_color = {label: color for label, color in zip(labels_list, colors)}

# Define the directory to search for files
# directory = 'DR_DATA/' + str(DRtype_for_dir)  # Specify the directory path
directory = '../Case2-tSNE-Data'  # Specify the directory path

# Compile a regex pattern to filter files
pattern = re.compile(
    rf'{NNtype}.*{layer_type}.*{type_of_DR}.*{hp_name}{h_param}(?=\D).*\.txt$'
)

# Filter files in the directory that match the pattern
files = [os.path.join(directory, f) for f in os.listdir(directory) if pattern.search(f)]

print(files)
print(NNtype, layer_type, type_of_DR)

if not files:
    print("No files found matching the specified keywords.", str(DRtype_for_dir))
    exit(1)

all_data = []
labels = []

# Load and store data from each file
for file in files:
    data = load_data(file)
    label = extract_label(os.path.basename(file), rock_types)
    all_data.append(data)
    labels.extend([label] * len(data))

# Plot the graph if plot_flag is True
if plot_flag:
    plt.figure(figsize=(10, 8))
    for data, label in zip(all_data, labels):
        plt.scatter(data[:, 0], data[:, 1], label=label, color=label_to_color[label])

    # Set legend in alphabetical order (if handles exist)
    handles, legend_labels = plt.gca().get_legend_handles_labels()
    if handles:
        sorted_legend_labels, sorted_handles = zip(*sorted(zip(legend_labels, handles), key=lambda x: x[0]))
        # Uncomment the following line to display the legend:
        # plt.legend(sorted_handles, sorted_legend_labels, fontsize=15)

    # Set axis labels and tick font sizes
    plt.xlabel('Component 1', fontsize=20)
    plt.ylabel('Component 2', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # Adjust layout to minimize margins
    plt.tight_layout()

    image_file = os.path.join(output_dir, f"{type_of_DR}_{NNtype}{layer_type}{hp_name}{h_param}.png")
    plt.savefig(image_file, dpi=200)
    print(f"Plot saved to {image_file}")

# Calculate clustering evaluation metrics if score_flag is True and data exists
if score_flag and all_data:
    all_data = np.vstack(all_data)
    unique_labels = list(set(labels))
    label_mapping_numeric = {label: idx for idx, label in enumerate(sorted(unique_labels))}  # Sorted alphabetically
    numeric_labels = [label_mapping_numeric[label] for label in labels]

    silhouette_avg = silhouette_score(all_data, numeric_labels)
    davies_bouldin = davies_bouldin_score(all_data, numeric_labels)
    calinski_harabasz = calinski_harabasz_score(all_data, numeric_labels)

    # Output scores in CSV format
    score_file = os.path.join(output_dir, f"scores_{type_of_DR}_{NNtype}{layer_type}{hp_name}{h_param}.csv")
    file_exists = os.path.isfile(score_file)
    with open(score_file, 'a') as f:
        if not file_exists:
            f.write("NNtype,LayerType,TypeOfDR,HParam,SilhouetteScore,DaviesBouldinIndex,CalinskiHarabaszIndex\n")
        f.write(f"{NNtype},{layer_type},{type_of_DR},{h_param},{silhouette_avg},{davies_bouldin},{calinski_harabasz}\n")

    print(f"Scores saved to {score_file}")
else:
    if not all_data:
        print("No data to process for scoring.")

