import os
import glob
import pandas as pd
import umap
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler

args = sys.argv
normalized = str(args[1])
n_components = int(args[2])
n_neighbors = int(args[3])
inputfile_directory = str(args[4])
outputfile_directory = str(args[5])
keyword_patterns = str(args[6]).split(',')

os.makedirs(outputfile_directory, exist_ok=True)

all_vectors = []
all_keywords = []
original_indices = []

for keyword in keyword_patterns:
    directory_path = inputfile_directory
    file_paths = glob.glob(f"{directory_path}/*{keyword}*")

    if not file_paths:
        print(f"No files found for keyword '{keyword}' in {directory_path}.")
        continue
    for file_path in file_paths:
        if not os.path.isfile(file_path):
            print(f"File not found: {file_path}")
            continue
        try:
            df = pd.read_csv(file_path, sep='\s+', header=None)
            print(df.head())
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            continue
        if df.empty:
            print(f"File is empty or cannot be read: {file_path}")
            continue

        vectors = df.values.tolist()
        all_vectors.extend(vectors)
        all_keywords.extend([keyword] * len(vectors))
        original_indices.extend(list(range(len(vectors))))

if not all_vectors:
    print("No valid vectors found. Exiting the program.")
    sys.exit("No valid vectors found. Please check the input files.")

if normalized == "yes":
    scaler = StandardScaler()
    processed_vectors = scaler.fit_transform(all_vectors)
else:
    processed_vectors = all_vectors


umap_model = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=42)
umap_results = umap_model.fit_transform(processed_vectors)


for keyword in set(all_keywords):
    keyword_indices = [i for i, k in enumerate(all_keywords) if k == keyword]
    keyword_results = [(original_indices[i], umap_results[i]) for i in keyword_indices]
    keyword_results.sort(key=lambda x: x[0])

    output_file = os.path.join(outputfile_directory, f'{keyword}_umap_nn{n_neighbors}_results.txt')
    with open(output_file, 'w') as f_out:
        for idx, point in keyword_results:
            f_out.write(f"{idx} {point[0]} {point[1]}\n")


