import os
import glob
import pandas as pd
import logging
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

args = sys.argv
normalized = str(args[1])
n_components = int(args[2])  # 
perplexity = int(args[3])  # t-SNE parameter
base_directory_path = str(args[4])
output_directory_path = str(args[5])
keyword_patterns = str(args[6]).split(',')  # Assuming keywords are passed as a command-line argument

logging.basicConfig(filename='computation_log.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

os.makedirs(output_directory_path, exist_ok=True)

all_vectors = []
all_keywords = []  # To store keywords for each vector
original_indices = []  # To track the original row index of each vector

# Collect data for all keywords
for keyword in keyword_patterns:
    directory_path = base_directory_path
    file_paths = glob.glob(f"{directory_path}/*{keyword}*")
    logging.info(f"Starting to process files containing '{keyword}' in {directory_path}. Found {len(file_paths)} files.")
    print(f"Starting to process files containing '{keyword}' in {directory_path}. Found {len(file_paths)} files.")

    if not file_paths:
        logging.warning(f"No files found for keyword '{keyword}' in {directory_path}.")
        print(f"No files found for keyword '{keyword}' in {directory_path}.")
        continue

    for file_path in file_paths:
        if not os.path.isfile(file_path):
            logging.error(f"File not found: {file_path}")
            print(f"File not found: {file_path}")
            continue

        logging.info(f"Processing file: {file_path}")
        print(f"Processing file: {file_path}")
        try:
            df = pd.read_csv(file_path, sep='\s+', header=None)
            print(f"File {file_path} read successfully with shape {df.shape}.")
            print(df.head())  # Display the first few rows of the data
        except Exception as e:
            logging.error(f"Error reading file {file_path}: {e}")
            print(f"Error reading file {file_path}: {e}")
            continue

        if df.empty:
            logging.error(f"File is empty or cannot be read: {file_path}")
            print(f"File is empty or cannot be read: {file_path}")
            continue

        vectors = df.values.tolist()
        #if any(len(vector) != nline for vector in vectors):  # Check if all vectors are of the same length
        #    logging.error(f"Data format error in file: {file_path}")
        all_vectors.extend(vectors)
        all_keywords.extend([keyword] * len(vectors))  # Append the keyword for each vector
        original_indices.extend(list(range(len(vectors))))  # Record the original row indices

# Check if all_vectors is empty
if not all_vectors:
    logging.error("No valid vectors found. Exiting the program.")
    print("No valid vectors found. Exiting the program.")
    sys.exit("No valid vectors found. Please check the input files.")

# Standardizing the data
if normalized == "yes":
    scaler = StandardScaler()
    processed_vectors = scaler.fit_transform(all_vectors)
else:
    processed_vectors = all_vectors

logging.info("Completed processing all files. Starting t-SNE computation.")
print("Completed processing all files. Starting t-SNE computation.")

# Applying t-SNE to the combined data from all keywords
tsne_model = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
tsne_results = tsne_model.fit_transform(processed_vectors)
logging.info("t-SNE computation completed.")
print("t-SNE computation completed.")

# Output the t-SNE results in separate files per keyword, maintaining the original row order
for keyword in set(all_keywords):
    keyword_indices = [i for i, k in enumerate(all_keywords) if k == keyword]
    keyword_results = [(original_indices[i], tsne_results[i]) for i in keyword_indices]

    # We need to sort based on original indices only, not including tsne_results directly in the sort key
    keyword_results.sort(key=lambda x: x[0])  # Sort by original index, x[0] is the index

    output_file = os.path.join(output_directory_path, f'{keyword}_tsne_pp{perplexity}_results.txt')
    with open(output_file, 'w') as f_out:
        for idx, point in keyword_results:
            # Write output, converting point (which is a NumPy array) to string explicitly
            f_out.write(f"{idx} {point[0]} {point[1]}\n")

    logging.info(f"t-SNE results for keyword '{keyword}' saved to {output_file}")
    print(f"t-SNE results for keyword '{keyword}' saved to {output_file}")

