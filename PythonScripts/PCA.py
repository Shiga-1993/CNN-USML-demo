import glob
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys
import os

args = sys.argv
normalized = str(args[1])
n_components = int(args[2])
inputfile_directory = str(args[3])
outputfile_directory = str(args[4])
keyword_patterns = str(args[5]).split(',')

os.makedirs(outputfile_directory, exist_ok=True)

all_vectors = []
all_keywords = []
original_indices = []

for keyword in keyword_patterns:
    directory_path = inputfile_directory
    file_paths = glob.glob(f"{directory_path}/*{keyword}*")

    for file_path in file_paths:
        df = pd.read_csv(file_path, sep='\s+', header=None)
        vectors = df.values.tolist()
        all_vectors.extend(vectors)
        all_keywords.extend([keyword] * len(vectors))
        original_indices.extend(list(range(len(vectors))))

if not all_vectors:
    sys.exit("No valid vectors found. Please check the input files.")

if normalized == "yes":
    scaler = StandardScaler()
    processed_vectors = scaler.fit_transform(all_vectors)
else:
    processed_vectors = all_vectors

pca = PCA(n_components=n_components)
pca_results = pca.fit_transform(processed_vectors)

for keyword in set(all_keywords):
    keyword_indices = [i for i, k in enumerate(all_keywords) if k == keyword]
    keyword_results = [(original_indices[i], pca_results[i]) for i in keyword_indices]
    keyword_results.sort(key=lambda x: x[0])

    output_file = os.path.join(outputfile_directory, f'{keyword}_pca_results.txt')
    with open(output_file, 'w') as f_out:
        for idx, point in keyword_results:
            f_out.write(f"{idx} {point[0]} {point[1]}\n")

