import glob
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import sys
import os

args = sys.argv
normalized = str(args[1])
n_components = int(args[2])
perplexity = int(args[3])
keyword_patterns = str(args[4]).split(',')

inputfile_directory = '/path/to/your/inputfile_directory'
outputfile_directory = '/path/to/your/outputfile_directory'
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

if normalized == "yes":
    scaler = StandardScaler()
    processed_vectors = scaler.fit_transform(all_vectors)
else:
    processed_vectors = all_vectors

tsne = TSNE(n_components=n_components, perplexity=perplexity)
tsne_results = tsne.fit_transform(processed_vectors)

for keyword in set(all_keywords):
    keyword_indices = [i for i, k in enumerate(all_keywords) if k == keyword]
    keyword_results = [(original_indices[i], tsne_results[i]) for i in keyword_indices]
    keyword_results.sort(key=lambda x: x[0])

    output_file = os.path.join(outputfile_directory, f'{keyword}_tsne_pp{perplexity}_results.txt')
    with open(output_file, 'w') as f_out:
        for idx, point in keyword_results:
            f_out.write(f"{idx} {point[0]} {point[1]}\n")


