import pandas as pd
import numpy as np
import torch

# Calculate descriptive features for each cluster
def calculate_descriptive_features(features, cluster_id):
    cluster_data = features[features['cluster'] == cluster_id].drop('cluster', axis=1)
    descriptive_features = cluster_data.mean()
    descriptive_features = pd.DataFrame({
        str(cluster_id): descriptive_features
    })
    descriptive_features = descriptive_features.sort_values(by=str(cluster_id), ascending=False)
    return descriptive_features.head(10)

# Calculate discriminating features for each cluster
def calculate_discriminating_features(features, cluster_id):
    cluster_data = features[features['cluster'] == cluster_id].drop('cluster', axis=1)
    other_data = features[features['cluster'] != cluster_id].drop('cluster', axis=1)
    mean_diff = cluster_data.mean() - other_data.mean()
    abs_diff = abs(cluster_data.mean() - other_data.mean())
    discriminating_features = pd.DataFrame({
        'mean_diff': mean_diff,
        'abs_diff': abs_diff
    })
    discriminating_features = discriminating_features.sort_values(by='abs_diff', ascending=False)
    return discriminating_features.head(10)

def jaccard_similarity_matrix(binary_matrix):
    n_samples = binary_matrix.shape[0]
    jaccard_matrix = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(n_samples):
            intersection = np.sum(binary_matrix[i] & binary_matrix[j])
            union = np.sum(binary_matrix[i] | binary_matrix[j])
            jaccard_matrix[i, j] = intersection / union if union != 0 else 0

    return jaccard_matrix

def save_sparse_matrix_to_cluto_graph(file_name, sparse_matrix):
    with open(file_name, 'w') as f:
        num_vertices = sparse_matrix.shape[0]
        num_edges = sparse_matrix.nnz

        # Write the number of vertices and edges
        f.write(f"{num_vertices} {num_edges}\n")

        # Write the adjacency list
        for i in range(num_vertices):
            row = sparse_matrix.getrow(i)
            # print(row)
            elements = row.nonzero()[1]
            # print(elements)
            line_elements = []
            for j in elements:
                line_elements.extend([str(j+1), str(row[0, j])])  # CLUTO uses 1-based indexing
            # print(line_elements)
            f.write(" ".join(line_elements) + "\n")

def save_dense_matrix_to_cluto_graph(file_name, similarity_matrix):
    with open(file_name, 'w') as f:
        num_vertices = similarity_matrix.shape[0]
        f.write(f"{num_vertices}\n")
        
        for i in range(num_vertices):
            row = similarity_matrix[i]
            line_elements = []
            for j in range(len(row)):
                line_elements.append(str(row[j]))
            f.write(" ".join(line_elements) + "\n")

def save_sparse_matrix_to_cluto_mat(file_name, data):
    with open(file_name, 'w') as f:
        rows, cols = data.shape
        f.write(f"{rows} {cols}\n")
        for row in data:
            f.write(" ".join(map(str, row)) + "\n")

def calculate_internal_external_similarity(similarity_matrix, cluster_assignments):
    unique_clusters = np.unique(cluster_assignments)
    internal_similarities = {}
    external_similarities = np.zeros((len(unique_clusters), len(unique_clusters)))

    for cluster in unique_clusters:
        # Internal similarity for the cluster
        cluster_indices = np.where(cluster_assignments == cluster)[0]
        if len(cluster_indices) > 1:
            pairwise_similarities = similarity_matrix[np.ix_(cluster_indices, cluster_indices)]
            np.fill_diagonal(pairwise_similarities, 0)
            internal_similarities[cluster] = np.sum(pairwise_similarities) / (len(cluster_indices) * (len(cluster_indices) - 1))
        else:
            internal_similarities[cluster] = 1.0

    for i, cluster_i in enumerate(unique_clusters):
        for j, cluster_j in enumerate(unique_clusters):
            if i != j:
                indices_i = np.where(cluster_assignments == cluster_i)[0]
                indices_j = np.where(cluster_assignments == cluster_j)[0]
                pairwise_similarities = similarity_matrix[np.ix_(indices_i, indices_j)]
                external_similarities[i, j] = np.mean(pairwise_similarities)

    return internal_similarities, external_similarities

def aggregate_inputs(input_tensor):
    column_sum = torch.sum(input_tensor, dim=0)
    binary_results = (column_sum >= 1).int()
    return binary_results

def save_class_file(path, file_name, class_list):
    with open(os.path.join(path, file_name), 'w') as f:
        for rclass in class_list:
            f.write(str(rclass) + '\n')