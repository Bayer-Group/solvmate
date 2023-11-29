from rdkit import Chem
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from rdkit.Chem import Draw
from rdkit.Chem import rdFingerprintGenerator

import math
import random
from typing import List

"""
Provides easy access to the butina clustering.
"""

def tanimoto_distance_matrix(fp_list):
    """Calculate distance matrix for fingerprint list"""
    dissimilarity_matrix = []
    for i in range(1, len(fp_list)):
        similarities = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[:i])
        dissimilarity_matrix.extend([1 - x for x in similarities])
    return dissimilarity_matrix


def simple_butina_clustering(fps,cutoff=0.4,):
    dist_mat = tanimoto_distance_matrix(fps)
    clusters = Butina.ClusterData(dist_mat, len(fps), cutoff, isDistData=True)
    clusters = sorted(clusters, key=len, reverse=True)
    return clusters



