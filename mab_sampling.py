import numpy as np
from collections import defaultdict

def contingency_sample(scores, n = 1000):     
    scores = scores.astype('int')
    print(scores.shape)
    # Build the contingence matrix
    # -> Map an attribute combination (1,0,0,1, ...,) to a list of indices
    contingence = defaultdict(lambda: [])
    for idx, a in enumerate(scores):
        contingence[tuple(a)].append(idx)
    N, k = scores.shape
    # List of indices that we will keep
    sampled_idx = []
    # Number of positives samples for each attribute in the current sampling
    counts = np.zeros(k, dtype="int")
    
    # Loop for each target sample
    for i in np.arange(n):
        # Extract possible attribute combinations from the contigence matrix
        # (impossible combinations are empty lists, i.e. len == 0)
        candidates = [np.array(k) for k, v in contingence.items() if len(v) > 0]
        # Create the current target vector by thresholding over n/2
        # If t_counts[i] is True, then we have more negatives than positives for attribute i
        #target = counts < len(sampled_idx) / 2
        
        # Get random group
        best = candidates[np.random.randint(0, len(candidates))]
        # Get a sample from the corresponding entry of the contingence matrix
        # (and remove it so that it cannot be sampled anymore)
        r = np.random.randint(0, len(contingence[tuple(best)]))
        chosen_one = contingence[tuple(best)].pop(r) # <- select and remove random sample

        # Add the indice of the chosen sample to the list
        sampled_idx.append(chosen_one)
        
        # Update the counts of positives
        counts += scores[chosen_one]
    
    print(counts / len(sampled_idx))
    
    return sampled_idx