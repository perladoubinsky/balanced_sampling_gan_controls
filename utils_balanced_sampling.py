import numpy as np
from sklearn import svm

from mab_sampling import contingency_sample

def get_most_confident_samples(scores_dict, t):
    confident_scores = []
    for k,v in scores_dict.items():
        if scores_dict[k].ndim == 1:
            scores_dict[k] = scores_dict[k][:,np.newaxis]
        tmp_scores = np.zeros(scores_dict[k].shape, dtype=bool)
        tmp_scores[np.where(scores_dict[k]>t)[0]] = True
        tmp_scores[np.where(scores_dict[k]<(1-t))[0]] = True
        confident_scores.append(tmp_scores)
    confident_scores = np.concatenate(confident_scores, axis=1)
    confident_scores = confident_scores.all(axis=1)
    idx = np.where(confident_scores==True)[0]
    np.random.shuffle(idx)
    return idx

def dict_to_np(scores_dict):
    scores = []
    for k,v in scores_dict.items():
        if scores_dict[k].ndim == 1:
            scores_dict[k] = scores_dict[k][:,np.newaxis]
        scores.append(scores_dict[k])            
    scores = np.concatenate(scores, axis=1)
    return scores
    
def get_boundary(latent_codes, labels, method):
    if method=='svm':
        clf = svm.SVC(kernel='linear')
        classifier = clf.fit(latent_codes, labels)
        boundary = classifier.coef_.reshape(1, latent_codes.shape[1]).astype(np.float32)
    elif method=='centroids':
        g_class_0 = np.mean(latent_codes[np.where(labels==0)[0]], axis = 0)
        g_class_1 = np.mean(latent_codes[np.where(labels==1)[0]], axis = 0)
        boundary = g_class_1 - g_class_0
        boundary = boundary.reshape(1, latent_codes.shape[1]).astype(np.float32)
    else:
        raise ValueError('This method is not defined')    
    return boundary / np.linalg.norm(boundary)

def train_boundary(latent_codes, scores_dict, attribute, confidence_t, n, method_boundary='centroids'):
    
    scores = dict_to_np(scores_dict)
    
    ### Select confident samples
    if confidence_t:
        idx_confident = get_most_confident_samples(scores_dict, confidence_t)
        scores = scores[idx_confident]
        latent_codes = latent_codes[idx_confident]
    
    ### Select a balanced subset among confident samples
    labels = np.round(scores)
    idx_balanced = contingency_sample(labels, n)
    latent_codes = latent_codes[idx_balanced]
    tgt_labels = labels[idx_balanced, list(scores_dict.keys()).index(attribute)]
    
    ### Compute the boundary
    boundary = get_boundary(latent_codes, tgt_labels, method_boundary)
    
    return boundary