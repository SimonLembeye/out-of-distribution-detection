import numpy as np

# labels = 1 if id, 0 if ood

def tfpn(labels, predictions):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i in range(len(labels)):
        if predictions[i] == 1:
            if labels[i] == 1:
                tp += 1
            else : 
                fp += 1
        else :
            if labels[i] == 1:
                fn += 1
            else : 
                tn += 1
    return tp, fp, fn, tn

def fpr95(labels, scores, precision = .5, num_samples = 1000):
    max_score = np.max(scores)
    min_score = np.min(scores)
    samples = 0
    fpr = 0
    for threshold in np.linspace(min_score, max_score, num_samples, endpoint = True):
        predictions = (scores > threshold)
        tp, fp, _, tn = tfpn(labels, predictions)
        tpr = tp / (tp + tn)
        if abs(tpr - .95) < precision:
            samples += 1
            fpr = fp / (fp + tn)
    return (fpr / samples)

def auroc(labels, scores, num_samples = 10000):
    max_score = np.max(scores)
    min_score = np.min(scores)
    high_fpr = 0
    area = 0
    for threshold in np.linspace(min_score, max_score, num_samples, endpoint = True):
        predictions = (scores > threshold)
        tp, fp, _, tn = tfpn(labels, predictions)
        tpr = tp / (tp + tn)
        low_fpr = high_fpr
        high_fpr = fp / (fp + tn)
        area += tpr * (high_fpr - low_fpr)
    area += (1 - high_fpr) * tpr
    return area

def aupr_in(labels, scores, num_samples = 10000):
    max_score = np.max(scores)
    min_score = np.min(scores)
    high_recall = 0
    area = 0
    for threshold in np.linspace(min_score, max_score, num_samples, endpoint = True):
        predictions = (scores > threshold)
        tp, fp, fn, _ = tfpn(labels, predictions)
        if tp != 0 :
            precision = tp / (tp + fp)
            low_recall = high_recall
            high_recall = tp / (tp + fn)
            area += precision * (high_recall - low_recall)
    area += (1 - high_recall) * precision
    return area

def aupr_out(labels, scores, num_samples = 10000):
    scores = np.ones(len(scores)) - scores
    labels = np.ones(len(labels)) - labels
    return aupr_in(labels, scores, num_samples)

def detection_error(labels, scores, num_samples = 10000):
    max_score = np.max(scores)
    min_score = np.min(scores)
    error = 1
    for threshold in np.linspace(min_score, max_score, num_samples, endpoint = True):
        predictions = (scores > threshold)
        tp, fp, _, tn = tfpn(labels, predictions)
        tpr = tp / (tp + tn)
        fpr = fp / (fp + tn)
        pe = .5 * (1 - tpr) + .5 * fpr
        error = min(error, pe)
    return error