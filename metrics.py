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
            else:
                fp += 1
        else:
            if labels[i] == 1:
                fn += 1
            else:
                tn += 1
    return tp, fp, fn, tn


def fpr95(labels, scores, precision=0.0005, num_samples=10000):
    max_score = np.max(scores)
    min_score = np.min(scores)
    samples = 0
    fpr = 0
    for threshold in np.linspace(min_score, max_score, num_samples, endpoint=True):
        predictions = scores > threshold
        tp, fp, fn, tn = tfpn(labels, predictions)
        tpr = tp / (tp + fn)
        if abs(tpr - 0.95) < precision:
            samples += 1
            fpr += fp / (fp + tn)
    if samples == 0:
        return fpr95(labels, scores, precision * 10)
    return fpr / samples


def auroc(labels, scores, num_samples=10000):
    max_score = np.max(scores)
    min_score = np.min(scores)
    high_fpr = 0
    area = 0
    for threshold in np.linspace(max_score, min_score, num_samples, endpoint=True):
        predictions = scores > threshold
        tp, fp, fn, tn = tfpn(labels, predictions)
        tpr = tp / (tp + fn)
        low_fpr = high_fpr
        high_fpr = fp / (fp + tn)
        area += tpr * (high_fpr - low_fpr)
    area += (1 - high_fpr) * tpr
    return area


def aupr_in(labels, scores, num_samples=10000):
    max_score = np.max(scores)
    min_score = np.min(scores)
    high_recall = 0
    area = 0
    for threshold in np.linspace(max_score, min_score, num_samples, endpoint=True):
        predictions = scores > threshold
        tp, fp, fn, _ = tfpn(labels, predictions)
        if tp != 0:
            precision = tp / (tp + fp)
            low_recall = high_recall
            high_recall = tp / (tp + fn)
            area += precision * (high_recall - low_recall)
    area += (1 - high_recall) * precision
    return area


def aupr_out(labels, scores, num_samples=10000):
    scores = np.ones(len(scores)) - scores
    labels = np.ones(len(labels)) - labels
    return aupr_in(labels, scores, num_samples)


def detection_error(labels, scores, num_samples=10000):
    max_score = np.max(scores)
    min_score = np.min(scores)
    error = 1
    for threshold in np.linspace(min_score, max_score, num_samples, endpoint=True):
        predictions = scores > threshold
        tp, fp, fn, tn = tfpn(labels, predictions)
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        pe = 0.5 * (1 - tpr) + 0.5 * fpr
        error = min(error, pe)
    return error


def get_metrics(labels, scores, num_thresholds=10000):
    max_score = np.max(scores)
    min_score = np.min(scores)

    tpr_samples = 0
    backup_tpr_samples = 0
    fpr_95 = 0
    backup_fpr_95 = 0

    high_fpr = 0
    auroc = 0

    high_recall = 0
    auprin = 0

    error = 1

    for threshold in np.linspace(max_score, min_score, num_thresholds, endpoint=True):
        predictions = scores > threshold
        tp, fp, fn, tn = tfpn(labels, predictions)
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)

        if abs(tpr - 0.95) < 0.0005:
            tpr_samples += 1
            fpr_95 += fpr

        if abs(tpr - 0.95) < 0.005:
            backup_tpr_samples += 1
            backup_fpr_95 += fpr

        low_fpr = high_fpr
        high_fpr = fpr
        auroc += tpr * (high_fpr - low_fpr)

        if tp != 0:
            precision = tp / (tp + fp)
            low_recall = high_recall
            high_recall = tp / (tp + fn)
            auprin += precision * (high_recall - low_recall)

        pe = 0.5 * (1 - tpr) + 0.5 * fpr
        error = min(error, pe)

    if tpr_samples == 0:
        fpr_95 = "Problem"
        # fpr_95 = backup_fpr_95 / backup_tpr_samples
    else:
        fpr_95 = fpr_95 / tpr_samples

    auroc += (1 - high_fpr) * tpr

    auprin += (1 - high_recall) * precision

    auprout = aupr_out(labels, scores, num_thresholds)

    return fpr_95, auroc, auprin, auprout, error
