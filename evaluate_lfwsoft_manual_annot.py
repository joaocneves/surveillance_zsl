import os
import numpy as np
from keras.utils.np_utils import to_categorical
from load_lfwsoft_annot_lib import load_lfwsoft_automatic_annot
from scipy.spatial import distance
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
import random


def compute_eer(fpr,tpr,thresholds):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    fnr = 1-tpr
    abs_diffs = np.abs(fpr - fnr)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((fpr[min_index], fnr[min_index]))
    return eer, thresholds[min_index]



gt_scores = []
pred_soft_scores = []
vgg_fc6_scores = []
vgg_fc7_scores = []

LFW_EVALUATION_FOLDS_PATH = 'data\\LFW_SoftBiometrics\\files'
MANUAL_ANNOT_FILE_PATH = 'data\\LFW_SoftBiometrics\\files\\LFW_AutomaticAnnotations.txt'
MAX_SOFT_LABELS_PER_TRAIT = 5 # useful for hamming distance

manual_soft_biom = load_lfwsoft_automatic_annot(MANUAL_ANNOT_FILE_PATH)

soft_mask = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

NON_CATEGORICAL_TRAITS_IDX = [1]

for split in range(1,10):

    for pair_type in ['genuine', 'impostor']:

        with open(os.path.join(LFW_EVALUATION_FOLDS_PATH, 'vggface_{:02d}_{}.txt'.format(split, pair_type))) as f:
            for line in f:

                line_parts = line.split()
                img_name_1 = line_parts[0]
                img_name_2 = line_parts[1]
                fc6_score = float(line_parts[2])
                fc7_score = float(line_parts[3])

                img_feat_1 = manual_soft_biom[img_name_1]
                img_feat_2 = manual_soft_biom[img_name_2]

                feat_dists = []
                for idx in range(len(img_feat_1)):

                    feat_val_1 = img_feat_1[idx]
                    feat_val_2 = img_feat_2[idx]

                    if idx in NON_CATEGORICAL_TRAITS_IDX:

                        dist = abs(feat_val_1 - feat_val_2)*soft_mask[idx]

                    else:

                        #img_feat_1_hot_enc = to_categorical(feat_val_1, num_classes=MAX_SOFT_LABELS_PER_TRAIT)
                        #img_feat_2_hot_enc = to_categorical(feat_val_2, num_classes=MAX_SOFT_LABELS_PER_TRAIT)

                        #dist = soft_mask[idx]*distance.hamming(img_feat_1_hot_enc, img_feat_2_hot_enc)
                        dist = soft_mask[idx]*(1 - int(feat_val_1==feat_val_2))

                    feat_dists.append(dist)

                if np.average(feat_dists) == 0.0:
                    soft_score = 1
                else:
                    soft_score = np.sum(soft_mask)/np.sum(feat_dists)

                vgg_fc6_scores.append(fc6_score)
                vgg_fc7_scores.append(fc7_score)
                pred_soft_scores.append(soft_score)
                if pair_type == 'genuine':
                    gt_scores.append(1)
                else:
                    gt_scores.append(0)


print(gt_scores)

# calculate roc curves
soft_fpr, soft_tpr, soft_thresholds = roc_curve(gt_scores, pred_soft_scores)
fcg_fpr, fc6_tpr, fc6_thresholds = roc_curve(gt_scores, vgg_fc6_scores)
# plot the roc curve for the model
pyplot.plot(soft_fpr, soft_tpr, linestyle='--', label='Soft biometrics (manual)')
pyplot.plot(fcg_fpr, fc6_tpr, marker='.', label='VGG Face fc6')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()

print(compute_eer(soft_fpr,soft_tpr,soft_thresholds))





