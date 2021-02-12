import numpy as np
from sklearn.metrics import accuracy_score
from scipy.spatial import distance_matrix
from load_lfwsoft_annot_lib import load_lfwsoft_annot_per_id

def create_gallery_probe_sets(id2feat):

    gallery_id = []
    gallery_feats = []
    probe_id = []
    probe_feats = []

    id_num = 0
    for id in id2feat:

        id_feat_list = id2feat[id]

        gallery_id.append(id_num)
        gallery_feats.append(id_feat_list[0])

        for feat in id_feat_list[1:]:

            probe_id.append(id_num)
            #probe_feats.append(feat)
            # test with completly perfect soft labels
            probe_feats.append(id_feat_list[0])

        id_num = id_num + 1

    return gallery_id, gallery_feats, probe_id, probe_feats


MANUAL_ANNOT_FILE_PATH = 'data\\LFW_SoftBiometrics\\files\\LFW_ManualAnnotations.txt'


id2feat = load_lfwsoft_annot_per_id(MANUAL_ANNOT_FILE_PATH)
gallery_id, gallery_feats, probe_id, probe_feats = create_gallery_probe_sets(id2feat)

gallery_feats = np.array(gallery_feats)
print(len(np.unique(gallery_feats, axis=0)))
probe_feats = np.array(probe_feats)

dist = distance_matrix(gallery_feats, probe_feats)
argmin_dist = np.argmin(dist, axis=0)
pred_id = np.array(gallery_id)[argmin_dist]

print(accuracy_score(probe_id, pred_id))