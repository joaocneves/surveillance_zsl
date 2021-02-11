import numpy as np
from sklearn.metrics import accuracy_score
from scipy.spatial import distance_matrix
from matplotlib import pyplot

def cmc_curve(ID_pred, ID_gt):

    """
    :param ID_pred: NxM array, containing the predicted ID for ith rank (out of M) for all N subjects
    :param ID_gt: Nx1 array containing the real ID of the N subjects
    :return cmc: Mx1 array
    """

    N = ID_pred.shape[0]
    M = ID_pred.shape[1]

    acc_per_rank = np.zeros((N, M))

    for idx in range(N):

        best_rank = np.where(ID_pred[idx]==ID_gt[idx])[0][0]
        acc_per_rank[idx, best_rank:]  = 1

    cmc = np.average(acc_per_rank, axis=0)

    return cmc

def hit_penetration(query2gallery_dist, softclass2id, gallery_id_arr, gallery_softclass_arr):

    """

    :param query2gallery_dist:
    :param softclass2id:
    :param gallery_id_arr:
    :return:
    """


    N_imgs = len(gallery_id_arr)
    N_id = len(np.unique(gallery_id_arr))
    M_id = query2gallery_dist.shape[0]

    hit_per_ids_retrieved = np.zeros((M_id, N_id))

    softclass_pred_mat = np.argsort(query2gallery_dist, axis=1)

    for i in range(M_id): # for each query
        ids_retrieved = set()
        softclass_pred_arr = []
        for j in range(N_imgs): # for each image

            softclass_pred = gallery_softclass_arr[softclass_pred_mat[i][j]]

            if softclass_pred in softclass_pred_arr:
                continue

            softclass_pred_arr.append(softclass_pred)
            gallery_id = gallery_id_arr[i]
            for el in softclass2id[softclass_pred]:
                ids_retrieved.add(el)

            if gallery_id in softclass2id[softclass_pred]:
                hit_per_ids_retrieved[i,len(ids_retrieved)-1:] = 1
                break


    hp_curve = np.average(hit_per_ids_retrieved, axis=0)

    return hp_curve


def build_softclass2id(X_labels, X_ID):

    """
    Constructs a dictionary that maps the soft class to a list of person IDs having this class

    :param X_labels:
    :param X_ID:
    :return:
    """

    softclass2id = {}
    for idx in range(len(X_ID)):

        softclass = X_labels[idx]
        persons_id = X_ID[idx]

        if softclass not in softclass2id:
            softclass2id[softclass] = [persons_id]
        else:
            softclass2id[softclass].append(persons_id)

    return softclass2id

X_labels = np.loadtxt('test-labels.txt', dtype=int)
S_test = np.loadtxt('test-atts.txt', dtype=int)
X_ID = np.loadtxt('test-person_id.txt', dtype=int)

softclass2id = build_softclass2id(X_labels, X_ID)

with open("test-imgs.txt") as f:
    X_img_names = f.read().splitlines()

# gallery
S_persons_to_test = [S_test[int(X_labels[idx])-1] for idx in range(len(X_labels))]

S_pred = []
for idx, img_name in enumerate(X_img_names):

    # load image
    # extract features
    # map features to atts

    S_pred.append(S_test[int(X_labels[idx])-1])

#print(S_pred)

query2gallery_dist = distance_matrix(S_test, S_pred)


hp_curve = hit_penetration(query2gallery_dist, softclass2id, X_ID, X_labels)
print(hp_curve)

# plot the roc curve for the model
pyplot.plot(list(range(len(hp_curve))), hp_curve, linestyle='--', label='Soft biometrics (manual)')

# axis labels
pyplot.xlabel('Penetration Rate')
pyplot.ylabel('Hit Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()