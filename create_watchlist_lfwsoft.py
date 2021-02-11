import random
import numpy as np
from operator import itemgetter
from load_lfwsoft_annot_lib import load_lfwsoft_annot_per_id
from load_lfwsoft_annot_lib import  load_lfwsoft_data
random.seed(10)

def write_data(eval_split, lfwsoft_data):

    f_train_imgs = open('{0}-imgs.txt'.format(eval_split), 'w')
    f_train_labels = open('{0}-labels.txt'.format(eval_split), 'w')
    f_train_atts = open('{0}-atts.txt'.format(eval_split), 'w')
    f_person_id = open('{0}-person_id.txt'.format(eval_split), 'w')

    label = 0
    soft_class = -1
    for inst in lfwsoft_data:

        if inst[4] != soft_class:
            soft_class = inst[4]
            label = label + 1
            for val in inst[3]:
                f_train_atts.write(str(int(val)) + ' ')
            f_train_atts.write('\n')

        f_train_imgs.write(inst[2] + '\n')
        f_train_labels.write(str(label) + '\n')
        f_person_id.write(str(inst[0]) + '\n')

    f_train_imgs.close()
    f_train_labels.close()
    f_train_atts.close()

def add_softclass_col(lfwsoft_data):

    soft_traits_array = np.array([inst[3] for inst in lfwsoft_data])

    # determine unique soft traits
    unique_soft_traits = np.unique(soft_traits_array, axis=0)

    soft_classes = find_rows_in_matrix(soft_traits_array, unique_soft_traits)

    for i in range(len(lfwsoft_data)):
        lfwsoft_data[i].append(soft_classes[i])

    return lfwsoft_data

def find_rows_in_matrix(query, M):

    indices = []
    for q in query:
       indices.append(np.where((M == q).all(axis=1))[0][0])

    return indices

def ensure_match_train_test_labels(train_soft_traits, test_soft_traits):

    D = train_soft_traits.shape[1]

    for i in range(D):
        if set(np.unique(test_soft_traits[:,i])).issubset(set(np.unique(train_soft_traits[:,i]))):
            print('ok')
        else:
            print('err')


def create_watchlist_lfwsoft(lfwsoft_data):

    soft_traits_array = np.array([inst[3] for inst in lfwsoft_data])

    # determine unique soft traits
    unique_soft_traits = np.unique(soft_traits_array, axis=0)

    # select 25% for the test set
    all_s_idx = list(range(unique_soft_traits.shape[0]))
    test_s_idx = random.sample(all_s_idx, int(0.25*len(all_s_idx)))
    train_s_idx = list(set(all_s_idx) - set(test_s_idx))
    test_soft_traits = unique_soft_traits[test_s_idx,:]
    train_soft_traits = unique_soft_traits[train_s_idx,:]
    ensure_match_train_test_labels(train_soft_traits, test_soft_traits)

    test_soft_traits_list = test_soft_traits.tolist()
    test_idx = []
    test_id = []
    for idx,inst in enumerate(lfwsoft_data):
        if list(inst[3]) in test_soft_traits_list:
            test_idx.append(idx)
            test_id.append(inst[0])

    lfwsoft_data_test = [lfwsoft_data[idx] for idx in test_idx]

    lfwsoft_data_train = [inst for inst in lfwsoft_data if inst[0] not in test_id]


    lfwsoft_data_test_sorted = sorted(lfwsoft_data_test, key=itemgetter(4))
    lfwsoft_data_train_sorted = sorted(lfwsoft_data_train, key=itemgetter(4))

    return lfwsoft_data_train_sorted, lfwsoft_data_test_sorted


MANUAL_ANNOT_FILE_PATH = 'W:\\LFW_SoftBiometrics\\files\\LFW_ManualAnnotations.txt'

# load lfw soft biometrics data in table format
lfwsoft_data = load_lfwsoft_data(MANUAL_ANNOT_FILE_PATH)

# add a new column with the soft class label
lfwsoft_data = add_softclass_col(lfwsoft_data)

# divide data into train/test according to Watchlist ZSL
#lfwsoft_data_train_sorted, lfwsoft_data_test_sorted = create_watchlist_lfwsoft(lfwsoft_data)

# divide data into train/test according to Watchlist ZSL
#lfwsoft_data_train_sorted, lfwsoft_data_val_sorted = create_watchlist_lfwsoft(lfwsoft_data_train_sorted)

lfwsoft_data_test_sorted = sorted(lfwsoft_data, key=itemgetter(4))
#write_data('train', lfwsoft_data_train_sorted)
#write_data('val', lfwsoft_data_val_sorted)
write_data('test', lfwsoft_data_test_sorted)
