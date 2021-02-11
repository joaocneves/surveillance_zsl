import numpy as np




# manual_soft_biom[img_name_1.jpg] -> [att_1 att_2 att_3 ... att_n]
def load_lfwsoft_manual_annot(annot_file_path):

    """
    Reads the soft biometric labels of the LFW Soft Biometrics Dataset (http://atvs.ii.uam.es/atvs/LFW_SoftBiometrics.html)
    :param annot_file_path: fullpath to the annotation file
    :return: dictionary mapping each image to its soft biometric traits (string to list)

    INPUT

    Example Annot file:
    ------ (General Version)
    img_name_1.jpg att_1 att_2 att_3 ... att_n
    img_name_2.jpg att_1 att_2 att_3 ... att_n
    ...
    img_name_N.jpg att_1 att_2 att_3 ... att_n

    ------ (Example Version)
    AJ_Cook_0001.jpg 1 2 0 0 2 0 0 0 1 1 0
    AJ_Lamas_0001.jpg 0 2 0 0 1 1 0 0 1 1 0
    Aaron_Eckhart_0001.jpg 0 3 0 0 2 0 0 1 1 1 0
    ------

    OUTPUT

    manual_soft_biom[img_name_1.jpg] -> [att_1 att_2 att_3 ... att_n]

    """

    manual_soft_biom = {}
    with open(annot_file_path) as f:
        for line in f:
           line_parts = line.split()
           img_name = line_parts[0].replace('.jpg','')
           feat_vec = np.array([int(i) for i in line_parts[1:]])
           manual_soft_biom[img_name] = feat_vec


    return manual_soft_biom

def load_lfwsoft_automatic_annot(annot_file_path):

    manual_soft_biom = {}
    with open(annot_file_path) as f:
        for line in f:
           line_parts = line.split()
           img_name = line_parts[0].replace('.jpg','')
           feat_vec = np.array([float(i) for i in line_parts[1:]])
           manual_soft_biom[img_name] = feat_vec


    return manual_soft_biom


def load_lfwsoft_annot_per_id(annot_file_path):

    id2feat = {}
    with open(annot_file_path) as f:
        for line in f:
           line_parts = line.split()
           img_name = line_parts[0]
           img_id = img_name[:img_name.rfind('_')]
           feat_vec = np.array([float(i) for i in line_parts[1:]])
           if img_id in id2feat.keys():
               id2feat[img_id].append(feat_vec)
           else:
               id2feat[img_id] = [feat_vec]



    return id2feat


def load_lfwsoft_data(annot_file_path):

    """
    Reads the soft biometric labels of the LFW Soft Biometrics Dataset (http://atvs.ii.uam.es/atvs/LFW_SoftBiometrics.html)
    :param annot_file_path: fullpath to the annotation file
    :return: N-dim table (list of lists) containing the information for each of the N images

    INPUT

    Example Annot file:
    ------ (General Version)
    img_name_1.jpg att_1 att_2 att_3 ... att_n
    img_name_2.jpg att_1 att_2 att_3 ... att_n
    ...
    img_name_N.jpg att_1 att_2 att_3 ... att_n

    ------ (Example Version)
    AJ_Cook_0001.jpg 1 2 0 0 2 0 0 0 1 1 0
    AJ_Lamas_0001.jpg 0 2 0 0 1 1 0 0 1 1 0
    Aaron_Eckhart_0001.jpg 0 3 0 0 2 0 0 1 1 1 0
    ------

    OUTPUT

    person_id is assigned based on name of the person in the image name
    person_name is the name of the person in the image name

    lfwsoft_data[0] = [person_id, person_name, img_name . [att_1 att_2 att_3 ... att_k]]
    lfwsoft_data[1] = [person_id, person_name, img_name . [att_1 att_2 att_3 ... att_k]]
    ...
    lfwsoft_data[N] = [person_id, person_name, img_name . [att_1 att_2 att_3 ... att_k]]
    """


    person_id = 0
    unique_idname_list = {''}
    lfwsoft_data = []

    with open(annot_file_path) as f:
        for line in f:
           line_parts = line.split()
           img_name = line_parts[0]
           person_name = img_name[:img_name.rfind('_')]
           feat_vec = np.array([float(i) for i in line_parts[1:]])
           #feat_vec = feat_vec[[0,1,2,6,8,9,10]]

           if person_name not in unique_idname_list:
               unique_idname_list.add(person_name)
               person_id = person_id + 1

           lfwsoft_data.append([person_id, person_name, img_name, feat_vec])

    return lfwsoft_data


if __name__ == '__main__':

    MANUAL_ANNOT_FILE_PATH = 'W:\\LFW_SoftBiometrics\\files\\LFW_ManualAnnotations.txt'
    print(load_lfwsoft_manual_annot(MANUAL_ANNOT_FILE_PATH))



