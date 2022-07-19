from __future__ import print_function, absolute_import
from collections import OrderedDict
import torch
import torch.nn.functional as F
from reid.feature_extraction import extract_cnn_feature
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from .rerank import re_ranking
import random
import time
import gc

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def extract_features_per_cam(model, data_loader):
    model.eval()
    per_cam_features = {}
    per_cam_fname = {}
    print("Start extract features per camera")
    for imgs, fnames, _, camid in tqdm(data_loader):
        camid = list(camid)
        for cam in camid:
            cam = cam.item()
            if cam not in per_cam_features.keys():
                per_cam_features[cam] = []
                per_cam_fname[cam] = []
        with torch.no_grad():
            outputs = extract_cnn_feature(model, imgs)

        for fname, output, cam in zip(fnames, outputs, camid):
            cam = cam.item()
            per_cam_features[cam].append(output)
            per_cam_fname[cam].append(fname)
    return per_cam_features, per_cam_fname


def extract_features_cross_cam(model, data_loader):
    model.eval()
    cross_cam_features = []
    cross_cam_fnames = []
    cross_cam_distribute = []
    cams = []
    cam_number = len(model.classifier)
    print("Start extract features cross camera")
    for imgs, fnames, _, camid in tqdm(data_loader):

        with torch.no_grad():
            outputs = extract_cnn_feature(model, imgs, norm=False)
            fnorm = torch.norm(outputs, p=2, dim=1, keepdim=True)
            norm_outputs = outputs.div(fnorm.expand_as(outputs))
            for i in range(cam_number):

                x = model.classifier[i](outputs)
                if i == 0:
                    distribute = F.softmax(x.data, dim=1)
                else:
                    distribute_tmp = F.softmax(x.data, dim=1)
                    distribute = torch.cat((distribute, distribute_tmp), dim=1)

        for fname, output, cam, dis in zip(fnames, norm_outputs, camid,
                                           distribute):
            cam = cam.item()
            cross_cam_fnames.append(fname)
            cross_cam_features.append(output)
            cams.append(cam)
            cross_cam_distribute.append(dis.cpu().numpy())
    return cross_cam_features, cross_cam_fnames, cross_cam_distribute, cams


def jaccard_sim_cross_cam(cross_cam_distribute):
    print("Start calculate jaccard similarity cross camera, this step may cost a lot of time")
    n = cross_cam_distribute.size(0)
    jaccard_sim = torch.zeros((n, n))
    for i in range(n):
        distribute = cross_cam_distribute[i]
        abs_sub = torch.abs(distribute - cross_cam_distribute)
        sum_distribute = distribute + cross_cam_distribute
        intersection = (sum_distribute - abs_sub).sum(dim=1) / 2
        union = (sum_distribute + abs_sub).sum(dim=1) / 2
        jaccard_sim[i, :] = intersection / union
    return to_numpy(jaccard_sim)


def cluster_cross_cam(cross_cam_dist,
                      cross_cam_fname,
                      class_number,
                      linkage="average",
                      cams=None,
                      mix_rate=0.,
                      jaccard_sim=None):
    cluster_results = OrderedDict()
    print("Start cluster cross camera according to distance")
    if mix_rate > 0:
        assert jaccard_sim is not None, "if mix_rate > 0, the jaccard sim is needed"
        assert cams is not None, "if mix_rate > 0, the cam is needed"
        n = len(cross_cam_fname)
        cams = np.array(cams).reshape((n, 1))
        expand_cams = np.tile(cams, n)
        mask = np.array(expand_cams != expand_cams.T, dtype=np.float32)
        cross_cam_dist -= mask * jaccard_sim * mix_rate
    cross_cam_dist = re_ranking(cross_cam_dist)
    Ag = AgglomerativeClustering(n_clusters=class_number,
                                 affinity="precomputed",
                                 linkage=linkage)
    labels = Ag.fit_predict(cross_cam_dist)
    for fname, label in zip(cross_cam_fname, labels):
        cluster_results[fname] = label
    return cluster_results

def distance_cross_cam(features, use_cpu=False):
    print("Start calculate pairwise distance cross camera")
    n = len(features)
    x = torch.cat(features)
    x = x.view(n, -1)
    if use_cpu:
        dist = 1 - np.matmul(x.cpu().numpy(), x.cpu().numpy().T)
    else:
        dist = 1 - torch.mm(x, x.t())

    return to_numpy(dist)


def distane_per_cam(per_cam_features):
    per_cam_dist = {}
    print("Start calculate pairwise distance per camera")
    for k, features in per_cam_features.items():
        n = len(features)
        x = torch.cat(features)
        x = x.view(n, -1)

        per_cam_dist[k] = 1 - torch.mm(x, x.t())
    return per_cam_dist


def cluster_per_cam(per_cam_dist,
                    per_cam_fname,
                    class_number,
                    linkage="average"):
    cluster_results = {}
    print("Start cluster per camera according to distance")
    for k, dist in per_cam_dist.items():
        cluster_results[k] = OrderedDict()

        # handle the number of samples is small
        dist = dist.cpu().numpy()
        n = dist.shape[0]
        if n < class_number:
            class_number = n // 2
            # double the number of samples
            dist = np.tile(dist, (2, 2))
            per_cam_fname[k] = per_cam_fname[k] + per_cam_fname[k]

        dist = re_ranking(dist)
        Ag = AgglomerativeClustering(n_clusters=class_number,
                                     affinity="precomputed",
                                     linkage=linkage)
        labels = Ag.fit_predict(dist)
        for fname, label in zip(per_cam_fname[k], labels):
            cluster_results[k][fname] = label
    return cluster_results

def merge_list(L):
    lenth = len(L)
    for i in range(1, lenth):
        for j in range(i):
            if L[i]=={0}or L[j]=={0}:
                continue
            x = L[i].union(L[j])
            y = len(L[i])+len(L[j])
            if len(x)<y:
                L[i] = x
                L[j] = {0}
    return [i for i in L if i !={0}]


# 📚
def denoise_labels_zmh(labels, dist):  # per_cam_features :[tensor1,tensor2,...,tensor n]

    dir_count = 0
    dir_count_b = 0

    #select cluster samples corresponding to their class
    label_to_images = {}  #Save the label corresponding index, using a dictionary


    for idx, l in enumerate(labels):
        label_to_images[l] = label_to_images.get(l, []) + [idx]


    c_l_index = {}  #cluster samples table : label to index
    c_l = []  #   cluster samples table
    for l in label_to_images:
        labs = label_to_images[l]  # labels to images
        len_labs = len(labs)  # nums of images
        dist_l = []  # similarities list
        idx1 = []  # index1
        idx2 = []  # index2
        if len_labs > 1:
            for i in range((len_labs - 1)):
                for j in range((i + 1), (len_labs)):
                    dist_l.append(dist[labs[i]][labs[j]])
                    idx1.append(labs[i])
                    idx2.append(labs[j])
            sorted_id = sorted(range(len(dist_l)), key=lambda k: dist_l[k], reverse=False)  #False is in ascending order

            # select any one of the most similar pair of samples as the cluster sample
            a = sorted_id[0]
            r = random.random()
            if r > 0.5:
                c_l_index[l] = idx1[a]  # Index of cluster sample
                c_l.append(idx1[a])
            else:
                c_l_index[l] = idx2[a]
                c_l.append(idx2[a])

        # If there is only one sample, the sample feature is directly set to the cluster feature
        elif len_labs == 1:
            c_l_index[l] = labs[0]
            c_l.append(labs[0])
    print("cluster samples table:")
    print(c_l)
    Set_merge = []
    # Avoid similar samples from being assigned to each other without achieving the effect of being assigned together caous_void
    caous_void = {}
    for l in range(len(np.unique(labels))):  # Walk through each class
        indexs_features = label_to_images[l]
        P = []
        len_c = len(np.unique(labels))  # class number

        #in: index of sample out: cleanliness(rank) probability of sample
        sample_index = c_l_index[l]  # The index of the sample corresponding to the cluster feature of the currently traversed category
        Index_P = []

        for i in indexs_features:
            a = []
            label_a = []
            dict_dist = {}
            Temp_merge = []
            for j in c_l:  #The distance between sample i and sample l is added to the dictionary first
                if labels[j] == l:
                    dict_dist[l] = dist[i][j]
                    break
            for j in c_l:
                a.append(dist[i][j])
                label_a.append(labels[j]) #Labels for each distance
                if labels[j] == l:  #If you see a sample of l again, you skip it, because the distance between i and l already exists in the dictionary
                    continue
                dict_dist[j] = dist[i][j]
            a.append(dist[i][sample_index])  #Splicing, with c+1 similarity
            sorted_id = sorted(range(len_c + 1), key=lambda k: a[k], reverse=False)  #ascending sort
            b = sorted_id.index(len_c) + 1  #Obtain the rank of the class similarity in C +1
            #If I is equidistant from many cluster classes, merge all the equal classes! The classes to be merged are marked and merged after they go out. Normally if i is the most like l, then b should be equal to 2
            if b!= 2 and a[sorted_id[0]] == a[len_c]: #There are other classes that are as distant from i as l is from i.
                print("出现样本i与{}个簇特征一样相似！".format(b-1))
                for i in range(b-1):
                    Temp_merge.append(label_a[sorted_id[i]])    # Remember the preceding cluster class tags by the same distance in Temp Merge
                Set_merge.append(set(Temp_merge))
            P = P + [b]
            # Index of cluster samples that most resemble a class
            c = min(dict_dist, key=dict_dist.get)
            Index_P.append(labels[c]) #Assign the Index of the first ranked class to Index P

        # ranking threshold
        # threshold = 1.00 / (len_c + 1)
        threshold = 2
        for i, j, ll in zip(indexs_features, P, Index_P):  # index of sample i, similarity ranking and label to reassign if i is noise.
            if j > threshold:
                #avoid two class reassign lables each other. If the sample corresponding to LL has allocated its noise data here, if so, there is no need to allocate the noise data here
                if labels[i] in caous_void:
                    if ll in caous_void[labels[i]]:
                        dir_count_b += 1
                        continue
                if ll not in caous_void:
                    caous_void[ll] = [labels[i]]
                else:
                    caous_void[ll].append(labels[i])
                labels[i] = ll
                dir_count += 1

    pecent_dir = (dir_count + dir_count_b) / len(labels)
    print("The percent of noise：{:.2%}, where noise_image_count: {} and {} images reassign each others.".format(pecent_dir, dir_count, dir_count_b))

    return labels, pecent_dir, Set_merge



# 📚
def cluster_per_cam_zmh(per_cam_dist,
                        per_cam_fname,
                        class_number,
                        linkage="average"):
    cluster_results = {}
    print("Start cluster per camera according to distance")
    for k, dist in per_cam_dist.items():
        cluster_results[k] = OrderedDict()
        # handle the number of samples is small
        dist = dist.cpu().numpy()
        n = dist.shape[0]
        if n < class_number:
            class_number = n // 2
            # double the number of samples
            dist = np.tile(dist, (2, 2))
            per_cam_fname[k] = per_cam_fname[k] + per_cam_fname[k]
        cosine_dist = dist
        dist = re_ranking(dist)
        Ag = AgglomerativeClustering(n_clusters=class_number,
                                     affinity="precomputed",
                                     linkage=linkage)
        labels = Ag.fit_predict(dist)
        c_num = len(set(labels))

        # The start of our the pseudo-labels redistribution method
        print("start to denoise")
        resdist = cosine_dist + dist
        n = cosine_dist.shape[0]
        for i in range(n):
          cosine_dist[i][i] == 0
        t_1 = time.time()
        labels, pecent_dir, Set_merge = denoise_labels_zmh(labels, cosine_dist)
        # merge classes which similarity are ranking the same when denoising
        Set_merge = merge_list(Set_merge)  #the set to merge
        for i in Set_merge:
            j = list(i)[0]
            labels = [j if x in i else x
                      for x in labels]
        c_num_later = len(np.unique(labels))
        # assure the labels are correct
        unique_label = np.sort(np.unique(labels))
        if c_num_later < c_num:
            for i in range(len(unique_label)):
                label_now = unique_label[i]
                labels = [i if x == label_now else x for x in labels]
        c_num_last = len(np.unique(labels))
        print("The cluster numbers after denoising is: {}.".format(c_num_last))
        t_2 = time.time()
        time_denoise = t_2 - t_1
        print("denoising time: {} second.".format(time_denoise))
        # The end of our the pseudo-labels redistribution method

        for fname, label in zip(per_cam_fname[k], labels):
            cluster_results[k][fname] = label
    return cluster_results

# 📚
def cluster_cross_cam_zmh(cross_cam_dist,
                      cross_cam_fname,
                      class_number,
                      linkage="average",
                      cams=None,
                      mix_rate=0.,
                      jaccard_sim=None):
    old_dist = cross_cam_dist
    cluster_results = OrderedDict()
    print("Start cluster cross camera according to distance")
    if mix_rate > 0:
        assert jaccard_sim is not None, "if mix_rate > 0, the jaccard sim is needed"
        assert cams is not None, "if mix_rate > 0, the cam is needed"
        n = len(cross_cam_fname)
        cams = np.array(cams).reshape((n, 1))
        expand_cams = np.tile(cams, n)
        mask = np.array(expand_cams != expand_cams.T, dtype=np.float32)
        cross_cam_dist -= mask * jaccard_sim * mix_rate
    cross_cam_dist = re_ranking(cross_cam_dist)
    resdist = cross_cam_dist + old_dist
    Ag = AgglomerativeClustering(n_clusters=class_number,
                                 affinity="precomputed",
                                 linkage=linkage)
    labels = Ag.fit_predict(cross_cam_dist)
    c_num = len(set(labels))

    # The start of our the pseudo-labels redistribution method
    print("start to denoise")
    t_1 = time.time()
    n = resdist.shape[0]
    for i in range(n):
      resdist[i][i] == 0
    labels, pecent_dir, Set_merge = denoise_labels_zmh(labels, resdist)
    #merge classes which similarity are ranking the same when denoising
    Set_merge = merge_list(Set_merge) #the set to merge
    for i in Set_merge:
        if len(list(i))==0:
          continue
        j = list(i)[0]
        labels = [j if x in i else x
                 for x in labels]
    c_num_later = len(np.unique(labels))

    # assure the labels are correct
    unique_label = np.sort(np.unique(labels))
    if c_num_later < c_num:
        for i in range(len(unique_label)):
            label_now = unique_label[i]
            labels = [i if x == label_now else x for x in labels]
    c_num_last = len(np.unique(labels))
    print("The cluster numbers after denoising is: {}.".format(c_num_last))
    t_2 = time.time()
    time_denoise = t_2 - t_1
    print("denoising time: {} second.".format(time_denoise))
    # The end of our the pseudo-labels redistribution method

    for fname, labels in zip(cross_cam_fname, labels):
        cluster_results[fname] = labels
    return cluster_results

def get_intra_cam_cluster_result(model, data_loader, class_number, linkage):
    per_cam_features, per_cam_fname = extract_features_per_cam(
        model, data_loader)

    per_cam_dist = distane_per_cam(per_cam_features)

    # cluster_results = cluster_per_cam(per_cam_dist, per_cam_fname, class_number, linkage)

    #IICS with our denoise method
    #print the sample numbers of each camera
    len_camera = len(per_cam_dist)
    for i in range(len_camera):
         print("The sample numbers of camera:{} are{}".format(i, len(per_cam_features[i])))
    cluster_results = cluster_per_cam_zmh(per_cam_dist, per_cam_fname, class_number, linkage)


    return cluster_results


def get_inter_cam_cluster_result(model,
                                 data_loader,
                                 class_number,
                                 linkage,
                                 mix_rate=0., use_cpu=False ):
    features, fnames, cross_cam_distribute, cams = extract_features_cross_cam(
        model, data_loader)

    cross_cam_distribute = torch.Tensor(np.array(cross_cam_distribute)).cuda()
    cross_cam_dist = distance_cross_cam(features, use_cpu=use_cpu)

    if mix_rate > 0:
        jaccard_sim = jaccard_sim_cross_cam(cross_cam_distribute)
    else:
        jaccard_sim = None


    # IICS
    # cluster_results = cluster_cross_cam(cross_cam_dist,
    #                                     fnames,
    #                                     class_number,
    #                                     linkage=linkage,
    #                                     cams=cams,
    #                                     mix_rate=mix_rate,
    #                                     jaccard_sim=jaccard_sim,
    #                                     )

    #IICS with our denoise method
    cluster_results = cluster_cross_cam_zmh(cross_cam_dist,
                                        fnames,
                                        class_number,
                                        linkage=linkage,
                                        cams=cams,
                                        mix_rate=mix_rate,
                                        jaccard_sim=jaccard_sim,
                                        )


    return cluster_results
