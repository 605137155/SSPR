from __future__ import print_function, absolute_import
from collections import OrderedDict
import torch
import torch.nn.functional as F
from reid.feature_extraction import extract_cnn_feature
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from .rerank import re_ranking
import math
import random
import time

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
                      eph,
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
    t1 = time.time()
    cross_cam_dist = re_ranking(cross_cam_dist)
    t2 = time.time()
    t_rerank = t2-t1
    print("reranking时间为{}".format(t_rerank))
    tri_mat = np.triu(cross_cam_dist, 1)
    tri_mat = tri_mat[np.nonzero(tri_mat)]
    tri_mat = np.sort(tri_mat, axis=None)
    top_num = np.round(eph * tri_mat.size).astype(int)
    eps = tri_mat[top_num]
    print(eps)

    Ag = AgglomerativeClustering(n_clusters=None,
                                 affinity="precomputed",
                                 linkage=linkage,
                                 distance_threshold=eps)
    labels = Ag.fit_predict(cross_cam_dist)
    print(len(set(labels)))
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
                    eph,
                    linkage="average"):
    cluster_results = {}
    print("Start cluster per camera according to distance")
    for k, dist in per_cam_dist.items():
        cluster_results[k] = OrderedDict()

        # handle the number of samples is small
        dist = dist.cpu().numpy()
        n = dist.shape[0]
        if n < eph:
            eph = n // 2
            # double the number of samples
            dist = np.tile(dist, (2, 2))
            per_cam_fname[k] = per_cam_fname[k] + per_cam_fname[k]

        dist = re_ranking(dist)
        tri_mat = np.triu(dist, 1)
        tri_mat = tri_mat[np.nonzero(tri_mat)]
        tri_mat = np.sort(tri_mat, axis=None)
        top_num = np.round(eph * tri_mat.size).astype(int)
        eps = tri_mat[top_num]
        print(eps)

        Ag = AgglomerativeClustering(n_clusters=None,
                                     affinity="precomputed",
                                     linkage=linkage,
                                     distance_threshold=eps)
        labels = Ag.fit_predict(dist)
        print(len(set(labels)))
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



##📚📚
def cluster_per_cam_zmh(per_cam_dist,
                        per_cam_fname,
                        eph,
                        linkage="average"):
    cluster_results = {}
    print("Start cluster per camera according to distance")
    for k, dist in per_cam_dist.items():
        cluster_results[k] = OrderedDict()

        # handle the number of samples is small
        dist = dist.cpu().numpy()
        n = dist.shape[0]
        if n < eph:
            eph = n // 2
            # double the number of samples
            dist = np.tile(dist, (2, 2))
            per_cam_fname[k] = per_cam_fname[k] + per_cam_fname[k]
        cosine_dist = dist
        
        
        dist = re_ranking(dist)
        tri_mat = np.triu(dist, 1)
        tri_mat = tri_mat[np.nonzero(tri_mat)]
        tri_mat = np.sort(tri_mat, axis=None)
        top_num = np.round(eph * tri_mat.size).astype(int)
        eps = tri_mat[top_num]
        print(eps)
        resdist = dist + cosine_dist
        

        Ag = AgglomerativeClustering(n_clusters=None,
                                     affinity="precomputed",
                                     linkage=linkage,
                                     distance_threshold=eps)
        labels = Ag.fit_predict(dist)

        c_num = len(set(labels))
        print(c_num)
        # 这里可以加入改进,input为
        resdist = cosine_dist + dist
        # 降噪算法开始
        print("开始降噪")
        n = cosine_dist.shape[0]
        for i in range(n):
          cosine_dist[i][i] == 0
        t_1 = time.time()

        #input:labels and dist, output:redistributed labels
        labels, pecent_dir, Set_merge = denoise_labels_zmh(labels, cosine_dist)

        
        # 合并类似的类
        Set_merge = merge_list(Set_merge)  # 合并有交集的集合
        for i in Set_merge:
            if len(list(i))==0:
              print("有空集！！！")
            j = list(i)[0]
            labels = [j if x in i else x
                      for x in labels]
        c_num_later = len(np.unique(labels))
        # 如果聚类后数量减少，则重新编排标签使得标签完整而不会训练出错
        unique_label = np.sort(np.unique(labels))
        if c_num_later < c_num:
            for i in range(len(unique_label)):
                label_now = unique_label[i]
                labels = [i if x == label_now else x for x in labels]
        c_num_last = len(np.unique(labels))
        print("聚類后簇的類為{}個".format(c_num_last))
        t_2 = time.time()
        time_denoise = t_2 - t_1
        print("降噪时间为:{}秒".format(time_denoise))
        # 降噪算法结束

        for fname, label in zip(per_cam_fname[k], labels):
            cluster_results[k][fname] = label
    return cluster_results



##📚📚
def cluster_cross_cam_zmh(cross_cam_dist, features,
                          cross_cam_fname,
                          eph,
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
    cosine_dist = cross_cam_dist
    time1 = time.time()
    cross_cam_dist = re_ranking(cross_cam_dist)
    time2 = time.time()
    print("re_ranking时间为:{}分钟".format((time2-time1)/60))

    tri_mat = np.triu(cross_cam_dist, 1)
    tri_mat = tri_mat[np.nonzero(tri_mat)]
    tri_mat = np.sort(tri_mat, axis=None)
    top_num = np.round(eph * tri_mat.size).astype(int)
    eps = tri_mat[top_num]
    print(eps)

    Ag = AgglomerativeClustering(n_clusters=None,
                                 affinity="precomputed",
                                 linkage=linkage,
                                 distance_threshold=eps)
    labels = Ag.fit_predict(cross_cam_dist)

    resdist = cosine_dist + cross_cam_dist
    n = resdist.shape[0]
    for i in range(n):
        resdist[i][i] == 0


    time3 = time.time()

    c_num = len(set(labels))
    
    # 降噪算法开始
    print(c_num)
    print("开始降噪")
    t_1 = time.time()
    # input:labels and dist, output:redistributed labels
    labels, pecent_dir, Set_merge = denoise_labels_zmh(labels, resdist)

    # 合并类似的类
    Set_merge = merge_list(Set_merge)  # 合并有交集的集合
    for i in Set_merge:
        if len(list(i))==0:
          print("有空集！！！")
        j = list(i)[0]
        labels = [j if x in i else x
                  for x in labels]
    c_num_later = len(np.unique(labels))
    # 如果聚类后数量减少，则重新编排标签使得标签完整而不会训练出错
    unique_label = np.sort(np.unique(labels))
    if c_num_later < c_num:
        for i in range(len(unique_label)):
            label_now = unique_label[i]
            labels = [i if x == label_now else x for x in labels]
    c_num_last = len(np.unique(labels))
    print("聚類后簇的類為{}個".format(c_num_last))
    t_2 = time.time()
    time_denoise = t_2 - t_1
    print("降噪时间为:{}秒".format(time_denoise))
    # 降噪算法结束
    time4 = time.time()

    print("降噪耗时:{}分钟".format((time4-time3)/60))
    print(len(set(labels)))
    for fname, label in zip(cross_cam_fname, labels):
        cluster_results[fname] = label
    return cluster_results


# 📚 our method
def denoise_labels_zmh(labels, dist):  # per_cam_features :[tensor1,tensor2,...,tensor n]

    dir_count = 0
    dir_count_b = 0


    # 计算每种标签对应的簇特征
    label_to_images = {}  # 保存标签对应索引，用字典
    for idx, l in enumerate(labels):
        label_to_images[l] = label_to_images.get(l, []) + [idx]

    # 通过reranking_dist 寻找簇特征
    c_l_index = {}  # 作为簇特征的样本的索引
    c_l = []  # 簇特征对应的样本list
    for l in label_to_images:
        labs = label_to_images[l]  # 标签对应图像的idx
        len_labs = len(labs)  # 图像的数量
        dist_l = []  # 距离列表
        idx1 = []  # 索引1
        idx2 = []  # 索引2
        if len_labs > 1:
            for i in range((len_labs - 1)):
                for j in range((i + 1), (len_labs)):
                    dist_l.append(dist[labs[i]][labs[j]])
                    idx1.append(labs[i])
                    idx2.append(labs[j])
            sorted_id = sorted(range(len(dist_l)), key=lambda k: dist_l[k], reverse=False)  # False为升序，这里的平均特征为最像的特征

            # 将最近两个特征求平均方法
            # a = sorted_id[0]
            # feature_avg[l] = (pc_f[idx1[a]] + pc_f[idx2[a]]) / 2

            # 任意挑选最近两个特征的单特征作为簇特征方法
            a = sorted_id[0]
            r = random.random()
            if r > 0.5:
                c_l_index[l] = idx1[a]  # 簇特征的索引
                c_l.append(idx1[a])
            else:
                c_l_index[l] = idx2[a]
                c_l.append(idx2[a])
            # 指数个特征的平均特征方法
            # a = sorted_id[:int(math.log2(len(sorted_id)))]
            # features_sum = torch.FloatTensor(2048)
            # for id in a:
            #     features_sum += (pc_f[idx1[id]] + pc_f[idx2[id]]) / 2
            # feature_avg[l] = features_sum/len(a)

        # 假如只有一个样本，则将该样本特征直接设为该簇类特征
        elif len_labs == 1:
            c_l_index[l] = labs[0]
            c_l.append(labs[0])

    Set_merge = []
    # 避免互相相似的样本被互相分配而达不到分配到一起的效果 caous_void
    caous_void = {}
    for l in range(len(np.unique(labels))):  # 遍历每个类
        indexs_features = label_to_images[l]
        P = []
        len_c = len(np.unique(labels))  # 类数

        # 基于dist计算样本干净概率  in:样本的索引  out:样本的干净概率
        sample_index = c_l_index[l]  # 当前遍历类别的簇特征对应样本的索引
        Index_P = []

        for i in indexs_features:
            a = []
            label_a = []
            dict_dist = {}
            Temp_merge = []
            for j in c_l:  # 优先将样本i跟l对应样本的距离加到字典
                if labels[j] == l:
                    dict_dist[l] = dist[i][j]
                    break
            for j in c_l:
                a.append(dist[i][j])
                label_a.append(labels[j]) #每个距离对应的标签
                if labels[j] == l:  # 如果再次遇到l对应样本，则跳过，因为字典里已经存在了i与l的距离
                    continue
                dict_dist[j] = dist[i][j]
            a.append(dist[i][sample_index])  # 拼接，此时有c+1个相似度
            sorted_id = sorted(range(len_c + 1), key=lambda k: a[k], reverse=False)  # 升序排序
            b = sorted_id.index(len_c) + 1  # 获得该类相似度在c+1个中的排位
            #如果i与很多簇类距离相等，则把相等的类都合并！要合并的类标记出来,出去之后再合并。正常来说若i最像l，则b应该等于2
            if b > 2 and a[sorted_id[0]] == a[len_c]: #这是说明存在其他类与i的距离和l与i的一样！
                print("出现样本i与{}个簇特征一样最相似！".format(b-1))
                for i in range(b-1):
                    Temp_merge.append(label_a[sorted_id[i]])    #将排在前面相同距离的簇类标签记住在Temp_merge
                Set_merge.append(set(Temp_merge))
            P = P + [b]
            #最像一类的簇样本索引
            c = min(dict_dist, key=dict_dist.get)
            Index_P.append(labels[c]) #将排位第一的类的索引赋给Index_P

        # 相似度排位阈值
        # threshold = 1.00 / (len_c + 1)
        threshold = 2
        for i, j, ll in zip(indexs_features, P, Index_P):  # 对该类的每个样本的索引i及其对应特征的干净概率
            if j > threshold:
                # 判断是否ll对应的样本已经分配其噪声数据到此，若有则不需要再将此处的噪声数据分配过去
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
    print("筛错比例为：{:.2%},共有{}张被筛错,其中有{}张互相分配。".format(pecent_dir, dir_count, dir_count_b))
    return labels, pecent_dir, Set_merge




def get_intra_cam_cluster_result(model, data_loader, eph, linkage):
    per_cam_features, per_cam_fname = extract_features_per_cam(
        model, data_loader)

    per_cam_dist = distane_per_cam(per_cam_features)

    # 原论文方法
    cluster_results = cluster_per_cam(per_cam_dist, per_cam_fname, eph, linkage)
    # 改进方法
    # 打印每个摄像机的样本
    # len_camera = len(per_cam_dist)
    # for i in range(len_camera):
    #     print("摄像机:{}的样本数为{}个".format(i, len(per_cam_features[i])))
    # cluster_results = cluster_per_cam_zmh(per_cam_dist, per_cam_fname, eph, linkage)

    return cluster_results


def get_inter_cam_cluster_result(model,
                                 data_loader,
                                 eph,
                                 linkage,
                                 mix_rate=0., use_cpu=False):
    features, fnames, cross_cam_distribute, cams = extract_features_cross_cam(
        model, data_loader)

    cross_cam_distribute = torch.Tensor(np.array(cross_cam_distribute)).cuda()
    cross_cam_dist = distance_cross_cam(features, use_cpu=use_cpu)

    t1 = time.time()
    if mix_rate > 0:
        jaccard_sim = jaccard_sim_cross_cam(cross_cam_distribute)
    else:
        jaccard_sim = None
    t2 = time.time()
    ttt = t2-t1
    print("计算jaccard相似度的时间:{}秒".format(ttt))
    # 原文算法
    cluster_results = cluster_cross_cam(cross_cam_dist,
                                        fnames,
                                        eph,
                                        linkage=linkage,
                                        cams=cams,
                                        mix_rate=mix_rate,
                                        jaccard_sim=jaccard_sim,
                                        )

    # 改进算法 with our method

    # cluster_results = cluster_cross_cam_zmh(cross_cam_dist,features,
    #                                     fnames,
    #                                     eph,
    #                                     linkage=linkage,
    #                                     cams=cams,
    #                                     mix_rate=mix_rate,
    #                                     jaccard_sim=jaccard_sim,
    #                                     )
    return cluster_results
