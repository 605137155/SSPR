
#import 25 images of 5 person

from __future__ import print_function, absolute_import
import re
import shutil
from collections import OrderedDict
from glob import glob

import torch
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
import os.path as osp


# identities = []
# all_pids = {}
# exdir = r'C:\Users\Minghui Zhang\Desktop\论文源代码\IICS_with_denoise'
# images_dir = r'C:\Users\Minghui Zhang\Desktop\论文源代码\IICS_with_denoise\降噪可视化行人图像_25'
# def register(subdir, pattern=re.compile(r'([-\d]+)_c(\d)')):
#     fpaths = sorted(glob(osp.join(exdir, subdir, '*.jpg')))
#     pids = set()
#     fnames = []
#     for fpath in tqdm(fpaths):
#         fname = osp.basename(fpath)
#         pid, cam = map(int, pattern.search(fname).groups())
#         assert 1 <= cam <= 8
#         cam -= 1
#         if pid not in all_pids:
#             all_pids[pid] = len(all_pids)
#         pid = all_pids[pid]
#         pids.add(pid)
#         if pid >= len(identities):
#             assert pid == len(identities)
#             identities.append([[] for _ in range(8)])  # 8 camera views
#         fname = ('{:08d}_{:02d}_{:04d}.jpg'.format(
#             pid, cam, len(identities[pid][cam])))
#         identities[pid][cam].append(fname)
#         fnames.append(fname)
#         shutil.copy(fpath, osp.join(images_dir, fname))
#     return pids, fnames
#
#
# trainval_pids, filenames = register(images_dir)
# print(trainval_pids) #5 class
# print(filenames) #25 filenames




#get mini25Duke data
from reid.cluster_utils.rerank import re_ranking
from reid.utils.serialization import load_checkpoint

root = r'C:\Users\Minghui Zhang\Desktop\论文源代码\IICS_with_denoise\example\data'
from reid import datasets
from example.iics import get_data
dataset, num_classes, train_loader, val_loader, test_loader = \
    get_data('miniDukeMTMC', split_id=0, data_dir=root, height=256,
             width=128, batch_size=64, workers=0,
             )

#model
# Create model
from reid import models
model = models.create("ft_net_inter_resnet",
                      num_classes=5, stride=1)


# Load from checkpoint

model.model.load_param(r'C:\Users\Minghui Zhang\Desktop\论文源代码\IICS_with_denoise\checkpoint\resnet50-19c8e357.pth')

# checkpoint_continue = load_checkpoint(osp.join(r'C:\Users\Minghui Zhang\Desktop\论文源代码\IICS_with_denoise\example\logs', 'model_best.pth.tar'))
# new_state_dict = {}
# for k in checkpoint_continue['state_dict'].keys():
#     if 'model' in k:
#         new_state_dict[k] = checkpoint_continue['state_dict'][k]
# model.load_state_dict(new_state_dict, strict=False)



#extract_features_cross_camera
from reid.cluster_utils.cluster import extract_features_cross_cam, distance_cross_cam, jaccard_sim_cross_cam, \
    denoise_labels_zmh, merge_list

features, fnames, cross_cam_distribute, cams = extract_features_cross_cam(
    model, train_loader)
cross_cam_dist = distance_cross_cam(features)
cross_cam_distribute = torch.Tensor(np.array(cross_cam_distribute))
jaccard_sim = jaccard_sim_cross_cam(cross_cam_distribute)

cluster_results = OrderedDict()
print("Start cluster cross camera according to distance")
old_distance = cross_cam_dist
n = len(fnames)
cams = np.array(cams).reshape((n, 1))
expand_cams = np.tile(cams, n)
mask = np.array(expand_cams != expand_cams.T, dtype=np.float32)
cross_cam_dist -= mask * jaccard_sim * (0.0195**0.6)

cross_cam_dist = re_ranking(cross_cam_dist)

Ag = AgglomerativeClustering(n_clusters=5,
                                 affinity="precomputed",
                                 linkage='average')
labels = Ag.fit_predict(cross_cam_dist)

print(labels)
c_num = len(set(labels))
resdist = old_distance + cross_cam_dist
import time
# 降噪算法开始
print("开始降噪")
t_1 = time.time()
n = cross_cam_dist.shape[0]
for i in range(n):
    cross_cam_dist[i][i] == 0
labels, pecent_dir, Set_merge = denoise_labels_zmh(labels, cross_cam_dist)

#合并类似的类
# Set_merge = merge_list(Set_merge) #合并有交集的集合
# for i in Set_merge:
#         if len(list(i))==0:
#           continue
#         j = list(i)[0]
#         labels = [j if x in i else x
#                  for x in labels]
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


print(labels)

print(features)
print(fnames)

