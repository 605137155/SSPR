
#the loss of IICS at  inter-camera stage is softmax cross entroy loss and triplet loss [10].


#单后cos降噪
loss_Triple_c = [0.467,0.438,0.383,0.328,0.389,0.280,0.249,0.145,0.199,0.228,   0.255,0.099,0.161,0.098,0.185,0.156,0.072,0.142,0.060,0.127,    0.115,0.082,0.071,0.150,0.100,0.162,0.082,0.069,0.042,0.035,   0.015,0.082,0.034,0.0001,0.053,0.064,0.015,0.052,0.025,0.039]
#单后rerank
loss_Triple_b = [0.483,0.363,0.228,0.245,0.135,0.129,0.089,0.183,0.132,0.102,   0.127,0.158,0.090,0.095,0.122,0.103,0.075,0.095,0.025,0.025,    0.166,0.080,0.071,0.039,0.050,0.032,0.064,0.025,0.067,0.111,   0.044,0.068,0.039,0.032,0.022,0.040,0.001,0.013,0.024,0.050]

#单后resdist降噪

loss_Entropy_a = [1.156,0.843,0.308,0.504,0.412,0.207,0.157,0.209,0.171,0.227,  0.124,0.051,0.244,0.172,0.144,0.108,0.123,0.154,0.207,0.191,    0.336,0.216,0.025,0.215,0.160,0.121,0.077,0.044,0.090,0.117,     0.106,0.131,0.048,0.088,0.260,0.309,0.099,0.055,0.084,0.108]
loss_Triple_a = [0.464,0.390,0.197,0.286,0.138, 0.115,0.073,0.157,0.098,0.142,  0.060,0.034,0.063,0.110,0.031,0.062,0.025,0.032,0.045,0.065,    0.131,0.074,0.006,0.026,0.009,0.062,0.032,0.008,0.043,0.026,     0.053,0.030,0.025,0.004,0.044,0.003,0.016,0.015,0.017,0.016 ]


#原论文
loss_Entropy = [0.608,0.418,0.466,0.552,0.517,0.261,0.259,0.189,0.102,0.230,   0.143,0.199,0.176,0.256,0.180,0.268,0.195,0.341,0.122,0.164,   0.195,0.079,0.058,0.052,0.124,0.242,0.088,0.091,0.240,0.048,     0.118,0.082,0.208,0.069,0.083,0.109,0.171,0.146,0.237,0.069]
loss_Triple = [0.415,0.250,0.241,0.257,0.306,0.178,0.228,0.142,0.066,0.184,    0.077,0.126,0.072,0.099,0.099,0.081,0.076,0.081,0.087,0.071,   0.084,0.036,0.021,0.036,0.075,0.063,0.017,0.037,0.035,0.044,     0.023,0.082,0.079,0.009,0.042,0.036,0.041,0.041,0.025,0.013]


target = loss_Triple
sum = 0
for i in target:
    sum +=i
print(sum)



#原论文loss_Entropy sum=8.155
#原论文loss_Triple sum=3.975
#loss_Entropy sum=8.401
#本loss_Triple sum=3.226999

import matplotlib.pyplot as plt

#绘画曲线
x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]
fig, axes = plt.subplots(1, 1, figsize=(8, 4))
# 折线图
y = loss_Triple_a
y2 = loss_Triple
y3 = loss_Triple_b
y4 = loss_Triple_c
axes.plot(x, y2, label='no_denoise_triplet loss',linestyle='-', color='b', marker='*', linewidth=1.5)
axes.plot(x, y4, label='cosine_denoised_triplet loss',linestyle='-', color='y', marker='p', linewidth=1.5)
axes.plot(x, y3, label='reranking_denoised_triplet loss',linestyle='-', color='c', marker='o', linewidth=1.5)
axes.plot(x, y, label='resdist_denoised_triplet loss', linestyle='-', color='r', marker='.', linewidth=1.5)

# 设置x、y轴标签
axes.set_ylabel("Loss")
axes.set_xlabel("Training Epoch")
# 设置y轴的刻度
axes.set_yticks([0.00, 0.25, 0.5])

# plt.legend(['resdist_denoised_triploss','no_denoise_triploss'])
plt.legend(['no_denoise_triplet loss','cosine_denoised_triplet loss','reranking_denoised_triplet loss','resdist_denoised_triplet loss'])
# 展示图片
plt.show()