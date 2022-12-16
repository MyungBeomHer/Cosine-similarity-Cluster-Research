import math
import sys
import os
import numpy as np
from numpy import dot
from numpy.linalg import norm
from operator import itemgetter, attrgetter

def get_data(file):
    data = []
    with open(file, 'r') as f:
        for line in f:
            data.append(line)

    return data


def read_data_ground_truth(file):
    data = []
    ground_truth = []
    with open(file, 'r') as f:
        for line in f:
            tmp1, *tmp2 = list(map(float, line.split()))
            data.append(tmp2)
            ground_truth.append(tmp1)
    return ground_truth, data


def cos_sim(A, B):
    return dot(A, B) / (norm(A) * norm(B))


# ------------------------------------------------------------------------------------------------------------#
# Ground truth는 각 gene이 어느 클러스터에 들어있는지만 보여줌.
ground_truth, data = read_data_ground_truth("project_input.txt")
absolute_value = list()
pred_cluster = list()

for d in data:
    ab_d = []
    cur = d[0]
    for i in range(len(d)):
        if i == 0:
            ab_d.append(cur)
        else:
            prev = cur
            cur = round(prev * d[i], 5)
            ab_d.append(cur)
    absolute_value.append(ab_d)

# print(*absolute_value, sep='\n')
# print(len(absolute_value))
# print(*data, sep='\n')

# 두 gene의 cos 유사도를 사용한 유사성 check
total_cos_res = []

A = absolute_value[0]  # 제일 첫번째 ID 값 저장
# 2개씩 비교함.

cos_res = []

for idx_B, B in enumerate(absolute_value):
    avg_cos = 0
    cos = []
    # cos 유사도 계산
    for i in range(len(A) - 1):
        doc1 = np.array([i, A[i], i + 1, A[i + 1]])
        doc2 = np.array([i, B[i], i + 1, B[i + 1]])
        # cos 유사도 계산 (2개 사이의)
        result = round(cos_sim(doc1, doc2), 5)
        cos.append(result)
        # 평균
    avg_cos = round(sum(cos) / len(cos), 5)
    total_cos_res.append([idx_B, avg_cos])

###Cluster 중심점 지정
sort_list = sorted(total_cos_res,key=itemgetter(1))
label = 10  ##label의 갯수 지정
interV = (len(sort_list) - 1)/ label

#app = []
#print(*sort_list,sep='\n')
idx_list = [0] * label
for i in range(label):
    idx_list[i] = sort_list[round(interV * i)][0]

min = math.inf
min_index = 0
total_cos_res_ans = []
###제일 유사한 Cluster 중심점을 찾음
for idx_B, B in enumerate(absolute_value):
    max_cluster_val = 0
    max_cluster_idx = 1
    #cluster 중심점들과의 거리 계산
    for l in range(len(idx_list)):
        avg_cos = 0
        cos = []
        for i in range(len(A) - 1):
            doc1 = np.array([i, absolute_value[idx_list[l]][i], i + 1, absolute_value[idx_list[l]][i + 1]])
            doc2 = np.array([i, B[i], i + 1, B[i + 1]])
            # cos 유사도 계산 (2개 사이의)
            result = round(cos_sim(doc1, doc2), 5)
            cos.append(result)

        # 평균
        avg_cos = round(sum(cos) / len(cos), 5)
        if max_cluster_val < avg_cos:
            max_cluster_val = avg_cos
            max_cluster_idx = l
    total_cos_res_ans.append([idx_B, max_cluster_idx])

# ------------------------------------------------------------------------------------------------------------#
# ------------------------------------------------MB----------------------------------------------------------#
"""
# def classifier(리스트,분류해주는 레이블의 갯수,최대 cos 유사도,최소 cos 유사도):
def classifier(total_cos_res, thresh_hold):
    a = []
    for i in range(len(total_cos_res)):
        for j in range(len(thresh_hold) - 1):
            if thresh_hold[j] <= total_cos_res[i][1] <= thresh_hold[j + 1]:
                # print(thresh_hold[j],"<=" ,total_cos_res[i][1] ,"<=" ,thresh_hold[j + 1])
                a.append([i, j + 1])  # ID번호, 레이블 번호
                break
    return a


# def threshold(최소 cos 유사도값, 레이블의 갯수)
def threshold(min, label):
    # 자기 자신 - 최솟 값 / 레이블 갯수로 범위를 지정
    interval = round((1 - min) / label, 5)
    thres_hold = [0] * (label + 1)

    for i in range(label):
        thres_hold[i] = min + interval * i
    thres_hold[label] = 1.00000
    return thres_hold


label = 10  ##label의 갯수 지정
thresh_hold_interval = threshold(min, label)

ans = classifier(total_cos_res, thresh_hold_interval)
"""
print(*total_cos_res_ans, sep='\n')

with open("project_output.txt", 'w', encoding='cp949') as file:
    for i in range(len(total_cos_res_ans)):
        sentence = str(total_cos_res_ans[i][1]) + '\n'
        file.writelines(sentence)

