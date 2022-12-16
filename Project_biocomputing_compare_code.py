import sys
import os
import numpy as np

def get_data(file):
    data = []
    with open(file, 'r') as f:
        for line in f:
            data.append(line)
    
    return data

def read_data_EC(file):
    data = []
    ground_truth = []
    with open(file, 'r') as f:
        for line in f:
            tmp1, *tmp2 = list(map(float, line.split()))
            data.append(tmp2)
            ground_truth.append(tmp1)
    return ground_truth, data

# Precision
def precision(ground_truth, prediction): # 각 cluster를 비교

    ground_truth_set = set(ground_truth)
    prediction_set = set(prediction)

    inter_sec = ground_truth_set.intersection(prediction_set)
    inter_sec_len = len(inter_sec)
    prediction_len = len(prediction_set)

    if prediction_len == 0:
        return 0

    return (inter_sec_len / prediction_len)

# Recall
def recall(ground_truth, prediction):

    ground_truth_set = set(ground_truth)
    prediction_set = set(prediction)

    inter_sec = ground_truth_set.intersection(prediction_set)
    inter_sec_len = len(inter_sec)
    ground_truth_len = len(ground_truth_set)

    if ground_truth_len == 0:
        return 0
    
    return (inter_sec_len / ground_truth_len)

# f-score
def f1score(ground_truth, prediction):
    prec = precision(ground_truth, prediction)
    rec = recall(ground_truth, prediction)

    if (prec + rec) == 0:
        return 0

    f1 = 2 * ((prec * rec)/(prec + rec))

    return f1

# ------------------------------------------------------------------------------------------------------------#
# Ground truth는 각 gene이 어느 클러스터에 들어있는지만 보여줌.
ground_truth, data = read_data_EC("project_input.txt")
for i in range(len(ground_truth)):
    ground_truth[i] = int(ground_truth[i])

ground_truth_kind = list(set(ground_truth))
# print(ground_truth)
# print(*data, sep='\n')
print("Ground_truth_kind = ", ground_truth_kind)

# ground_truth 값을 gene index로 분류되게 저장 (compare 하기 쉽게 변환)
ground_truth_cluster = [[] for _ in range(len(ground_truth_kind))]

for i in range(len(ground_truth_kind)):
    for idx, clu in enumerate(ground_truth):
        if clu == ground_truth_kind[i]:
            ground_truth_cluster[i].append(str(idx))

# outlier인 cluster 삭제
del ground_truth_cluster[-1]

# print(*ground_truth_cluster, sep='\n')
# ------------------------------------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------------------------------------#
'''
total_len_sum = 0
for i in ground_truth_cluster:
    print(len(i))
    total_len_sum += len(i)

print(total_len_sum)
'''
# ------------------------------------------------------------------------------------------------------------#
# Result 반환 -> prediction value -> 위의 ground_truth 값과 유사한 형태로 만듦.
prediction_result = get_data("project_output.txt")
for i in range(len(prediction_result)):
    prediction_result[i] =  prediction_result[i].replace('\n', '')
    prediction_result[i] =  int(prediction_result[i])

prediction_result_kind = list(set(prediction_result))
prediction_result_kind.sort()
#print("Prediction_result_kind = ", prediction_result_kind)

# 1~10, 마지막 cluster는 outlier값들이 모여있음.
pred_cluster = [[] for _ in range(len(prediction_result_kind))]

for i in range(len(prediction_result_kind)):
    for idx, clu in enumerate(prediction_result):
        if clu == prediction_result_kind[i]:
            pred_cluster[i].append(str(idx))

# print(*pred_cluster, sep='\n')
# -------------------------------------------------------------------------------------------------------------#

total_list = [pred_cluster]
total_name_list = ['project_pred_value']
i = 0

f_score_list = list()
f_score_avg_list = list()

for name in total_list:
    for p in name:
        f_score_max = 0
        for g in ground_truth_cluster:
            f1_score = f1score(g, p)
            if f1_score > f_score_max:
                f_score_max = f1_score

        f_score_list.append(f_score_max)

    #print(total_name_list[i] + "_f_score_list_length = ", len(f_score_list))
    f_score_avg = sum(f_score_list) / len(f_score_list)
    i += 1

    f_score_avg_list.append(f_score_avg)
    #print("f_score_result : ", f_score_avg)
    f_score_list.clear()

#print("Similarity : ", round(f_score_avg_list[0] * 100, 5), "%")

with open("project_result_compare.txt", 'w', encoding='cp949') as file:
    title = 'Compare each Algorithm using f-score'
    file.write(title)
    file.write('\n\n')
    #print("flag2")
    for idx, result in enumerate(f_score_avg_list):
        sentence = total_name_list[idx] + ' : ' + str(round(result * 100, 5)) + '%'
        print(str(round(result * 100, 5)))
        file.write(sentence)
        file.write('\n')




