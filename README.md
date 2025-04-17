## Gene Clustering   

팀원 : [허명범](https://github.com/MyungBeomHer), [양지웅], [현승민]

### 프로젝트 주제 (연세대학교 바이오 컴퓨팅 수업)
Cosine 유사도를 이용한 클러스팅 분류기를 통해 시계열 데이터를 분류

### 프로젝트 언어 및 환경
프로젝트 언어 : Pytorch

### Run 
```bash
project_biocomputing_multiple.py
Project_biocomputing_compare_code.py
```

### Algorithm
<p align="center">
  <img src="/figure/model.png" width=100%> <br>
</p>

```
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
     
```
[project_biocomputing_multiple.py](project_biocomputing_multiple.py)

## Result
### Overall Accuracy 
<p align="center">
  <img src="/figure/acc.png" width=100%> <br>
</p>

### validation loss 
<p align="center">
  <img src="/figure/loss.png" width=100%> <br>
</p>

### Confusion Matrix 
<p align="center">
  <img src="/figure/confusion matrix.png" width=100%> <br>
</p>

