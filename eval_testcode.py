import numpy as np
import random

def ap(gt, rec, k):
    rec = rec[:k]
    
    precision = []
    p = 0.0
    true_count = 0
    for i in range(1,len(rec)+1):
        if rec[i-1] in gt:
            true_count += 1
            p = p*(i-1)/i+1/i
            precision.append(p)
        else:
            p = p*(i-1)/i    
    
    if true_count == 0:
        return 0
    
    ap = sum(precision)/len(precision)

    return ap

########################################
def ndcg(rel, k, idealrel):
    idcgs = [sum((idealrel[i] / np.log2(i + 2) for i in range(l+1))) for l in range(k)]
    rel = rel[:k]
    
    dcg = 0.0
    for i in range(k):
        dcg += rel[i] / np.log2(i + 2)

    #print("dcg:", dcg)
    #print("idcg:", idcgs[k-1])
    return dcg / idcgs[k-1]
########################################
gt1 = [1,3,4,5,6,10]
rec1 = [1,2,3,4,5,6,7,8,9,10]

gt2 = [2,5,6,7,9,10]
rec2 = [1,2,3,4,5,6,7,8,9,10]

answer1 = []
for k in range(1,len(rec1)+1):
    answer1.append(ap(gt1,rec1,k))

answer2 = []
for k in range(1,len(rec2)+1):
    answer2.append(ap(gt2,rec2,k))

#print(answer1)
#print(answer2)

idealrel = [3,3,2,2,1,0]
predictrel = [3,2,3,0,1,2]

#print("ndcg:",ndcg(predictrel, 6, idealrel))

############

songs = [i for i in range(1,101)]
ground_truth = random.sample(songs,10)

for k in [5,10,20,100]:
    map = 0.0
    for _ in range(1000):
        recommend = random.sample(songs,k)
        map += ap(ground_truth,recommend,k)
    map/=1000

    m_ndcg = 0.0
    for _ in range(1000):
        rel = [1 if i in ground_truth else 0 for i in range(1,101)]
        ideal = ([1 for _ in range(min(10,k))] + [0 for _ in range(100-min(10,k))])[:k]
        m_ndcg += ndcg(rel,k,ideal)
    m_ndcg/=1000
    
    print("MAP:",round(map,4),"NDCG:",round(m_ndcg,4),"k:",k)
        
