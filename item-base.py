# import cac thu vien
import numpy as np
from numpy import dot
from numpy.linalg import norm

#
userRating = np.array([[4.0, 2.0, 0.0, 5.0, 4.0], [5.0, 3.0, 4.0, 0.0, 3.0], [3.0, 0.0, 4.0, 4.0, 3.0]])
print(userRating)
similarityUser = np.zeros((userRating.shape[1], userRating.shape[1]))

#
def knn_search(D, K):
    ind = np.argpartition(D, -K)[-K:]
    return ind

for i in range(userRating.shape[1]):
    for j in range(i, userRating.shape[1]):
        a = []
        b = []
        for k in range(userRating.shape[0]):
            if(userRating[k][i] != 0) and (userRating[k][j] != 0):
                a.append(userRating[k][i])
                b.append(userRating[k][j])

        similarityUser[i][j] = dot(a,b)/(norm(a) * norm(b))
        similarityUser[j][i] = similarityUser[i][j]

print(similarityUser)
result = userRating
for j in range (userRating.shape[1]):
    for i in range(userRating.shape[0]):
        if userRating[i][j] == 0:
            tu = 0
            mau = -1
            neig_idx  = knn_search(similarityUser[i], 3)
            for k in neig_idx:
                tu += similarityUser[j][k] * userRating[i][k]
                mau += similarityUser[j][k]
            result[i][j] = tu/mau
print(result)
