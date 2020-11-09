import numpy as np
import random
U = 3
I = 4
K = int((U + I)/2)

def matrixFactoration(D, beta, alpha, iterations):
    W = np.random.rand( U, K)
    H = np.random.rand( K, I)
    for j in range(iterations):
        for u in range(U):
            for i in range(I):
                r = D[u][i]
                if r <=0: continue
                r1 = 0
                for k in range(K):
                    r1 += W[u][k] * H[k][i]
                eui = r - r1
                for k in range(K):
                    W[u][k] += beta * (eui * H[k][i] - alpha * W[u][k])
                    H[k][i] += beta * (eui * W[u][k] - alpha * H[k][i])
    return W, H
Dtrain = np.random.randint(5, size=(U, I))
W,H = matrixFactoration(Dtrain, 0.02, 0.1, 100000)
print(Dtrain)
print(np.dot(W,H))

#D   : user-item rating matrix
#K     : number of latent dimensions
#alpha: learning rate
#beta : delta x

