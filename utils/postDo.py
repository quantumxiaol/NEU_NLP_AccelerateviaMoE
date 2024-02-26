import numpy as np
wM = np.load("./Expert/wIndex.npy").astype(np.int64)
n = wM.shape[0]
N = wM.shape[-1]
index = 0
for w in wM:
    fileName = "./Expert/w" + str(index) + ".txt"
    f = open(fileName, "w")
    index += 1
    nEdge = 0
    for i in range(N):
        for j in range(N):
            if w[i, j] != 0:
                nEdge += 1
    f.write(str(N) + " " + str(nEdge) + " " + "001\n")
    for i in range(N):
        for j in range(N):
            if w[i, j] != 0:
                f.write(str(j) + " " + str(w[i, j]) + " ")
        f.write("\n")
    f.close()
