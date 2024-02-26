import numpy as np
import networkx as nx

wM = np.load("./Expert/wIndex.npy").astype(np.int64)
# [3,1024,1024]
# 获得W1，W2，W3的权重矩阵
W1 = wM[0]
W2 = wM[1]
W3 = wM[2]

# print(W1.shape)

# def get_weight_matrix():
# 读取图的权值并可视化

# print(W1)
# 随机创建一个对称矩阵，值为整数
# W1 = np.random.randint(0, 10, (24, 24))
# print(W1)
# 将W1对称化
# W1 = (W1 + W1.T) / 2
# 转为整型
# W1 = W1.astype(np.int64)
# print(W1)

def graph_clustering(weight_matrix, n_clusters):
    # 将权重取负，以便找到最大生成树
    negative_weight_matrix = -1 * weight_matrix
    # 创建图
    G = nx.from_numpy_matrix(negative_weight_matrix)
    # 找到最小生成树，此时的最小生成树实际上是原始权重下的最大生成树
    T = nx.minimum_spanning_tree(G)
    # 找到最大的n-1条边
    edges = sorted(T.edges(data=True), key=lambda t: t[2].get('weight', 1))
    # 去掉最大的n-1条边
    for i in range(n_clusters - 1):
        T.remove_edge(*edges[i][:2])
    # 找到连通分量，即划分好的神经元组
    groups = list(nx.connected_components(T))
    # 创建最后的结果矩阵
    result = np.zeros((n_clusters, weight_matrix.shape[0]))
    for i, group in enumerate(groups):
        for node in group:
            result[i, node] = 1
    return result

# index = graph_clustering(W1, 8)
# print(index.shape)

# 将result拼接为1维，其中值代表其属于第几个专家
def get_index(result):
    index = []
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if result[i][j] == 1:
                index.append(i)
    # return np.array(index)
    return index

result = graph_clustering(W1, 8)
print(result)
index = get_index(result)
print(index)
# print(index.shape)
# print(index)

# 保存index到文件中，以换行分割每个元素
np.savetxt('./Expert/index.txt', index, fmt='%d', newline='\n')


