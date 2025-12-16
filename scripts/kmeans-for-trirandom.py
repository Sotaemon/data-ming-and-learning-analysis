import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics

def plot_SSE(x, clusters):
    K = range(1, clusters + 1)
    TSSE = []
    for k in K:
        SSE = []
        kmeans = KMeans(k)
        kmeans.fit(x)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        for label in set(labels):
            SSE.append(np.sum((x.loc[labels == label,] - centers[label, :]) ** 2))
        TSSE.append(np.sum(SSE))
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.style.use('ggplot')
    plt.plot(K, TSSE, 'b*-')
    plt.xlabel('图2   簇的个数K')
    plt.ylabel('簇内离差平方和之和（SSE）')
    plt.show()

def k_silhouette(x, clusters):
    KK = range(2, clusters + 1)
    S = []
    for k in KK:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(x)
        labels = kmeans.labels_
        S.append(metrics.silhouette_score(x, labels, metric='euclidean'))
    arg_max = np.array(S).argmax()
    num = int(KK[arg_max])
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.style.use('ggplot')
    plt.plot(KK, S, 'b*-')
    plt.xlabel('图3簇的个数K')
    plt.ylabel('silhouette_score')
    plt.text(KK[arg_max], S[arg_max], '最佳k值为%s' % num)
    plt.show()
    return num

def main():
    np.random.seed(1234)
    mean_1 = [0.5, 0.5]
    cov_1 = [[0.3, 0], [0, 0.3]]
    x_1, y_1 = np.random.multivariate_normal(mean_1, cov_1, 1000).T

    mean_2 = [0, 8]
    cov_2 = [[1.5, 0], [0, 1]]
    x_2, y_2 = np.random.multivariate_normal(mean_2, cov_2, 1000).T

    mean_3 = [8, 4]
    cov_3 = [[1.5, 0], [0, 1]]
    x_3, y_3 = np.random.multivariate_normal(mean_3, cov_3, 1000).T

    plt.scatter(x_1, y_1)
    plt.scatter(x_2, y_2)
    plt.scatter(x_3, y_3)
    plt.show()

    group1 = np.column_stack((x_1, y_1))
    group2 = np.column_stack((x_2, y_2))
    group3 = np.column_stack((x_3, y_3))
    data = np.vstack([group1, group2, group3])
    x = pd.DataFrame(data, columns=['x', 'y'])

    plot_SSE(x, 15)

    k_num = k_silhouette(x, 15)

    kmeans = KMeans(n_clusters=k_num, random_state=1234)
    kmeans.fit(x)
    labels = kmeans.labels_

    plt.figure(figsize=(8, 6))
    plt.scatter(x.iloc[labels == 0, 0], x.iloc[labels == 0, 1], s=50, c='red', label='Cluster 1')
    plt.scatter(x.iloc[labels == 1, 0], x.iloc[labels == 1, 1], s=50, c='blue', label='Cluster 2')
    plt.scatter(x.iloc[labels == 2, 0], x.iloc[labels == 2, 1], s=50, c='green', label='Cluster 3')
    plt.legend()
    plt.xlabel('图4  数据的聚类结果')
    plt.show()

if __name__=='__main__':
    main()