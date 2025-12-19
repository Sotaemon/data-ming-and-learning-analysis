#encode=utf-8
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def main():
    iris = pd.read_csv('./../data/iris.csv')

    x = iris.drop(labels='Species', axis=1)
    print(x)
    print(x.shape)
    iris.head()

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(x)
    x['cluster'] = kmeans.labels_
    x.cluster.value_counts()

    centers = kmeans.cluster_centers_

    # 使用 scatterplot 代替 lmplot
    plt.figure(figsize=(10,6))
    sns.scatterplot(
        x='Petal.Length',
        y='Petal.Width',
        hue='cluster',
        data=x,
        alpha=0.8,
        palette='bright'
    )
    plt.scatter(centers[:, 2], centers[:, 3], marker='*', color='black', s=130)
    plt.xlabel('花瓣长度')
    plt.ylabel('花瓣宽度')
    plt.legend(title='Cluster')
    plt.title('K-Means 聚类分析')
    plt.show()

    # 增加一个辅助列，将不同的花种映射到0,1,2三种值，目的方便后面图形的对比
    iris['Species_map'] = iris.Species.map({'virginica': 0, 'setosa': 1, 'versicolor': 2})

    # 绘制原始数据三个类别的散点图，同样使用 scatterplot
    plt.figure(figsize=(10,6))
    sns.scatterplot(
        x='Petal.Length',
        y='Petal.Width',
        hue='Species_map',
        data=iris,
        alpha=0.8,
        palette='bright'
    )
    plt.xlabel('花瓣长度')
    plt.ylabel('花瓣宽度')
    plt.legend(title='Species')
    plt.title('原始数据类别分布')
    plt.show()

if __name__=='__main__':
    main()