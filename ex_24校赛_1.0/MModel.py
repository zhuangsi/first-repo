import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class MModel:
    def __init__(self, problem_id=1):
        if problem_id == 1:
            self.M_f = np.load('data/M_f1.npy')
            self.M_d = np.load('data/M_d1.npy')
        elif problem_id == 2:
            self.M_f = np.load('data/M_f2.npy')
            self.M_d = np.load('data/M_d2.npy')
            self.coordinates = np.load('data/coordinates2.npy')
        else:
            raise ValueError('Invalid problem_id')

        self.max_node = self.M_f.shape[0]
        self.labels = None
        self.centers = None

    def reset_by_Kmeans(self, num_clusters=3):
        M_c = self.M_f * self.M_d

        # 使用K-means聚类
        kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init='auto').fit(M_c)
        labels = kmeans.labels_

        # 找到每个簇的中心节点
        centers = np.array([np.argmin(np.sum(self.M_d[labels == i], axis=0)) for i in range(num_clusters)])

        self.centers = centers
        self.labels = labels

    def H(self, x):
        cluster_label = self.labels[x]
        return self.centers[cluster_label]

    def compute_MD(self, X=1.0):
        # 计算每个节点到其中心节点的距离
        distances_to_centers = self.M_d[np.arange(self.max_node), self.centers[self.labels]]

        # 计算中心节点之间的距离矩阵
        center_distances = self.M_d[np.ix_(self.centers, self.centers)] * X

        # 计算 MD 矩阵
        MD = distances_to_centers[:, np.newaxis] + center_distances[self.labels][:, self.labels] + distances_to_centers

        return MD

    def compute_MD_slow(self, X=1.0):
        # 计算每个节点到其中心节点的距离
        distances_to_centers = self.M_d[np.arange(self.max_node), self.centers[self.labels]]

        # 计算中心节点之间的距离矩阵
        center_distances = self.M_d[np.ix_(self.centers, self.centers)] * X

        # 计算 MD 矩阵
        MD = np.zeros((self.max_node, self.max_node))
        for i in range(self.max_node):
            for j in range(self.max_node):
                MD[i, j] = (distances_to_centers[i] +
                            center_distances[self.labels[i], self.labels[j]] +
                            distances_to_centers[j])
        return MD

    def change(self):
        change_mode = np.random.randint(0, 2)
        if change_mode == 0:  # 交换不同类别的节点
            change_center_index = np.random.choice(np.arange(len(self.centers)), size=2)
            change_node1 = np.random.choice(np.where(self.labels == change_center_index[0])[0])
            change_node2 = np.random.choice(np.where(self.labels == change_center_index[1])[0])
            self.labels[change_node1] = change_center_index[1]
            self.labels[change_node2] = change_center_index[0]

        else:  # 同类别内改变中心
            change_center_index = np.random.choice(np.arange(len(self.centers)))
            change_node = np.random.choice(np.where(self.labels == change_center_index)[0])
            self.centers[change_center_index] = change_node

    def objective_function(self, X=1.0):
        # 计算 M_c
        M_c = self.M_f * self.compute_MD(X)
        return np.sum(M_c)  # 可以使用其他适当的目标函数

    def simulated_annealing(self, initial_temp=1000, min_temp=1, alpha=0.95, max_iterations=1000, X=1.0):
        current_temp = initial_temp
        best_labels = self.labels.copy()
        best_centers = self.centers.copy()
        best_cost = self.objective_function()

        for iteration in range(max_iterations):
            # 生成新状态
            self.change()
            new_cost = self.objective_function(X)

            # 计算接受概率
            if new_cost < best_cost or np.random.rand() < np.exp((best_cost - new_cost) / current_temp):
                best_labels = self.labels.copy()
                best_centers = self.centers.copy()
                best_cost = new_cost

            # 降低温度
            current_temp = max(min_temp, alpha * current_temp)

            # 每迭代一定次数输出当前最优值
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Cost = {best_cost}")

        self.labels = best_labels
        self.centers = best_centers
        print(f"Final cost: {best_cost}")

        return best_cost

    def plot_result(self):
        plt.figure()
        for i in range(len(self.centers)):
            plt.scatter(self.coordinates[self.labels == i, 0], self.coordinates[self.labels == i, 1])
            plt.scatter(self.coordinates[self.centers[i], 0], self.coordinates[self.centers[i], 1], marker='x', c='r')
        plt.show()


if __name__ == '__main__':
    model = MModel(problem_id=1)
    model.reset_by_Kmeans(num_clusters=3)
    MD = model.compute_MD(X=0.8)
    MD_ = model.compute_MD_slow(X=0.8)

    print(np.allclose(MD, MD_))



