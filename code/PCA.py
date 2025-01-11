import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
import numpy as np
import seaborn as sns
import pandas as pd
import os

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage

from matplotlib.backends.backend_pdf import PdfPages
import tensorflow as tf

from sklearn.metrics.pairwise import euclidean_distances

# ======================== 绘图辅助函数 ========================

# font_path = "D:\\Shanghaitech\\courses\\CS182\\CS182_project\\ShanghaiTech-CS182\\code\\files\\Times New Roman - Bold.ttf"
font_path = "./files/Times New Roman - Bold.ttf"
font_prop = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)

fontdict = {
    "family": "Times New Roman",
    "weight": "bold",
    "size": 24,
}

ticks_fontdict = {
    "family": "Times New Roman",
    "weight": "bold",
    "size": 16,
}

plt_properties = {
    "color": [
        "#00b894",
        "#0984e3",
        "#6c5ce7",
        "#d63031",
        "#fdcb6e",
        "#e84393",
        "#81ecec",
        "#a29bfe",
        "#fab1a0",
        "#636e72",
    ],
    "marker": ["*", "o", "s", "^", ".", "D", "v", "x"],
    "markersize": [16, 10, 10, 10, 10, 10, 10, 10],
}

def plot_lines(
    data,
    file_name=None,
    aspect_ratio=0.666,
    xlabel=None,
    ylabel=None,
    y_range=None,
    x_range=None,
    title=None,
    legend_title=None,
    legend_loc=None,
    legend_ncol=1,
    split_index=None,
    split_marks=None,
    hide_right_top=True,
    fontdict=fontdict,
    ticks_fontdict=ticks_fontdict,
    font_size=None,
    ticks_font_size=None,
):
    if font_size is not None:
        fontdict["size"] = font_size
    if ticks_font_size is not None:
        ticks_fontdict["size"] = ticks_font_size

    fig_width = math.sqrt(24 / aspect_ratio)
    plt.figure(figsize=(fig_width, fig_width * aspect_ratio))  # 创建新图形

    data_items = ["color", "marker", "markersize"]

    for i, value in enumerate(data):
        if "x" not in value:
            value["x"] = range(len(value["y"]))
        for item in data_items:
            if item not in value:
                value[item] = plt_properties[item][i % len(plt_properties[item])]
        plt.plot(
            value["x"],
            value["y"],
            **{k: v for k, v in value.items() if k not in ["x", "y", "label"]},  # 排除 'label'
            label=value.get("label", f"Line {i+1}")
        )
    if y_range is not None:
        plt.ylim(bottom=y_range[0], top=y_range[1])
    if x_range is not None:
        plt.xlim(left=x_range[0], right=x_range[1])
    if xlabel is not None:
        plt.xlabel(xlabel, fontdict=fontdict)
    if ylabel is not None:
        plt.ylabel(ylabel, fontdict=fontdict)
    if title is not None:
        plt.title(title, fontdict=fontdict)
    plt.xticks(**ticks_fontdict)
    plt.yticks(**ticks_fontdict)
    plt.grid(True, alpha=0.3)
    if split_index is not None:
        x_min, x_max = plt.xlim()
        plt.axvspan(
            x_min, split_index, label=split_marks[0], color="lightcoral", alpha=0.3
        )
        plt.axvspan(
            split_index, x_max, label=split_marks[1], color="lightblue", alpha=0.3
        )
    plt.legend(
        title=legend_title,
        loc=legend_loc,
        ncol=legend_ncol,
        title_fontproperties=fontdict,
        prop=ticks_fontdict,
    )  # 显示图例
    if hide_right_top is True:
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"./outputs_PCA/figures/{file_name}.pdf", bbox_inches="tight")
    plt.close()


def plot_heatmap(
    data,
    file_name=None,
    aspect_ratio=0.825,
    xtick_labels=None,
    ytick_labels=None,
    xlabel=None,
    ylabel=None,
    title=None,
    rotation=45,
    cmap="coolwarm",
    colorbar_label=None,
    annotate=True,
    fontdict=fontdict,
    ticks_fontdict=ticks_fontdict,
    font_size=None,
    ticks_font_size=None,
):
    """
    Generates a heatmap given a 2D numpy array `data`.

    Parameters:
    - data: 2D numpy array, where each element represents a value in the heatmap (Z-axis).
    - xlabel: Label for the X-axis.
    - ylabel: Label for the Y-axis.
    - title: Title of the heatmap.
    - xtick_labels: Custom labels for the X-axis.
    - ytick_labels: Custom labels for the Y-axis.
    - rotation: Rotation angle for tick labels.
    - cmap: Color map for the heatmap.
    - colorbar_label: Label for the colorbar.
    - file_name: File name to save the plot.
    - fontdict: Dictionary of font properties for plot labels and titles.
    - annotate: Boolean, whether to annotate the heatmap with data values.

    Returns:
    - A heatmap plot.
    """
    if font_size is not None:
        fontdict["size"] = font_size
    if ticks_font_size is not None:
        ticks_fontdict["size"] = ticks_font_size

    # 定义 X 和 Y 轴的索引
    y_indices = np.arange(data.shape[0])
    x_indices = np.arange(data.shape[1])

    # 创建热力图
    fig_width = math.sqrt(24 / aspect_ratio)
    plt.figure(figsize=(fig_width, fig_width * aspect_ratio))  # 创建新图形

    # 使用 pcolormesh 绘制热力图
    heatmap = plt.pcolormesh(x_indices, y_indices, data, cmap=cmap, shading="auto")

    # 添加颜色条和标签
    if colorbar_label is not None:
        cbar = plt.colorbar(heatmap, label=colorbar_label)
        cbar.set_label(colorbar_label, **fontdict)
    else:
        cbar = plt.colorbar(heatmap)
    cbar.ax.tick_params(labelsize=ticks_fontdict["size"])  # 设置颜色条刻度标签字体大小

    if xlabel is not None:
        plt.xlabel(xlabel, fontdict=fontdict)
    if ylabel is not None:
        plt.ylabel(ylabel, fontdict=fontdict)

    # 设置 X 和 Y 轴的自定义刻度标签并旋转
    if xtick_labels is not None:
        plt.xticks(x_indices+0.5, xtick_labels, rotation=rotation, fontdict=ticks_fontdict)
    if ytick_labels is not None:
        plt.yticks(y_indices+0.5, ytick_labels, rotation=rotation, fontdict=ticks_fontdict)

    plt.title(title, fontdict=fontdict)

    # 如果需要，注释热力图的每个数据值
    if annotate:
        annot_fontdict = {
            "fontsize": ticks_fontdict["size"],
            "color": "white",
            "ha": "center",
            "va": "center",
            "family": ticks_fontdict["family"],
            "weight": ticks_fontdict["weight"],
        }
        for i in y_indices:
            for j in x_indices:
                plt.text(j , i , f"{data[i, j]:.2f}", **annot_fontdict)

    plt.tight_layout()
    plt.savefig(f"./outputs_PCA/figures/{file_name}.pdf", bbox_inches="tight")
    plt.close()


def plot_bar_chart(
    data,
    file_name=None,
    aspect_ratio=0.825,
    bar_width=0.30,
    xlabel=None,
    ylabel=None,
    y_range=None,
    title=None,
    legend_title=None,
    legend_loc=None,
    legend_ncol=1,
    hide_right_top=True,
    fontdict=fontdict,
    ticks_fontdict=ticks_fontdict,
    font_size=None,
    ticks_font_size=None,
):
    if font_size is not None:
        fontdict["size"] = font_size
    if ticks_font_size is not None:
        ticks_fontdict["size"] = ticks_font_size

    categories = data["categories"]
    values = data["values"]

    fig_width = math.sqrt(24 / aspect_ratio)
    plt.figure(figsize=(fig_width, fig_width * aspect_ratio))  # 创建新图形

    x = np.arange(len(categories))

    for i, (label, val) in enumerate(values.items()):
        plt.bar(
            x + (i - 1) * bar_width,
            val,
            width=bar_width,
            label=label,
            color=plt_properties["color"][i % len(plt_properties["color"])],
        )

    if xlabel is not None:
        plt.xlabel(xlabel, fontdict=fontdict)
    if ylabel is not None:
        plt.ylabel(ylabel, fontdict=fontdict)
    if y_range is not None:
        plt.ylim(bottom=y_range[0], top=y_range[1])
    if title is not None:
        plt.title(title, fontdict=fontdict)

    plt.xticks(x, categories, **ticks_fontdict)
    plt.yticks(**ticks_fontdict)
    plt.grid(True, alpha=0.3)
    plt.legend(
        title=legend_title,
        loc=legend_loc,
        ncol=legend_ncol,
        title_fontproperties=fontdict,
        prop=ticks_fontdict,
    )  # 显示图例
    if hide_right_top is True:
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"./outputs_PCA/figures/{file_name}.pdf", bbox_inches="tight")
    plt.close()

# ======================== 主要函数 ========================

def dunn(X, labels):
    """
    计算聚类的 Dunn 指数
    :param X: 数据点矩阵 (n_samples, n_features)
    :param labels: 聚类标签 (n_samples,)
    :return: Dunn 指数
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters < 2:
        raise ValueError("Dunn index is not defined for fewer than 2 clusters.")
    
    # 计算簇内最大距离
    intra_distances = []
    for label in unique_labels:
        cluster_points = X[labels == label]
        if len(cluster_points) > 1:
            intra_distances.append(np.max(euclidean_distances(cluster_points)))
        else:
            intra_distances.append(0)
    max_intra_distance = np.max(intra_distances)

    # 计算簇间最小距离
    inter_distances = []
    for i, label_i in enumerate(unique_labels):
        for j, label_j in enumerate(unique_labels):
            if i < j:
                cluster_i = X[labels == label_i]
                cluster_j = X[labels == label_j]
                inter_distances.append(np.min(euclidean_distances(cluster_i, cluster_j)))
    min_inter_distance = np.min(inter_distances)

    # 计算 Dunn 指数
    return min_inter_distance / max_intra_distance


def set_random_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # For KMeans from scikit-learn, setting random state directly
    import random
    random.seed(seed)


def main():
    set_random_seed(42)
    plots_dir = 'plots'
    figures_dir = "./outputs_PCA/figures/"
    os.makedirs(figures_dir, exist_ok=True)
    
    # 步骤1：数据预处理
    print("步骤1：数据预处理")
    data = pd.read_csv('car_price.csv')
    print("数据加载完成。")
    
    # 1.2 处理分类变量：独热编码（One-hot Encoding）
    categorical_cols = ['symboling', 'fueltype', 'aspiration', 'doornumber',
                        'carbody', 'drivewheel', 'enginelocation',
                        'enginetype', 'cylindernumber', 'fuelsystem']
    
    data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    # 保留car_ID和CarName在前两列
    columns = ['car_ID', 'CarName'] + [col for col in data_encoded.columns if col not in ['car_ID', 'CarName']]
    data_encoded = data_encoded[columns]
    print("独热编码完成。")
    
    # 1.3 标准化数值型特征
    numeric_cols = ['wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight',
                    'enginesize', 'boreratio', 'stroke', 'compressionratio',
                    'horsepower', 'peakrpm', 'citympg', 'highwaympg', 'price']
    
    # scaler = StandardScaler()
    scaler = MinMaxScaler()
    data_encoded[numeric_cols] = scaler.fit_transform(data_encoded[numeric_cols])
    print("数值型特征标准化完成。")
    
    # 1.4 检查并处理异常值
    z_scores = np.abs(stats.zscore(data_encoded[numeric_cols]))
    threshold = 3
    outliers = (z_scores > threshold).sum(axis=0)
    print("异常值数量：")
    print(outliers)
    
    # 替换异常值为中位数
    data_cleaned = data_encoded.copy()
    for col in numeric_cols:
        z = np.abs(stats.zscore(data_cleaned[col]))
        median = data_cleaned[col].median()
        data_cleaned.loc[z > threshold, col] = median
    print("异常值处理完成。")

    # 1.5.1 数值型特征的分布图
    print("生成数值型特征的分布图并保存到单个 PDF 文件...")
    pdf_path_dist = os.path.join(figures_dir, 'numeric_features_distribution.pdf')
    with PdfPages(pdf_path_dist) as pdf:
        fig, axes = plt.subplots(2, 7, figsize=(20, 8))  # 2行7列的布局
        axes = axes.flatten()  # 将子图矩阵展平为1D数组
        for i, col in enumerate(numeric_cols):
            if i < len(axes):
                sns.histplot(data_cleaned[col], kde=True, bins=30, ax=axes[i], color=plt_properties["color"][i % len(plt_properties["color"])])
                # axes[i].set_title(f'Distribution of {col}', fontdict=fontdict)
                axes[i].set_xlabel(col, fontdict=fontdict)
                axes[i].set_ylabel('Frequency', fontdict=fontdict)
                axes[i].tick_params(axis='both', which='major', labelsize=ticks_fontdict["size"])
            else:
                break
        # 删除多余的子图
        for j in range(len(numeric_cols), len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    print(f"数值型特征的分布图已保存到 {pdf_path_dist}。")

    # 1.5.2 异常值检测箱线图
    print("生成数值型特征的箱线图并保存到单个 PDF 文件...")
    pdf_path_box = os.path.join(figures_dir, 'numeric_features_boxplots.pdf')
    with PdfPages(pdf_path_box) as pdf:
        fig, axes = plt.subplots(2, 7, figsize=(20, 8))  # 2行7列的布局
        axes = axes.flatten()  # 将子图矩阵展平为1D数组
        for i, col in enumerate(numeric_cols):
            if i < len(axes):
                sns.boxplot(x=data_cleaned[col], ax=axes[i], color=plt_properties["color"][i % len(plt_properties["color"])])
                # axes[i].set_title(f'Boxplot of {col}', fontdict=fontdict)
                axes[i].set_xlabel(col, fontdict=fontdict)
                axes[i].tick_params(axis='both', which='major', labelsize=ticks_fontdict["size"])
            else:
                break
        # 删除多余的子图
        for j in range(len(numeric_cols), len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    print(f"数值型特征的箱线图已保存到 {pdf_path_box}。")


    # 1.6 保存预处理后的数据
    data_cleaned.to_csv('processed.csv', index=False)
    print("预处理后的数据已保存为 'processed.csv'")
    
    # 步骤2：相关性分析
    print("\n步骤2：相关性分析")
    processed_data = pd.read_csv('processed.csv')
    
    # 2.2 计算相关性矩阵
    corr_matrix = processed_data[numeric_cols].corr()
    print("相关性矩阵计算完成。")
    
    # 2.3 绘制相关性热力图
    print("绘制相关性热力图并保存...")
    heatmap_data = corr_matrix.values
    plot_heatmap(
        data=heatmap_data,
        file_name='correlation_heatmap',
        aspect_ratio=0.825,
        xtick_labels=corr_matrix.columns.tolist(),
        ytick_labels=corr_matrix.columns.tolist(),
        xlabel='Features',
        ylabel='Features',
        title='Correlation Heatmap',
        rotation=45,
        cmap='coolwarm',
        colorbar_label='Correlation Coefficient',
        annotate=True,
        font_size=20,
        ticks_font_size=4,
    )
    print("相关性热力图已保存。")
    
    # 2.4 删除高度相关的特征
    print("删除高度相关的特征...")
    threshold_corr = 0.85
    to_drop = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold_corr:
                to_drop.add(corr_matrix.columns[i])
    
    print("需要删除的特征：", to_drop)
    
    # 根据业务意义选择保留的特征
    # 这里以删除 'carlength' 和 'horsepower' 为例，实际操作中请根据具体业务选择
    features_to_drop = list(to_drop)
    print(f"删除的特征：{features_to_drop}")
    
    processed_deleted = processed_data.drop(columns=features_to_drop)
    print("高度相关的特征已删除。")
    
    # 更新numeric_cols_deleted
    numeric_cols_deleted = [col for col in numeric_cols if col not in features_to_drop]
    
    # 保存处理后的数据
    processed_deleted.to_csv('processed_deleted.csv', index=False)
    print("处理后的数据已保存为 'processed_deleted.csv'")
    
    # 步骤3：降维处理
    print("\n步骤3：降维处理")
    processed_deleted = pd.read_csv('processed_deleted.csv')
    
    # 3.1 提取特征列
    features = processed_deleted.drop(columns=['car_ID', 'CarName'])
    
    # 3.2 PCA降维
    print("进行PCA降维...")
    pca = PCA(n_components=0.95, random_state=42)
    principal_components = pca.fit_transform(features)
    pca_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(principal_components.shape[1])])
    pca_df = pd.concat([processed_deleted[['car_ID', 'CarName']], pca_df], axis=1)
    pca_df.to_csv('pca.csv', index=False)
    print("PCA降维后的数据已保存为 'pca.csv'")
    
    # 步骤4：聚类分析
    print("\n步骤4：聚类分析")
    pca_data = pd.read_csv('pca.csv')
    pca_features = pca_data.drop(columns=['car_ID', 'CarName'])
    
    # 4.1 K-means聚类
    print("进行K-means聚类分析...")
    sse = {}
    for k in range(1, 21):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(pca_features)
        sse[k] = kmeans.inertia_
    
    # 使用 plot_lines 绘制肘部图
    print("绘制肘部图并保存...")
    elbow_data = [{"y": list(sse.values()), "label": "SSE"}]
    plot_lines(
        data=elbow_data,
        file_name="elbow_method",
        aspect_ratio=0.825,
        xlabel="Number of clusters (K)",
        ylabel="SSE",
        x_range=(1, 30),
        y_range=(min(sse.values()), max(sse.values())),
        title="Elbow Method for Optimal K",
        legend_title="Metrics",
        legend_loc="best",
        legend_ncol=1,
        hide_right_top=True,
        font_size=20,
        ticks_font_size=16,
    )
    print("肘部图已保存。")
    
    # 根据肘部图选择K值，这里假设K=8
    optimal_k = 8
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(pca_features)
    pca_data['KMeans_Cluster'] = kmeans_labels
    #pca_data.to_csv('kmeans_clusters.csv', index=False)
    print("K-means聚类结果已保存为 'kmeans_clusters.csv'")

    # 替换 K-means 聚类结果可视化部分
    print("使用 t-SNE 可视化 K-means 聚类结果...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    tsne_results = tsne.fit_transform(pca_features)

    # 创建 t-SNE 数据框
    tsne_df = pd.DataFrame(data=tsne_results, columns=['TSNE1', 'TSNE2'])
    tsne_df['KMeans_Cluster'] = kmeans_labels

    # 可视化 t-SNE 降维后的 K-means 聚类结果
    # 由于 plot_lines 主要用于折线图，这里仍需使用原始绘图方式
    # 但应用统一的字体和样式
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=tsne_df, x='TSNE1', y='TSNE2',
                    hue='KMeans_Cluster', palette='Set1', legend='full', ax=ax)
    ax.set_title('K-means Clustering Results (2D t-SNE) using PCA features', fontdict=fontdict)
    ax.set_xlabel('t-SNE Dimension 1', fontdict=fontdict)
    ax.set_ylabel('t-SNE Dimension 2', fontdict=fontdict)
    ax.legend(title='Cluster', loc='best', prop=ticks_fontdict)
    if True:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'PCA_kmeans_tsne_2d.pdf'), bbox_inches="tight")
    plt.close()
    print("t-SNE K-means 聚类结果图已保存。")

        
    # 4.2 基于层次的聚类分析
    print("进行基于层次的聚类分析...")
    linked = linkage(pca_features, method='ward')
    
    # 绘制树状图
    print("绘制树状图并保存...")
    fig, ax = plt.subplots(figsize=(10, 7))
    dendrogram(linked,
               orientation='top',
               distance_sort='descending',
               show_leaf_counts=False,
               ax=ax)
    ax.set_title('Hierarchical Clustering Dendrogram (using PCA features)', fontdict=fontdict)
    ax.set_xlabel('Sample index', fontdict=fontdict)
    ax.set_ylabel('Distance', fontdict=fontdict)
    ax.tick_params(axis='both', which='major', labelsize=ticks_fontdict["size"])

    ax.set_xticks([])
    
    if True:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'hierarchical_dendrogram.pdf'), bbox_inches="tight")
    plt.close()
    print("树状图已保存。")
    
    # 选择K=8进行剪切
    agg_clustering = AgglomerativeClustering(n_clusters=optimal_k, metric='euclidean', linkage='ward')
    agg_labels = agg_clustering.fit_predict(pca_features)
    pca_data['Agglomerative_Cluster'] = agg_labels
    #pca_data.to_csv('agglomerative_clusters.csv', index=False)
    print("基于层次的聚类结果已保存为 'agglomerative_clusters.csv'")
    
    # 可视化层次聚类结果
    print("使用 t-SNE 可视化 层次聚类 结果...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    tsne_results = tsne.fit_transform(pca_features)

    # 创建 t-SNE 数据框
    tsne_df = pd.DataFrame(data=tsne_results, columns=['TSNE1', 'TSNE2'])
    tsne_df['Agglomerative_Cluster'] = agg_labels

    # 可视化 t-SNE 降维后的 层次聚类 结果
    # 由于 plot_lines 主要用于折线图，这里仍需使用原始绘图方式
    # 但应用统一的字体和样式
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=tsne_df, x='TSNE1', y='TSNE2',
                    hue='Agglomerative_Cluster', palette='Set1', legend='full', ax=ax)
    ax.set_title('HC Clustering Results (2D t-SNE) using PCA features', fontdict=fontdict)
    ax.set_xlabel('t-SNE Dimension 1', fontdict=fontdict)
    ax.set_ylabel('t-SNE Dimension 2', fontdict=fontdict)
    ax.legend(title='Cluster', loc='best', prop=ticks_fontdict)
    if True:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'PCA_hc_tsne_2d.pdf'), bbox_inches="tight")
    plt.close()
    print("t-SNE HC 聚类结果图已保存。")

    # 步骤5：聚类效果评估（轮廓系数、CH 指数、DB 指数、Dunn 指数）
    print("\n步骤5：聚类效果评估")

    # 计算 Silhouette Score
    kmeans_silhouette = silhouette_score(pca_features, pca_data['KMeans_Cluster'])
    agg_silhouette = silhouette_score(pca_features, pca_data['Agglomerative_Cluster'])

    # 计算 Calinski-Harabasz 指数
    kmeans_ch = calinski_harabasz_score(pca_features, pca_data['KMeans_Cluster'])
    agg_ch = calinski_harabasz_score(pca_features, pca_data['Agglomerative_Cluster'])

    # 计算 Davies-Bouldin 指数
    kmeans_db = davies_bouldin_score(pca_features, pca_data['KMeans_Cluster'])
    agg_db = davies_bouldin_score(pca_features, pca_data['Agglomerative_Cluster'])

    # 计算 Dunn 指数
    # 由于 Dunn 指数计算量较大，可能需要一定时间
    print("正在计算 Dunn 指数，这可能需要一些时间...")
    kmeans_dunn = dunn(pca_features.values, pca_data['KMeans_Cluster'].values)
    agg_dunn = dunn(pca_features.values, pca_data['Agglomerative_Cluster'].values)

    # 输出结果
    print("聚类效果评估指标：")
    print(f'K-means: Silhouette Score = {kmeans_silhouette:.4f}, CH Score = {kmeans_ch:.4f}, DB Score = {kmeans_db:.4f}, Dunn Index = {kmeans_dunn:.4f}')
    print(f'Agglomerative Clustering: Silhouette Score = {agg_silhouette:.4f}, CH Score = {agg_ch:.4f}, DB Score = {agg_db:.4f}, Dunn Index = {agg_dunn:.4f}')

    original_data = pd.read_csv('car_price.csv')
    # 统计 K-means 聚类结果的每个簇的汽车数量及价格均值
    print("\n基于 K-means 聚类结果的分类簇统计分析：")
    cluster_stats_kmeans = []
    for cluster in range(optimal_k):
        cluster_data = pca_data[pca_data['KMeans_Cluster'] == cluster]
        cluster_car_ids = cluster_data['car_ID'].tolist()
        
        # 获取原始数据中的价格信息
        cluster_prices = original_data[original_data['car_ID'].isin(cluster_car_ids)]['price']
        cluster_size = len(cluster_car_ids)
        cluster_mean_price = cluster_prices.mean()
        cluster_stats_kmeans.append((cluster, cluster_size, cluster_mean_price))
        
        print(f"K-means 簇 {cluster}: 汽车数量 = {cluster_size}, 平均价格 = {cluster_mean_price:.2f}")

    # 检查 K-means 聚类结果中 Volkswagen 所属簇的分布情况
    vw_cars_kmeans = pca_data[pca_data['CarName'].str.contains('volkswagen', case=False, na=False)]
    if not vw_cars_kmeans.empty:
        print("\n基于 K-means 聚类结果的 Volkswagen 分布统计：")
        vw_clusters_kmeans = vw_cars_kmeans['KMeans_Cluster'].unique()
        for vw_cluster in vw_clusters_kmeans:
            vw_cluster_data = vw_cars_kmeans[vw_cars_kmeans['KMeans_Cluster'] == vw_cluster]
            vw_car_ids = vw_cluster_data['car_ID'].tolist()
            vw_prices = original_data[original_data['car_ID'].isin(vw_car_ids)]['price']
            vw_mean_price = vw_prices.mean()
            print(f"K-means 簇 {vw_cluster}: 包含的 Volkswagen 数量 = {len(vw_car_ids)}, Volkswagen 平均价格 = {vw_mean_price:.2f}")

            # 筛选与 Volkswagen 同簇且价格波动不超过10%的车型
            competitors_data_kmeans = pca_data[pca_data['KMeans_Cluster'] == vw_cluster]
            competitors_ids_kmeans = competitors_data_kmeans['car_ID'].tolist()
            competitors_prices_kmeans = original_data[original_data['car_ID'].isin(competitors_ids_kmeans)]
            
            # competitors_within_price_range_kmeans = competitors_prices_kmeans[
            #     (competitors_prices_kmeans['price'] >= 0.9 * vw_mean_price) &
            #     (competitors_prices_kmeans['price'] <= 1.1 * vw_mean_price)
            # ]
            
            competitors_within_price_range_kmeans = competitors_prices_kmeans[
                (competitors_prices_kmeans['price'] >= 0)
            ]
            
            print(f"K-means 簇 {vw_cluster}: 竞品车型数量 = {len(competitors_within_price_range_kmeans)}")
            print(competitors_within_price_range_kmeans[['car_ID', 'CarName', 'price']])

            final_competitors_kmeans_df = competitors_within_price_range_kmeans.copy()
            #final_competitors_kmeans_df.to_csv(f'PCA_final_competitors_kmeans_cluster{vw_cluster}.csv', index=False)

    # 统计层次聚类结果的每个簇的汽车数量及价格均值
    print("\n基于层次聚类结果的分类簇统计分析：")
    cluster_stats_hierarchical = []
    for cluster in range(optimal_k):
        cluster_data = pca_data[pca_data['Agglomerative_Cluster'] == cluster]
        cluster_car_ids = cluster_data['car_ID'].tolist()
        
        # 获取原始数据中的价格信息
        cluster_prices = original_data[original_data['car_ID'].isin(cluster_car_ids)]['price']
        cluster_size = len(cluster_car_ids)
        cluster_mean_price = cluster_prices.mean()
        cluster_stats_hierarchical.append((cluster, cluster_size, cluster_mean_price))
        
        print(f"层次聚类 簇 {cluster}: 汽车数量 = {cluster_size}, 平均价格 = {cluster_mean_price:.2f}")

    # 检查层次聚类结果中 Volkswagen 所属簇的分布情况
    vw_cars_hierarchical = pca_data[pca_data['CarName'].str.contains('volkswagen', case=False, na=False)]
    if not vw_cars_hierarchical.empty:
        print("\n基于层次聚类结果的 Volkswagen 分布统计：")
        vw_clusters_hierarchical = vw_cars_hierarchical['Agglomerative_Cluster'].unique()
        for vw_cluster in vw_clusters_hierarchical:
            vw_cluster_data = vw_cars_hierarchical[vw_cars_hierarchical['Agglomerative_Cluster'] == vw_cluster]
            vw_car_ids = vw_cluster_data['car_ID'].tolist()
            vw_prices = original_data[original_data['car_ID'].isin(vw_car_ids)]['price']
            vw_mean_price = vw_prices.mean()
            print(f"层次聚类 簇 {vw_cluster}: 包含的 Volkswagen 数量 = {len(vw_car_ids)}, Volkswagen 平均价格 = {vw_mean_price:.2f}")

            # 筛选与 Volkswagen 同簇且价格波动不超过10%的车型
            competitors_data_hierarchical = pca_data[pca_data['Agglomerative_Cluster'] == vw_cluster]
            competitors_ids_hierarchical = competitors_data_hierarchical['car_ID'].tolist()
            competitors_prices_hierarchical = original_data[original_data['car_ID'].isin(competitors_ids_hierarchical)]
            
            # competitors_within_price_range_hierarchical = competitors_prices_hierarchical[
            #     (competitors_prices_hierarchical['price'] >= 0.9 * vw_mean_price) &
            #     (competitors_prices_hierarchical['price'] <= 1.1 * vw_mean_price)
            # ]
            
            competitors_within_price_range_hierarchical = competitors_prices_hierarchical[
                (competitors_prices_hierarchical['price'] >= 0)
            ]
            
            print(f"层次聚类 簇 {vw_cluster}: 竞品车型数量 = {len(competitors_within_price_range_hierarchical)}")
            print(competitors_within_price_range_hierarchical[['car_ID', 'CarName', 'price']])

            final_competitors_hierarchical_df = competitors_within_price_range_hierarchical.copy()
            #final_competitors_hierarchical_df.to_csv(f'PCA_final_competitors_hierarchical_cluster{vw_cluster}.csv', index=False)


    print("K-means 和层次聚类的竞品车型分析结果已分别保存")

    print("\n所有步骤已完成。")

if __name__ == "__main__":
    main()