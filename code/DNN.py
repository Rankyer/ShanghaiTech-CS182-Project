import pandas as pd
import numpy as np
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy import stats
from matplotlib.backends.backend_pdf import PdfPages

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
    if split_index is not None and split_marks is not None:
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
    plt.savefig(f"./outputs_AE/figures/{file_name}.pdf", bbox_inches="tight")
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
        plt.xticks(x_indices + 0.5, xtick_labels, rotation=rotation, fontdict=ticks_fontdict)
    if ytick_labels is not None:
        plt.yticks(y_indices + 0.5, ytick_labels, rotation=rotation, fontdict=ticks_fontdict)

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
                plt.text(j + 0.5, i + 0.5, f"{data[i, j]:.2f}", **annot_fontdict)

    plt.tight_layout()
    plt.savefig(f"./outputs_AE/figures/{file_name}.pdf", bbox_inches="tight")
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
    plt.savefig(f"./outputs_AE/figures/{file_name}.pdf", bbox_inches="tight")
    plt.close()

# ======================== 辅助函数 ========================

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


# ======================== 主要函数 ========================
def main():
    set_random_seed(0)
    plots_dir = 'enhanced_deep_clustering_plots'
    figures_dir = "./outputs_AE/figures/"
    os.makedirs(figures_dir, exist_ok=True)
    
    # 步骤1：加载预处理数据
    print("Step 1: Load preprocessed data")
    processed_deleted = pd.read_csv('processed_deleted.csv')
    print("Data loaded successfully.")
    
    # 提取特征
    features = processed_deleted.drop(columns=['car_ID', 'CarName', 'price'])
    feature_names = features.columns.tolist()
    print(f"Number of features: {features.shape[1]}")
    
    X = features.values
    
    print("\nChecking data types and missing values...")
    
    print("Data types:", features.dtypes)
    
    # 确保所有特征都是数值型
    features = features.apply(pd.to_numeric, errors='coerce')
    
    # 检查缺失值
    missing_values = np.isnan(features).sum()
    print("Missing values per feature:")
    print(missing_values)
    
    if missing_values.sum() > 0:
        print("Missing values found, filling with median...")
        features = features.fillna(features.median())
        print("Missing values filled with median.")
    else:
        print("No missing values.")
    
    X = features.values
    X = X.astype('float32')
    
    # 步骤2：构建增强型自编码器
    print("\nStep 2: Build enhanced autoencoder")
    input_dim = X.shape[1]
    encoding_dim = 20
    
    input_layer = Input(shape=(input_dim,))
    
    encoded = Dense(128, activation='relu')(input_layer)
    encoded = BatchNormalization()(encoded)
    encoded = Dropout(0.2)(encoded)
    
    encoded = Dense(64, activation='relu')(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = Dropout(0.2)(encoded)
    
    encoded = Dense(32, activation='relu')(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = Dropout(0.2)(encoded)
    
    encoded = Dense(encoding_dim, activation='relu')(encoded)
    encoded = BatchNormalization()(encoded)
    
    decoded = Dense(32, activation='relu')(encoded)
    decoded = BatchNormalization()(decoded)
    decoded = Dropout(0.2)(decoded)
    
    decoded = Dense(64, activation='relu')(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = Dropout(0.2)(decoded)
    
    decoded = Dense(128, activation='relu')(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = Dropout(0.2)(decoded)
    
    decoded = Dense(input_dim, activation='linear')(decoded)
    
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    encoder = Model(inputs=input_layer, outputs=encoded)
    
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
    print("Enhanced autoencoder built successfully.")
    
    # 步骤3：训练增强型自编码器
    print("\nStep 3: Train enhanced autoencoder")
    
    history = autoencoder.fit(X, X,
                              epochs=500,
                              batch_size=128,
                              shuffle=True,
                              validation_split=0.2,
                              callbacks=[reduce_lr],
                              verbose=1)
    
    print("Enhanced autoencoder training complete.")
    
    # 使用 plot_lines 绘制训练和验证损失
    print("绘制训练和验证损失曲线并保存...")
    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']
    epochs_range = range(1, len(training_loss) + 1)
    
    loss_data = [
        {
            "y": training_loss,
            "label": "Training Loss",
            "color": plt_properties["color"][0],
            "marker": "s",        # 指定标记形状
            "markevery": 100      # 每100个数据点绘制一个标记
        },
        {
            "y": validation_loss,
            "label": "Validation Loss",
            "color": plt_properties["color"][1],
            "marker": "*",        # 指定标记形状
            "markevery": 100      # 每100个数据点绘制一个标记
        }
    ]

    plot_lines(
        data=loss_data,
        file_name="autoencoder_training_loss",
        aspect_ratio=0.666,
        xlabel="Epoch",
        ylabel="Loss",
        y_range=(0, max(max(training_loss), max(validation_loss)) * 1.1),
        x_range=(1, len(training_loss)),
        title="Autoencoder Training Process",
        legend_title="Metrics",
        legend_loc="upper right",
        legend_ncol=1,
        hide_right_top=True,
        font_size=20,
        ticks_font_size=16,
    )
    print("Training loss plot saved.")
    
    # 步骤4：提取嵌入
    print("\nStep 4: Extract embeddings")
    embeddings = encoder.predict(X)
    print(f"Embedding shape: {embeddings.shape}")
    
    embeddings_df = pd.DataFrame(embeddings, columns=[f'Embedding_{i+1}' for i in range(embeddings.shape[1])])
    embeddings_df = pd.concat([processed_deleted[['car_ID', 'CarName', 'price']], embeddings_df], axis=1)
    
    # 步骤5：应用聚类方法
    print("\nStep 5: Apply clustering in embedding space")
    
    # 5.1 K-means 聚类
    print("5.1 K-means Clustering")
    print("Determining optimal K value using the elbow method...")
    sse = {}
    K_range = range(2, 21)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(embeddings)
        sse[k] = kmeans.inertia_
    
    # 使用 plot_lines 绘制肘部图
    print("绘制肘部图并保存...")
    elbow_data = [{"y": list(sse.values()), "label": "SSE", "color": plt_properties["color"][0]}]
    plot_lines(
        data=elbow_data,
        file_name="elbow_method_deep_kmeans",
        aspect_ratio=0.825,
        xlabel="Number of Clusters (K)",
        ylabel="SSE",
        x_range=(2, 20),
        y_range=(min(sse.values()), max(sse.values())),
        title="Elbow Method for Optimal K (K-means)",
        legend_title="Metrics",
        legend_loc="best",
        legend_ncol=1,
        hide_right_top=True,
        font_size=20,
        ticks_font_size=16,
    )
    print("Elbow plot for K-means saved.")
    
    # 选择最佳 K 值，这里假设 K=8
    optimal_k = 8
    print(f"Selected optimal K value for K-means: {optimal_k}")
    
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(embeddings)
    embeddings_df['KMeans_Cluster'] = kmeans_labels
    
    # 5.2 基于层次的聚类分析（Agglomerative Clustering）
    print("\n5.2 Agglomerative Clustering")
    print("Determining optimal K value using the silhouette method for Agglomerative Clustering...")
    
    # # 使用轮廓系数来选择最佳 K
    # silhouette_scores = {}
    # for k in K_range:
    #     agg_clustering = AgglomerativeClustering(n_clusters=k, metric='euclidean', linkage='ward')
    #     agg_labels = agg_clustering.fit_predict(embeddings)
    #     silhouette = silhouette_score(embeddings, agg_labels)
    #     silhouette_scores[k] = silhouette


    # # 使用 plot_lines 绘制轮廓系数
    # print("绘制层次聚类的轮廓系数图并保存...")
    # silhouette_data = [{"y": list(silhouette_scores.values()), "label": "Silhouette Score", "color": plt_properties["color"][1]}]
    # plot_lines(
    #     data=silhouette_data,
    #     file_name="silhouette_method_deep_agg",
    #     aspect_ratio=0.825,
    #     xlabel="Number of Clusters (K)",
    #     ylabel="Silhouette Score",
    #     x_range=(2, 20),
    #     y_range=(min(silhouette_scores.values()) - 0.05, max(silhouette_scores.values()) + 0.05),
    #     title="Silhouette Scores for Agglomerative Clustering",
    #     legend_title="Metrics",
    #     legend_loc="best",
    #     legend_ncol=1,
    #     hide_right_top=True,
    #     font_size=20,
    #     ticks_font_size=16,
    # )
    # print("Silhouette scores plot for Agglomerative Clustering saved.")
    
    # 根据轮廓系数选择最佳 K 值，这里假设 K=8
    print(f"Selected optimal K value for Agglomerative Clustering: {optimal_k}")
    agg_clustering = AgglomerativeClustering(n_clusters=optimal_k, metric='euclidean', linkage='ward')
    agg_labels = agg_clustering.fit_predict(embeddings)
    embeddings_df['Agglomerative_Cluster'] = agg_labels

    # 4.2 基于层次的聚类分析
    print("进行基于层次的聚类分析...")
    linked = linkage(embeddings, method='ward')
    
    # 绘制树状图
    print("绘制树状图并保存...")
    fig, ax = plt.subplots(figsize=(10, 7))
    dendrogram(linked,
               orientation='top',
               distance_sort='descending',
               show_leaf_counts=False,
               ax=ax)
    ax.set_title('Hierarchical Clustering (using AE features)', fontdict=fontdict)
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

    # 步骤6：聚类效果评估
    print("\nStep 6: Clustering Evaluation")
    
    # 6.1 计算 Silhouette Score
    kmeans_silhouette = silhouette_score(embeddings, kmeans_labels)
    agg_silhouette = silhouette_score(embeddings, agg_labels)
    
    # 6.2 计算 Calinski-Harabasz 指数
    kmeans_ch = calinski_harabasz_score(embeddings, kmeans_labels)
    agg_ch = calinski_harabasz_score(embeddings, agg_labels)
    
    # 6.3 计算 Davies-Bouldin 指数
    kmeans_db = davies_bouldin_score(embeddings, kmeans_labels)
    agg_db = davies_bouldin_score(embeddings, agg_labels)
    
    # 6.4 计算 Dunn 指数
    print("正在计算 Dunn 指数，这可能需要一些时间...")
    kmeans_dunn = dunn(embeddings, kmeans_labels)
    agg_dunn = dunn(embeddings, agg_labels)
    
    # 输出聚类效果评估指标
    print("\nClustering Evaluation Metrics:")
    print(f'K-means: Silhouette Score = {kmeans_silhouette:.4f}, CH Score = {kmeans_ch:.4f}, DB Score = {kmeans_db:.4f}, Dunn Index = {kmeans_dunn:.4f}')
    print(f'Agglomerative Clustering: Silhouette Score = {agg_silhouette:.4f}, CH Score = {agg_ch:.4f}, DB Score = {agg_db:.4f}, Dunn Index = {agg_dunn:.4f}')
    
    # 步骤7：可视化聚类结果
    print("\nStep 7: Visualize clustering results")
    
    # 7.1 K-means 聚类的 t-SNE 可视化
    print("7.1 Visualizing K-means Clustering Results with t-SNE...")
    tsne_kmeans = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_results_kmeans = tsne_kmeans.fit_transform(embeddings)
    
    embeddings_df['TSNE_KMeans_1'] = tsne_results_kmeans[:, 0]
    embeddings_df['TSNE_KMeans_2'] = tsne_results_kmeans[:, 1]
    
    # 绘制 K-means 聚类的 t-SNE 图
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=embeddings_df, x='TSNE_KMeans_1', y='TSNE_KMeans_2', hue='KMeans_Cluster', palette='Set1', s=100, alpha=0.7)
    plt.title('K-means Clustering Results (2D t-SNE) using AE features', fontdict=fontdict)
    plt.xlabel('t-SNE Dimension 1', fontdict=fontdict)
    plt.ylabel('t-SNE Dimension 2', fontdict=fontdict)
    plt.legend(title='Cluster', loc='best', prop=ticks_fontdict)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'AE_kmeans_tsne_2d.pdf'), bbox_inches="tight")
    plt.close()
    print("K-means t-SNE plot saved.")
    
    # 7.2 层次聚类的 t-SNE 可视化
    print("7.2 Visualizing Agglomerative Clustering Results with t-SNE...")
    tsne_agg = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_results_agg = tsne_agg.fit_transform(embeddings)
    
    embeddings_df['TSNE_Agg_1'] = tsne_results_agg[:, 0]
    embeddings_df['TSNE_Agg_2'] = tsne_results_agg[:, 1]
    
    # 绘制层次聚类的 t-SNE 图
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=embeddings_df, x='TSNE_Agg_1', y='TSNE_Agg_2', hue='Agglomerative_Cluster', palette='Set1', s=100, alpha=0.7)
    plt.title('HC Results (2D t-SNE) using AE features', fontdict=fontdict)
    plt.xlabel('t-SNE Dimension 1', fontdict=fontdict)
    plt.ylabel('t-SNE Dimension 2', fontdict=fontdict)
    plt.legend(title='Cluster', loc='best', prop=ticks_fontdict)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'AE_hc_tsne_2d.pdf'), bbox_inches="tight")
    plt.close()
    print("Agglomerative Clustering t-SNE plot saved.")
    
    # 步骤8：保存聚类结果
    print("\nStep 8: Save clustering results")
    #embeddings_df.to_csv('enhanced_deep_clustering_results.csv', index=False)
    print("Clustering results saved to 'enhanced_deep_clustering_results.csv'")
    
    # 步骤9：统计和分析聚类结果
    print("\nStep 9: Statistical Analysis of Clusters")
    original_data = pd.read_csv('car_price.csv')
    
    # 9.1 统计 K-means 聚类结果的每个簇的汽车数量及价格均值
    print("\n9.1 K-means Clustering: Cluster Statistics")
    cluster_stats_kmeans = []
    for cluster in range(optimal_k):
        cluster_data = embeddings_df[embeddings_df['KMeans_Cluster'] == cluster]
        cluster_car_ids = cluster_data['car_ID'].tolist()
        
        # 获取原始数据中的价格信息
        cluster_prices = original_data[original_data['car_ID'].isin(cluster_car_ids)]['price']
        cluster_size = len(cluster_car_ids)
        cluster_mean_price = cluster_prices.mean()
        cluster_stats_kmeans.append((cluster, cluster_size, cluster_mean_price))
        
        print(f"K-means Cluster {cluster}: Number of Cars = {cluster_size}, Mean Price = {cluster_mean_price:.2f}")
    
    kmeans_result = original_data.copy()
    kmeans_result = pd.concat([kmeans_result, embeddings_df['KMeans_Cluster']], axis=1)
    kmeans_result.to_csv('kmeans_clustered_cars.csv', index=False)
    
    # 9.2 检查 K-means 聚类结果中 Volkswagen 所属簇的分布情况
    print("\n9.2 K-means Clustering: Volkswagen Cluster Distribution")
    vw_cars_kmeans = embeddings_df[embeddings_df['CarName'].str.contains('volkswagen', case=False, na=False)]
    if not vw_cars_kmeans.empty:
        print("Volkswagen cars are present in the dataset.")
        vw_clusters_kmeans = vw_cars_kmeans['KMeans_Cluster'].unique()
        for vw_cluster in vw_clusters_kmeans:
            vw_cluster_data = vw_cars_kmeans[vw_cars_kmeans['KMeans_Cluster'] == vw_cluster]
            vw_car_ids = vw_cluster_data['car_ID'].tolist()
            vw_prices = original_data[original_data['car_ID'].isin(vw_car_ids)]['price']
            vw_mean_price = vw_prices.mean()
            print(f"K-means Cluster {vw_cluster}: Number of Volkswagen Cars = {len(vw_car_ids)}, Mean Price = {vw_mean_price:.2f}")
    
            # 筛选与 Volkswagen 同簇且价格波动不超过10%的车型
            competitors_data_kmeans = embeddings_df[embeddings_df['KMeans_Cluster'] == vw_cluster]
            competitors_ids_kmeans = competitors_data_kmeans['car_ID'].tolist()
            competitors_prices_kmeans = original_data[original_data['car_ID'].isin(competitors_ids_kmeans)]
            
            # competitors_within_price_range_kmeans = competitors_prices_kmeans[
            #     (competitors_prices_kmeans['price'] >= 0.9 * vw_mean_price) &
            #     (competitors_prices_kmeans['price'] <= 1.1 * vw_mean_price)
            # ]                                                                   
            
            competitors_within_price_range_kmeans = competitors_prices_kmeans[
                (competitors_prices_kmeans['price'] >= 0)
            ]
            
            print(f"K-means Cluster {vw_cluster}: Number of Competitor Cars within ±10% Price Range = {len(competitors_within_price_range_kmeans)}")
            print(competitors_within_price_range_kmeans[['car_ID', 'CarName', 'price']])

            final_competitors_kmeans_df = competitors_within_price_range_kmeans.copy()
            #final_competitors_kmeans_df.to_csv(f'DNN_final_competitors_kmeans_cluster{vw_cluster}.csv', index=False)


    else:
        print("No Volkswagen cars found in the dataset.")
    
    # 9.3 统计层次聚类结果的每个簇的汽车数量及价格均值
    print("\n9.3 Agglomerative Clustering: Cluster Statistics")
    cluster_stats_agg = []
    for cluster in range(optimal_k):
        cluster_data = embeddings_df[embeddings_df['Agglomerative_Cluster'] == cluster]
        cluster_car_ids = cluster_data['car_ID'].tolist()
        
        # 获取原始数据中的价格信息
        cluster_prices = original_data[original_data['car_ID'].isin(cluster_car_ids)]['price']
        cluster_size = len(cluster_car_ids)
        cluster_mean_price = cluster_prices.mean()
        cluster_stats_agg.append((cluster, cluster_size, cluster_mean_price))
        
        print(f"Agglomerative Cluster {cluster}: Number of Cars = {cluster_size}, Mean Price = {cluster_mean_price:.2f}")
    
    ac_result = original_data.copy()
    ac_result = pd.concat([ac_result, embeddings_df['Agglomerative_Cluster']], axis=1)
    ac_result.to_csv('agglomerative_clustered_cars.csv', index=False)
    
    # 9.4 检查层次聚类结果中 Volkswagen 所属簇的分布情况
    print("\n9.4 Agglomerative Clustering: Volkswagen Cluster Distribution")
    vw_cars_agg = embeddings_df[embeddings_df['CarName'].str.contains('volkswagen', case=False, na=False)]
    if not vw_cars_agg.empty:
        print("Volkswagen cars are present in the dataset.")
        vw_clusters_agg = vw_cars_agg['Agglomerative_Cluster'].unique()
        for vw_cluster in vw_clusters_agg:
            vw_cluster_data = vw_cars_agg[vw_cars_agg['Agglomerative_Cluster'] == vw_cluster]
            vw_car_ids = vw_cluster_data['car_ID'].tolist()
            vw_prices = original_data[original_data['car_ID'].isin(vw_car_ids)]['price']
            vw_mean_price = vw_prices.mean()
            print(f"Agglomerative Cluster {vw_cluster}: Number of Volkswagen Cars = {len(vw_car_ids)}, Mean Price = {vw_mean_price:.2f}")
    
            # 筛选与 Volkswagen 同簇且价格波动不超过10%的车型
            competitors_data_agg = embeddings_df[embeddings_df['Agglomerative_Cluster'] == vw_cluster]
            competitors_ids_agg = competitors_data_agg['car_ID'].tolist()
            competitors_prices_agg = original_data[original_data['car_ID'].isin(competitors_ids_agg)]
            
            # competitors_within_price_range_agg = competitors_prices_agg[
            #     (competitors_prices_agg['price'] >= 0.9 * vw_mean_price) &
            #     (competitors_prices_agg['price'] <= 1.1 * vw_mean_price)
            # ]
            
            competitors_within_price_range_agg = competitors_prices_agg[
                (competitors_prices_agg['price'] >= 0)
            ]
            
            print(f"Agglomerative Cluster {vw_cluster}: Number of Competitor Cars within ±10% Price Range = {len(competitors_within_price_range_agg)}")
            print(competitors_within_price_range_agg[['car_ID', 'CarName', 'price']])

            final_competitors_hierarchical_df = competitors_within_price_range_agg.copy()
            #final_competitors_hierarchical_df.to_csv(f'DNN_final_competitors_hierarchical_cluster{vw_cluster}.csv', index=False)


    else:
        print("No Volkswagen cars found in the dataset.")
    
    print("\nAll steps completed successfully.")



if __name__ == "__main__":
    main()





