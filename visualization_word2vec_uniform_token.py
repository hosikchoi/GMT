# -----------------------------
# Step 1~2: 데이터 생성 동일
# -----------------------------
import numpy as np
import pandas as pd

np.random.seed(42)
n = 1000
rho = 0.9
n_bins = 5
#x1 = np.random.normal(0, 1, n)
#epsilon = np.random.normal(0, 1, n)
#x2 = np.sqrt(rho) * x1 + np.sqrt(1 - rho) * epsilon

def generate_uniform_correlated_data(n, rho):
    # Step 1: Generate x1 and epsilon as independent uniform(0,1)
    x1 = np.random.uniform(0, 1, n)
    epsilon = np.random.uniform(0, 1, n)
    
    # Step 2: Apply Gaussian copula approach to induce correlation
    # Convert uniform to standard normal
    x1_norm = np.sqrt(2) * erfinv(2 * x1 - 1)
    epsilon_norm = np.sqrt(2) * erfinv(2 * epsilon - 1)
    
    # Step 3: Create x2 using correlated normal variables
    x2_norm = np.sqrt(rho) * x1_norm + np.sqrt(1 - rho) * epsilon_norm
    
    # Step 4: Transform back to uniform
    from scipy.stats import norm
    x2 = norm.cdf(x2_norm)

    return pd.DataFrame({'x1': x1, 'x2': x2})

from scipy.special import erfinv
df_uniform = generate_uniform_correlated_data(n=1000, rho=rho)
x1 = df_uniform["x1"]
x2 = df_uniform["x2"]

#import ace_tools as tools; tools.display_dataframe_to_user(name="Uniform Correlated Data", dataframe=df_uniform)


df = pd.DataFrame({'x1': x1, 'x2': x2})

def bin_to_token(x, n_bins=n_bins):
    bins = np.linspace(np.min(x), np.max(x), n_bins + 1)
    labels = [f"level{i}" for i in range(n_bins)]
    return pd.cut(x, bins=bins, labels=labels, include_lowest=True)

df['token_x1'] = bin_to_token(df['x1'])
df['token_x2'] = bin_to_token(df['x2'])

# -----------------------------
# Step 3: 문장화 - token에 prefix 붙이기
# -----------------------------
df['token_x1'] = 'x1_' + df['token_x1'].astype(str)
df['token_x2'] = 'x2_' + df['token_x2'].astype(str)

sentences = df.apply(lambda row: [row["token_x1"], row["token_x2"]], axis=1).tolist()

# -----------------------------
# Step 4: Word2Vec 학습
# -----------------------------
from gensim.models import Word2Vec

model = Word2Vec(
    sentences=sentences,
    vector_size=50,
    window=2,
    min_count=1,
    workers=2,
    sg=1,
    epochs=100
)

# -----------------------------
# Step 5: 'x1_level*', 'x2_level*' 임베딩
# -----------------------------
n_bins = 5
tokens_x1 = [f"x1_level{i}" for i in range(n_bins)]
tokens_x2 = [f"x2_level{i}" for i in range(n_bins)]

data = []

for token in tokens_x1:
    data.append({
        'token': token.replace("x1_", ""),
        'source': 'x1',
        'vector': model.wv[token]
    })

for token in tokens_x2:
    data.append({
        'token': token.replace("x2_", ""),
        'source': 'x2',
        'vector': model.wv[token]
    })

# -----------------------------
# Step 6: t-SNE 시각화
# -----------------------------
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

vecs = np.stack([d['vector'] for d in data])
tsne_result = TSNE(n_components=2, random_state=42, perplexity=n_bins-1).fit_transform(vecs)

tsne_df = pd.DataFrame(tsne_result, columns=['tsne_1', 'tsne_2'])
tsne_df['token'] = [d['token'] for d in data]
tsne_df['source'] = [d['source'] for d in data]

# -----------------------------
# Step 7: Plot with color & style
# -----------------------------
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=tsne_df,
    x="tsne_1",
    y="tsne_2",
    hue="token",      # 색깔로 level 구분
    style="source",   # 모양으로 x1/x2 구분
    s=100,
    palette="Spectral"
)
plt.xlim(-150, 150)  # Set x-axis range
plt.ylim(-80, 80)    
for i, row in tsne_df.iterrows():
    plt.text(row["tsne_1"] + 0.3, row["tsne_2"], f"{row['source']}_{row['token']}", fontsize=13)
plt.title("t-SNE of Word2Vec Embeddings (x1 & x2 Tokens)")
plt.tight_layout()
plt.show()


######################

# ------------------------------------------
# 1. 토큰 목록: x1_level0~19, x2_level0~19
# ------------------------------------------
tokens_x1 = [f"x1_level{i}" for i in range(n_bins)]
tokens_x2 = [f"x2_level{i}" for i in range(n_bins)]
all_tokens = tokens_x1 + tokens_x2

# ------------------------------------------
# 2. 벡터 추출
# ------------------------------------------
vectors = {token: model.wv[token] for token in all_tokens}

# ------------------------------------------
# 3. L2 거리 행렬 계산
# ------------------------------------------
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd

vec_matrix = np.stack([vectors[token] for token in all_tokens])
distance_matrix = cdist(vec_matrix, vec_matrix, metric='euclidean')
dist_df = pd.DataFrame(distance_matrix, index=all_tokens, columns=all_tokens)

# ------------------------------------------
# 4. Heatmap + Clustermap 시각화
# ------------------------------------------
#plt.figure(figsize=(12, 10))
#sns.clustermap(
#    dist_df,
#    cmap="mako",        # 색상맵은 취향에 따라 변경 가능
#    linewidths=0.5,
#    figsize=(13, 13),
#    xticklabels=True,
#    yticklabels=True,
#    dendrogram_ratio=0.2,
#    cbar_pos=(0.02, 0.8, 0.03, 0.18)
#)
#plt.suptitle("L2 Distance Clustermap of x1 & x2 Ordinal Tokens", y=1.05, fontsize=16)
#plt.show()

# -------------------------------
# 1. Readable label로 변환
# -------------------------------
def prettify_label(token):
    var, level = token.split("_")
    return f"{var}\n{level}"

pretty_labels = [prettify_label(tok) for tok in all_tokens]

# -------------------------------
# 2. DataFrame에 적용
# -------------------------------
dist_df.index = pretty_labels
dist_df.columns = pretty_labels

# -------------------------------
# 3. Clustermap 시각화
# -------------------------------
group_labels = ["x1" if "x1_" in tok else "x2" for tok in all_tokens]
row_colors = ['royalblue' if g == "x1" else 'darkorange' for g in group_labels]

sns.set(font_scale=1.2)  # 전체 폰트 스케일 업
g = sns.clustermap(
    dist_df,
    cmap="mako",
    row_colors=row_colors,
    col_colors=row_colors,
    linewidths=0.5,
    figsize=(13, 13),
    xticklabels=True,
    yticklabels=True,
    dendrogram_ratio=0.2,
    cbar_pos=(0.02, 0.8, 0.03, 0.18)
)

# 세부 tick label 조정 (추가적으로 직접 조절)
g.ax_heatmap.tick_params(axis='x', labelsize=20)
g.ax_heatmap.tick_params(axis='y', labelsize=20)
plt.suptitle("L2 Distance Clustermap of x1 & x2 Ordinal Tokens", y=1.05, fontsize=20)
plt.show()


















































