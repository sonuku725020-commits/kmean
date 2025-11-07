import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score, confusion_matrix
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go

# Optional (for label alignment in confusion matrix)
try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# =============== Helpers ===============
def make_four_clusters_3d(n_per_cluster=250, std_each=1.0, seed=123, anisotropic=False):
    from sklearn.datasets import make_blobs
    centers = np.array([
        [-6.0, -6.0, -6.0],
        [ 6.0,  6.0, -6.0],
        [ 6.0, -6.0,  6.0],
        [-6.0,  6.0,  6.0],
    ], dtype=float)
    if np.isscalar(std_each):
        std_each = [float(std_each)] * 4
    X, y = make_blobs(
        n_samples=[n_per_cluster] * 4,
        centers=centers,
        n_features=3,
        cluster_std=list(std_each),
        random_state=seed
    )
    if anisotropic:
        A = np.array([[0.8, -0.4, 0.0],
                      [0.2,  0.7, -0.3],
                      [0.1,  0.2,  0.9]])
        X = X @ A
    df = pd.DataFrame(X, columns=["x1", "x2", "x3"])
    df["target"] = y.astype(int)
    return df

def align_pred_to_true(y_true, y_pred):
    if HAS_SCIPY:
        cm = confusion_matrix(y_true, y_pred)
        row_ind, col_ind = linear_sum_assignment(-cm)
        mapping = {pred: true for true, pred in zip(col_ind, row_ind)}
        y_aligned = np.array([mapping.get(lbl, lbl) for lbl in y_pred])
        return y_aligned, mapping
    # Fallback (greedy)
    mapping = {}
    for p in np.unique(y_pred):
        mode_vals = pd.Series(y_true[y_pred == p]).mode()
        mapping[p] = int(mode_vals.iloc[0]) if len(mode_vals) else p
    y_aligned = np.array([mapping.get(lbl, lbl) for lbl in y_pred])
    return y_aligned, mapping

def imshow_heatmap(Z, x, y, title, xlab, ylab):
    try:
        fig = px.imshow(Z, x=x, y=y, text_auto=True, color_continuous_scale="Blues",
                        labels=dict(color="Count"), title=title)
    except TypeError:
        fig = px.imshow(Z, x=x, y=y, color_continuous_scale="Blues",
                        labels=dict(color="Count"), title=title)
    fig.update_xaxes(title=xlab)
    fig.update_yaxes(title=ylab)
    return fig

def line_plot(x, y, xlab, ylab, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers"))
    fig.update_layout(xaxis_title=xlab, yaxis_title=ylab, title=title)
    return fig

def create_kmeans(
    n_clusters, init_mode, init_array, n_init_mode, n_init_val,
    max_iter, tol, algorithm, random_state, copy_x, verbose
):
    # Handle custom init
    if init_mode == "custom-array" and init_array is not None:
        init_param = np.array(init_array, dtype=float)
        n_init_param = 1
    else:
        init_param = init_mode  # 'k-means++' or 'random'
        n_init_param = "auto" if n_init_mode == "auto" else int(n_init_val)

    try:
        km = KMeans(
            n_clusters=int(n_clusters),
            init=init_param,
            n_init=n_init_param,
            max_iter=int(max_iter),
            tol=float(tol),
            algorithm=algorithm,
            random_state=None if random_state is None else int(random_state),
            copy_x=bool(copy_x),
            verbose=int(verbose),
        )
    except TypeError:
        # Older sklearn back-compat (no n_init='auto' or 'algorithm' kw)
        if isinstance(n_init_param, str):
            n_init_param = 10
        try:
            km = KMeans(
                n_clusters=int(n_clusters),
                init=init_param,
                n_init=int(n_init_param),
                max_iter=int(max_iter),
                tol=float(tol),
                random_state=None if random_state is None else int(random_state),
                copy_x=bool(copy_x),
                verbose=int(verbose),
            )
        except TypeError:
            km = KMeans(
                n_clusters=int(n_clusters),
                init=init_param,
                n_init=int(n_init_param),
                max_iter=int(max_iter),
                tol=float(tol),
                random_state=None if random_state is None else int(random_state),
            )
    return km

# =============== App ===============
st.set_page_config(page_title="KMeans 3D Dashboard (all sklearn params)", layout="wide")
st.title("KMeans Dashboard (3D) — all scikit-learn parameters")
st.caption("Generate or upload your data, tune all KMeans parameters, visualize in 3D, and evaluate clustering.")

with st.sidebar:
    st.header("Data")
    data_src = st.selectbox("Dataset source", ["Synthetic: 4 clusters (3D)", "Upload CSV"])

    if data_src == "Synthetic: 4 clusters (3D)":
        n_per = st.slider("Samples per cluster", 20, 10000, 250, step=10)
        std_all = st.slider("Cluster std (same for all)", 0.1, 3.0, 1.0, step=0.1)
        anis = st.checkbox("Anisotropic (elliptical)", value=False)
        data_seed = st.number_input("Data random seed", 0, 100000, 123, step=1)
        df = make_four_clusters_3d(n_per_cluster=n_per, std_each=std_all, seed=data_seed, anisotropic=anis)
        target_col = "target"
        st.caption(f"Generated: {df.shape[0]} rows, 3 features + target")
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            st.write(f"Loaded shape: {df.shape}")
            # Optional ground truth label
            guess = [c for c in df.columns if c.lower() in ("target", "label", "class", "y")]
            target_col = st.selectbox("Target column (optional)", [None] + list(df.columns),
                                      index=0 if not guess else 1 + list(df.columns).index(guess[0]))
        else:
            df, target_col = None, None

    st.header("Preprocessing")
    scale_flag = st.checkbox("Standardize features (recommended)", value=True)

    # Choose 3D plotting space
    vis_mode = st.radio("3D visualization space", ["Use 3 data columns", "PCA(3D)"], index=0)
    if df is not None:
        num_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in num_cols_all:
            num_cols_all.remove(target_col)
        if vis_mode == "Use 3 data columns":
            default_cols = ["x1", "x2", "x3"]
            defaults = [c for c in default_cols if c in num_cols_all]
            if len(defaults) < 3 and len(num_cols_all) >= 3:
                defaults = num_cols_all[:3]
            plot_cols = st.multiselect("Pick 3 numeric columns to plot (x,y,z)", num_cols_all, default=defaults)
            if len(plot_cols) != 3:
                st.warning("Select exactly 3 columns, or switch to PCA(3D).")
        else:
            plot_cols = None

    st.header("KMeans Parameters (all)")
    k = st.slider("n_clusters (k)", 2, 50, 4, step=1)

    init_choice = st.selectbox("init", ["k-means++", "random", "From data (random rows)"], index=0)
    init_mode = "k-means++" if init_choice == "k-means++" else ("random" if init_choice == "random" else "custom-array")
    init_seed = st.number_input("Init sampling seed (for 'From data')", 0, 100000, 42, step=1, disabled=(init_mode != "custom-array"))

    n_init_mode = st.radio("n_init", ["auto", "set value"], index=0, horizontal=True, help="When custom init is used, n_init=1 is enforced.")
    n_init_val = st.slider("n_init value", 1, 200, 10, disabled=(n_init_mode == "auto" or init_mode == "custom-array"))

    max_iter = st.slider("max_iter", 50, 5000, 300, step=50)
    tol = st.number_input("tol", min_value=1e-9, max_value=1e-1, value=1e-4, format="%.6f")
    algorithm = st.selectbox("algorithm", ["lloyd", "elkan"], index=0)
    random_state = st.number_input("random_state (model)", 0, 100000, 42, step=1)
    copy_x = st.checkbox("copy_x", value=True)
    verbose = st.selectbox("verbose", [0, 1, 2], index=0)

    st.header("Analysis")
    do_elbow = st.checkbox("Elbow & Silhouette curves", value=True)
    k_range = st.slider("k range", 2, 50, (2, 10))
    sample_weight_col = None
    if df is not None:
        num_cols_sw = df.select_dtypes(include=[np.number]).columns.tolist()
        sample_weight_col = st.selectbox("Sample weight column (optional)", [None] + num_cols_sw, index=0)

# Stop if no data
if df is None:
    st.info("Load or generate a dataset to continue.")
    st.stop()

# Build feature matrix X
X_all = df.select_dtypes(include=[np.number]).copy()
if target_col and target_col in X_all.columns:
    X_all = X_all.drop(columns=[target_col])

if X_all.shape[1] == 0:
    st.error("No numeric features found. Please provide numeric columns.")
    st.stop()

# Scale
scaler = StandardScaler() if scale_flag else None
X_scaled = scaler.fit_transform(X_all.values) if scaler is not None else X_all.values

# Custom init array if selected
rng = np.random.RandomState(int(init_seed)) if init_mode == "custom-array" else None
init_array = None
if init_mode == "custom-array":
    if X_scaled.shape[0] < k:
        st.error(f"Not enough samples ({X_scaled.shape[0]}) to pick {k} unique initial centers from data.")
        st.stop()
    idx = rng.choice(X_scaled.shape[0], size=int(k), replace=False)
    init_array = X_scaled[idx, :]

# Fit KMeans
km = create_kmeans(
    n_clusters=k, init_mode=init_mode, init_array=init_array,
    n_init_mode=n_init_mode, n_init_val=n_init_val,
    max_iter=max_iter, tol=tol, algorithm=algorithm,
    random_state=random_state, copy_x=copy_x, verbose=verbose
)

sample_weight = None
if sample_weight_col:
    sw = df[sample_weight_col].values
    if np.any(sw < 0):
        st.warning("Sample weights contain negative values; clipping to >= 0.")
        sw = np.clip(sw, 0, None)
    sample_weight = sw

y_pred = km.fit_predict(X_scaled, sample_weight=sample_weight)
inertia = float(km.inertia_)

# Metrics
y_true = df[target_col].values if target_col and target_col in df.columns else None
sil = None
try:
    if len(np.unique(y_pred)) > 1:
        sil = float(silhouette_score(X_scaled, y_pred))
except Exception:
    sil = None
ari = float(adjusted_rand_score(y_true, y_pred)) if y_true is not None else None
ami = float(adjusted_mutual_info_score(y_true, y_pred)) if y_true is not None else None

# Cluster sizes
sizes_df = pd.Series(y_pred).value_counts().sort_index().rename("count").to_frame().reset_index(names="cluster")

# 3D plotting data
if vis_mode == "Use 3 data columns":
    if plot_cols is None or len(plot_cols) != 3:
        st.warning("Switching to PCA(3D) for visualization.")
        vis_mode = "PCA(3D)"

if vis_mode == "PCA(3D)":
    pca = PCA(n_components=3, random_state=0)
    X_plot = pca.fit_transform(X_scaled)
    centers_3d = pca.transform(km.cluster_centers_)
    plot_labels = ["pc1", "pc2", "pc3"]
    X_plot_df = pd.DataFrame(X_plot, columns=plot_labels)
    centers_df = pd.DataFrame(centers_3d, columns=plot_labels)
else:
    plot_labels = plot_cols
    X_plot_df = df[plot_cols].copy()
    # centers to original scale
    centers_full = scaler.inverse_transform(km.cluster_centers_) if scaler is not None else km.cluster_centers_
    idxs = [list(X_all.columns).index(c) for c in plot_cols]
    centers_sel = centers_full[:, idxs]
    centers_df = pd.DataFrame(centers_sel, columns=plot_cols)

# 3D plots
left, right = st.columns([1.25, 1])
with left:
    st.subheader("KMeans clusters (3D)")
    fig = px.scatter_3d(
        X_plot_df.assign(cluster=y_pred.astype(int)),
        x=plot_labels[0], y=plot_labels[1], z=plot_labels[2],
        color="cluster",
        opacity=0.85,
        title=f"k={k} | inertia={inertia:.2f}" + (f", silhouette={sil:.3f}" if sil is not None else ""),
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    fig.update_traces(marker=dict(size=4))
    fig.add_trace(go.Scatter3d(
        x=centers_df[plot_labels[0]], y=centers_df[plot_labels[1]], z=centers_df[plot_labels[2]],
        mode="markers", name="Centers",
        marker=dict(size=10, color="black", symbol="x")
    ))
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Cluster sizes")
    st.dataframe(sizes_df, use_container_width=True)
    st.subheader("Scores")
    st.metric("Inertia (SSE)", f"{inertia:.2f}")
    st.metric("Silhouette", f"{sil:.4f}" if sil is not None else "N/A")
    if y_true is not None:
        st.metric("ARI", f"{ari:.4f}")
        st.metric("AMI", f"{ami:.4f}")

# Ground truth comparison (if labels provided)
if y_true is not None:
    st.subheader("Ground truth vs KMeans (aligned labels)")
    y_aligned, mapping = align_pred_to_true(y_true, y_pred)
    cm = confusion_matrix(y_true, y_aligned)
    fig_cm = imshow_heatmap(cm, x=sorted(np.unique(y_aligned)), y=sorted(np.unique(y_true)),
                            title="Confusion Matrix (Aligned)", xlab="Predicted", ylab="True")
    st.plotly_chart(fig_cm, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_true = px.scatter_3d(
            X_plot_df.assign(target=y_true.astype(int)),
            x=plot_labels[0], y=plot_labels[1], z=plot_labels[2],
            color="target", opacity=0.85, title="Ground truth (3D)",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_true.update_traces(marker=dict(size=4))
        st.plotly_chart(fig_true, use_container_width=True)
    with c2:
        fig_pred2 = px.scatter_3d(
            X_plot_df.assign(cluster=y_aligned.astype(int)),
            x=plot_labels[0], y=plot_labels[1], z=plot_labels[2],
            color="cluster", opacity=0.85, title="KMeans (aligned labels)",
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        fig_pred2.update_traces(marker=dict(size=4))
        st.plotly_chart(fig_pred2, use_container_width=True)

# Elbow & Silhouette analysis
if do_elbow:
    st.subheader("Elbow & Silhouette analysis across k")
    ks = list(range(k_range[0], k_range[1] + 1))
    inertias, silhouettes, aris, amis = [], [], [], []
    with st.spinner("Computing..."):
        for kk in ks:
            # Build custom init if requested
            init_arr_k = None
            if init_mode == "custom-array":
                rng_k = np.random.RandomState(int(init_seed))
                idx_k = rng_k.choice(X_scaled.shape[0], size=int(kk), replace=False)
                init_arr_k = X_scaled[idx_k, :]
            km_k = create_kmeans(
                n_clusters=kk, init_mode=init_mode, init_array=init_arr_k,
                n_init_mode=n_init_mode, n_init_val=n_init_val,
                max_iter=max_iter, tol=tol, algorithm=algorithm,
                random_state=random_state, copy_x=copy_x, verbose=0
            )
            yp = km_k.fit_predict(X_scaled, sample_weight=sample_weight)
            inertias.append(float(km_k.inertia_))
            try:
                silhouettes.append(silhouette_score(X_scaled, yp) if len(np.unique(yp)) > 1 else np.nan)
            except Exception:
                silhouettes.append(np.nan)
            if y_true is not None:
                aris.append(adjusted_rand_score(y_true, yp))
                amis.append(adjusted_mutual_info_score(y_true, yp))
    cA, cB = st.columns(2)
    with cA:
        st.plotly_chart(line_plot(ks, inertias, "k", "Inertia (SSE)", "Elbow curve"), use_container_width=True)
    with cB:
        st.plotly_chart(line_plot(ks, silhouettes, "k", "Silhouette", "Silhouette vs k"), use_container_width=True)
    if y_true is not None:
        cC, cD = st.columns(2)
        with cC:
            st.plotly_chart(line_plot(ks, aris, "k", "ARI", "Adjusted Rand Index vs k"), use_container_width=True)
        with cD:
            st.plotly_chart(line_plot(ks, amis, "k", "AMI", "Adjusted Mutual Info vs k"), use_container_width=True)

# Download results
out = df.copy()
out["kmeans_cluster"] = y_pred
st.download_button("Download data + KMeans labels (CSV)", data=out.to_csv(index=False).encode("utf-8"),
                   file_name="data_with_kmeans.csv", mime="text/csv")

st.success("Dashboard ready — tweak KMeans parameters from the sidebar to explore different clusterings!")