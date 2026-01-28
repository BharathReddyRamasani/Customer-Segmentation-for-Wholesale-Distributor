import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Wholesale Customer Segmentation",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- TITLE ----------------
st.title("📦 Wholesale Customer Segmentation Dashboard")
st.caption("Unsupervised Learning | K-Means | Business Intelligence")

st.markdown("---")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("data/Wholesale customers data.csv")

spending_features = [
    'Fresh', 'Milk', 'Grocery',
    'Frozen', 'Detergents_Paper', 'Delicassen'
]

X = df[spending_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Clustering Controls")
k = st.sidebar.slider("Number of Clusters (K)", 2, 8, 4)

# ---------------- CLUSTERING ----------------
kmeans = KMeans(n_clusters=k, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# ---------------- KPIs ----------------
st.subheader("📊 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", df.shape[0])
col2.metric("Clusters", k)
col3.metric("Avg Monthly Spend", f"{int(df[spending_features].sum(axis=1).mean()):,}")
col4.metric("Highest Spend (₹)", f"{int(df[spending_features].sum(axis=1).max()):,}")

st.markdown("---")

# ---------------- ELBOW & SCATTER ----------------
left, right = st.columns(2)

# ----- Elbow Plot -----
with left:
    st.subheader("🔍 Optimal Cluster Selection (Elbow Method)")
    inertia = []
    for i in range(2, 9):
        km = KMeans(n_clusters=i, random_state=42)
        km.fit(X_scaled)
        inertia.append(km.inertia_)

    fig1, ax1 = plt.subplots()
    ax1.plot(range(2,9), inertia, marker='o', linewidth=2)
    ax1.set_xlabel("Number of Clusters (K)")
    ax1.set_ylabel("WCSS")
    ax1.grid(alpha=0.3)
    st.pyplot(fig1)

# ----- Scatter Plot -----
with right:
    st.subheader("📈 Customer Clusters (Milk vs Grocery)")

    fig2, ax2 = plt.subplots()

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for c in sorted(df["Cluster"].unique()):
        subset = df[df["Cluster"] == c]
        ax2.scatter(
            subset["Grocery"],
            subset["Milk"],
            s=35,
            alpha=0.7,
            label=f"Cluster {c}",
            color=colors[c % len(colors)]
        )

    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    ax2.scatter(
        centers[:,2], centers[:,1],
        c="black", s=250, marker="X", label="Centroids"
    )

    ax2.set_xlabel("Grocery Spend")
    ax2.set_ylabel("Milk Spend")
    ax2.legend()
    ax2.grid(alpha=0.3)

    st.pyplot(fig2)

st.markdown("---")

# ---------------- CLUSTER PROFILING ----------------
st.subheader("🧩 Cluster Profiles (Average Spend)")

profile = df.groupby("Cluster")[spending_features].mean()
st.dataframe(profile.style.format("{:.0f}"))

# ---------------- BUSINESS INSIGHTS ----------------
st.subheader("💡 Business Insights & Strategies")

insights = {
    0: "🛒 **Retail Stores** → Bulk pricing, high inventory allocation",
    1: "🍽️ **Restaurants** → Combo offers, predictable restocking",
    2: "☕ **Cafés** → Loyalty programs, small-batch supplies",
    3: "🏨 **Hotels** → Premium pricing, freshness & priority delivery"
}

for c, text in insights.items():
    if c in df["Cluster"].unique():
        st.markdown(f"**Cluster {c}:** {text}")

st.markdown("---")

# ---------------- FOOTER ----------------
st.caption(
    "📌 Project Highlight: Customer Segmentation using Unsupervised Learning "
    "| Scalable | Explainable | Business-Driven"
)
