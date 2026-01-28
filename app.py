# # import streamlit as st
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # from sklearn.preprocessing import StandardScaler
# # from sklearn.cluster import KMeans
# # from sklearn.metrics import silhouette_score

# # # ---------------- PAGE CONFIG ----------------
# # st.set_page_config(
# #     page_title="Wholesale Customer Segmentation",
# #     layout="wide",
# #     initial_sidebar_state="expanded"
# # )

# # # ---------------- TITLE ----------------
# # st.title("📦 Wholesale Customer Segmentation Dashboard")
# # st.caption("Unsupervised Learning | K-Means | Business Intelligence")

# # st.markdown("---")

# # # ---------------- LOAD DATA ----------------
# # df = pd.read_csv("data/Wholesale customers data.csv")

# # spending_features = [
# #     'Fresh', 'Milk', 'Grocery',
# #     'Frozen', 'Detergents_Paper', 'Delicassen'
# # ]

# # X = df[spending_features]
# # scaler = StandardScaler()
# # X_scaled = scaler.fit_transform(X)

# # # ---------------- SIDEBAR ----------------
# # st.sidebar.header("⚙️ Clustering Controls")
# # k = st.sidebar.slider("Number of Clusters (K)", 2, 8, 4)

# # # ---------------- CLUSTERING ----------------
# # kmeans = KMeans(n_clusters=k, random_state=42)
# # df["Cluster"] = kmeans.fit_predict(X_scaled)

# # # ---------------- KPIs ----------------
# # st.subheader("📊 Key Metrics")

# # col1, col2, col3, col4 = st.columns(4)

# # col1.metric("Total Customers", df.shape[0])
# # col2.metric("Clusters", k)
# # col3.metric("Avg Monthly Spend", f"{int(df[spending_features].sum(axis=1).mean()):,}")
# # col4.metric("Highest Spend (₹)", f"{int(df[spending_features].sum(axis=1).max()):,}")

# # st.markdown("---")

# # # ---------------- ELBOW & SCATTER ----------------
# # left, right = st.columns(2)

# # # ----- Elbow Plot -----
# # with left:
# #     st.subheader("🔍 Optimal Cluster Selection (Elbow Method)")
# #     inertia = []
# #     for i in range(2, 9):
# #         km = KMeans(n_clusters=i, random_state=42)
# #         km.fit(X_scaled)
# #         inertia.append(km.inertia_)

# #     fig1, ax1 = plt.subplots()
# #     ax1.plot(range(2,9), inertia, marker='o', linewidth=2)
# #     ax1.set_xlabel("Number of Clusters (K)")
# #     ax1.set_ylabel("WCSS")
# #     ax1.grid(alpha=0.3)
# #     st.pyplot(fig1)

# # # ----- Scatter Plot -----
# # with right:
# #     st.subheader("📈 Customer Clusters (Milk vs Grocery)")

# #     fig2, ax2 = plt.subplots()

# #     colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# #     for c in sorted(df["Cluster"].unique()):
# #         subset = df[df["Cluster"] == c]
# #         ax2.scatter(
# #             subset["Grocery"],
# #             subset["Milk"],
# #             s=35,
# #             alpha=0.7,
# #             label=f"Cluster {c}",
# #             color=colors[c % len(colors)]
# #         )

# #     centers = scaler.inverse_transform(kmeans.cluster_centers_)
# #     ax2.scatter(
# #         centers[:,2], centers[:,1],
# #         c="black", s=250, marker="X", label="Centroids"
# #     )

# #     ax2.set_xlabel("Grocery Spend")
# #     ax2.set_ylabel("Milk Spend")
# #     ax2.legend()
# #     ax2.grid(alpha=0.3)

# #     st.pyplot(fig2)

# # st.markdown("---")

# # # ---------------- CLUSTER PROFILING ----------------
# # st.subheader("🧩 Cluster Profiles (Average Spend)")

# # profile = df.groupby("Cluster")[spending_features].mean()
# # st.dataframe(profile.style.format("{:.0f}"))

# # # ---------------- BUSINESS INSIGHTS ----------------
# # st.subheader("💡 Business Insights & Strategies")

# # insights = {
# #     0: "🛒 **Retail Stores** → Bulk pricing, high inventory allocation",
# #     1: "🍽️ **Restaurants** → Combo offers, predictable restocking",
# #     2: "☕ **Cafés** → Loyalty programs, small-batch supplies",
# #     3: "🏨 **Hotels** → Premium pricing, freshness & priority delivery"
# # }

# # for c, text in insights.items():
# #     if c in df["Cluster"].unique():
# #         st.markdown(f"**Cluster {c}:** {text}")

# # st.markdown("---")

# # # ---------------- FOOTER ----------------
# # st.caption(
# #     "📌 Project Highlight: Customer Segmentation using Unsupervised Learning "
# #     "| Scalable | Explainable | Business-Driven"
# # )


# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler

# # ---------------- PAGE CONFIG ----------------
# st.set_page_config(
#     page_title="Customer Segmentation Dashboard",
#     layout="wide"
# )

# # ---------------- TITLE & DESCRIPTION ----------------
# st.title("🟢 Customer Segmentation Dashboard")

# st.markdown(
#     """
#     **This system uses K-Means Clustering to group customers based on their
#     purchasing behavior and similarities.**

#     👉 Discover hidden customer groups **without predefined labels**.
#     """
# )

# st.markdown("---")

# # ---------------- LOAD DATA ----------------
# df = pd.read_csv("data/Wholesale customers data.csv")

# numeric_features = df.select_dtypes(include="number").columns.tolist()

# # ---------------- SIDEBAR (INPUT SECTION) ----------------
# st.sidebar.header("⚙️ Clustering Controls")

# feature_1 = st.sidebar.selectbox(
#     "Select Feature 1",
#     numeric_features,
#     index=numeric_features.index("Grocery")
# )

# feature_2 = st.sidebar.selectbox(
#     "Select Feature 2",
#     numeric_features,
#     index=numeric_features.index("Milk")
# )

# k = st.sidebar.slider(
#     "Number of Clusters (K)",
#     min_value=2,
#     max_value=10,
#     value=4
# )

# random_state = st.sidebar.number_input(
#     "Random State (optional)",
#     min_value=0,
#     max_value=999,
#     value=42
# )

# run_button = st.sidebar.button("🟦 Run Clustering")

# # ---------------- VALIDATION ----------------
# if feature_1 == feature_2:
#     st.warning("⚠️ Please select **two different features** for clustering.")
#     st.stop()

# # ---------------- CLUSTERING ACTION ----------------
# if run_button:

#     # Prepare data
#     X = df[[feature_1, feature_2]]
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     # Run KMeans
#     kmeans = KMeans(n_clusters=k, random_state=random_state)
#     df["Cluster"] = kmeans.fit_predict(X_scaled)

#     centers = scaler.inverse_transform(kmeans.cluster_centers_)

#     # ---------------- VISUALIZATION SECTION ----------------
#     st.subheader("📊 Cluster Visualization")

#     fig, ax = plt.subplots(figsize=(7, 5))

#     for c in sorted(df["Cluster"].unique()):
#         subset = df[df["Cluster"] == c]
#         ax.scatter(
#             subset[feature_1],
#             subset[feature_2],
#             s=40,
#             alpha=0.7,
#             label=f"Cluster {c}"
#         )

#     ax.scatter(
#         centers[:, 0],
#         centers[:, 1],
#         c="black",
#         s=300,
#         marker="X",
#         label="Cluster Centers"
#     )

#     ax.set_xlabel(feature_1)
#     ax.set_ylabel(feature_2)
#     ax.set_title("Customer Groups Based on Selected Features")
#     ax.legend()
#     ax.grid(alpha=0.3)

#     st.pyplot(fig)

#     # ---------------- CLUSTER SUMMARY ----------------
#     st.subheader("📋 Cluster Summary")

#     summary = (
#         df.groupby("Cluster")
#         .agg(
#             Customers=("Cluster", "count"),
#             Avg_Feature_1=(feature_1, "mean"),
#             Avg_Feature_2=(feature_2, "mean")
#         )
#         .reset_index()
#     )

#     st.dataframe(summary.style.format("{:.2f}"))

#     # ---------------- BUSINESS INTERPRETATION ----------------
#     st.subheader("💡 Business Interpretation")

#     for _, row in summary.iterrows():
#         st.markdown(
#             f"""
#             🟢 **Cluster {int(row['Cluster'])}:**
#             Customers in this group show similar spending patterns in
#             **{feature_1}** and **{feature_2}**, indicating a distinct
#             purchasing behavior segment.
#             """
#         )

#     # ---------------- USER GUIDANCE BOX ----------------
#     st.info(
#         "📌 Customers in the same cluster exhibit similar purchasing behaviour "
#         "and can be targeted with similar business strategies."
#     )

# else:
#     st.info("👈 Select features, choose K, and click **Run Clustering** to begin.")


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Wholesale Customer Segmentation Dashboard",
    layout="wide"
)

# ---------------- HEADER ----------------
st.title("📦 Wholesale Customer Segmentation Dashboard")
st.caption("Unsupervised Learning | K-Means | Business Intelligence")

st.markdown(
    """
    This dashboard uses **K-Means Clustering** to group customers based on their
    purchasing behaviour and similarities.

    👉 **Goal:** Discover hidden customer segments without predefined labels.
    """
)

st.markdown("---")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("data/Wholesale customers data.csv")

spending_features = [
    'Fresh', 'Milk', 'Grocery',
    'Frozen', 'Detergents_Paper', 'Delicassen'
]

# ---------------- SIDEBAR (INPUTS) ----------------
st.sidebar.header("⚙️ Clustering Controls")

feature_1 = st.sidebar.selectbox(
    "Select Feature 1",
    spending_features,
    index=spending_features.index("Grocery")
)

feature_2 = st.sidebar.selectbox(
    "Select Feature 2",
    spending_features,
    index=spending_features.index("Milk")
)

k = st.sidebar.slider("Number of Clusters (K)", 2, 10, 4)

random_state = st.sidebar.number_input(
    "Random State (optional)",
    min_value=0,
    max_value=999,
    value=42
)

run_btn = st.sidebar.button("🟦 Run Clustering")

# ---------------- VALIDATION ----------------
if feature_1 == feature_2:
    st.warning("⚠️ Please select **two different features**.")
    st.stop()

# ---------------- RUN CLUSTERING ----------------
if run_btn:

    # ---------------- PREPROCESS ----------------
    X = df[spending_features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---------------- MODEL ----------------
    kmeans = KMeans(n_clusters=k, random_state=random_state)
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    centers = scaler.inverse_transform(kmeans.cluster_centers_)

    # ---------------- KPIs ----------------
    st.subheader("📊 Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Customers", df.shape[0])
    col2.metric("Clusters", k)
    col3.metric(
        "Avg Monthly Spend",
        f"{int(df[spending_features].sum(axis=1).mean()):,}"
    )
    col4.metric(
        "Highest Spend (₹)",
        f"{int(df[spending_features].sum(axis=1).max()):,}"
    )

    st.markdown("---")

    # ---------------- ELBOW + SCATTER ----------------
    left, right = st.columns(2)

    # ---- Elbow Method ----
    with left:
        st.subheader("🔍 Optimal Cluster Selection (Elbow Method)")
        inertia = []
        for i in range(2, 11):
            km = KMeans(n_clusters=i, random_state=42)
            km.fit(X_scaled)
            inertia.append(km.inertia_)

        fig1, ax1 = plt.subplots()
        ax1.plot(range(2, 11), inertia, marker="o", linewidth=2)
        ax1.set_xlabel("Number of Clusters (K)")
        ax1.set_ylabel("WCSS")
        ax1.grid(alpha=0.3)
        st.pyplot(fig1)

    # ---- Scatter Plot ----
    with right:
        st.subheader(f"📈 Customer Clusters ({feature_2} vs {feature_1})")

        fig2, ax2 = plt.subplots()

        for c in sorted(df["Cluster"].unique()):
            subset = df[df["Cluster"] == c]
            ax2.scatter(
                subset[feature_1],
                subset[feature_2],
                s=40,
                alpha=0.7,
                label=f"Cluster {c}"
            )

        ax2.scatter(
            centers[:, spending_features.index(feature_1)],
            centers[:, spending_features.index(feature_2)],
            c="black",
            s=300,
            marker="X",
            label="Cluster Centers"
        )

        ax2.set_xlabel(feature_1)
        ax2.set_ylabel(feature_2)
        ax2.legend()
        ax2.grid(alpha=0.3)

        st.pyplot(fig2)

    st.markdown("---")

    # ---------------- CLUSTER PROFILES ----------------
    st.subheader("🧩 Cluster Profiles (Average Spend)")

    profile = df.groupby("Cluster")[spending_features].mean()
    st.dataframe(profile.style.format("{:.0f}"))

    # ---------------- DYNAMIC BUSINESS INSIGHTS ----------------
    st.subheader("💡 Business Insights & Strategies")

    cluster_summary = profile.copy()
    cluster_summary["Total_Spend"] = cluster_summary.sum(axis=1)

    max_spend = cluster_summary["Total_Spend"].max()
    min_spend = cluster_summary["Total_Spend"].min()

    for cluster_id, row in cluster_summary.iterrows():

        spend_ratio = (
            (row["Total_Spend"] - min_spend)
            / (max_spend - min_spend + 1e-6)
        )

        if spend_ratio > 0.66:
            emoji = "🟢"
            insight = (
                "High-spending customers across multiple categories. "
                "Ideal for premium pricing, priority inventory allocation, "
                "and long-term contracts."
            )

        elif spend_ratio > 0.33:
            emoji = "🔵"
            insight = (
                "Moderate spenders with consistent purchasing patterns. "
                "Suitable for bundled offers, loyalty programs, "
                "and predictable supply planning."
            )

        else:
            emoji = "🟡"
            insight = (
                "Budget-conscious or selective buyers. "
                "Best targeted with discounts, entry-level packages, "
                "and flexible order quantities."
            )

        st.markdown(
            f"""
            {emoji} **Cluster {cluster_id}:**  
            {insight}
            """
        )

    # ---------------- GUIDANCE ----------------
    st.info(
        "📌 Customers in the same cluster exhibit similar purchasing behaviour "
        "and can be targeted with similar business strategies."
    )

    # ---------------- FOOTER ----------------
    st.caption(
        "📌 Project Highlight: Customer Segmentation using Unsupervised Learning "
        "| Scalable | Explainable | Business-Driven"
    )

else:
    st.info("👈 Select features, choose K, and click **Run Clustering** to begin.")
