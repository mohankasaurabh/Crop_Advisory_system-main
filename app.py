import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Crop Advisory System", layout="wide")

# ---------------- PATH ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "data", "Crop_recommendation.csv")

# ---------------- LOAD ----------------
@st.cache_resource
def load_models():
    model = joblib.load("models/crop_recommendation_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    return model, scaler, label_encoder

@st.cache_data
def load_data():
    return pd.read_csv(data_path)

model, scaler, label_encoder = load_models()
df = load_data()

# ---------------- SIDEBAR ----------------
st.sidebar.title("🌱 Crop Advisory System")
page = st.sidebar.radio("", ["🏠 Dashboard", "🌾 Prediction", "📊 Clustering"])

# ---------------- DASHBOARD ----------------
if page == "🏠 Dashboard":
    st.title("🌱 Crop Advisory Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Model", "Random Forest")
    col2.metric("Accuracy", "99.3%")
    col3.metric("Data Points", len(df))

    st.markdown("---")
    st.subheader("📌 About")
    st.write("""
    - 🌾 Crop Prediction using Machine Learning  
    - 📊 Clustering for pattern discovery  
    - 🧠 Custom scoring for smarter recommendations  
    """)

# ---------------- PREDICTION ----------------
elif page == "🌾 Prediction":
    st.title("🌾 Smart Crop Recommendation")

    input_mode = st.radio("Select Input Mode:", ["Manual Input", "Use Sample Data"])

    if input_mode == "Manual Input":
        col1, col2 = st.columns(2)

        with col1:
            N = st.number_input("Nitrogen", 0.0)
            P = st.number_input("Phosphorus", 0.0)
            K = st.number_input("Potassium", 0.0)
            temperature = st.number_input("Temperature", 0.0)

        with col2:
            humidity = st.number_input("Humidity", 0.0)
            ph = st.number_input("pH", 0.0)
            rainfall = st.number_input("Rainfall", 0.0)

    else:
        sample = df.sample(1).iloc[0]
        N, P, K = sample['N'], sample['P'], sample['K']
        temperature, humidity = sample['temperature'], sample['humidity']
        ph, rainfall = sample['ph'], sample['rainfall']

        st.dataframe(sample)

    if st.button("🚀 Predict Crop"):
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)
        crop = label_encoder.inverse_transform(prediction)

        st.success(f"🌱 Recommended Crop: **{crop[0]}**")

    st.markdown("---")

    st.subheader("📊 Feature Importance")
    features = ['N', 'P', 'K', 'Temp', 'Humidity', 'pH', 'Rainfall']
    importances = model.feature_importances_

    fig = px.bar(
        x=importances,
        y=features,
        orientation='h',
        color=importances
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------- CLUSTERING ----------------
elif page == "📊 Clustering":
    st.title("📊 Clustering Insights")

    features = ['N','P','K','temperature','humidity','ph','rainfall']
    X = df[features]

    scaler_cluster = StandardScaler()
    X_scaled = scaler_cluster.fit_transform(X)

    kmeans = KMeans(n_clusters=5, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    sil_score = silhouette_score(X_scaled, df['cluster'])
    st.metric("Silhouette Score (K=5)", round(sil_score, 4))

    tab1, tab2, tab3, tab4 = st.tabs([
        "📌 PCA (K=5)",
        "📉 Elbow",
        "🌾 Distribution",
        "⚖️ K5 vs K8"
    ])

    # -------- PCA --------
    with tab1:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        pca_df = pd.DataFrame({
            "PCA1": X_pca[:, 0],
            "PCA2": X_pca[:, 1],
            "Cluster": df['cluster'].astype(str)
        })

        fig = px.scatter(
            pca_df,
            x="PCA1",
            y="PCA2",
            color="Cluster",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        st.plotly_chart(fig, use_container_width=True)

    # -------- ELBOW --------
    with tab2:
        wcss = []
        for i in range(1, 10):
            km = KMeans(n_clusters=i, random_state=42)
            km.fit(X_scaled)
            wcss.append(km.inertia_)

        fig = px.line(x=list(range(1,10)), y=wcss, markers=True)
        st.plotly_chart(fig, use_container_width=True)

        st.success("🎯 Optimal K = 5")
        st.info("Silhouette suggests K=8, but K=5 chosen for interpretability")

    # -------- DISTRIBUTION --------
    with tab3:
        cluster_crop = df.groupby(['cluster', 'label']).size().unstack().fillna(0)
        st.bar_chart(cluster_crop)
        st.dataframe(cluster_crop)

    # -------- COMPARISON --------
    with tab4:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        col1, col2 = st.columns(2)

        with col1:
            k5 = KMeans(n_clusters=5, random_state=42).fit_predict(X_scaled)
            df5 = pd.DataFrame({"PCA1": X_pca[:,0], "PCA2": X_pca[:,1], "Cluster": k5.astype(str)})
            fig5 = px.scatter(df5, x="PCA1", y="PCA2", color="Cluster", title="K=5")
            st.plotly_chart(fig5, use_container_width=True)

        with col2:
            k8 = KMeans(n_clusters=8, random_state=42).fit_predict(X_scaled)
            df8 = pd.DataFrame({"PCA1": X_pca[:,0], "PCA2": X_pca[:,1], "Cluster": k8.astype(str)})
            fig8 = px.scatter(df8, x="PCA1", y="PCA2", color="Cluster", title="K=8")
            st.plotly_chart(fig8, use_container_width=True)

    # -------- CLUSTER INPUT --------
    st.markdown("---")
    st.subheader("🧠 Cluster-Based Recommendation")

    input_mode_cluster = st.radio(
        "Select Input Mode:",
        ["Manual Input", "Use Sample Data"],
        key="cluster_input"
    )

    if input_mode_cluster == "Manual Input":
        col1, col2 = st.columns(2)

        with col1:
            N = st.number_input("Nitrogen", 0.0)
            P = st.number_input("Phosphorus", 0.0)
            K = st.number_input("Potassium", 0.0)
            temperature = st.number_input("Temperature", 0.0)

        with col2:
            humidity = st.number_input("Humidity", 0.0)
            ph = st.number_input("pH", 0.0)
            rainfall = st.number_input("Rainfall", 0.0)

    else:
        sample = df.sample(1).iloc[0]

        N, P, K = sample['N'], sample['P'], sample['K']
        temperature, humidity = sample['temperature'], sample['humidity']
        ph, rainfall = sample['ph'], sample['rainfall']

        st.dataframe(sample)

    if st.button("🔍 Get Recommendation"):
        user = np.array([[N,P,K,temperature,humidity,ph,rainfall]])
        user_scaled = scaler_cluster.transform(user)

        distances = np.linalg.norm(kmeans.cluster_centers_ - user_scaled, axis=1)
        nearest = np.argsort(distances)[:3]

        results = {}

        for cid in nearest:
            cluster_data = df[df['cluster']==cid]
            total = len(cluster_data)

            counts = cluster_data['label'].value_counts()

            for crop, count in counts.items():
                prop = count/total
                score = (1/(distances[cid]+1e-5))*prop*(1/(cid+1))
                results[crop] = results.get(crop,0)+score

        sorted_res = sorted(results.items(), key=lambda x:x[1], reverse=True)

        st.subheader("🌾 Top Crops")
        for crop, score in sorted_res[:5]:
            st.write(f"🌱 {crop} → {score:.4f}")

    st.code("Score ∝ (1 / Distance) × Crop Proportion × Cluster Weight")
    