# 🌾 Crop Advisory System | Hybrid Machine Learning Approach

A complete end-to-end Crop Recommendation System that combines **Supervised Learning** and **Unsupervised Learning** to provide intelligent and flexible crop suggestions based on soil and environmental conditions.

This project goes beyond traditional ML models by introducing a **Hybrid Clustering + Distance-Based Recommendation System**.

---

## 📌 Problem Statement

Selecting the right crop based on soil and climate conditions is critical for maximizing agricultural productivity.

This system uses:

* Nitrogen (N)
* Phosphorus (P)
* Potassium (K)
* Temperature
* Humidity
* pH
* Rainfall

to recommend the most suitable crops.

---

# 🧠 APPROACH OVERVIEW

This project has **TWO APPROACHES**:

---

## 🔹 1️⃣ Supervised Learning Approach (Baseline)

### ✔ Models Used:

* Logistic Regression
* Decision Tree
* Random Forest
* Support Vector Machine (SVM)
* AdaBoost
* Gradient Boosting

### 🏆 Best Model:

👉 **Random Forest Classifier**

### 📊 Performance:

* Test Accuracy: ~99%
* Cross Validation Accuracy: ~99%
* Very low variance → Highly stable model

---

### 🌱 Top-3 Recommendation (Classification)

* Used `predict_proba()`
* Returns Top 3 crops with probability
* Adds confidence level (Very High / High / Low)

---

## 🔥 2️⃣ Hybrid Clustering-Based Recommendation (CORE INNOVATION)

This is the **main highlight of the project**.

---

### 📌 Step 1: Clustering

* Applied **KMeans Clustering**
* Optimal K selected using:

  * Elbow Method
  * Silhouette Score

---

### 📌 Step 2: Cluster Understanding

* PCA visualization of clusters
* Crop distribution inside each cluster
* Identification of dominant crops per cluster

---

### 💡 Step 3: Distance-Based Recommendation (Custom Formula)

```
Score = (1 / Distance) × Crop_Proportion × Cluster_Weight
```

---

### 🔍 Explanation (Simple Terms)

* **Distance** → Measures how close the input soil conditions are to a cluster center
* **Crop Proportion** → Percentage of a crop inside that cluster
* **Cluster Weight** → Importance of cluster based on number of samples

---

### ⚙️ How Prediction Works

For each new input:

1. Compute distance to all cluster centroids
2. Select **Top 3 nearest clusters**
3. Calculate:

   * Cluster weight
   * Crop proportion
4. Apply scoring formula
5. Rank crops

👉 Final output = **Top recommended crops (not just one prediction)**

---

## 📊 Key Observations

| Concept          | Insight                         |
| ---------------- | ------------------------------- |
| Silhouette Score | Measures cluster quality        |
| Accuracy         | Measures prediction performance |

### ⚠️ Important Finding:

* Lower number of clusters → Better cluster separation
* Higher number of clusters → Better prediction accuracy

👉 Clustering quality and prediction accuracy are different objectives

---



---

## 📊 Model Evaluation

* Accuracy Score (for clustering-based prediction)
* Confusion Matrix
* Adjusted Rand Index (ARI)
* Normalized Mutual Information (NMI)
* Silhouette Score

---

## 🔬 Feature Importance (From Random Forest)

Top influencing features:

* Rainfall 🌧️
* Humidity 💧
* Potassium (K)

---

## 🛠 Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Jupyter Notebook

---

## 📂 Project Structure

```
Crop_Advisory_system/
│
├── data/
│   └── Crop_recommendation.csv
│
├── notebooks/
│   ├── Crop_Model_Tuning.ipynb
│   ├── CropAdvisory.ipynb
│   ├── Crop_clustering_analysis.ipynb
│   └── experiment_clustering.ipynb   ⭐ FINAL MODEL
│
├── models/
│   ├── crop_recommendation_model.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
│
├── requirements.txt
└── README.md
```

---

## 🧠 Key Contributions

✔ Converted clustering into a recommendation system
✔ Designed a custom scoring formula
✔ Combined distance + probability + cluster importance
✔ Compared supervised vs unsupervised approaches
✔ Built an interpretable AI system

---

## 📌 Conclusion

This project demonstrates how machine learning can be extended beyond traditional classification to solve real-world problems.

By combining:

* Clustering (pattern discovery)
* Distance-based reasoning
* Probability distributions

we created a **hybrid intelligent crop advisory system** that is:

✔ Flexible
✔ Interpretable
✔ Practical

---

## 🔮 Future Scope

* Integration with real-time soil sensors
* Weather API integration
* Web or mobile deployment
* Advanced clustering methods (DBSCAN, GMM)
* Hybrid ensemble (Clustering + Classification)

---

## 👨‍💻 Author

**Amrit**
Machine Learning & AI Enthusiast

---

⭐ If you found this useful, please star the repository!
