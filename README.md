🧠 ML Weekly Progress Notes
This repository contains weekly notes and experiments from my machine learning journey. Each week covers different topics, including code, visualizations, and key takeaways.

📘 Week 1: ML Fundamentals
Topics Covered:

Supervised vs. Unsupervised Learning

Classification vs. Regression

Data preprocessing basics (handling missing values, encoding, scaling)

Exploratory Data Analysis (EDA)

Code Highlights:

Used Pandas and Seaborn for data analysis

Applied StandardScaler, LabelEncoder, and visualized correlations

Key Learnings:

Importance of clean, scaled data

How feature distributions impact model performance

📘 Week 2: Classical ML Algorithms
Topics Covered:

K-Nearest Neighbors (KNN)

Decision Trees

Naive Bayes

Model Evaluation Metrics (Accuracy, F1 Score, Confusion Matrix)

Code Highlights:

Trained and compared KNN, NB, and DT using cross_val_predict

Visualized confusion matrices with seaborn

Printed classification reports

Key Learnings:

Trade-offs between simplicity and accuracy

Decision Trees can overfit without pruning

KNN sensitive to feature scaling and k value

📘 Week 3: Unsupervised Learning (K-Means Clustering)
Topics Covered:

K-Means Clustering

Feature scaling and dimensionality impact

Cluster interpretation

Labeling clusters (e.g., "Frugal Elders", "Impulsive Youth")

Code Highlights:

Used StandardScaler and KMeans

Plotted elbow method to find optimal k

Integrated model into a FastAPI web app with prediction capability

Key Learnings:

Unsupervised learning requires intuition in interpreting clusters

Cluster centers can give insights into consumer segmentation

Web integration makes ML more interactive and user-friendly

🔧 Running the App
bash
Copy
Edit
uvicorn app:app --reload
Visit: http://127.0.0.1:8000

Choose model, enter values, and predict segment