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






1-Month Deep Learning Roadmap 
Theme: Learn → Apply → Build (with weekly hands-on use case) 
Goals by End of Month: - Understand DL fundamentals - Implement MLP, CNN, RNN/LSTM, and NLP pipelines - Complete 4 hands-on projects - Use real-world datasets 
Phase 1: Foundations of Neural Networks (Basic NN) - 1 Week 
Goal: Understand the math, structure, and intuition of basic fully connected neural 
networks (MLP). 
Topics: - Intro to Deep Learning and Neural Nets - Neuron, weights, bias, activation functions (ReLU, Sigmoid, Tanh) - Loss functions (MSE, CrossEntropy) - Optimizers (SGD, Adam) - Forward and backward propagation - Overfitting, underfitting, regularization 
Hands-On: - Build a neural network from scratch using NumPy - Re-implement using PyTorch or TensorFlow - Visualize loss curves and accuracy 
Project: Binary and Multi-class Classifier on Tabular Data - Dataset: Iris / Breast Cancer / Titanic - Build and evaluate MLP - Apply dropout, regularization 
Phase 2: Convolutional Neural Networks (CNNs) - 1-1.5 Week 
Goal: Master image data representation and how CNNs learn spatial hierarchies. 
Topics: - Convolution, filters, feature maps - Padding, stride, pooling - Dropout & Batch Normalization - Architecture: LeNet, VGG, ResNet (intro) 
Hands-On: - Build CNN from scratch for image classification - Use data augmentation - Try transfer learning 
Project: Image Classifier on CIFAR-10 or Fashion-MNIST - Add batch norm, dropout - Bonus: Transfer learning on custom dataset 
Phase 3: Sequential Models (RNNs, LSTMs, GRUs) - 1-1.5 Week 
Goal: Understand how DL handles sequential/temporal data. 
Topics: - Sequential data overview - RNNs and their limitations - LSTM and GRU - Sequence modeling patterns 
Hands-On: - RNNs/LSTMs on synthetic/time series data - Pattern classification - Visualize hidden states 
Project: Stock Price Prediction / Sequence Pattern Recognition - Dataset: Stock data or synthetic sequences - Use LSTM/GRU - Plot predictions over time 