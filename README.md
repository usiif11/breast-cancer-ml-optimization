# -Breast-Cancer-Classification-using-Genetic-Algorithm-Random-Forest
Breast Cancer Classification using Genetic Algorithm + Random Forest 

📌 Overview 

This project builds a Machine Learning pipeline for Breast Cancer classification (malignant vs benign) using: 

Genetic Algorithm (GA) for feature selection  

Random Forest Classifier for model training  

Stratified K-Fold Cross Validation for reliable evaluation  

Optimization objective: maximize the classification accuracy. 

The dataset used is the Breast Cancer Wisconsin Dataset provided by sklearn.datasets. 

 

🧠 Project Pipeline 

1️⃣ Load Dataset 

569 samples  

30 numerical features  

Target:  

0 → Benign  

1 → Malignant 

2️⃣ Genetic Algorithm Setup 

Generations: 20  

Population size: 50  

Mutation rate: 0.05  

Feature probability (INDPB): 0.02  

Fitness metric: Mean accuracy across Stratified K-Fold (CV=5) 

3️⃣ Chromosome Structure 

Binary vector of size = number of features  

1 → Feature INCLUDED  

0 → Feature EXCLUDED 

4️⃣ Fitness Function 

For each chromosome: 

Select the included features  

Train a RandomForestClassifier  

Evaluate using Stratified K-Fold  

Return mean accuracy as the fitness score 

 

🚀 Optimization Goal 

The Genetic Algorithm searches for: 

Best subset of features  

Best Random Forest hyperparameters  

Best classification accuracy 

 

📊 Results (from your notebook) 

Selected features count: 9  

Selected feature names:  

mean texture  

mean concave points  

mean symmetry  

mean fractal dimension  

area error  

worst texture  

worst area  

worst smoothness  

worst concavity 

Best Hyperparameters Found: 

n_estimators: 474 

max_depth: 24 

min_samples_split: 3 

min_samples_leaf: 4 

max_features: 0.155 

Final test AUC: 0.9937 

Accuracy: 0.9825  

All-features 

 test AUC: 0.9937 

Accuracy: 0.9561 

 

 
