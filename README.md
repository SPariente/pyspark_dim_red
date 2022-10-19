# Dimensionality reduction using PySpark and AWS cloud

Small project to showcase the use of PySpark for data preprocessing and simple ML modelling.
The project was setup on AWS, using S3 to store all data and EC2 (t2.micro) for processing.
The [Fruits 360](https://www.kaggle.com/datasets/moltean/fruits) dataset was used, using only a small sample (5 random images by fruit) for this POC. 

Contents:
- 1. pca-aws.ipynb: dimensionality reduction using a PCA
- 2. logreg-aws.ipynb: logistic regression following this PCA-based dimensionality reduction
- 3. pca-aws-tf.ipynb: dimensionality reduction using a ResNet50-based embedding followed by a PCA
- functions.py: custom functions used in the project
- pca_transform.csv: output from the PCA-based dimensionality reduction
