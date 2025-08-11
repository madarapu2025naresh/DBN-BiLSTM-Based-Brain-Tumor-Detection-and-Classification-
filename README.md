# DBN-BiLSTM-Based-Brain-Tumor-Detection-and-Classification-
Deep Belief Network with Bidirectional Long Short-Term Memory (BiLSTM) Based Brain Tumor Detection and Classification Using Magnetic Resonance Imaging (MRI) Images



Description –Overview of the code/dataset

The provided code implements a complete pipeline for brain tumor detection and classification using MRI images, integrating Deep Belief Networks (DBN) for hierarchical spatial feature extraction and Bidirectional Long Short-Term Memory (BiLSTM) for capturing sequential dependencies. 
It includes modules for image preprocessing (noise reduction, contrast enhancement), segmentation, feature extraction via Barnacles Mating Optimizer (BMO), and final classification.


Dataset Information
The dataset consists of 3,264 MRI brain images sourced from publicly available repositories (e.g., Kaggle), categorized into four classes: no tumor (1,426 images), meningioma (708), pituitary (930), and glioma (708). 
All images are resized to 224×224 pixels for model compatibility. The code supports reproducible training, evaluation (accuracy, precision, recall, F1-score, RMSE, processing time), and visualization of performance metrics.

Code
The code implements a full deep learning workflow for brain tumor detection and classification using DBN-BiLSTM. 
It includes modules for data loading, preprocessing (resizing, normalization, noise reduction), segmentation, feature extraction via Barnacles Mating Optimizer (BMO), and classification. 
The training phase uses accuracy, precision, recall, and F1-score as metrics, with visualizations for performance evaluation. The implementation is in Python using TensorFlow/Keras and supports reproducibility with configurable parameters.


Usage Instructions – How to Use or Load the Dataset and Code
Download the dataset
Obtain the brain MRI dataset from Kaggle: Brain Tumor MRI Dataset
Ensure the dataset folder contains subdirectories for each class:
Copy
Edit
dataset/
  no_tumor/
  meningioma/
  pituitary/
  glioma/
Place dataset in project directory
Move or extract the dataset folder into the same directory as the code or update the dataset path in the notebook/script.
Install dependencies
bash
Copy
Edit
pip install tensorflow keras opencv-python scikit-image numpy pandas matplotlib seaborn
Load and preprocess dataset
The code automatically loads images from the dataset path, resizes them to 224×224, normalizes pixel values, and applies optional preprocessing such as denoising and contrast enhancement.
Run the notebook/script
Open the Jupyter Notebook (Brain tumor.ipynb) or Python script.
Execute cells sequentially to perform training, evaluation, and visualization.
Evaluate results
Review accuracy, precision, recall, F1-score, RMSE, and execution time metrics.
Visualize confusion matrices, training curves, and Grad-CAM heatmaps.


Requirements
Flask
flask-cors
gunicorn
pandas
scikit-learn
numpy
matplotlib
Pillow
opencv-python-headless
keras
tensorflow
fastapi
uvicorn
pybase64
typing
pydantic

Citations
https://www.kaggle.com/datasets/bilalakgz/brain-tumor-mri-dataset

