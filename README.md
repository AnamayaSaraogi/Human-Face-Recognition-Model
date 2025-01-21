# Human-Face-Recognition-Model

This repository contains the implementation of a **Convolutional Neural Network (CNN)** for human face recognition. The project is designed to classify images into two categories: **human face** and **not a face**.  

## Repository Structure  

1. **`face_recognition.keras`**  
   - This file contains the pre-trained CNN model saved in Keras format.  
   - It can be directly used to perform face recognition tasks without needing to retrain the model.  

2. **`implementation.ipynb`**  
   - A Google Colab notebook demonstrating how to load and use the `face_recognition.keras` file for face recognition.  
   - This file walks users through the process of loading the model, feeding an input image, and obtaining predictions.  

3. **`Human_Face_Recognition_Model.ipynb`**  
   - A Google Colab notebook containing the complete code for building and training the CNN model from scratch.  
   - It includes dataset preprocessing, model architecture, training, and evaluation.  

## Dataset Structure  

The dataset used for training and evaluation is structured as follows:  
dataset/
│
├── human_face/ # Contains images of human faces
├── not_a_face/ # Contains images that are not human faces


## How to Use  

### 1. Use the Pre-trained Model  
To directly use the pre-trained model, follow these steps:  

- Open the `implementation.ipynb` notebook in Google Colab.  
- Upload the `face_recognition.keras` file from this repository.  
- Run the code to load the model and classify your input images.  

### 2. Train the Model from Scratch  
If you wish to train the model from scratch:  

- Open the `Human_Face_Recognition_Model.ipynb` notebook in Google Colab.  
- Ensure your dataset is organized as described in the "Dataset Structure" section.  
- Follow the steps in the notebook to preprocess the data, build the model, and train it.  

## Requirements  

- Python 3.x  
- TensorFlow >= 2.x  
- OpenCV  
- NumPy  

## Model Overview  

The CNN model is designed to extract features from images and classify them into two categories: **human face** and **not a face**. It uses:  

- Convolutional layers for feature extraction.  
- Max-pooling layers for dimensionality reduction.  
- Fully connected layers for classification.  

## Acknowledgments  

- **Keras**: Used for building and saving the model.  
- **Google Colab**: Used for implementation and training.  

## Future Scope  

- Expand the dataset to include more diverse images for improved generalization.  
- Optimize the model for deployment in real-time applications.  

---

Feel free to contribute or raise issues for further improvements!  

