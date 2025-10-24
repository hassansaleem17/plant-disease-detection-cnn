
# 🌿 Plant Disease Detection using CNN

**Repository Name:** `plant-disease-detection-cnn`  
**Description:** Deep Learning model to detect plant leaf diseases using **TensorFlow** and **Keras**, trained on Kaggle’s *New Plant Diseases Dataset (Augmented)*. Includes a **Streamlit web app** for real-time leaf disease prediction.

---

## 🧠 Overview
This project uses a **Convolutional Neural Network (CNN)** to automatically classify plant leaf images as *healthy* or *diseased*.  
It helps in early detection of plant diseases and supports farmers and researchers in plant health monitoring using AI.

---

## 📊 Dataset
Dataset used:  
🔗 [New Plant Diseases Dataset (Augmented)](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)  

- Contains **87,000+ RGB images** of healthy and diseased crop leaves.  
- Divided into **38 categories** across multiple plant species.  
- Images are augmented for better generalization and robust model training.

---

## ⚙️ Project Structure
```

📂 Plant-Disease-Detection
│
├── pdd-01-train-plant-disease.ipynb     # Model training and evaluation
├── pdd-02-test-plant-disease.ipynb      # Model testing and visualization
├── main.py                              # Streamlit app for real-time prediction
└── README.md                            # Project documentation

````

---

## 🚀 How to Run

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/plant-disease-detection-cnn.git
cd plant-disease-detection-cnn
````

### 2️⃣ Install Dependencies

### 3️⃣ Train the Model

```bash
jupyter notebook pdd-01-train-plant-disease.ipynb
```

This will train the CNN model and save it as:

```
pdd_trained_model.keras
```

### 4️⃣ Test the Model

```bash
jupyter notebook pdd-02-test-plant-disease.ipynb
```

### 5️⃣ Launch the Streamlit App

```bash
streamlit run main.py
```

Upload a plant leaf image to get real-time predictions of whether it’s **healthy** or **diseased**, and view the predicted **disease name**.

---

## 🧩 Model Architecture

| Layer Type           | Details                         |
| -------------------- | ------------------------------- |
| Input                | 128×128×3 RGB image             |
| Convolutional Layers | 3 layers with ReLU activation   |
| Pooling Layers       | MaxPooling2D                    |
| Dense Layers         | Fully connected, ReLU + Softmax |
| Optimizer            | Adam                            |
| Loss Function        | Categorical Crossentropy        |
| Accuracy             | ~97% on validation data         |

---

## 🌱 Features

* ✅ High accuracy plant disease classification
* ✅ Real-time prediction using Streamlit
* ✅ Trained on a large, augmented dataset
* ✅ Easy to deploy locally or on cloud

---

## 📦 Dependencies

```
tensorflow
numpy
matplotlib
pandas
streamlit
opencv-python
scikit-learn
```

---

## 📈 Results

* **Training Accuracy:** ~98%
* **Validation Accuracy:** ~97%
* **Loss:** Low cross-entropy — strong generalization observed

---

## 🧾 Example Usage

Once the Streamlit app is running:

1. Upload a leaf image.
2. The app preprocesses the image and feeds it to the CNN model.
3. You’ll see the predicted **disease class** and **confidence score**.

---

## 👨‍💻 Author

**Hassan Saleem**
📧 [chhassan1041@gmail.com](mailto:chhassan1041@gmail.com)

---
