# 🌸 Flower Classification Using Deep Learning  
### Derin Öğrenme ile Çiçek Türü Tespiti

---

## 📌 Description | Açıklama

**English:**  
This project implements a **flower classification system** using **Deep Learning (MobileNetV2)**.  
The model is trained to recognize multiple flower species by applying **transfer learning**, **data augmentation**, and optimization techniques such as **early stopping**.  
It demonstrates an end-to-end ML workflow: dataset preparation, training, evaluation, and prediction.

**Türkçe:**  
Bu proje, **Derin Öğrenme (MobileNetV2)** kullanarak **çiçek türü sınıflandırma sistemi** uygular.  
Model, farklı çiçek türlerini tanıyacak şekilde eğitilmiş olup, **transfer learning**, **veri artırma** ve **early stopping** gibi yöntemlerle optimize edilmiştir.  
Proje, uçtan uca bir makine öğrenimi iş akışını (veri hazırlama, eğitim, değerlendirme ve tahmin) göstermektedir.

---

## 🚀 Features | Özellikler
- Pretrained **MobileNetV2** with transfer learning  
- Data preprocessing and augmentation (rotation, flip, normalization)  
- Training with **Adam optimizer**, **early stopping**  
- High accuracy and robust performance on test data  
- Easy inference on new flower images  

---

## 🛠️ Tech Stack | Teknolojiler
- Python  
- TensorFlow / Keras  
- NumPy, Pandas, Matplotlib  
- scikit-learn  

---

## 📂 Project Structure | Proje Yapısı
├── data/ # Flower dataset
├── notebooks/ # Jupyter notebooks for experiments
├── models/ # Saved trained models
├── flower_classification.py # Main training & evaluation script
├── predict.py # Script for testing with new images
└── README.md

---

## ▶️ Usage | Kullanım

**English:**  
1. Clone the repository:  
   ```bash
   git clone https://github.com/gokcebaykaal/Flower-Classification-Using-Deep-Learning-Derin-Ogrenme-ile-Cicek-Turu-Tespiti.git
2.Install dependencies:

pip install -r requirements.txt

3.Train the model:

python flower_classification.py


4.Predict on a new image:

python predict.py --image sample.jpg

**Türkçe:**

1.Depoyu klonlayın:

git clone https://github.com/gokcebaykaal/Flower-Classification-Using-Deep-Learning-Derin-Ogrenme-ile-Cicek-Turu-Tespiti.git


2.Gerekli kütüphaneleri yükleyin:

pip install -r requirements.txt


3.Modeli eğitmek için:

python flower_classification.py


4.Yeni bir görsel üzerinde tahmin yapmak için:

python predict.py --image ornek.jpg
