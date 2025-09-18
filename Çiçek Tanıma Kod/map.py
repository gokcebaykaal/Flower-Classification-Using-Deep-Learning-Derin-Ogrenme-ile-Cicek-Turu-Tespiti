import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import average_precision_score

# --- AYARLAR ---
model_path = "flower_mobilenet.h5"
folder_dir = r"C:\Users\EXCALIBUR\OneDrive\Masaüstü\Çiçek Tanıma Kod\archive\flowers"
SIZE       = 128
batch_size = 32

# --- MODELİ YÜKLE ---
model = load_model(model_path)

# --- VAL DATA (SADECE RESCALE) ---
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_gen = val_datagen.flow_from_directory(
    folder_dir,
    target_size=(SIZE, SIZE),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

categories = list(val_gen.class_indices.keys())

# --- mAP HESAPLAMA ---
y_true = []
y_pred = []

val_gen.reset()
for i in range(len(val_gen)):
    x_batch, y_batch = val_gen[i]
    y_true.append(y_batch)
    y_pred.append(model.predict(x_batch, verbose=0))

# Stack veriler
y_true = np.vstack(y_true)
y_pred = np.vstack(y_pred)

# Her sınıf için AP hesapla
average_precisions = []
for i in range(y_true.shape[1]):
    ap = average_precision_score(y_true[:, i], y_pred[:, i])
    average_precisions.append(ap)
    print(f"Sınıf: {categories[i]} → AP: {ap:.4f}")

# Ortalama mAP
mAP = np.mean(average_precisions)
print(f"\n📊 Ortalama mAP: {mAP:.4f}")
