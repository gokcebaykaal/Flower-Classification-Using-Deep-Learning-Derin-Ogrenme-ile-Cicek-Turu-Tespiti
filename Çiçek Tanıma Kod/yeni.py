import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


folder_dir    = r"C:\Users\EXCALIBUR\OneDrive\Masaüstü\Çiçek Tanıma Kod\archive\flowers"
SIZE          = 128
batch_size    = 32
model_path    = "flower_mobilenet.h5"


train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.4,
    height_shift_range=0.4,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    folder_dir, target_size=(SIZE, SIZE),
    batch_size=batch_size, class_mode='categorical',
    subset='training', shuffle=True
)
val_gen = val_datagen.flow_from_directory(
    folder_dir, target_size=(SIZE, SIZE),
    batch_size=batch_size, class_mode='categorical',
    subset='validation', shuffle=False
)

categories  = list(train_gen.class_indices.keys())
num_classes = len(categories)
print("Eğitilen sınıflar:", categories)


es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
rl = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)


if os.path.exists(model_path):
    print("Kayıtlı model bulundu. Yükleniyor…")
    model = load_model(model_path)
    history1, history2 = None, None
else:
    print("Yeni model oluşturuluyor: MobileNetV2 + Fine-Tuning")
    base = MobileNetV2(input_shape=(SIZE,SIZE,3), include_top=False, weights='imagenet')
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    preds = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=preds)

    
    for layer in base.layers:
        layer.trainable = False
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    history1 = model.fit(
        train_gen, initial_epoch=0, epochs=10,
        validation_data=val_gen, callbacks=[es, rl], verbose=1
    )

    
    for layer in base.layers[-50:]:
        layer.trainable = True
    model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    history2 = model.fit(
        train_gen, initial_epoch=10, epochs=50,
        validation_data=val_gen, callbacks=[es, rl], verbose=1
    )

    model.save(model_path)
    print("Model eğitildi ve kaydedildi!")


if history1 is not None and history2 is not None:
    
    h1, h2 = history1.history, history2.history
    history = {k: h1[k] + h2[k] for k in h1}

    epochs = range(1, len(history['accuracy']) + 1)

    plt.figure()
    plt.plot(epochs, history['accuracy'], label='Train Acc')
    plt.plot(epochs, history['val_accuracy'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(epochs, history['loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


val_iter = iter(val_gen)

fig, ax = plt.subplots(6, 6, figsize=(18,18))
plt.subplots_adjust(wspace=0.5, hspace=0.6)

for i in range(6):
    for j in range(6):
        imgs, lbls = next(val_iter)
        img       = imgs[0]
        true_idx  = np.argmax(lbls[0])
        pred_idx  = np.argmax(model.predict(img[np.newaxis,...], verbose=0)[0])
        color = 'green' if pred_idx == true_idx else 'red'
        rect  = patches.Rectangle((0,0), SIZE, SIZE, linewidth=3,
                                  edgecolor=color, facecolor='none')
        ax[i,j].imshow(img); ax[i,j].add_patch(rect); ax[i,j].axis('off')
        ax[i,j].set_title(f"T:{categories[true_idx]}\nP:{categories[pred_idx]}",
                          color=color, fontsize=10)

plt.tight_layout()
plt.show()


def predict_flower_interactive():
    while True:
        path = input("Resim yolunu gir (çıkmak için 'q'): ")
        if path.lower() == 'q': break
        if not os.path.exists(path):
            print("❌ Dosya bulunamadı, tekrar deneyin."); continue
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (SIZE, SIZE)) / 255.0
        cls = categories[np.argmax(model.predict(img[np.newaxis,...], verbose=0)[0])]
        plt.imshow(img); plt.title(cls, color='green'); plt.axis('off'); plt.show()
        print(f"Tahmin edilen çiçek: {cls}\n")

predict_flower_interactive()
