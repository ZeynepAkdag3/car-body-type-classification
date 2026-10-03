import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Yolları ayarla
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'car_classification_model.h5')
VAL_DIR = os.path.join(BASE_DIR, 'dataset', 'validation')
CLASSES = ["SUV", "VAN", "STATION_WAGON", "MICRO", "OPEN_WHEEL", "SEDAN", "HATCHBACK", "PICKUP"]

print("✓ Doğru model yükleniyor...")
model = load_model(MODEL_PATH)

# Test verisini KARIŞTIRMADAN oku (Sihirli shuffle=False kuralımız)
test_datagen = ImageDataGenerator()
test_gen = test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    classes=CLASSES,
    shuffle=False
)

print("✓ Pikseller analiz ediliyor (1-2 dakika sürebilir)...")
y_true = test_gen.classes
y_pred_proba = model.predict(test_gen)
y_pred = np.argmax(y_pred_proba, axis=1)

# Karmaşıklık matrisini hesapla ve 0-1 arasına normalize et
cm = confusion_matrix(y_true, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Şık bir Isı Haritası (Heatmap) Çizdir
plt.figure(figsize=(10, 8))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
plt.title('Normalized Confusion Matrix (%79.16 Doğruluk)')
plt.ylabel('Gerçek Sınıf')
plt.xlabel('Tahmin Edilen Sınıf')
plt.xticks(rotation=45)
plt.tight_layout()

# Resmi reports klasörüne kaydet
kayit_yolu = os.path.join(BASE_DIR, 'reports', 'dogru_confusion_matrix.png')
plt.savefig(kayit_yolu)
print(f"✅ HARİKA! Yeni ve doğru grafik başarıyla kaydedildi: {kayit_yolu}")