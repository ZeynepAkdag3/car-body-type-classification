import os
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Klasör yollarını senin proje yapına göre otomatik bulma
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'car_classification_model.h5')
JSON_PATH = os.path.join(BASE_DIR, 'models', 'class_mapping.json')
import sys
# Kullanim: python src/test_model.py [fotograf_yolu]
FOTO_YOLU = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE_DIR, 'test_arabasi.jpg')

# 1. Model ve Etiketleri Yükle
print("✓ Yapay zeka beyni yükleniyor...")
model = load_model(MODEL_PATH)

with open(JSON_PATH, 'r') as f:
    class_mapping = json.load(f)

# JSON key'leri string'dir, onları integer'a çevirelim
class_labels = {int(k): v for k, v in class_mapping.items()}

# 2. Test Edeceğimiz Fotoğrafı Hazırla
if not os.path.exists(FOTO_YOLU):
    print(f"HATA: {FOTO_YOLU} bulunamadı! Lütfen fotoğrafı proje ana klasörüne atın.")
else:
    img = image.load_img(FOTO_YOLU, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Modele uygun formata getiriyoruz

    # 3. Tahmin Et!
    tahmin_oranlari = model.predict(img_array)
    kazanan_index = np.argmax(tahmin_oranlari)
    araba_turu = class_labels[kazanan_index]
    guven_orani = np.max(tahmin_oranlari) * 100

    print("\n" + "="*35)
    print(f"🚗 TAHMİN EDİLEN ARAÇ: {araba_turu}")
    print(f"📊 Emin Olma Oranı:  %{guven_orani:.2f}")
    print("="*35)
