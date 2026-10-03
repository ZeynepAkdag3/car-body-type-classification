import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import time
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Araç Gövde Sınıflandırma", page_icon="🏎️", layout="wide")

# --- MİNİMALİST TASARIM İÇİN CSS ---
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {padding-top: 2rem; padding-bottom: 0rem;}
    .stButton>button {
        width: 100%; 
        border-radius: 8px; 
        height: 3em; 
        font-weight: bold; 
        background-color: #2C3E50; 
        color: white;
        border: 2px solid #2C3E50;
    }
    .stButton>button:hover {
        background-color: white;
        color: #2C3E50;
    }
    </style>
    """, unsafe_allow_html=True)

# Klasör yolları
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'car_classification_model.h5')
JSON_PATH = os.path.join(BASE_DIR, 'models', 'class_mapping.json')

# Modeli bellekte tutmak için
@st.cache_resource
def load_ai_model():
    model = load_model(MODEL_PATH)
    with open(JSON_PATH, 'r') as f:
        class_mapping = json.load(f)
    class_labels = {int(k): v for k, v in class_mapping.items()}
    return model, class_labels

# --- BAŞLIK ---
st.markdown("<h1 style='text-align: center; color: #2C3E50;'>Araç Gövde Tipi Sınıflandırma Sistemi</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #7F8C8D;'>Derin Öğrenme (EfficientNet) tabanlı yapay zeka analiz arayüzü</p>", unsafe_allow_html=True)
st.markdown("---")

# Model yükleme kontrolü
try:
    model, class_labels = load_ai_model()
except Exception as e:
    st.error("Model dosyaları bulunamadı. Lütfen 'models' klasörünü kontrol edin.")
    st.stop()

# BÖLÜM 1: Görüntü Yükleme Alanı
st.markdown("### 1. Adım: Test Görseli Yükleme")
uploaded_file = st.file_uploader("Lütfen analiz edilecek araç fotoğrafını sürükleyip bırakın (JPG, PNG)", type=["jpg", "jpeg", "png"])

# BÖLÜM 4: Yüklenen Görsel ve Sonuç Yan Yana
if uploaded_file is not None:
    st.markdown("---")
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 📷 Yüklenen Görüntü")
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_container_width=True)
        
        # BÖLÜM 2: Tahmin Butonu
        st.markdown("<br>", unsafe_allow_html=True)
        tahmin_et = st.button("🔍 SINIFLANDIR")

    with col2:
        st.markdown("### 📊 Analiz Sonuçları")
        
        if tahmin_et:
            # Süre ölçümünü başlat
            baslangic_zamani = time.time()
            
            with st.spinner('Model görüntü üzerinde çalışıyor...'):
                # Görüntüyü hazırla
                img = image.resize((224, 224))
                img_array = img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)

                # Tahmin yap
                tahmin_oranlari = model.predict(img_array)[0]
                kazanan_index = np.argmax(tahmin_oranlari)
                araba_turu = class_labels[kazanan_index]
                
                # Güven oranını hesapla
                guven_orani = tahmin_oranlari[kazanan_index] * 100
                
            # Süre ölçümünü bitir
            bitis_zamani = time.time()
            gecen_sure = bitis_zamani - baslangic_zamani

            # BÖLÜM 3: Tahmin Sonucu Çıktısı
            st.success(f"## **TESPİT EDİLEN SINIF: {araba_turu}**")
            
            # --- YENİ EKLENEN KISIM: GÜVEN ORANI ---
            st.info(f"**Doğruluk / Güven Oranı:** %{guven_orani:.2f}")
            
            # ... (önceki kısımlar aynı kalacak)
            st.caption(f"⏱️ Model Yanıt Süresi: {gecen_sure:.3f} saniye") 
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### Tüm Sınıflar İçin Olasılık Dağılımı")
            
            # Verileri Hazırlama
            df_data = {"Sınıf": [], "Olasılık (%)": []}
            for i, prob in enumerate(tahmin_oranlari):
                df_data["Sınıf"].append(class_labels[i])
                df_data["Olasılık (%)"].append(float(prob * 100))
            
            df = pd.DataFrame(df_data).set_index("Sınıf")
            
            # 1. YENİLİK: Sayısal Oranları Tablo Olarak Göster (Yüzde formatında)
            st.dataframe(df.style.format("{:.2f}%"), use_container_width=True)
            
            # 2. Çubuk Grafik (Görsellik İçin)
            st.bar_chart(df, color="#3498DB", height=250)
            
        else:
            st.info("👆 Analizi başlatmak için soldaki 'SINIFLANDIR' butonuna tıklayın.")
else:
    # Sayfa boş görünmesin diye yer tutucu bilgi mesajı
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.info("💡 **Bilgi:** Sistemin arayüzü (Tahmin butonu, analiz grafikleri ve sonuç paneli) test görüntüsü yüklendikten sonra otomatik olarak açılacaktır. Sunum sırasında görüntüler yüklendiğinde analiz modülü devreye girecektir.")