"""
Araba Gövde Tipi Sınıflandırma - Veri Toplama Scripti
Bu script Kaggle veri setlerinden görüntü indirmek için yardımcı olur.

Kullanım:
1. Kaggle API kimlik bilgilerini ~/.kaggle/kaggle.json'a koy
2. Aşağıdaki datasetleri indir ve dataset/ klasörüne taşı
"""

import os
import shutil
import urllib.request
from pathlib import Path

# 8 araba sınıfı
CLASSES = [
    "SUV",
    "VAN",
    "STATION_WAGON",
    "MICRO",
    "OPEN_WHEEL",
    "SEDAN",
    "HATCHBACK",
    "PICKUP"
]

def create_dataset_structure():
    """Veri seti klasör yapısını oluştur"""
    base_path = Path("dataset")
    
    for split in ["train", "validation", "test"]:
        for class_name in CLASSES:
            class_path = base_path / split / class_name
            class_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ {class_path} oluşturuldu")

def download_kaggle_datasets():
    """Kaggle veri setlerini indir (CLI yöntemi)"""
    print("\n=== Kaggle Veri Setlerini İndirme ===")
    print("Aşağıdaki komutları terminalde çalıştır:")
    print("\n1. Cars Body Type Cropped dataset:")
    print("   kaggle datasets download -d ademboukhris/cars-body-type-cropped")
    print("\n2. Stanford Car Body Type Data:")
    print("   kaggle datasets download -d mayurmahurkar/stanford-car-body-type-data")
    print("\nKaggle API kurulumu:")
    print("   pip install kaggle")
    print("   https://www.kaggle.com/account/login?phase=signin adresinden API token indir")
    print("   ~/.kaggle/kaggle.json dosyasına kaydet")

def info():
    """Veri toplama talimatları"""
    print("\n=== ARABA GÖVDESİ TİPİ SINIFLANDIRMASI - VERİ TOPLAMA KILAVUZU ===\n")
    print("Her sınıf için en az 150-200 görüntü toplayınız.\n")
    
    print("Önerilen kaynaklar:")
    print("• Google Images")
    print("• Kaggle Datasets (referans)")
    print("• Unsplash, Pexels")
    print("• Car manufacturer websites\n")
    
    print("Sınıflar:")
    for i, cls in enumerate(CLASSES, 1):
        print(f"{i}. {cls}")
    
    print("\nYönerge:")
    print("1. Görselleri indirdikten sonra sınıf klasörlerine taşıyın")
    print("2. Her sınıf için train/, validation/, test/ klasörlerine dağıtın")
    print("3. Ön işleme scripti görselleri otomatik olarak işler")

if __name__ == "__main__":
    create_dataset_structure()
    info()
    download_kaggle_datasets()
    print("\n✓ Veri seti yapısı hazır. Görselleri indirip ilgili klasörlere yerleştirin.")
