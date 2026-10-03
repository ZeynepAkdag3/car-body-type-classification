"""
Araba Gövde Tipi Sınıflandırma - Model Eğitimi
Transfer Learning ile EfficientNetB0 kullanarak
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import pickle
import json

# Ayarlar
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001

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

NUM_CLASSES = len(CLASSES)

def create_model():
    """Transfer Learning modelini oluştur"""
    # Base model: EfficientNetB0 (pretrained ImageNet)
    base_model = EfficientNetB0(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Sadece son 30 katmanı (yaklaşık 1-2 blok) eğitime aç
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    # Custom top layers
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu', name='dense_256'),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu', name='dense_128'),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation='softmax', name='output')
    ])
    
    # Öğrenme hızını düşür ve modeli derle (1e-4 genellikle fine-tuning için idealdir)
    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model

def prepare_data():
    """Veri setini hazırla ve augmentation uygula"""
    
    # Data Augmentation
    # NOT: EfficientNetB0 kendi icinde normalizasyon yapar (0-255 girdi bekler).
    # Bu yuzden rescale=1./255 KULLANILMAZ; kullanilirsa model ogrenemez.
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.15,
        fill_mode='nearest'
    )
    
    validation_datagen = ImageDataGenerator()
    
    # Train generator
    train_generator = train_datagen.flow_from_directory(
        'dataset/train',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASSES
    )
    
    # Validation generator
    validation_generator = validation_datagen.flow_from_directory(
        'dataset/validation',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASSES
    )
    
    return train_generator, validation_generator

def train_model(model, base_model, train_gen, val_gen):
    """Modeli eğit"""
    
    # Early Stopping
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Model Checkpoint
    checkpoint = keras.callbacks.ModelCheckpoint(
        'models/best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    # YENİ: Akıllı Öğrenme Hızı Kontrolü
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.2, 
        patience=3, 
        min_lr=1e-6, 
        verbose=1
    )
    
    # Eğitim
    print("✓ Model eğitimi başlıyor...")
    history = model.fit(
        train_gen,
        steps_per_epoch=len(train_gen),
        validation_data=val_gen,
        validation_steps=len(val_gen),
        epochs=EPOCHS, # İstersen süreyi artırmak için EPOCHS'u 20 veya 30 yapabilirsin.
        callbacks=[early_stop, checkpoint, reduce_lr], # reduce_lr'ı buraya ekledik
        verbose=1
    )
    
    return history

def fine_tune_model(model, base_model, train_gen, val_gen):
    """Fine-tuning: Base modelin üst katmanlarını eğit"""
    
    # Base modelin son katmanlarını unloke et
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    # Daha düşük learning rate ile yeniden derle
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE / 10)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Kısa bir süre daha eğit
    print("✓ Fine-tuning başlıyor...")
    history_finetune = model.fit(
        train_gen,
        steps_per_epoch=len(train_gen),
        validation_data=val_gen,
        validation_steps=len(val_gen),
        epochs=10,
        verbose=1
    )
    
    return history_finetune

def evaluate_model(model):
    """Modeli test seti üzerinde değerlendir"""
    
    test_datagen = ImageDataGenerator()
    
    # Test klasöründe görüntü yoksa validation seti üzerinde test et
    try:
        test_generator = test_datagen.flow_from_directory(
            'dataset/test',
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            classes=CLASSES,
            shuffle=False
        )
        
        # Eğer test seti boş ise validasyon seti kullan
        if len(test_generator) == 0:
            print("Test seti bos, validasyon seti kullaniliyor...")
            test_generator = test_datagen.flow_from_directory(
                'dataset/validation',
                target_size=(IMG_SIZE, IMG_SIZE),
                batch_size=BATCH_SIZE,
                class_mode='categorical',
                classes=CLASSES,
                shuffle=False
            )
    except:
        print("Test klasoru bulunamadi, validasyon seti kullaniliyor...")
        test_generator = test_datagen.flow_from_directory(
            'dataset/validation',
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            classes=CLASSES,
            shuffle=False
        )
    
    # Tahminler
    y_true = test_generator.classes
    y_pred_proba = model.predict(test_generator)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Metrikleri hesapla
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrikleri
    report = classification_report(
        y_true, y_pred,
        target_names=CLASSES,
        output_dict=True
    )
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'report': report,
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_normalized': cm_normalized.tolist()
    }
    
    return metrics, cm_normalized, y_pred, y_true

def plot_results(history, cm_normalized):
    """Sonuçları görselleştir"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss grafiği
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy grafiği
    axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reports/training_history.png', dpi=300, bbox_inches='tight')
    print("✓ Training history grafiği kaydedildi: reports/training_history.png")
    
    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES, cbar_kws={'label': 'Frequency'})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('reports/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Confusion Matrix grafiği kaydedildi: reports/confusion_matrix.png")

def save_model(model):
    """Modeli kaydet"""
    model_path = 'models/car_classification_model.h5'
    model.save(model_path)
    
    # Model boyutunu kontrol et
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"✓ Model kaydedildi: {model_path}")
    print(f"  Model boyutu: {model_size_mb:.2f} MB (Limit: 95 MB)")
    
    # Class mapping
    class_mapping = {i: cls for i, cls in enumerate(CLASSES)}
    with open('models/class_mapping.json', 'w') as f:
        json.dump(class_mapping, f)
    
    return model_size_mb

if __name__ == "__main__":
    print("=== ARABA GÖVDE SINIFLANDIRMASI - MODEL EĞITIMI ===\n")
    
    # Dizinleri oluştur
    Path('models').mkdir(exist_ok=True)
    Path('reports').mkdir(exist_ok=True)
    
    # Modeli oluştur
    print("✓ Model oluşturuluyor...")
    model, base_model = create_model()
    model.summary()
    
    # Veriyi hazırla
    print("\n✓ Veri yükleniyor...")
    train_gen, val_gen = prepare_data()
    
    # Eğitim
    history = train_model(model, base_model, train_gen, val_gen)
    
    # Fine-tuning (opsiyonel)
    # history_finetune = fine_tune_model(model, base_model, train_gen, val_gen)
    
    # Modeli kaydet
    model_size = save_model(model)
    
    # Değerlendirme
    print("\n✓ Model değerlendiriliyor...")
    metrics, cm_normalized, y_pred, y_true = evaluate_model(model)
    
    # Sonuçları görselleştir
    plot_results(history, cm_normalized)
    
    # Metrikleri yazdır
    print("\n=== MODELİN PERFORMANSI ===")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    
    # Metrikleri kaydet
    with open('reports/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print("\n✓ Tüm sonuçlar reports/ klasörüne kaydedildi")
