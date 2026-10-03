import matplotlib.pyplot as plt
import os

# Zeynep'in Colab loglarından alınan GERÇEK eğitim verileri
# 1-10 arası Warm-Up, 11-20 arası Fine-Tuning
acc = [0.6112, 0.7371, 0.7666, 0.7927, 0.8101, 0.8128, 0.8364, 0.8368, 0.8412, 0.8499, 0.6195, 0.6842, 0.7176, 0.7476, 0.7622, 0.7778, 0.7870, 0.7993, 0.8055, 0.8066]
val_acc = [0.7448, 0.8249, 0.8447, 0.8550, 0.8700, 0.8716, 0.8938, 0.8803, 0.9017, 0.9089, 0.8320, 0.8154, 0.8162, 0.8273, 0.8455, 0.8542, 0.8605, 0.8669, 0.8748, 0.8803]
loss = [1.0202, 0.6739, 0.6013, 0.5503, 0.4968, 0.4923, 0.4463, 0.4429, 0.4269, 0.4116, 1.2940, 0.9389, 0.8058, 0.7104, 0.6675, 0.6126, 0.6061, 0.5377, 0.5136, 0.5155]
val_loss = [0.6707, 0.4677, 0.3871, 0.3786, 0.3281, 0.3313, 0.2863, 0.3114, 0.2651, 0.2610, 0.4700, 0.5234, 0.4900, 0.4488, 0.4012, 0.3775, 0.3530, 0.3388, 0.3222, 0.3130]

epochs = range(1, len(acc) + 1)

# Çift pencereli profesyonel grafik oluşturma
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 1. LOSS GRAFİĞİ
ax1.plot(epochs, loss, 'b-', label='Training Loss', linewidth=2)
ax1.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
ax1.axvline(x=10.5, color='gray', linestyle='--', alpha=0.7) # İki aşamayı ayıran çizgi
ax1.text(5, max(loss)*0.9, 'Aşama 1:\nWarm-up', ha='center', color='gray')
ax1.text(15, max(loss)*0.9, 'Aşama 2:\nFine-Tuning', ha='center', color='gray')
ax1.set_title('Training and Validation Loss', fontweight='bold')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.grid(True, alpha=0.3)
ax1.legend()

# 2. ACCURACY GRAFİĞİ
ax2.plot(epochs, acc, 'b-', label='Training Accuracy', linewidth=2)
ax2.plot(epochs, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
ax2.axvline(x=10.5, color='gray', linestyle='--', alpha=0.7)
ax2.set_title('Training and Validation Accuracy', fontweight='bold')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()

# Doğru grafiği reports klasörüne kaydet
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
kayit_yolu = os.path.join(BASE_DIR, 'reports', 'dogru_training_history.png')
plt.savefig(kayit_yolu, dpi=300, bbox_inches='tight')

print(f"✅ HARİKA! Yeni ve %90 barajını aşan doğru eğitim grafiği başarıyla kaydedildi: {kayit_yolu}")