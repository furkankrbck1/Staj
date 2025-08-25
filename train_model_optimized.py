import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt

print("TensorFlow versiyonu:", tf.__version__)
print("Kullanılabilir GPU'lar:", tf.config.list_physical_devices('GPU'))

# GPU bellek optimizasyonu
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # TF bellek büyümesini optimize et
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # GPU konfigürasyonunu ayarla
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)]  # 6GB GPU bellek
        )
        print("GPU başarıyla konfigüre edildi!")
        
        # Mixed precision training aktifleştir
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Mixed precision training aktif!")
        
    except RuntimeError as e:
        print("GPU konfigürasyonu sırasında hata:", e)
else:
    print("GPU bulunamadı, CPU kullanılacak")

# Veri seti yapılandırması
DATA_DIR = 'data'
IMG_SIZE = 128
NUM_CLASSES = 33
BATCH_SIZE = 64
AUTOTUNE = tf.data.AUTOTUNE

@tf.function
def preprocess_image(img_path):
    """GPU-optimized görüntü önişleme"""
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.image.rgb_to_grayscale(img)
    img = tf.cast(img, tf.float32) / 255.0
    return img

def load_data():
    """Optimize edilmiş veri yükleme"""
    images = []
    labels = []
    
    print("Veri seti yükleniyor...")
    for class_folder in range(NUM_CLASSES):
        folder_path = os.path.join(DATA_DIR, str(class_folder))
        print(f"Sınıf {class_folder}/{NUM_CLASSES-1} yükleniyor...")
        
        image_paths = [os.path.join(folder_path, img_file) 
                      for img_file in os.listdir(folder_path)]
        
        # Paralel görüntü yükleme
        dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        dataset = dataset.map(preprocess_image, num_parallel_calls=AUTOTUNE)
        
        # Görüntüleri topla
        batch_images = list(dataset.as_numpy_iterator())
        images.extend(batch_images)
        labels.extend([class_folder] * len(batch_images))
    
    return np.array(images), np.array(labels)

def create_model():
    """GPU-optimized model mimarisi"""
    model = models.Sequential([
        # Giriş katmanı
        layers.Conv2D(64, (3, 3), padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # İkinci konvolüsyon bloğu
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Üçüncü konvolüsyon bloğu
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Dördüncü konvolüsyon bloğu
        layers.Conv2D(512, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fully connected katmanları
        layers.Flatten(),
        layers.Dense(512),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model

def main():
    # Veri yükleme
    print("Veri seti yükleniyor...")
    images, labels = load_data()
    
    # Veri bölme
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Model oluşturma
    print("Model oluşturuluyor...")
    model = create_model()
    
    # Optimizer ayarları
    initial_learning_rate = 0.001
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
    
    if tf.config.list_physical_devices('GPU'):
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    # Model derleme
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger('training_log.csv')
    ]
    
    # Veri pipeline optimize etme
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.cache()
    train_ds = train_ds.shuffle(1000)
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(AUTOTUNE)
    
    val_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    val_ds = val_ds.batch(BATCH_SIZE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    
    # Model eğitimi
    print("Model eğitimi başlıyor...")
    history = model.fit(
        train_ds,
        epochs=30,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )
    
    # Model kaydetme
    model.save('sign_language_model.h5')
    print("Model kaydedildi: sign_language_model.h5")
    
    # Eğitim visualizasyonu
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
    plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
    plt.title('Model Doğruluğu')
    plt.xlabel('Epoch')
    plt.ylabel('Doğruluk')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Eğitim Kaybı')
    plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
    plt.title('Model Kaybı')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    # Test değerlendirmesi
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest doğruluğu: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()
