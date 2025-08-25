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
        
        # GPU konfigürasyonunu ayarla - bellek limitini GPU'nuza göre ayarlayın
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
def preprocess_image(img):
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.image.rgb_to_grayscale(img)
    img = tf.cast(img, tf.float32) / 255.0
    return img
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # Normalize et
    img = img.astype('float32') / 255.0
    
    return img

def load_data():
    images = []
    labels = []
    
    # tf.data.Dataset kullanarak veri yüklemeyi optimize et
    @tf.function
    def load_and_preprocess_image(img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.cast(img, tf.float32) / 255.0
        return img
    
    print("Veri seti yükleniyor...")
    total_classes = NUM_CLASSES
    
    # Tüm sınıfları dolaş
    for class_folder in range(total_classes):
        folder_path = os.path.join(DATA_DIR, str(class_folder))
        print(f"Sınıf {class_folder}/{total_classes-1} yükleniyor...")
        
        # Her sınıftaki görüntüleri dolaş
        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            
            # Görüntüyü oku ve işle
            img = cv2.imread(img_path)
            if img is not None:
                img = preprocess_image(img)
                
                # Veri artırma - ayna görüntüsü
                img_flipped = cv2.flip(img, 1)
                
                # Orijinal ve ayna görüntüsünü ekle
                images.append(img)
                images.append(img_flipped)
                labels.extend([class_folder, class_folder])
    
    return np.array(images), np.array(labels)

def create_model():
    model = models.Sequential()
    
    # Giriş ve ilk konvolüsyon bloğu
    model.add(layers.Conv2D(64, (3, 3), padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    # İkinci konvolüsyon bloğu
    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    # Üçüncü konvolüsyon bloğu
    model.add(layers.Conv2D(256, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(256, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    # Yoğun katmanlar
    model.add(layers.Flatten())
    model.add(layers.Dense(512))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(NUM_CLASSES, activation='softmax'))
        layers.Conv2D(64, (3, 3), padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # İkinci konvolüsyon bloğu
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # Üçüncü konvolüsyon bloğu
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # Yoğun katmanlar
        layers.Flatten(),
        layers.Dense(512),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        layers.Dense(256),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # İlk konvolüsyon bloğu
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # İkinci konvolüsyon bloğu
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Üçüncü konvolüsyon bloğu
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Yoğun katmanlar
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    # Model derlemesi
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    print("Veri seti yükleniyor...")
    images, labels = load_data()
    
    # Veriyi eğitim ve test setlerine ayır
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Benzersiz sınıf sayısı: {len(np.unique(labels))}")
    print(f"Her sınıftaki örnek sayısı:")
    for i in range(NUM_CLASSES):
        print(f"Sınıf {i}: {np.sum(labels == i)} örnek")
    
    print("Model oluşturuluyor...")
    model = create_model()
    
    # Optimize edilmiş model derlemesi
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # Mixed precision için optimizer'ı sar
    if tf.config.list_physical_devices('GPU'):
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model eğitimi başlıyor...")
    
    # Optimize edilmiş veri artırma
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        fill_mode='nearest',
        validation_split=0.2,
        preprocessing_function=lambda x: tf.cast(x, tf.float16) if tf.config.list_physical_devices('GPU') else x
    )

    # Gelişmiş eğitim kontrolleri için callbacks
    callbacks = [
        # En iyi modeli kaydet
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        # Erken durdurma
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,  # Daha fazla sabır
            restore_best_weights=True,
            verbose=1
        ),
        # Öğrenme oranını dinamik olarak ayarla
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        # Eğitim durumunu kaydet
        tf.keras.callbacks.CSVLogger('training_log.csv')
    ]
    
    # Veri setini normalize et
    print("Veri normalize ediliyor...")
    mean = X_train.mean()
    std = X_train.std()
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    
    print(f"Eğitim veri seti boyutu: {X_train.shape}")
    print(f"Test veri seti boyutu: {X_test.shape}")
    
    # GPU için optimize edilmiş eğitim konfigürasyonu
    BATCH_SIZE = 64  # GPU için daha büyük batch size
    AUTOTUNE = tf.data.AUTOTUNE
    
    # Veri pipeline'ını optimize et
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.cache()
    train_ds = train_ds.shuffle(1000)
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(AUTOTUNE)
    
    val_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    val_ds = val_ds.batch(BATCH_SIZE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    
    # Optimize edilmiş eğitim
    history = model.fit(
        train_ds,
        epochs=30,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )
    
    # Modeli kaydet
    model.save('sign_language_model.h5')
    print("Model kaydedildi: sign_language_model.h5")
    
    # Eğitim grafiklerini çiz
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
    
    # Test seti üzerinde değerlendirme
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest doğruluğu: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()
