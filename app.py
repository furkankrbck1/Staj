import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tempfile
import time

# Sayfanın temel ayarları
st.set_page_config(
    page_title="İşaret Dili Tanıma",
    page_icon="👋",
    layout="wide"
)

def load_sign_language_model():
    try:
        return load_model('sign_language_model.h5')
    except:
        st.error("Model dosyası bulunamadı! Önce train_model.py'yi çalıştırın.")
        return None

def preprocess_frame(frame, target_size=(64, 64)):
    # ROI (Region of Interest) belirleme
    roi = cv2.resize(frame, target_size)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    return roi / 255.0

def main():
    st.title("İşaret Dili Tanıma Sistemi")
    
    # Sidebar
    st.sidebar.title("Ayarlar")
    detection_confidence = st.sidebar.slider("Tespit Güven Eşiği", 0.0, 1.0, 0.5)
    
    # Ana sayfa düzeni
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Kamera Görüntüsü")
        run = st.checkbox('Kamerayı Başlat')
        
        # Kamera görüntüsü için yer tutucu
        FRAME_WINDOW = st.image([])
        
    with col2:
        st.header("Sonuçlar")
        result_placeholder = st.empty()
        confidence_placeholder = st.empty()
        
    # Model yükleme
    model = load_sign_language_model()
    if model is None:
        return
        
    # İşaret dili harfleri ve sembolleri
    labels = {
        0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G',
        7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N',
        14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
        20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z',
        26: 'Boşluk', 27: 'Sil', 28: 'Hiçbiri', 29: '0', 30: '1',
        31: '2', 32: '3'
    }
    
    # Kamera
    cap = cv2.VideoCapture(0)
    
    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Kamera görüntüsü alınamıyor!")
            break
            
        # Görüntüyü aynala
        frame = cv2.flip(frame, 1)
        
        # ROI'yi çiz
        cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)
        roi = frame[100:400, 100:400]
        
        # Tahmin
        processed_roi = preprocess_frame(roi)
        prediction = model.predict(np.expand_dims(processed_roi, axis=0), verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        # Sonuçları göster
        if confidence > detection_confidence:
            result_placeholder.subheader(f"Tespit Edilen İşaret: {labels[predicted_class]}")
            confidence_placeholder.progress(confidence)
        
        # Kamera görüntüsünü göster
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        time.sleep(0.1)  # CPU kullanımını azaltmak için
        
    cap.release()

if __name__ == "__main__":
    main()
