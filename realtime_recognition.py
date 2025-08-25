import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

def preprocess_frame(frame, target_size=(128, 128)):  # Boyutu 128x128'e çıkardık
    try:
        # Görüntü işleme
        # BGR'den RGB'ye dönüştür
        roi = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Görüntüyü yeniden boyutlandır
        roi = cv2.resize(roi, target_size)
        
        # Gürültü azaltma
        roi = cv2.GaussianBlur(roi, (3, 3), 0)
        
        # Kontrast artırma
        lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl,a,b))
        roi = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        # Keskinleştirme
        kernel = np.array([[-1,-1,-1],
                         [-1, 9,-1],
                         [-1,-1,-1]])
        roi = cv2.filter2D(roi, -1, kernel)
        
        # Normalize et
        roi = roi.astype('float32') / 255.0
        
        return roi
        
    except Exception as e:
        print(f"Görüntü işleme hatası: {e}")
        return np.zeros((target_size[0], target_size[1], 3))

def main():
    # Tahmin geçmişi için liste
    predictions = []
    current_text = ""
    current_confidence = 0
    
    # Modeli yükle
    try:
        model = load_model('sign_language_model.h5')
        print("Model başarıyla yüklendi!")
        print("\nKullanım Kılavuzu:")
        print("1. Elinizi yeşil kare içinde tutun")
        print("2. Elinizi sabit tutun ve iyi aydınlatın")
        print("3. İşaretleri veri setindeki gibi gösterin")
        print("4. İşaret algılanana kadar pozisyonu ayarlayın")
        print("5. Çıkmak için 'q' tuşuna basın\n")
    except:
        print("Model dosyası bulunamadı! Önce train_model.py'yi çalıştırın.")
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

    # Kamerayı başlat
    cap = cv2.VideoCapture(0)
    
    # Kamera çözünürlüğünü ayarla
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Pencere oluştur ve boyutunu ayarla
    cv2.namedWindow('Sign Language Recognition', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Sign Language Recognition', 800, 600)
    
    while True:
        # Kameradan görüntü al
        ret, frame = cap.read()
        if not ret:
            print("Kamera görüntüsü alınamıyor!")
            break
            
        # Görüntüyü aynala
        frame = cv2.flip(frame, 1)
        
        # Görüntü boyutlarını al
        height, width = frame.shape[:2]
        
        # ROI koordinatlarını hesapla (merkezi bir kare)
        box_size = 200
        x = (width - box_size) // 2
        y = (height - box_size) // 2
        
        # ROI'yi işle ve tahmin yap
        roi = frame[y:y+box_size, x:x+box_size].copy()
        processed_roi = preprocess_frame(roi)
        
        # Modele gönder
        try:
            prediction = model.predict(np.expand_dims(processed_roi, axis=0), verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]
            
            # Sadece yeşil kareyi çiz
            cv2.rectangle(frame, (x, y), (x + box_size, y + box_size), (0, 255, 0), 2)
            
            # Sonuçları ekrana yaz
            if confidence > 0.7:  # Güven eşiğini artırdık
                # Son 5 tahmini sakla
                predictions.append(predicted_class)
                if len(predictions) > 5:
                    predictions.pop(0)
                
                # En çok tekrar eden sınıfı bul
                if len(predictions) == 5:
                    most_common = max(set(predictions), key=predictions.count)
                    if predictions.count(most_common) >= 3:  # En az 3 kez tekrar etmeli
                        current_text = labels[most_common]
                        current_confidence = confidence
                
                # Harfi büyük ve ortada göster
                if current_text:
                    font_scale = 6.0
                    thickness = 5
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    
                    # Yazının boyutunu hesapla
                    (text_width, text_height), _ = cv2.getTextSize(current_text, font, font_scale, thickness)
                    text_x = (width - text_width) // 2
                    text_y = height - 50  # Alt kısımda göster
                    
                    # Harfi yaz
                    cv2.putText(frame, current_text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)
                    
                    # Doğruluk oranını daha küçük yaz
                    accuracy_text = f"%{current_confidence*100:.1f}"
                    cv2.putText(frame, accuracy_text, (text_x, text_y + 40), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
        except Exception as e:
            print(f"Tahmin hatası: {e}")
        
        # ROI'yi işle
        roi = preprocess_frame(frame)
        
        # Tahmin yap
        prediction = model.predict(np.expand_dims(roi, axis=0), verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        # Güven eşiği kontrolü (0.6 = %60 güven)
        if confidence > 0.6:
            text = f"İşaret: {labels[predicted_class]} (Güven: %{confidence*100:.1f})"
            color = (0, 255, 0)  # Yeşil
        else:
            text = "İşaret bekleniyor... El pozisyonunu ayarlayın"
            color = (0, 165, 255)  # Turuncu
            
        # Sonucu ekrana yaz
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Algılanan işaretin referans görüntüsünü göster
        if confidence > 0.6:
            text2 = "Algılanan İşaret Örneği"
            cv2.putText(frame, text2, (frame.shape[1] - 300, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Yardım metni
        help_text = "Çıkış için 'q' tuşuna basın"
        cv2.putText(frame, help_text, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Görüntüyü göster
        cv2.imshow('Sign Language Recognition', frame)
        
        # 'q' tuşuna basılırsa çık
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
    
    # Temizle
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # Ek pencere temizleme
    cv2.waitKey(1)  # Bazı sistemlerde birden fazla çağrı gerekebilir
    exit()  # Programı tamamen sonlandır

if __name__ == "__main__":
    main()
