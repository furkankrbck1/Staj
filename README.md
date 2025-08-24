# İşaret Dili Tanıma ve Dönüştürme Sistemi

Bu proje, bilgisayar kamerası aracılığıyla işaret dili hareketlerini tanıyarak metne dönüştüren ve metinden işaret diline çeviri yapan çift yönlü bir sistem geliştirmeyi amaçlamaktadır. Görüntü işleme, derin öğrenme ve kullanıcı arayüzü bileşenlerini birleştiren kapsamlı bir uygulamadır.

---

## Proje Amacı ve Kapsamı

- Görsel veriler üzerinden işaret dili tanıma (kamera ile canlı tanıma)
- Metinden işaret diline dönüşüm (GIF veya animasyonlar)
- Kullanıcı dostu bir grafik arayüz (GUI)
- Gerçek zamanlı performans ve doğruluk takibi
- Erişilebilirliği artırmak için işitme engelli bireylere yönelik destekleyici sistem oluşturmak

---

## Kullanılacak Model ve Yaklaşım

- **Model Türü:** Convolutional Neural Network (CNN)
- **Veri Formatı:** Görüntü (kare/kamera görüntüsü) + Etiket (harf/kelime)
- **Tanıma Yöntemi:** Görüntü sınıflandırma (image classification) ve canlı kamera akış analizi

---

## Hedef Kullanıcı Profili

- İşitme engelli bireyler
- İşaret dili öğrenen kullanıcılar
- Engellilerle iletişim kurmak isteyen bireyler
- Geliştiriciler ve akademik araştırmacılar

---

## Kullanılan Kütüphane ve Araçlar

| Araç / Kütüphane | Açıklama |
|------------------|----------|
| `Python`         | Ana programlama dili |
| `OpenCV`         | Görüntü işleme |
| `TensorFlow / Keras` | Derin öğrenme modeli oluşturma ve eğitimi |
| `NumPy, Pandas`  | Veri işleme |
| `Matplotlib`     | Grafik çizimi ve analiz |
| `Streamlit / Tkinter` | GUI geliştirme |
| `scikit-learn`   | Veri bölme ve performans ölçme |

---

## Veri Kümesi

- **Kullanılan Veri Kümesi:** (https://www.kaggle.com/datasets/feronial/turkish-sign-languagefinger-spelling/data)
- **Alternatif:** El ile oluşturulan özel veri seti
- **Etiketler:** Harfler ve temel kelimeler
