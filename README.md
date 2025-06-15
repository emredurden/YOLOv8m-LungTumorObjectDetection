# YOLOv8m-LungTumorObjectDetection

Bu proje, akciğer tümörlerini tespit etmek için YOLOv8m tabanlı bir nesne tespit modeli geliştirmeyi amaçlar. DICOM medikal görüntüleri PNG formatına dönüştürme ve model eğitimi, çıkarımı (inference) ile hiperparametre optimizasyonu gibi temel fonksiyonları içerir.

Ana Özellikler
DICOM dosyalarını otomatik olarak algılayıp PNG formatına dönüştürme (dicomTopng.py)
YOLOv8m ile model eğitimi, toplu klasör çıkarımı ve Optuna ile hiperparametre optimizasyonu (main.py)
Kolay kullanılabilir komut satırı arayüzü

Klasör Yapısı

core/ : Eğitim, çıkarım ve optimizasyon modülleri

configs/ : Model ve veri seti yapılandırma dosyaları

data/ : Eğitim ve test veri setleri

runs/ : Model çıktı ve sonuçları

