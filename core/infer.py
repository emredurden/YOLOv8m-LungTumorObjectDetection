import os
import cv2
from ultralytics import YOLO
import numpy as np

def adjust_contrast(img, alpha):
    """
    Görüntünün kontrastını ayarlar.
    :param img: Giriş görüntüsü (NumPy dizisi).
    :param alpha: Kontrast çarpanı.
    :return: Ayarlanmış kontrastlı görüntü.
    """
    return np.clip(img * alpha, 0, 255).astype(np.uint8)


PIXEL_TO_MM_RATIO = 0.7 

def infer_folder(model_path, folder_path, confidence_threshold=0.25):
    """
    Belirtilen klasördeki görüntüler üzerinde model tahmini yapar ve sonuçları görselleştirir.
    :param model_path: Eğitilmiş YOLO modelinin yolu (örn. best.pt).
    :param folder_path: Tahmin yapılacak görüntülerin bulunduğu klasörün yolu.
    :param confidence_threshold: Güven eşiği (bu eşiğin altındaki tespitler gösterilmez).
    """
    model = YOLO(model_path)

    image_files = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png')) 
    ])

    if not image_files:
        print("❌ Klasörde görüntü bulunamadı.")
        return

    output_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(output_dir, exist_ok=True) 

    index = 0
    alpha = 1.0  

    while True:
        img_path = image_files[index]
        img = cv2.imread(img_path) # Görüntüyü yükler
        
        # Model tahmini yapılır, belirtilen güven eşiği kullanılır.
        # Burada conf parametresi, modelin dönüşünü filtreler.
        # Eğer tüm düşük güvenli tespitleri görmek istiyorsak, burada conf=0.0 yapılabilir.
        # Ancak görselleştirme ve alan hesaplama için yine de threshold kullanırız.
        raw_results = model(img, conf=0.0)[0] # Hata ayıklama için tüm tespitleri al (conf=0.0 ile)

        print(f"\n--- Görüntü: {os.path.basename(img_path)} ---") # Her yeni görüntü için başlık
        
        # --- Hata Ayıklama Çıktısı ---
        if len(raw_results.boxes) == 0:
            print("  ⚠️ Model bu görüntüde hiçbir nesne tespit etmedi (güven eşiği 0.0 olmasına rağmen).")
        else:
            print(f"  Toplam {len(raw_results.boxes)} adet ham tespit bulundu (conf=0.0):")
            for box in raw_results.boxes:
                cls_id_raw = int(box.cls[0])
                actual_class_name_raw = model.names[cls_id_raw]
                conf_raw = float(box.conf[0])
                print(f"    - {actual_class_name_raw}: Güven = {conf_raw:.4f}")
       
        # Sınırlama mantığı için set
        # 'tumor' (modelin tanıdığı singular isim) dışındaki sınıfların sadece bir kez çizildiğini takip eder
        drawn_once_non_tumor_classes = set()
        
        # Terminal çıktısı için bayrak: Tümör veya Timus tespiti yapılıp yapılmadığını kontrol eder
        tumor_or_thymus_detected_in_image = False

        # Sadece belirlenen confidence_threshold üzerindeki tespitleri işle
        filtered_results = model(img, conf=confidence_threshold)[0] # Burada gerçek eşik kullanılır

        # Her bir tespit edilen kutu için döngü (filtrelenmiş sonuçlar üzerinde)
        for box in filtered_results.boxes: # filtered_results üzerinde döngü
            cls_id = int(box.cls[0]) # Sınıf ID'si
            actual_class_name = model.names[cls_id] # Modelin tanıdığı gerçek sınıf adını alır (örn: 'tumor')
            
            # Görüntüleme için kullanılacak sınıf adı (örn: 'tumors' veya 'thymus')
            display_class_name = actual_class_name
            if actual_class_name == 'tumors':
                display_class_name = 'tumors' # Modelin 'tumor' olarak tanıdığını 'tumors' olarak göster

            # Sınırlama: 'tumor' dışındaki hiçbir sınıf 2 kere gösterilemesin (yani en fazla 1 kere)
            if actual_class_name != 'tumors' and cls_id in drawn_once_non_tumor_classes:
                continue # Bu non-tümör sınıfı zaten çizilmişse atla
            
            # Non-tümör sınıfı ilk kez çiziliyorsa, set'e ekle
            if actual_class_name != 'tumors':
                drawn_once_non_tumor_classes.add(cls_id)

            x1, y1, x2, y2 = map(int, box.xyxy[0]) # Sınırlayıcı kutu koordinatları (piksel)
            conf = float(box.conf[0]) # Güven skoru
            
            # Piksel cinsinden alan hesaplama
            pixel_width = x2 - x1
            pixel_height = y2 - y1
            pixel_area = pixel_width * pixel_height

            # mm^2 cinsinden alanı hesaplama
            # PIXEL_TO_MM_RATIO'nun doğru ayarlandığından emin olun!
            mm_area = pixel_area * (PIXEL_TO_MM_RATIO ** 2) 
            
            # Etiket metni oluşturma: Gösterim için güncellenmiş sınıf adı, güven skoru ve mm^2 cinsinden alan
            label_text = f"{display_class_name} {conf:.2f} | Alan: {mm_area:.2f} mm2"

            # 'tumor' veya 'thymus' (modelin tanıdığı isimler) ise terminale yazdır
            if actual_class_name in ['tumors', 'thymus']:
                print(f"  ✅ Tespit Edilen (Görselde Gösterilen): {label_text}")
                tumor_or_thymus_detected_in_image = True # Bayrağı ayarla

            # Tümör sınıfı için farklı renk ve kalınlık kullanma (modelin tanıdığı isme göre)
            color = (0, 255, 0) # Varsayılan yeşil renk (diğer organlar için)
            thickness = 2 # Varsayılan çizgi kalınlığı
            if actual_class_name == 'tumors':
                color = (0, 0, 255) # Tümör için kırmızı renk
                thickness = 3 # Tümör için daha kalın çizgi
            elif actual_class_name == 'thymus': # Timus için farklı bir renk tanımlayabiliriz
                color = (255, 0, 0) # Mavi renk
                thickness = 2
            
            # Sınırlayıcı kutuyu çizme
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            
            # Etiketi çizme (metin ve arka plan rengiyle)
            cv2.putText(img, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)

        # Eğer görüntüde tümör veya timus tespiti yapılmadıysa terminale bilgi ver (filtrelenmiş sonuçlara göre)
        if not tumor_or_thymus_detected_in_image:
            print(f"  ❌ Belirtilen güven eşiği ({confidence_threshold:.2f}) üzerinde bu görüntüde tümör veya timus tespiti yapılmadı.")

        # Kontrast ayarı uygulanmış görüntüyü oluşturur
        contrast_img = adjust_contrast(img.astype(np.float32), alpha)

        # Görüntü başlığını ve kontrast bilgisini gösterir
        display_text = f"[{index+1}/{len(image_files)}] {os.path.basename(img_path)} | Kontrast: {alpha:.2f}"
        cv2.imshow(display_text, contrast_img) # Görüntüyü pencerede gösterir

        # Çıktı görüntüsünü kaydeder
        output_path = os.path.join(output_dir, os.path.basename(img_path))
        cv2.imwrite(output_path, contrast_img)

        # Kullanıcıdan tuş girişi beklenir
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow(display_text) # Mevcut görüntü penceresini kapatır

        # Tuşlara göre navigasyon ve kontrast ayarı
        if key == ord('d'): # 'd' tuşu: Sonraki görüntü
            index = (index + 1) % len(image_files)
        elif key == ord('a'): # 'a' tuşu: Önceki görüntü
            index = (index - 1) % len(image_files)
        elif key == ord('w'): # 'w' tuşu: Kontrastı artır
            alpha = min(alpha + 0.1, 3.0)
        elif key == ord('s'): # 's' tuşu: Kontrastı azalt
            alpha = max(alpha - 0.1, 0.1)
        elif key == 27:  # ESC tuşu: Çıkış
            break

    cv2.destroyAllWindows() # Tüm açık pencereleri kapatır BRONŞLARI ETİKETLEYİP BRONŞLARI KANSERLE KARIŞTIRMASINI ENGELLEYECEĞİZ
