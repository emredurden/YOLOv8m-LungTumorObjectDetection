import os
import pydicom
import numpy as np
import cv2

def is_dicom_file(filepath):
    try:
        with open(filepath, 'rb') as f:
            return f.read(132)[128:132] == b'DICM'
    except:
        return False

def ensure_dcm_extensions(folder):
    renamed_count = 0
    for root, _, files in os.walk(folder):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.splitext(file)[1].lower() == '.dcm':
                continue
            if is_dicom_file(file_path):
                new_file_path = file_path + '.dcm'
                os.rename(file_path, new_file_path)
                print(f"✅ DICOM uzantısı eklendi: {new_file_path}")
                renamed_count += 1
            else:
                print(f"⛔ DICOM değil: {file_path}")
    return renamed_count

def dicom_to_png(dicom_folder, output_folder, contrast_factor=1.5):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, _, files in os.walk(dicom_folder):
        for file in files:
            if not file.lower().endswith('.dcm'):
                continue

            dicom_path = os.path.join(root, file)
            try:
                dicom_data = pydicom.dcmread(dicom_path)
                image = dicom_data.pixel_array.astype(np.float32)

                # Normalize ve kontrast artır
                image -= np.min(image)
                image /= np.max(image)
                image *= 255.0
                image = np.clip(image * contrast_factor, 0, 255).astype(np.uint8)

                png_filename = os.path.splitext(file)[0] + ".png"
                png_path = os.path.join(output_folder, png_filename)
                cv2.imwrite(png_path, image)
                print(f"🖼️ PNG kaydedildi: {png_path}")
            except Exception as e:
                print(f"❌ Hata oluştu ({dicom_path}): {e}")

def main():
    dicom_folder = r"C:\Users\Emre Duran\Desktop\PGMV3\dicom_input"
    output_folder = r"C:\Users\Emre Duran\Desktop\PGMV3\output_dicom"

    print("🚀 Script başlatıldı")
    print(f"📂 DICOM klasörü: {dicom_folder}")
    print(f"📂 PNG çıkış klasörü: {output_folder}")

    renamed = ensure_dcm_extensions(dicom_folder)
    print(f"\n🔁 Toplam yeniden adlandırılan dosya: {renamed}")

    dicom_to_png(dicom_folder, output_folder,contrast_factor=0.9)

if __name__ == "__main__":
    main()
