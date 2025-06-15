from ultralytics import YOLO
from configs.model_config import MODEL_NAME
import yaml
import os

def train_model(hyperparams=None):
    config_path = os.path.join("configs", "train_config.yaml")

    # ÖNERİLEN DEĞİŞİKLİK: 'utf-8' kodlamasıyla dosyayı aç
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)  # Burası dict döndürmeli

    if not isinstance(config, dict):
        raise TypeError("YAML dosyası düzgün yüklenemedi. Lütfen 'configs/train_config.yaml' dosyasının biçimini kontrol edin.")

    if hyperparams:
        config.update(hyperparams)

    model = YOLO(MODEL_NAME)
    model.train(**config)

