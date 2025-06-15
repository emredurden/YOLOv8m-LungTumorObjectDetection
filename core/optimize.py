import optuna
from ultralytics import YOLO
from functools import partial
import tempfile

def objective(trial, data_yaml):
    # Hiperparametreleri öner
    lr0 = trial.suggest_float("lr0", 1e-5, 1e-1, log=True)
    momentum = trial.suggest_float("momentum", 0.6, 0.98)
    weight_decay = trial.suggest_float("weight_decay", 0.0001, 0.01)
    batch = trial.suggest_categorical("batch", [4, 8, 16])
    imgsz = trial.suggest_categorical("imgsz", [416, 512, 640])

    # Modeli başlat
    # ÖNERİLEN DEĞİŞİKLİK: yolov8m.yaml yerine yolov8l.yaml kullanıldı
    model = YOLO("yolov8m.yaml")  # Config dosyası, .pt değil!

    # Geçici klasöre eğitim sonucu yazılır
    with tempfile.TemporaryDirectory() as tmpdir:
        result = model.train(
            data=data_yaml,
            imgsz=imgsz,
            epochs=20,  # Hızlı test için düşük epoch (sonra arttır)
            batch=batch,
            lr0=lr0,
            momentum=momentum,
            weight_decay=weight_decay,
            project=tmpdir,
            name="optuna_trial",
            device="cuda",
            verbose=False,
        )
        # Geriye validation mAP verelim (Optuna maksimize eder)
        metrics = model.metrics
        return metrics.box.map  # mAP@0.5:0.95 kullanılıyor

def run_optuna(data_yaml, n_trials=30):
    study = optuna.create_study(direction="maximize")
    study.optimize(partial(objective, data_yaml=data_yaml), n_trials=n_trials)

    print("✅ En iyi hiperparametreler:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"🎯 En iyi mAP: {study.best_value:.4f}")

