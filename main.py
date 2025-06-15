import argparse
from core.infer import infer_folder
from configs.model_config import TRAINED_MODEL_PATH, DATA_YAML_PATH
from core.train import train_model
from core.optimize import run_optuna

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'infer_folder', 'optuna'],
                        help='train: Model eğitimi | infer_folder: Klasör içi tahmin | optuna: Hiperparametre optimizasyonu')
    parser.add_argument('--folder', type=str, help='Folder path for batch inference')
    parser.add_argument('--model', type=str, default=TRAINED_MODEL_PATH, help='Path to trained model (best.pt)')
    parser.add_argument('--data', type=str, default=DATA_YAML_PATH, help='YOLO veri kümesi YAML dosyası (optuna için)')
    parser.add_argument('--trials', type=int, default=30, help='Optuna deneme sayısı')

    args = parser.parse_args()

    if args.mode == 'train':
        train_model()

    elif args.mode == 'infer_folder':
        if not args.folder:
            print("⚠️  Lütfen --folder parametresini verin.")
        else:
            infer_folder(args.model, args.folder)

    elif args.mode == 'optuna':
        run_optuna(data_yaml=args.data, n_trials=args.trials)

if __name__ == "__main__":
    main()
