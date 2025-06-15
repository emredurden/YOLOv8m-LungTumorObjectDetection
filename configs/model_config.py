import os

CLASSES = ['tumors', 'thymus', 'left_lung', 'right_lung', 'esophagus', 'heart', 'spinal_kord']

MODEL_NAME = 'yolov8m.pt'

BASE_DIR = os.path.abspath(os.path.dirname(__file__) + "/..")
DATA_YAML_PATH = os.path.join(BASE_DIR, 'data', 'data.yaml')
MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, 'runs', 'lung_model')
TRAINED_MODEL_PATH = os.path.join(MODEL_OUTPUT_DIR, 'weights', 'best.pt')
