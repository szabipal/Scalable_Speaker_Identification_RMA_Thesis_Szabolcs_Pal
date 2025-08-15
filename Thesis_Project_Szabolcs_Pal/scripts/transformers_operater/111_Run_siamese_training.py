import os
import sys


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT_DIR)

try:

    from ..training.train_siamese_nn import main as train_siamese_model
except ImportError:

    from scripts.training.train_siamese_nn import main as train_siamese_model

if __name__ == "__main__":
    train_siamese_model()