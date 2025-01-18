from datasets import load_dataset
from config import STORAGE_DIR_DATA_FLORES_PLUS

ds = load_dataset("openlanguagedata/flores_plus", cache_dir=STORAGE_DIR_DATA_FLORES_PLUS)