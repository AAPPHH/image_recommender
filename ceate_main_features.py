import os
from pathlib import Path
import logging

from create_color_vector import ColorVectorCreatorDB
import create_lpips_vector
from create_lpips_vector import LPIPSVektorCreatorDB
from create_dreamsim_vector import DreamSimImageIndexer

logging.basicConfig(
    level=logging.INFO,
    filename="pipeline.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)


def console_and_log(message, level="info"):
    print(message)
    level = level.lower()
    if level == "info":
        logging.info(message)
    elif level == "warning":
        logging.warning(message)
    elif level == "error":
        logging.error(message)
    else:
        logging.debug(message)


def main():
    DB_PATH = "images.db"
    BASE_FOLDER = Path(
        r"C:/Users/jfham/OneDrive/Dokumente/Workstation_Clones/"
        r"image_recomender/image_recommender/images_v3"
    )
    WORKERS = os.cpu_count()

    # 1) Color-Vektoren
    bins = 16
    color_batch = 8192
    color_indexer = ColorVectorCreatorDB(
        db_path=str(DB_PATH),
        base_folder=BASE_FOLDER,
        bins=bins,
        batch_size=color_batch,
        workers=WORKERS,
    )
    color_indexer.update_db()
    console_and_log("✅ Farb-Vektoren erstellt und gespeichert.", level="info")

    # 2) LPIPS-Vektoren (ersetzt den Hash-Indexer)

    create_lpips_vector.DB_BATCH = 8192
    create_lpips_vector.MODEL_BATCH = 128
    create_lpips_vector.DB_PATH = "images.db"
    create_lpips_vector.BASE_DIR = os.getcwd()

    lpips_indexer = LPIPSVektorCreatorDB() 
    lpips_indexer.index_images()
    console_and_log("✅ LPIPS-Vektoren erstellt und gespeichert.", level="info")

    ## 3) DreamSim-Features
    
    DB_BATCH_SIZE = 8192
    MODEL_BATCH_SIZE = 128
    BASE_DIR = r"C:\Users\jfham\OneDrive\Dokumente\Workstation_Clones\image_recomender\image_recommender\images_v3"
    indexer = DreamSimImageIndexer(
        db_path=DB_PATH,
        base_dir=BASE_DIR,
        db_batch_size=DB_BATCH_SIZE,
        model_batch_size=MODEL_BATCH_SIZE,
    )
    indexer.index_images()
    console_and_log("✅ DreamSim-Features indexiert und gespeichert.", level="info")

if __name__ == "__main__":
    main()
