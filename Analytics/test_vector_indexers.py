import unittest
import sqlite3
import random
from pathlib import Path
import sys
import traceback

base_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(base_dir / "vector_scripts"))

from create_color_vector import ColorVectorIndexer
from create_sift_vector import SIFTVLADVectorIndexer
from create_dreamsim_vector import DreamSimVectorIndexer
from search_from_image import ImageRecommender

# Konstanten
DB_PATH = str(base_dir / "images.db")
IMAGES_DIR = base_dir / "image_data"


class TestVectorIndexers(unittest.TestCase):
    passed_tests = []

    @classmethod
    def setUpClass(cls):
        cls.conn = sqlite3.connect(DB_PATH)
        cls.cursor = cls.conn.cursor()

        cls.recommender = ImageRecommender(db_path=DB_PATH, images_root=str(IMAGES_DIR))
        cls.recommender._plot_results = lambda *args, **kwargs: None

        cls.cursor.execute("SELECT path FROM images")
        rows = cls.cursor.fetchall()
        assert rows, "Keine Pfade in der Datenbank gefunden"
        cls.rel_path = random.choice(rows)[0].replace("\\", "/")
        cls.query_path = str(IMAGES_DIR / cls.rel_path)
        print(f"\n🎯 Query-Bild: {cls.query_path}")

    def test_color_indexing(self):
        """
        Test für die Erstellung von Farbvektoren
        """
        try:
            indexer = ColorVectorIndexer(db_path=DB_PATH, base_dir=str(IMAGES_DIR))
            indexer.run()
            self.cursor.execute("SELECT COUNT(*) FROM color_vectors")
            count = self.cursor.fetchone()[0]
            self.assertGreater(count, 0)
            self.__class__.passed_tests.append("color")
        except Exception as e:
            print("color indexing failed:", e)
            traceback.print_exc()
            raise

    def test_sift_indexing(self):
        """
        Test für die Erstellung von SIFT-Vektoren
        """
        try:
            indexer = SIFTVLADVectorIndexer(db_path=DB_PATH, base_dir=str(IMAGES_DIR))
            indexer.run()
            self.cursor.execute("SELECT COUNT(*) FROM sift_vectors")
            count = self.cursor.fetchone()[0]
            self.assertGreater(count, 0)
            self.__class__.passed_tests.append("sift")
        except Exception as e:
            print("sift indexing failed:", e)
            traceback.print_exc()
            raise

    def test_dreamsim_indexing(self):
        """
        Test für die Erstellung von DreamSim-Vektoren
        """
        try:
            indexer = DreamSimVectorIndexer(db_path=DB_PATH, base_dir=str(IMAGES_DIR))
            indexer.run()
            self.cursor.execute("SELECT COUNT(*) FROM dreamsim_vectors")
            count = self.cursor.fetchone()[0]
            self.assertGreater(count, 0)
            self.__class__.passed_tests.append("dreamsim")
        except Exception as e:
            print("dreamsim indexing failed:", e)
            traceback.print_exc()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()
        print("\nTest-Ergebnisse:")
        for name in ["color", "sift", "dreamsim"]:
            if name in cls.passed_tests:
                print(f"{name} indexing erfolgreich")
            else:
                print(f"{name} indexing fehlgeschlagen")


if __name__ == "__main__":
    print("Starte robuste VectorIndexer-Tests …\n")
    unittest.main(verbosity=2)