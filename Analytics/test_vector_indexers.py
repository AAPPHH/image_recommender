import unittest
import sqlite3
import random
import traceback
from pathlib import Path

from vector_scripts.create_color_vector import ColorVectorIndexer
from vector_scripts.create_sift_vector import SIFTVLADVectorIndexer
from vector_scripts.create_dreamsim_vector import DreamSimVectorIndexer
from main.search_from_image import ImageRecommender

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "images.db"
IMAGES_DIR = BASE_DIR / "image_data"


class TestVectorIndexers(unittest.TestCase):
    passed_tests = []

    @classmethod
    def setUpClass(cls):
        """
        Sets up the database connection and selects a random test image for use in tests.

        Raises:
            FileNotFoundError: If the SQLite database is not found.
            ValueError: If no image paths are found in the database.
        """
        if not DB_PATH.exists():
            raise FileNotFoundError(f"Datenbank nicht gefunden: {DB_PATH}")

        cls.conn = sqlite3.connect(DB_PATH)
        cls.cursor = cls.conn.cursor()

        cls.recommender = ImageRecommender(db_path=str(DB_PATH), images_root=str(IMAGES_DIR))
        cls.recommender._plot_results = lambda *args, **kwargs: None

        cls.cursor.execute("SELECT path FROM images")
        rows = cls.cursor.fetchall()
        if not rows:
            raise ValueError("Keine Bildpfade in der Datenbank gefunden")

        cls.rel_path = rows[random.randint(0, len(rows) - 1)][0].replace("\\", "/")
        cls.query_path = IMAGES_DIR / cls.rel_path
        print(f"\nQuery-Bild: {cls.query_path}")

    def test_color_indexing(self):
        """
        Tests the color feature extraction using ColorVectorIndexer.

        Asserts:
            That at least one entry exists in the `color_vectors` table after indexing.
        """
        try:
            indexer = ColorVectorIndexer(db_path=str(DB_PATH), base_dir=str(IMAGES_DIR))
            indexer.run()
            self.cursor.execute("SELECT COUNT(*) FROM color_vectors")
            count = self.cursor.fetchone()[0]
            self.assertGreater(count, 0)
            self.__class__.passed_tests.append("color")
        except Exception as e:
            print("Color indexing failed:", e)
            traceback.print_exc()
            raise

    def test_sift_indexing(self):
        """
        Tests the SIFT-VLAD feature extraction using SIFTVLADVectorIndexer.

        Asserts:
            That at least one entry exists in the `sift_vectors` table after indexing.
        """
        try:
            indexer = SIFTVLADVectorIndexer(db_path=str(DB_PATH), base_dir=str(IMAGES_DIR))
            indexer.run()
            self.cursor.execute("SELECT COUNT(*) FROM sift_vectors")
            count = self.cursor.fetchone()[0]
            self.assertGreater(count, 0)
            self.__class__.passed_tests.append("sift")
        except Exception as e:
            print("SIFT indexing failed:", e)
            traceback.print_exc()
            raise

    def test_dreamsim_indexing(self):
        """
        Tests the DreamSim feature extraction using DreamSimVectorIndexer.

        Asserts:
            That at least one entry exists in the `dreamsim_vectors` table after indexing.
        """
        try:
            indexer = DreamSimVectorIndexer(db_path=str(DB_PATH), base_dir=str(IMAGES_DIR))
            indexer.run()
            self.cursor.execute("SELECT COUNT(*) FROM dreamsim_vectors")
            count = self.cursor.fetchone()[0]
            self.assertGreater(count, 0)
            self.__class__.passed_tests.append("dreamsim")
        except Exception as e:
            print("DreamSim indexing failed:", e)
            traceback.print_exc()
            raise

    @classmethod
    def tearDownClass(cls):
        """
        Closes the database connection and prints a summary of passed indexer tests.
        """
        cls.conn.close()
        print("\nTest-Ergebnisse:")
        for name in ["color", "sift", "dreamsim"]:
            if name in cls.passed_tests:
                print(f"{name} indexing erfolgreich")
            else:
                print(f"{name} indexing fehlgeschlagen")


if __name__ == "__main__":
    """
    Entry point to execute the vector indexer unit tests.
    """
    print("Starte VectorIndexer-Tests …\n")
    unittest.main(verbosity=2)