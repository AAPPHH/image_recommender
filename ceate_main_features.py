import cv2
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Callable

# ==== Feature Extractors ==== #
def extract_sift_features(image: np.ndarray):
    sift = cv2.SIFT_create()
    return sift.detectAndCompute(image, None)

def extract_color_histogram(image: np.ndarray):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()

def extract_dreamsim_features(image: np.ndarray):
    # Dummy-Simulation für Modelllaufzeit
    time.sleep(0.2)
    return np.random.rand(512)

# ==== Feature Type Mapping ==== #
FEATURE_EXTRACTORS = {
    "sift": extract_sift_features,
    "color": extract_color_histogram,
    "dreamsim": extract_dreamsim_features,
}

# ==== Analyzer ==== #
class FeatureExtractionAnalyzer:
    def __init__(self, image_dir: str = "images_v3", num_images: int = 10):
        self.image_dir = Path(image_dir)
        self.num_images = num_images
        self.feature_types = list(FEATURE_EXTRACTORS.keys())
        self.results = {ftype: [] for ftype in self.feature_types}

        self.test_images = self._select_images()

    def _select_images(self):
        all_images = list(self.image_dir.rglob("*.[jp][pn]g")) + list(self.image_dir.rglob("*.webp"))
        if not all_images:
            raise RuntimeError(f"❌ Keine Bilder gefunden in: {self.image_dir}")
        random.seed(42)
        return random.sample(all_images, min(self.num_images, len(all_images)))

    def run(self):
        for ftype in self.feature_types:
            extractor: Callable[[np.ndarray], any] = FEATURE_EXTRACTORS[ftype]
            print(f"\n--- Feature-Typ: {ftype.upper()} ---")
            for img_path in self.test_images:
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"❌ Fehler beim Laden: {img_path.name}")
                    continue

                start = time.perf_counter()
                try:
                    _ = extractor(img)
                    duration = time.perf_counter() - start
                    self.results[ftype].append(duration)
                    print(f"{img_path.name}: {duration:.3f}s ✅")
                except Exception as e:
                    print(f"{img_path.name}: Fehler bei {ftype}: {e}")

        self.plot_results()

    def plot_results(self):
        avg_times = {ftype: np.mean(times) for ftype, times in self.results.items() if times}
        if not avg_times:
            print("Keine gültigen Messdaten zum Plotten.")
            return

        plt.figure(figsize=(10, 6))
        bars = plt.barh(list(avg_times.keys()), list(avg_times.values()), color="orange")
        for bar, val in zip(bars, avg_times.values()):
            plt.text(val + 0.005, bar.get_y() + bar.get_height() / 2, f"{val:.3f}s", va="center")
        plt.xlabel("Durchschnittliche Extraktionsdauer (s)")
        plt.title("⏱️ Feature-Erstellzeit pro Index-Typ")
        plt.grid(axis="x", linestyle="--", alpha=0.5)
        plt.tight_layout()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"feature_runtime_plot_{ts}.png"
        plt.savefig(filename)
        plt.show()
        print(f"📊 Plot gespeichert: {filename}")

# ==== Run Script ==== #
if __name__ == "__main__":
    analyzer = FeatureExtractionAnalyzer(image_dir="images_v3", num_images=10)
    analyzer.run()
