import time
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from contextlib import contextmanager
import traceback

from vector_scripts.create_color_vector import ColorVectorIndexer
from vector_scripts.create_sift_vector import SIFTVLADVectorIndexer
from vector_scripts.create_dreamsim_vector import DreamSimVectorIndexer

BASE_DIR = Path("images_v3")
DB_PATH = "images.db"


class Timer:
    def __init__(self):
        self.times = {}

    @contextmanager
    def measure(self, operation_name):
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.times.setdefault(operation_name, []).append(duration)


def find_test_images(images_dir=BASE_DIR, num_images=10):
    images_dir = Path(images_dir)
    if not images_dir.exists():
        print(f"Verzeichnis nicht gefunden: {images_dir}")
        return []
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    all_images = []
    for ext in image_extensions:
        all_images.extend(images_dir.rglob(f"*{ext}"))
        all_images.extend(images_dir.rglob(f"*{ext.upper()}"))
    random.seed(42)
    selected_images = random.sample(all_images, num_images)
    print(f"\n {num_images} Testbilder ausgewählt:")
    for i, img in enumerate(selected_images, 1):
        print(f"  {i:2d}. {img.relative_to(images_dir)}")
    return selected_images


def run_feature_extraction(test_images, index_types=None):
    if index_types is None:
        index_types = ["color", "sift", "dreamsim"]

    timer = Timer()
    indexers = {
        "color": ColorVectorIndexer(db_path=DB_PATH, base_dir="."),
        "sift": SIFTVLADVectorIndexer(db_path=DB_PATH, base_dir="."),
        "dreamsim": DreamSimVectorIndexer(db_path=DB_PATH, base_dir="."),
    }

    for index_type in index_types:
        extractor = indexers[index_type]
        for img_path in test_images:
            rel_path = str(img_path.relative_to(BASE_DIR))
            op_name = f"Total ({index_type})"
            try:
                with timer.measure(op_name):
                    _ = extractor.compute_vectors([rel_path])[0]
                print(f"{img_path.name} [{index_type}]: Erfolgreich")
            except Exception as e:
                print(f"{img_path.name} [{index_type}]: Fehler: {e}")
    return timer


def create_plot(timer):
    totals = {op[7:-1]: np.mean(times) for op, times in timer.times.items()
              if op.startswith('Total (') and op.endswith(')')}
    sorted_totals = sorted(totals.items(), key=lambda x: x[1])

    names = [name.upper() for name, _ in sorted_totals]
    values = [time for _, time in sorted_totals]
    colors = ['lightblue'] * len(sorted_totals)
    edge_colors = ['navy'] * len(sorted_totals)

    plt.figure(figsize=(12, 8))
    bars = plt.barh(names, values, color=colors, edgecolor=edge_colors, alpha=0.7, linewidth=1.5)
    for bar, value in zip(bars, values):
        plt.text(bar.get_width() + value * 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{value:.4f}s', va='center', fontweight='bold', fontsize=11)
    plt.xlabel('Durchschnittliche Extraktionszeit pro Bild (Sekunden)', fontsize=12)
    plt.title('Feature-Extraktionszeiten pro Index-Typ', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.xlim(left=0)
    plt.tight_layout()
    filename = "feature_runtime_analysis.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Plot gespeichert: {filename}")


if __name__ == "__main__":
    NUM_IMAGES = 10
    INDEX_TYPES = ["color", "sift", "dreamsim"]
    print("⏱️ Starte Feature-Runtime-Analyse")

    try:
        test_images = find_test_images(BASE_DIR, NUM_IMAGES)
        timer = run_feature_extraction(test_images, index_types=INDEX_TYPES)
        if timer:
            create_plot(timer)
    except Exception as e:
        print(f"Unerwarteter Fehler während Analyse: {e}")
        traceback.print_exc()