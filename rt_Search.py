import os
import sys
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from contextlib import contextmanager
from pathlib import Path
import traceback

base_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(base_dir))

from search_from_image import ImageRecommender

DB_PATH = str(base_dir / "images.db")
IMAGES_ROOT = base_dir / "image_data"


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


class MultiRecommender(ImageRecommender):
    def __init__(self, timer, *args, **kwargs):
        self.timer = timer
        super().__init__(*args, **kwargs)

    def search_similar_images(self, query_path, index_type="color"):
        with self.timer.measure(f"Total ({index_type})"):
            path_rel = str(Path(query_path).resolve().relative_to(self.images_root))
            ordered = self._get_ordered_index_types(index_type)
            if not ordered:
                return
            query_vec = self._extract_query_vector(path_rel, ordered)
            if query_vec is None:
                return
            canonical = "_".join(ordered)
            index, offset_table = self._load_faiss_index(canonical)
            if index is None:
                return
            distances, indices = index.search(query_vec, self.top_k)
            return self._fetch_results(indices, distances, offset_table)


def find_test_images(images_dir=IMAGES_ROOT, num_images=10):
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


def run_analysis(num_images=10, index_types=None):
    if index_types is None:
        index_types = ["color", "sift", "dreamsim"]
    test_images = find_test_images(IMAGES_ROOT, num_images)
    timer = Timer()
    rec = MultiRecommender(timer, images_root=str(IMAGES_ROOT), db_path=DB_PATH)
    for index_type in index_types:
        for test_image in test_images:
            try:
                rec.search_similar_images(str(test_image), index_type)
            except Exception as e:
                print(f"Fehler: {e}")
    return timer


def create_plot(timer):
    totals = {op[7:-1]: np.mean(times) for op, times in timer.times.items()
              if op.startswith('Total (') and op.endswith(')')}
    cumulative_total = sum(totals.values())
    sorted_totals = sorted(totals.items(), key=lambda x: x[1])

    names = [name.upper() for name, _ in sorted_totals] + ['GESAMTZEIT (ALLE 3)']
    values = [time for _, time in sorted_totals] + [cumulative_total]
    colors = ['lightblue'] * len(sorted_totals) + ['red']
    edge_colors = ['navy'] * len(sorted_totals) + ['darkred']

    plt.figure(figsize=(12, 8))
    bars = plt.barh(names, values, color=colors, edgecolor=edge_colors, alpha=0.7, linewidth=1.5)
    for bar, value in zip(bars, values):
        plt.text(bar.get_width() + value * 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{value:.4f}s', va='center', fontweight='bold', fontsize=11)
    plt.xlabel('Zeit (Sekunden)', fontsize=12)
    plt.title('Search-Performance: Einzelzeiten vs. Gesamtzeit für ein Bild', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.xlim(left=0)
    plt.tight_layout()
    plt.savefig("runtime_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("Starte Search-Runtime-Analyse")
    NUM_IMAGES = 10
    INDEX_TYPES = ["color", "sift", "dreamsim"]
    try:
        timer = run_analysis(num_images=NUM_IMAGES, index_types=INDEX_TYPES)
        if timer:
            create_plot(timer)
    except Exception as e:
        print(f"Unerwarteter Fehler während Analyse: {e}")
        traceback.print_exc()