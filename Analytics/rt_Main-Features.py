import time
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from contextlib import contextmanager

from vector_scripts.create_color_vector import ColorVectorIndexer
from vector_scripts.create_sift_vector import SIFTVLADVectorIndexer
from vector_scripts.create_dreamsim_vector import DreamSimVectorIndexer


BASE_DIR = Path(__file__).resolve().parents[1]
IMAGE_DIR = BASE_DIR / "image_data"
DB_PATH = BASE_DIR / "images.db"


class Timer:
    def __init__(self):
        self.times = {}

    @contextmanager
    def measure(self, operation_name):
        """
        Context manager to measure the execution time of a named operation.

        Args:
            operation_name (str): The name of the operation being timed.

        Yields:
            None
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.times.setdefault(operation_name, []).append(duration)


def find_test_images(images_dir: Path, num_images: int = 10):
    """
    Randomly selects a number of test images from a given directory.

    Args:
        images_dir (Path): Path to the root directory containing image files.
        num_images (int, optional): Number of images to select. Defaults to 10.

    Returns:
        List[Path]: A list of image file paths.

    Raises:
        ValueError: If fewer than `num_images` are found in the directory.
    """
    if not images_dir.exists():
        print(f"Verzeichnis nicht gefunden: {images_dir}")
        return []

    exts = {'.jpg', '.jpeg', '.png', '.webp'}
    all_images = [p for ext in exts for p in images_dir.rglob(f"*{ext}") + images_dir.rglob(f"*{ext.upper()}")]

    if len(all_images) < num_images:
        raise ValueError(f"Nur {len(all_images)} Bilder gefunden, aber {num_images} angefordert.")

    random.seed(42)
    selected = random.sample(all_images, num_images)

    print(f"{num_images} Testbilder ausgewählt:")
    for i, img in enumerate(selected, 1):
        print(f"{i:2d}. {img.relative_to(images_dir)}")
    return selected


def run_feature_extraction(test_images, index_types=None):
    """
    Runs feature extraction on a list of images for the specified index types.

    Args:
        test_images (List[Path]): List of image paths to process.
        index_types (List[str], optional): Index types to extract (e.g., "color", "sift", "dreamsim").
                                           Defaults to all three.

    Returns:
        Timer: A Timer instance containing the recorded durations.
    """
    if index_types is None:
        index_types = ["color", "sift", "dreamsim"]

    timer = Timer()
    indexers = {
        "color": ColorVectorIndexer(db_path=str(DB_PATH), base_dir=str(IMAGE_DIR)),
        "sift": SIFTVLADVectorIndexer(db_path=str(DB_PATH), base_dir=str(IMAGE_DIR)),
        "dreamsim": DreamSimVectorIndexer(db_path=str(DB_PATH), base_dir=str(IMAGE_DIR)),
    }

    for index_type in index_types:
        extractor = indexers[index_type]
        for img_path in test_images:
            rel_path = str(img_path.relative_to(IMAGE_DIR))
            op_name = f"Total ({index_type})"
            try:
                with timer.measure(op_name):
                    _ = extractor.compute_vectors([rel_path])[0]
                print(f"{img_path.name} [{index_type}]: Erfolgreich")
            except Exception as e:
                print(f"{img_path.name} [{index_type}]: Fehler: {e}")
    return timer


def create_plot(timer: Timer):
    """
    Creates a horizontal bar chart showing the average feature extraction time per index type.

    Args:
        timer (Timer): A Timer object with recorded execution times.

    Side Effects:
        - Saves the plot as a PNG file.
        - Displays the plot using matplotlib.
        - Prints the output file name.
    """
    totals = {
        op[7:-1]: np.mean(times)
        for op, times in timer.times.items()
        if op.startswith("Total (") and op.endswith(")")
    }
    sorted_totals = sorted(totals.items(), key=lambda x: x[1])

    names = [name.upper() for name, _ in sorted_totals]
    values = [val for _, val in sorted_totals]

    plt.figure(figsize=(12, 8))
    bars = plt.barh(names, values, color='lightblue', edgecolor='navy', alpha=0.7, linewidth=1.5)

    for bar, value in zip(bars, values):
        plt.text(bar.get_width() + value * 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{value:.4f}s", va="center", fontweight="bold", fontsize=11)

    plt.xlabel("Durchschnittliche Extraktionszeit pro Bild (Sekunden)", fontsize=12)
    plt.title("Feature-Extraktionszeiten pro Index-Typ", fontsize=14, fontweight="bold")
    plt.grid(axis="x", alpha=0.3)
    plt.xlim(left=0)
    plt.tight_layout()
    filename = "feature_runtime_analysis.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Plot gespeichert: {filename}")


if __name__ == "__main__":
    """
    Script entry point. Performs feature extraction runtime analysis.

    Steps:
        - Selects a random sample of images from the dataset.
        - Performs feature extraction for each specified index type.
        - Plots the average extraction time per index type.
    """
    print("Starte Feature-Runtime-Analyse")
    NUM_IMAGES = 10
    INDEX_TYPES = ["color", "sift", "dreamsim"]

    try:
        test_images = find_test_images(IMAGE_DIR, NUM_IMAGES)
        timer = run_feature_extraction(test_images, INDEX_TYPES)
        create_plot(timer)
    except Exception as e:
        print(f"Unerwarteter Fehler: {e}")