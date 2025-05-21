# 🖼️ Image Recommender with Color, SIFT-VLAD, DreamSim & FAISS

A high-performance image similarity search pipeline leveraging multiple embedding strategies and efficient indexing. Store image paths and embeddings in SQLite, then build a FAISS HNSW index for fast nearest-neighbor retrieval.

---

## 🚀 Key Features

* **Diverse Embedding Methods**: Color histograms, SIFT-VLAD descriptors, DreamSim embeddings (using a CLIP backbone)
* **SQLite Database**: Central storage of image filepaths and embeddings in structured tables
* **Batch Processing**: Scale effortlessly over large image collections with configurable batch sizes
* **GPU Acceleration**: Option to leverage CUDA for DreamSim feature extraction
* **FAISS HNSW Index**: Build and query an HNSW index for sub-second similarity searches
* **Flexible Search**: Combine any subset of supported embeddings (e.g., `color`, `sift`, `dreamsim`)
* **Built-in Visualization**: Quickly display top-K similar images using Matplotlib

---

## 📂 Project Structure

```bash
.
├── create_db.py               # Scan image folders and insert relative filepaths into SQLite
├── create_color_vector.py     # Compute and store color histogram embeddings
├── create_sift_vector.py      # Compute and store SIFT-VLAD embeddings
├── create_dreamsim_vector.py  # Compute and store DreamSim embeddings (with CLIP backbone)
├── ceate_main_features.py     # Orchestrate color, SIFT-VLAD & DreamSim extraction in a single pipeline
├── create_index.py            # Build a FAISS HNSW index over completed embeddings
├── search_from_image.py       # Perform similarity search and display results
└── README.md                  # Project documentation
```

---

## 🛠️ Installation & Dependencies

1. **Python Version**: 3.8 or higher
2. **Recommended**: CUDA-enabled GPU for accelerated feature extraction
3. **Core Libraries**:

```bash
pip install torch torchvision dreamsim faiss-cpu Pillow matplotlib opencv-python-headless joblib numpy scipy scikit-learn
```

> For GPU support with FAISS, install `faiss-gpu` instead of `faiss-cpu`.

---

## 📘 Usage Guide

### 1. Initialize the Database

```bash
python create_db.py --base-folder images_v3 --db-path images.db
```

Scans the specified folder recursively and populates the database with relative image paths.

### 2. Extract Feature Embeddings

Run all extractors in one go:

```bash
python ceate_main_features.py
```

Or execute each step individually:

```bash
python create_color_vector.py
python create_sift_vector.py
python create_dreamsim_vector.py
```

Each script processes images lacking its corresponding embeddings.

### 3. Build the FAISS Index

```bash
python create_index.py --vector-types color sift dreamsim --output index_hnsw.faiss
```

* **`--vector-types`**: Choose any combination of `color`, `sift`, `dreamsim`.
* **`--batch-size`**, **`--hnsw_M`**, **`--efConstruction`**, **`--efSearch`**: Tune indexing performance and accuracy.

### 4. Search for Similar Images

```bash
python search_from_image.py --query path/to/query.jpg --index combo_color_sift_dreamsim --top-k 5
```

Displays the top-K matches with their similarity metrics.

---

## ⚙️ Configuration Parameters

| Parameter          | Description                                        |
| ------------------ | -------------------------------------------------- |
| `BASE_DIR`         | Root folder containing images                      |
| `DB_PATH`          | SQLite database file path                          |
| `batch_size`       | Records processed per batch                        |
| `model_batch_size` | Number of images sent to DreamSim at once          |
| `bins`             | Number of bins per channel for color histograms    |
| `hnsw_M`           | FAISS HNSW "M" parameter                           |
| `efConstruction`   | FAISS index construction effort vs. accuracy trade |
| `efSearch`         | FAISS search speed vs. recall trade-off            |
| `TOP_K`            | Number of neighbors retrieved in search            |

---

## 🗄️ Database Schema

**Main Table**: `images`

| Column | Type    | Description                   |
| ------ | ------- | ----------------------------- |
| `id`   | INTEGER | Primary key                   |
| `path` | TEXT    | Relative path under BASE\_DIR |

**Embedding Tables**:

* `color_vectors`: Stores color histogram embeddings
* `sift_vectors`: Stores SIFT-VLAD embeddings
* `dreamsim_vectors`: Stores DreamSim embeddings

Each embedding table has a foreign key reference to the main `images` table.

---

## 📄 License

MIT License — freely use, modify, and distribute.

---

## 💡 Applications

Ideal for media archives, creative projects, and image libraries where rapid visual similarity search enhances recommendation systems and content curation workflows.
