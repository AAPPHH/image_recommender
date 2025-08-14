# 🖼️ Image Recommender with Color, SIFT-VLAD, DreamSim, FAISS & Interactive Visualization

A high-performance image similarity search pipeline leveraging multiple embedding strategies, efficient indexing, and modern interactive visualization.  
Store image paths and embeddings in SQLite, build a FAISS HNSW index for fast nearest-neighbor retrieval, and explore results visually with UMAP, HDBSCAN, Plotly, and Dash.

---

## 🚀 Key Features

* **Diverse Embedding Methods**:  
  - Color histograms  
  - SIFT-VLAD descriptors  
  - DreamSim embeddings (using a CLIP backbone)  
  
* **SQLite Database**: Central storage for image paths and embeddings  
* **Batch Processing**: Scale effortlessly over large image collections  
* **GPU Acceleration**: Option to leverage CUDA for DreamSim feature extraction 
* **FAISS HNSW Index**: Build and query an HNSW index for sub-second similarity searches
* **Interactive Visualization** with UMAP, HDBSCAN, Plotly & Dash   


---

## 📂 Project Structure

```bash
.
├── main/
│   ├── create_db.py                # Scan image folders and insert relative filepaths into SQLite
│   ├── create_index.py             # Build a FAISS HNSW index over completed embeddings
│   ├── create_main_features.py     # Orchestrate color, SIFT-VLAD & DreamSim extraction
│   ├── db_con.py                   # Lightweight SQLite connection utility
│   ├── search_from_image.py        # Perform similarity search and display results
│   └── visualization.py            # UMAP + clustering + similarity graph rendering
│ 
├── vector_scripts/
│   ├── create_color_vector.py      # Compute and store color histogram embeddings
│   ├── create_dreamsim_vector.py   # Compute and store DreamSim embeddings (with CLIP backbone)
│   ├── create_sift_vector.py       # Compute and store SIFT-VLAD embeddings
│   └── create_vector_base.py       # Shared base class and utility functions
│
├── analytics/
│   ├── rt_Main-Features.py         # Measure runtime of feature extraction
│   ├── rt_Search.py                # Measure runtime of image search
│   └── test_vector_indexers.py     # Unit tests for indexing and retrieval pipelines
│
├── autoencoder/
│   ├── autoencoder_grid_search.py  # Grid search for compression model tuning
│   ├── encoder_optu_tuner.py       # Optuna tuner for autoencoder hyperparameters
│   └── encoder_test.py             # Test autoencoder-based compression performance
│
├── README.md                       # Project documentation
└── requirements.txt                # Project Requirements
```

## 🛠️ Installation & Dependencies

1. **Python Version**: 3.8 or higher
2. **Recommended**: CUDA-enabled GPU for accelerated feature extraction
3. **Core Libraries**:

```bash
pip install dreamsim faiss-cpu joblib lpips matplotlib numpy opencv-python Pillow requests scikit-learn scipy seaborn scikit-image submitit torch torchvision tqdm umap-learn hdbscan networkx plotly dash
```

> For GPU support with FAISS, install `faiss-gpu` instead of `faiss-cpu`.

---

## 📘 Usage Guide

### 1. Initialize the Database

```bash
python -m main.create_db --base-folder image_data --db-path images.db
```

Scans the specified folder recursively and populates the database with relative image paths.

### 2. Extract Feature Embeddings

Run all extractors in one go:

```bash
python -m main.create_main_features --db-path images.db --images-root image_data
```

Or execute each step individually:

```bash
python -m vector_scripts.create_color_vector     --db-path images.db --images-root image_data
python -m vector_scripts.create_sift_vector      --db-path images.db --images-root image_data
python -m vector_scripts.create_dreamsim_vector  --db-path images.db --images-root image_data

```

Each script processes images lacking its corresponding embeddings.

### 3. Build the FAISS Index

```bash
python -m main.create_index --db-path images.db --vector-types color sift dreamsim --output index_hnsw.faiss
```

* **`--vector-types`**: Choose any combination of `color`, `sift`, `dreamsim`.
* **`--batch-size`**, **`--hnsw_M`**, **`--efConstruction`**, **`--efSearch`**: Tune indexing performance and accuracy.

### 4. Search for Similar Images

```bash
python -m main.search_from_image --db-path images.db --images-root image_data --query path/to/query.jpg --index combo_color_sift_dreamsim --top-k 5
```

### 5. Interactive Visualisation

```bash
python -m main.visualization \
  --db-path images.db \
  --images-root image_data \
  --table dreamsim_vectors \
  --limit 2000
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
