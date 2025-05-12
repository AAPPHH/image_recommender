# 🖼️ Image Recommender with Color, LPIPS, DreamSim & FAISS

A high-performance image similarity search pipeline leveraging multiple embedding strategies and efficient indexing. Store image paths and embeddings in SQLite, then build a FAISS HNSW index for fast nearest-neighbor retrieval.

---

## 🚀 Key Features

- **Diverse Embedding Methods**: Color histograms, LPIPS perceptual features, DreamSim embeddings (using a CLIP backbone)
- **SQLite Database**: Central storage of image filepaths and embedding blobs in a single `images` table
- **Batch Processing**: Scale effortlessly over large image collections with configurable batch sizes
- **GPU Acceleration**: Option to leverage CUDA for LPIPS and DreamSim feature extraction
- **FAISS HNSW Index**: Build and query an HNSW index for sub-second similarity searches
- **Flexible Search**: Combine any subset of supported embeddings (e.g., `color`, `lpips`, `dreamsim`)
- **Built-in Visualization**: Quickly display top-K similar images using Matplotlib

---

## 📂 Project Structure

```bash
.
├── create_db.py              # Scan image folders and insert relative filepaths into SQLite
├── create_color_vector.py    # Compute and store color histogram embeddings
├── create_lpips_vector.py    # Compute and store LPIPS perceptual embeddings
├── create_dreamsim_vector.py # Compute and store DreamSim embeddings (with CLIP backbone)
├── ceate_main_features.py    # Orchestrate color, LPIPS & DreamSim extraction in a single pipeline
├── create_index.py           # Build a FAISS HNSW index over completed embeddings
├── search_from_image.py      # Perform similarity search and display results
└── README.md                 # Project documentation
``` 

---

## 🛠️ Installation & Dependencies

1. **Python Version**: 3.8 or higher
2. **Recommended**: CUDA-enabled GPU for accelerated feature extraction
3. **Core Libraries**:

```bash
pip install torch torchvision lpips dreamsim faiss-cpu Pillow matplotlib opencv-python
```

> For GPU support with FAISS, install `faiss-gpu` instead of `faiss-cpu`.

---

## 📘 Usage Guide

### 1. Initialize the Database

```bash
python create_db.py --base-folder /path/to/images --db-path images.db
```

Recursively scans the specified folder and populates the `images` table with relative filepaths.

### 2. Extract Feature Embeddings

Run all extractors in one go:

```bash
python ceate_main_features.py
```

Or execute each step individually:

```bash
python create_color_vector.py
python create_lpips_vector.py
python create_dreamsim_vector.py
```

Each script locates records where its corresponding embedding blob is `NULL` and appends the new data.

### 3. Build the FAISS Index

```bash
python create_index.py --vector-types color lpips dreamsim --output index_hnsw.faiss
```

- **`--vector-types`**: Choose any combination of `color`, `lpips`, `dreamsim`.
- **`--batch-size`**, **`--hnsw_M`**, **`--efConstruction`**, **`--efSearch`**: Tune indexing speed vs. recall.

### 4. Search for Similar Images

```bash
python search_from_image.py --query /path/to/query.jpg --index combo_color_lpips_dreamsim --top-k 5
```

Renders the top-K matches with distance metrics.

---

## ⚙️ Configuration Parameters

| Parameter          | Description                                        |
|--------------------|----------------------------------------------------|
| `BASE_DIR`         | Root folder containing images                      |
| `DB_PATH`          | SQLite database file path                          |
| `batch_size`       | Records processed per batch                        |
| `model_batch_size` | Number of images sent to the model at once         |
| `bins`             | Number of bins per channel for color histograms    |
| `hnsw_M`           | FAISS HNSW "M" parameter                           |
| `efConstruction`   | FAISS index construction effort vs. accuracy trade |
| `efSearch`         | FAISS search speed vs. recall trade-off            |
| `TOP_K`            | Number of neighbors retrieved in search            |

---

## 🗄️ Database Schema

**Table `images`**

| Column                  | Type    | Description                         |
|-------------------------|---------|-------------------------------------|
| `id`                    | INTEGER | Primary key                         |
| `filepath`              | TEXT    | Relative path under `BASE_DIR`      |
| `color_vector_blob`     | BLOB    | Serialized color histogram vector   |
| `lpips_vector_blob`     | BLOB    | Serialized LPIPS feature vector     |
| `dreamsim_vector_blob`  | BLOB    | Serialized DreamSim embedding       |

**Offset Tables**

Each FAISS combination creates a table `faiss_index_offsets_<combo>` mapping `image_id` → index offset.

---

## 📄 License

MIT License — freely use, modify, and distribute.

---

## 💡 Applications

Ideal for creative archives, research projects, and media libraries where rapid visual similarity search enhances curation workflows and recommendation engines.
