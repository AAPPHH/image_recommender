# 🖼️ Image Recommender with OpenCLIP + FAISS

Ein hochperformantes System zur Bildähnlichkeitssuche basierend auf OpenCLIP-Embeddings, SQLite-Datenbank und FAISS-Indexierung. Ideal für Projekte, bei denen eine schnelle und GPU-beschleunigte Suche nach visueller Ähnlichkeit gefragt ist.

## 🚀 Features

- 🔍 **CLIP-basiertes Embedding** mit `ViT-L-14` (OpenAI pretrained)
- 🗂️ **Datenbank** (`SQLite`) zur Speicherung von Pfaden und Vektoren
- ⚡ **Vektorisierung** mit CUDA-Unterstützung
- 📦 **HNSW-Indexierung** mit FAISS
- 🎯 **Bildbasierte Ähnlichkeitssuche** mit visueller Vorschau (via Matplotlib)
- 🖼️ Unterstützung für `.jpg`, `.jpeg`, `.png`

---

## 🔧 Projektstruktur

```bash
.
├── create_db.py            # Erstellt und füllt die SQLite-Datenbank mit Bildpfaden
├── create_features.py      # Erstellt Vektoren aus Bildern via OpenCLIP
├── create_index.py         # Baut einen FAISS HNSW Index auf den Embeddings
└── search_from_image.py    # Führt eine bildbasierte Ähnlichkeitssuche durch
```

---

## 🛠️ Setup

### Voraussetzungen

- Python 3.8+
- CUDA-fähige GPU (optional, aber empfohlen)
- Empfohlene Pakete:

```bash
pip install torch torchvision open_clip_torch faiss-cpu Pillow matplotlib seaborn
```

> Für CUDA-Unterstützung `faiss-gpu` installieren und `torch` entsprechend deiner GPU-Version.

---

## 📸 Verwendung

### 1. Datenbank erstellen

```bash
python create_db.py
```

Durchsucht das Bilderverzeichnis rekursiv und legt relative Pfade in einer SQLite-Datenbank ab.

---

### 2. Feature-Extraktion (OpenCLIP)

```bash
python create_features.py
```

Lädt Bilder, erzeugt normalisierte Embeddings mit `ViT-L-14` und speichert sie als BLOBs in die Datenbank.

---

### 3. FAISS-Index aufbauen

```bash
python create_index.py
```

Lädt alle vorhandenen Vektoren aus der Datenbank und erstellt einen HNSW-Index (inkl. Speicherung von Offsets).

---

### 4. Bildähnlichkeitssuche

```bash
python search_from_image.py
```

Gibt zu einem eingegebenen Bild die `Top-K` ähnlichsten Bilder visuell aus.

---

## ⚙️ Konfigurierbare Parameter

In den jeweiligen Skripten anpassbar:

- `BASE_DIR`: Basisverzeichnis der Bilder
- `db_path`: Pfad zur SQLite-Datenbank
- `batch_size`, `model_batch_size`: Einfluss auf Speed vs. Speicherbedarf
- `hnsw_M`, `efConstruction`, `efSearch`: Feinjustierung des FAISS-Index
- `TOP_K`: Anzahl ähnlicher Bilder bei der Suche

---

## 📂 Datenbankstruktur

Tabelle `images`:

| id | filepath (relativ) | vector_blob | faiss_index_offset |
|----|--------------------|-------------|---------------------|
| 1  | `img/foo.jpg`      | BLOB        | 42                  |

---

## 🧠 Modell-Infos

- **Modell**: `ViT-L-14` (CLIP, pretrained by OpenAI)
- **Lib**: `open_clip_torch`
- **Vektordimension**: 768 (standard für dieses Modell)

---

## 📝 Lizenz

MIT License – feel free to use, modify and share.

---

## 💡 Inspiration

Diese Pipeline ist ideal, um große Bildmengen lokal effizient durchsuchbar zu machen – sei es in Forschung, Medienarchiven oder kreativen Projekten. ✨
