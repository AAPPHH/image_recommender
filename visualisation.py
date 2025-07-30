import sqlite3
import pickle
import numpy as np
import umap.umap_ as umap
import plotly.graph_objects as go


def load_vectors(table_name, vector_col, db_path="images.db", limit=1000):
    print(f"📥 Lade bis zu {limit} Vektoren aus Tabelle '{table_name}' …")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = f"""
        SELECT i.path, v.{vector_col}
        FROM images i
        JOIN {table_name} v ON i.id = v.image_id
        LIMIT {limit}
    """
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()

    image_paths, features, categories = [], [], []

    for path, blob in rows:
        try:
            vec = pickle.loads(blob)
            image_paths.append(path)
            features.append(vec)
            categories.append(path.split("image_data/")[1].split("/")[0])
        except Exception as e:
            print(f"❌ Fehler bei {path}: {e}")

    return image_paths, np.array(features), categories


def reduce_with_umap(features, n_neighbors=15, min_dist=0.1):
    reducer = umap.UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=min_dist)
    return reducer.fit_transform(features)


def plot_3d(embedding, image_paths, features, categories, title_prefix):
    print(f"📊 Visualisiere {title_prefix}-Vektoren …")
    unique_sources = sorted(set(categories))
    color_map = {
        src: f"hsl({i * 360 / len(unique_sources)},70%,50%)"
        for i, src in enumerate(unique_sources)
    }

    fig = go.Figure()
    for src in unique_sources:
        idx = [i for i, cat in enumerate(categories) if cat == src]
        fig.add_trace(go.Scatter3d(
            x=embedding[idx, 0], y=embedding[idx, 1], z=embedding[idx, 2],
            mode='markers', name=src,
            marker=dict(size=3, opacity=0.7, color=color_map[src]),
            text=[f"<b>{image_paths[i]}</b><br>{np.round(features[i][:5], 3)}…" for i in idx],
            hoverinfo='text'
        ))

    fig.update_layout(title=f"{title_prefix} – UMAP 3D", scene=dict(
        xaxis_title="UMAP 1", yaxis_title="UMAP 2", zaxis_title="UMAP 3"
    ), legend_title="Datenquelle")
    fig.show()


if __name__ == "__main__":
    LIMIT = 1_000_000

    # Color
    color_paths, color_vecs, color_cats = load_vectors("color_vectors", "color_vector_blob", limit=LIMIT)
    color_embed = reduce_with_umap(color_vecs)
    plot_3d(color_embed, color_paths, color_vecs, color_cats, "Color")

    # SIFT
    sift_paths, sift_vecs, sift_cats = load_vectors("sift_vectors", "sift_vector_blob", limit=LIMIT)
    sift_embed = reduce_with_umap(sift_vecs)
    plot_3d(sift_embed, sift_paths, sift_vecs, sift_cats, "SIFT")

    # DreamSim
    dream_paths, dream_vecs, dream_cats = load_vectors("dreamsim_vectors", "dreamsim_vector_blob", limit=LIMIT)
    dream_embed = reduce_with_umap(dream_vecs)
    plot_3d(dream_embed, dream_paths, dream_vecs, dream_cats, "DreamSim")
