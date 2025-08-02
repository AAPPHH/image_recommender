import sqlite3
import pickle
import numpy as np
import umap
import hdbscan
import threading
import socketserver
import http.server
import webbrowser
from pathlib import Path
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output

BASE_URL = "http://localhost:8000/image_data"

def load_vectors(table_name, vector_col, db_path="images.db", limit=1000):
    """
    Lädt Vektoren aus der angegebenen Tabelle und Spalte der SQLite-Datenbank.
    """
    print(f"Lade bis zu {limit} Vektoren aus Tabelle '{table_name}' …")
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

    image_paths, features = [], []

    for path, blob in rows:
        try:
            vec = pickle.loads(blob)
            image_paths.append(path)
            features.append(vec)
        except Exception as e:
            print(f"Fehler bei {path}: {e}")

    return image_paths, np.array(features)

def encode_image_as_url(image_path):
    """
    Generiert einen HTML-Code-Snippet für die Anzeige eines Bildes.
    """
    return f"""
        <div style='text-align:center;'>
            <img src="{BASE_URL}/{image_path}" 
                 style="width:120px; max-height:120px; object-fit:contain; border:1px solid #ccc;">
            <br><span style='font-size:10px'>{image_path}</span>
        </div>
    """

def reduce_with_umap(features, n_neighbors=15, min_dist=0.1):
    """
    Reduziert die Dimensionalität der Merkmale mit UMAP.
    """
    print("UMAP-Reduktion in 3D …")
    reducer = umap.UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=min_dist)
    return reducer.fit_transform(features)

def assign_clusters_hdbscan(features, min_cluster_size=10):
    """
    Führt HDBSCAN-Clustering auf den gegebenen Merkmalen durch.
    """
    print(f"HDBSCAN-Clustering mit min_cluster_size={min_cluster_size} …")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    return clusterer.fit_predict(features)

PORT = 8000
ROOT_DIR = Path(__file__).resolve().parent.parent

class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT_DIR), **kwargs)

def start_file_server():
    """
    Startet einen einfachen HTTP-Server, um Bilder anzuzeigen.
    """
    print(f"Starte Datei-Server unter http://localhost:{PORT}")
    print(f"Root-Verzeichnis: {ROOT_DIR}")
    threading.Thread(
        target=lambda: socketserver.TCPServer(("", PORT), CustomHandler).serve_forever(),
        daemon=True
    ).start()

app = dash.Dash(__name__)
app.title = "DreamSim mit Hover-Bildanzeige"

@app.callback(
    Output("image-preview", "children"),
    Input("umap-graph", "hoverData")
)
def show_image(hoverData):
    """
    Zeigt ein Bild an, wenn über einen Punkt im UMAP-Graphen gehovt wird.
    """
    if hoverData and "points" in hoverData:
        path = hoverData["points"][0]["customdata"]
        return html.Img(src=f"{BASE_URL}/{path}", style={"height": "200px", "border": "1px solid #ccc"})
    return "Hover über einen Punkt, um das Bild anzuzeigen."

if __name__ == "__main__":
    LIMIT = 1_000_000
    TABLE_NAME = "dreamsim_vectors"
    VECTOR_COLUMN = "dreamsim_vector_blob"
    MIN_CLUSTER_SIZE = 10

    dream_paths, dream_vecs = load_vectors(TABLE_NAME, VECTOR_COLUMN, limit=LIMIT)
    dream_embed = reduce_with_umap(dream_vecs)
    cluster_labels = assign_clusters_hdbscan(dream_embed, min_cluster_size=MIN_CLUSTER_SIZE)

    fig = go.Figure()
    unique_labels = sorted(set(cluster_labels))
    for label in unique_labels:
        idx = [i for i, lbl in enumerate(cluster_labels) if lbl == label]
        label_name = f"Cluster {label}" if label != -1 else "Noise"
        color = "gray" if label == -1 else f"hsl({(label * 360 / max(1, len(unique_labels)-1)) % 360},70%,50%)"
        fig.add_trace(go.Scatter3d(
            x=dream_embed[idx, 0], y=dream_embed[idx, 1], z=dream_embed[idx, 2],
            mode='markers', name=label_name,
            marker=dict(size=3, opacity=0.8, color=color),
            customdata=[str(dream_paths[i]) for i in idx],
            hoverinfo='skip',
            hovertemplate="%{customdata}<extra></extra>"
        ))

    fig.update_layout(
        height=700,
        title="DreamSim Cluster",
        scene=dict(
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            zaxis_title="UMAP 3"
        )
    )

    app.layout = html.Div([
        html.H2("DreamSim Cluster mit Bildvorschau"),
        html.Div([
            html.Div(id="image-preview", style={"width": "300px", "marginRight": "20px", "textAlign": "center"}),
            dcc.Graph(id="umap-graph", figure=fig, style={"flex": 1, "height": "700px"})
        ], style={"display": "flex", "flexDirection": "row"})
    ])

    start_file_server()
    webbrowser.open("http://127.0.0.1:8050")
    app.run(debug=True)
    
    fig.write_html("umap_output.html", include_plotlyjs="cdn")
    print("HTML-Datei gespeichert: umap_output.html")