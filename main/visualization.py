import hashlib
import http.server
import pickle
import socketserver
import sqlite3
import threading
import webbrowser
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import duckdb
import hdbscan
import numpy as np
import plotly.graph_objects as go
import umap
from dash import dcc, html, Input, Output

BASE_URL = "http://localhost:8000"
DASH_PORT = 8050
FILE_SERVER_PORT = 8000
ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = "images.db"
TABLE_NAME = "dreamsim_vectors"
VECTOR_COLUMN = "dreamsim_vector_blob"
CACHE_DIR = Path("cache")

# Data and Processing Parameters
LIMIT = 15_000
UMAP_PARAMS = {
    "n_neighbors": 15,
    "min_dist": 0.1,
    "n_components": 3,
    "random_state": 42
}
HDBSCAN_PARAMS = {
    "min_cluster_size": 10,
    "cluster_selection_method": 'eom'
}

dream_vectors = np.array([])
umap_embeddings = np.array([])
cluster_labels = np.array([])
relative_image_paths = []

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.SOLAR, dbc.icons.FONT_AWESOME]
)
app.title = "Interactive Cluster Analysis"


def load_vectors_from_sqlite(db_path, table_name, vector_col, limit):
    """Loads image paths and vectors from a SQLite database.

    Args:
        db_path (str): The path to the SQLite database file.
        table_name (str): The name of the table containing the vectors.
        vector_col (str): The name of the column with the vector BLOBs.
        limit (int): The maximum number of records to load.

    Returns:
        tuple[list[str], np.ndarray]: A tuple containing a list of image paths
                                      and a NumPy array of the vectors.
    """
    print("Reading vectors from SQLite DB.")
    image_paths, features = [], []
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        query = (
            f"SELECT i.path, v.{vector_col} FROM images i "
            f"JOIN {table_name} v ON i.id = v.image_id LIMIT {limit}"
        )
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()

        for path_str, blob in rows:
            try:
                image_paths.append(path_str)
                features.append(pickle.loads(blob))
            except pickle.UnpicklingError as e:
                print(f"Error deserializing blob for '{path_str}': {e}")

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return [], np.array([])

    return image_paths, np.array(features)


def cache_vectors_with_duckdb(cache_filename, limit):
    """Loads vectors from a DuckDB cache or creates it from SQLite.

    Args:
        cache_filename (str): Filename for the DuckDB cache file.
        limit (int): Maximum number of records to load if the cache needs
                     to be created.

    Returns:
        tuple[list[str], np.ndarray]: A tuple of image paths and vectors.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache_path = CACHE_DIR / cache_filename
    if cache_path.exists():
        print(f"Loading vectors from fast DuckDB cache: '{cache_path}'")
        with duckdb.connect(str(cache_path), read_only=True) as con:
            result_df = con.execute("SELECT path, vector FROM vectors").fetch_df()
        return result_df['path'].tolist(), np.array(result_df['vector'].tolist())

    print("DuckDB cache not found. Performing initial load...")
    image_paths_str, features = load_vectors_from_sqlite(
        DB_PATH, TABLE_NAME, VECTOR_COLUMN, limit
    )

    if features.size == 0:
        print("No vectors loaded. Skipping cache creation.")
        return image_paths_str, features

    print(f"Creating fast DuckDB cache: '{cache_path}'")
    with duckdb.connect(str(cache_path)) as con:
        vector_length = features.shape[1]
        con.execute(
            f"CREATE TABLE vectors (path VARCHAR, vector FLOAT[{vector_length}]);"
        )
        for path_str, vec in zip(image_paths_str, features):
            con.execute("INSERT INTO vectors VALUES (?, ?)", [path_str, vec.tolist()])
    return image_paths_str, features


def cache_data(filename, func, *args, **kwargs):
    """A generic caching function using Pickle.

    Checks if a cache file exists. If so, loads the result from the file.
    Otherwise, it executes the given function, saves the result, and then
    returns it.

    Args:
        filename (str): The filename for the Pickle cache file.
        func (callable): The function whose result should be cached.
        *args: Positional arguments for the function `func`.
        **kwargs: Keyword arguments for the function `func`.

    Returns:
        Any: The result of `func(*args, **kwargs)`.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache_path = CACHE_DIR / filename
    if cache_path.exists():
        print(f"Loading '{filename}' from Pickle cache.")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"Calculating '{filename}' and saving to Pickle cache.")
    result = func(*args, **kwargs)
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    return result


# --- Data Processing Wrappers ---

def reduce_with_umap(features, **kwargs):
    """Performs UMAP dimensionality reduction.

    Args:
        features (np.ndarray): The high-dimensional data array.
        **kwargs: Parameters for the umap.UMAP initializer.

    Returns:
        np.ndarray: The data array reduced to 3 dimensions.
    """
    print(f"Performing UMAP reduction with parameters: {kwargs}")
    reducer = umap.UMAP(**kwargs)
    return reducer.fit_transform(features)


def assign_clusters_hdbscan(features, **kwargs):
    """Performs HDBSCAN clustering.

    Args:
        features (np.ndarray): The data array to cluster on (typically the
                               UMAP embeddings).
        **kwargs: Parameters for the hdbscan.HDBSCAN initializer.

    Returns:
        np.ndarray: An array of cluster labels. Noise points are labeled -1.
    """
    print(f"Performing HDBSCAN clustering with parameters: {kwargs}")
    clusterer = hdbscan.HDBSCAN(gen_min_span_tree=True, **kwargs)
    return clusterer.fit_predict(features)


# --- Server and App Components ---

class CustomHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP handler to serve files from the project's root directory."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT_DIR), **kwargs)


def start_file_server():
    """Starts a simple file server in a separate thread."""
    print(
        f"Starting file server at {BASE_URL} for directory {ROOT_DIR}"
    )
    handler = CustomHandler
    server = socketserver.TCPServer(("", FILE_SERVER_PORT), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()


def _create_stats_table(cluster_id, umap_coords, original_vector):
    """Creates a Dash Bootstrap component table for vector statistics.

    Args:
        cluster_id (int): The cluster ID of the data point.
        umap_coords (np.ndarray): The 3D UMAP coordinates of the point.
        original_vector (np.ndarray): The original high-dimensional vector.

    Returns:
        dbc.Table: A Dash component displaying the statistics.
    """
    stats = {
        "Mean": f"{np.mean(original_vector):.4f}",
        "Std. Dev.": f"{np.std(original_vector):.4f}",
        "Min": f"{np.min(original_vector):.4f}",
        "Max": f"{np.max(original_vector):.4f}",
        "L2-Norm": f"{np.linalg.norm(original_vector):.4f}"
    }

    header = [html.Thead(html.Tr([html.Th("Metric"), html.Th("Value")]))]
    body = [html.Tbody([
        html.Tr([
            html.Td("Cluster ID"),
            html.Td(html.B(str(cluster_id)) if cluster_id != -1 else "Noise")
        ]),
        html.Tr([html.Td("UMAP X"), html.Td(f"{umap_coords[0]:.3f}")]),
        html.Tr([html.Td("UMAP Y"), html.Td(f"{umap_coords[1]:.3f}")]),
        html.Tr([html.Td("UMAP Z"), html.Td(f"{umap_coords[2]:.3f}")]),
        *[html.Tr([html.Td(key), html.Td(value)]) for key, value in stats.items()]
    ])]

    return dbc.Table(
        header + body,
        bordered=False,
        striped=True,
        hover=True,
        size="sm",
        responsive=True
    )

# --- Dash Callback ---

@app.callback(
    Output("image-preview", "children"),
    Output("vector-stats-table", "children"),
    Input("umap-graph", "hoverData")
)
def show_image_and_stats_on_hover(hoverData):
    """Updates the preview panel when a user hovers over a point in the graph.

    Args:
        hoverData (dict | None): Data from the hover event in the Plotly graph.
                                 Contains information about the hovered point.

    Returns:
        tuple[dash.development.base_component.Component, ...]:
            Components to update the image preview and stats table.
    """
    if not hoverData or "points" not in hoverData:
        placeholder_text = html.Div([
            html.I(className="fa-regular fa-hand-pointer fa-2x mb-2"),
            html.P("Hover over a point for image preview and details.")
        ], style={'padding': '20px', 'textAlign': 'center', 'color': 'grey'})
        return placeholder_text, None

    point = hoverData["points"][0]
    path_str = point["customdata"]

    try:
        idx = relative_image_paths.index(path_str)
    except ValueError:
        return html.P("Image not found in data."), None

    # Retrieve data using the found index from module-level variables
    original_vector = dream_vectors[idx]
    coords = umap_embeddings[idx]
    cluster_id = cluster_labels[idx]

    stats_table = _create_stats_table(cluster_id, coords, original_vector)

    image_preview = html.Img(
        src=f"{BASE_URL}/{path_str}",
        style={
            "display": "block",
            "margin": "auto",
            "maxHeight": "300px",
            "maxWidth": "100%",
        }
    )
    return image_preview, stats_table


# --- Main Application Logic ---

def load_and_process_data():
    """Loads vectors, reduces dimensions, and performs clustering."""
    global dream_vectors, umap_embeddings, cluster_labels, relative_image_paths

    vector_cache_file = f"vectors_limit{LIMIT}.duckdb"
    image_paths, dream_vectors = cache_vectors_with_duckdb(vector_cache_file, LIMIT)
    relative_image_paths = [Path(p).as_posix() for p in image_paths]

    vecs_hash = hashlib.sha256(dream_vectors.tobytes()).hexdigest()[:10]
    umap_cache_file = (
        f"umap_embed_{vecs_hash}_{UMAP_PARAMS['n_neighbors']}_"
        f"{UMAP_PARAMS['min_dist']}.pkl"
    )
    umap_embeddings = cache_data(
        umap_cache_file, reduce_with_umap, dream_vectors, **UMAP_PARAMS
    )

    embed_hash = hashlib.sha256(umap_embeddings.tobytes()).hexdigest()[:10]
    hdbscan_cache_file = (
        f"hdbscan_clusters_{embed_hash}_{HDBSCAN_PARAMS['min_cluster_size']}_"
        f"{HDBSCAN_PARAMS['cluster_selection_method']}.pkl"
    )
    cluster_labels = cache_data(
        hdbscan_cache_file,
        assign_clusters_hdbscan,
        umap_embeddings,
        **HDBSCAN_PARAMS
    )


def calculate_cluster_colors(labels, embeddings):
    """Calculates a unique color for each cluster based on its centroid.

    The color is derived from the normalized 3D coordinates of the cluster's
    centroid in the UMAP space.

    Args:
        labels (np.ndarray): Array of cluster labels.
        embeddings (np.ndarray): The 3D UMAP embeddings.

    Returns:
        dict[int, str]: A dictionary mapping cluster IDs to RGB color strings.
    """
    unique_labels = sorted(list(set(labels)))
    non_noise_labels = [l for l in unique_labels if l != -1]

    if not non_noise_labels:
        return {}

    centroids = {
        label: np.mean(embeddings[labels == label], axis=0)
        for label in non_noise_labels
    }
    centroid_coords = np.array(list(centroids.values()))
    min_coords = centroid_coords.min(axis=0)
    max_coords = centroid_coords.max(axis=0)
    coord_range = np.where((max_coords - min_coords) == 0, 1, max_coords - min_coords)

    normalized_centroids = (centroid_coords - min_coords) / coord_range

    cluster_colors = {
        label: f'rgb({int(norm[0]*255)}, {int(norm[1]*255)}, {int(norm[2]*255)})'
        for label, norm in zip(centroids.keys(), normalized_centroids)
    }
    return cluster_colors


def create_main_figure(embeddings, labels, paths, colors):
    """Creates the main 3D scatter plot figure.

    Args:
        embeddings (np.ndarray): 3D UMAP embeddings.
        labels (np.ndarray): Cluster labels.
        paths (list[str]): Relative paths to the images.
        colors (dict[int, str]): Dictionary mapping cluster IDs to colors.

    Returns:
        go.Figure: The Plotly Graph Object figure.
    """
    print("Creating Plotly figure...")
    fig = go.Figure()
    unique_labels = sorted(list(set(labels)))

    for label in unique_labels:
        idx = np.where(labels == label)[0]
        custom_data_for_trace = [paths[i] for i in idx]

        if label == -1:
            color, name, opacity = "dimgray", "Noise", 0.3
        else:
            color = colors.get(label, "white")
            name = f"Cluster {label}"
            opacity = 0.8

        fig.add_trace(go.Scatter3d(
            x=embeddings[idx, 0], y=embeddings[idx, 1], z=embeddings[idx, 2],
            mode='markers',
            name=name,
            marker=dict(size=3, opacity=opacity, color=color),
            customdata=custom_data_for_trace,
            hoverinfo='none'  # Custom hover handled by Dash callback
        ))

    fig.update_layout(
        title="Interactive UMAP Analysis of Image Vectors",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        scene=dict(
            xaxis_title="UMAP 1 (→ Red)",
            yaxis_title="UMAP 2 (→ Green)",
            zaxis_title="UMAP 3 (→ Blue)"
        ),
        showlegend=False,
    )
    return fig


def create_app_layout(figure):
    """Creates the main layout for the Dash application.

    Args:
        figure (go.Figure): The main plot to display.

    Returns:
        dbc.Container: The root component of the app layout.
    """
    return dbc.Container(fluid=True, className="p-4", children=[
        dbc.Row(dbc.Col(html.H1(
            "Interactive Cluster Analysis of Image Vectors",
            className="text-center mb-4"
        ))),
        dbc.Row([
            dbc.Col(md=3, children=[
                dbc.Card([
                    dbc.CardHeader(html.H4([
                        html.I(className="fa-solid fa-image me-2"), "Preview & Details"
                    ], className="mb-0")),
                    dbc.CardBody([
                        dcc.Loading(
                            id="loading-preview", type="circle",
                            children=html.Div(
                                id="image-preview",
                                style={"minHeight": "310px"}
                            )
                        ),
                        html.Hr(className="my-3"),
                        html.H6("Vector Statistics", className="mb-2"),
                        html.Div(id="vector-stats-table")
                    ])
                ], className="h-100")
            ]),
            dbc.Col(md=9, children=[
                dbc.Card([
                    dbc.CardHeader(html.H4([
                        html.I(className="fa-solid fa-cubes me-2"),
                        "UMAP 3D Visualization"
                    ], className="mb-0")),
                    dbc.CardBody(dcc.Loading(
                        id="loading-graph", type="circle", children=[
                            dcc.Graph(
                                id="umap-graph",
                                figure=figure,
                                style={"height": "80vh"}
                            )
                        ]
                    ))
                ], className="h-100")
            ]),
        ])
    ])


def main():
    """Main function to run the application."""
    load_and_process_data()
    
    cluster_colors = calculate_cluster_colors(cluster_labels, umap_embeddings)
    
    main_figure = create_main_figure(
        umap_embeddings, cluster_labels, relative_image_paths, cluster_colors
    )
    
    app.layout = create_app_layout(main_figure)
    
    start_file_server()
    webbrowser.open(f"http://127.0.0.1:{DASH_PORT}")
    app.run_server(port=DASH_PORT, debug=False)

if __name__ == "__main__":
    main()