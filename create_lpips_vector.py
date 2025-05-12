import os
import sqlite3
import pickle
import torch
import lpips
import gc
from torchvision.io import read_image
from torchvision.transforms.functional import resize
from torch.nn.functional import adaptive_avg_pool2d, normalize
from logging_utils import setup_logging, console_and_log

def init_logging():
    setup_logging("lpips_indexer.log")

class LPIPSVektorCreatorDB:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        console_and_log(f"Using device: {self.device}", level="info")
        self.net = lpips.LPIPS(net="alex").to(self.device).half().eval()
        console_and_log("LPIPS model loaded.", level="info")

        # Database connections
        self.read_conn = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
        self.read_conn.execute("PRAGMA journal_mode=WAL;")
        self.write_conn = sqlite3.connect(DB_PATH, timeout=30)
        self.write_conn.execute("PRAGMA journal_mode=WAL;")
        console_and_log(f"Connected to DB: {DB_PATH}", level="info")

    def index_images(self):
        rcur = self.read_conn.cursor()
        wcur = self.write_conn.cursor()
        total = rcur.execute(
            "SELECT COUNT(*) FROM images WHERE lpips_vector_blob IS NULL"
        ).fetchone()[0]
        console_and_log(f"Starting LPIPS indexing for {total} images…", level="info")

        last_id, processed, batch_no = 0, 0, 0
        while True:
            batch_no += 1
            rows = rcur.execute(
                "SELECT id, path FROM images "
                "WHERE lpips_vector_blob IS NULL AND id > ? "
                "ORDER BY id ASC LIMIT ?",
                (last_id, DB_BATCH),
            ).fetchall()
            self.read_conn.commit()

            if not rows:
                console_and_log("No more images.", level="info")
                break

            ids, paths = zip(*rows)
            last_id = ids[-1]
            console_and_log(
                f"Batch {batch_no}: loaded {len(paths)} paths (IDs {ids[0]}–{ids[-1]}).",
                level="info",
            )

            update_data = []
            total_new = 0

            for i in range(0, len(paths), MODEL_BATCH):
                sub = paths[i : i + MODEL_BATCH]
                console_and_log(
                    f"  Sub-batch {i+1}–{i+len(sub)} ({len(sub)} paths)…", level="debug"
                )

                tensors = []
                for rel in sub:
                    img_path = os.path.join(BASE_DIR, "images_v3", rel)
                    if not os.path.exists(img_path):
                        console_and_log(
                            f"    ⚠️ File not found: {img_path}", level="warning"
                        )
                        continue
                    try:
                        img = read_image(img_path).float() / 255.0
                    except Exception as e:
                        console_and_log(
                            f"    ⚠️ Error reading {img_path}: {e}", level="warning"
                        )
                        continue

                    # Ensure 3-channel RGB
                    if img.shape[0] == 1:
                        img = img.repeat(3, 1, 1)
                    elif img.shape[0] == 4:
                        img = img[:3, :, :]
                    elif img.shape[0] != 3:
                        console_and_log(
                            f"    ⚠️ Unexpected channel count ({img.shape[0]}) in {img_path}, skipped.",
                            level="warning",
                        )
                        continue

                    tensors.append(resize(img, [224, 224], antialias=True))

                console_and_log(f"    Tensors created: {len(tensors)}", level="debug")
                if not tensors:
                    continue

                # Move to GPU and normalize to [-1, 1]
                x = (torch.stack(tensors).to(self.device).half() - 0.5) / 0.5

                # Inference
                with torch.inference_mode():
                    feats = self.net.net(x)

                # Select the feature map, then pooling + normalization
                fmap = feats[-1] if isinstance(feats, (list, tuple)) else feats
                vecs = normalize(adaptive_avg_pool2d(fmap, 1).flatten(1), dim=1)

                # Bring to CPU as float16
                arr16 = vecs.cpu().numpy().astype("float16")
                console_and_log(f"    Feature shape: {arr16.shape}", level="debug")

                # Prepare update data
                before = len(update_data)
                for rec_id, vec in zip(ids[i : i + MODEL_BATCH], arr16):
                    blob = sqlite3.Binary(
                        pickle.dumps(vec, protocol=pickle.HIGHEST_PROTOCOL)
                    )
                    update_data.append((blob, rec_id))
                added = len(update_data) - before
                total_new += added
                console_and_log(f"    New entries for UPDATE: {added}", level="debug")

                # **Free GPU memory**
                del x, tensors, feats, fmap, vecs
                torch.cuda.empty_cache()
                gc.collect()

            if update_data:
                console_and_log(f"  → Writing {total_new} vectors to DB", level="info")
                wcur.executemany(
                    "UPDATE images SET lpips_vector_blob = ? WHERE id = ?", update_data
                )
                self.write_conn.commit()
            else:
                console_and_log(f"  → No update_data in batch {batch_no}", level="warning")

            processed += len(paths)
            console_and_log(f"Progress: {processed}/{total}", level="info")

        console_and_log("✅ LPIPS indexing finished.", level="info")


if __name__ == "__main__":
    BASE_DIR = os.getcwd()
    DB_PATH = "images.db"
    DB_BATCH = 8192
    MODEL_BATCH = 128

    init_logging()
    LPIPSVektorCreatorDB().index_images()
