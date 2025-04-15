import os
import sqlite3
import pickle
import torch
import open_clip
import logging
from PIL import Image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Configure logging once:
logging.basicConfig(
    level=logging.INFO,
    filename="clip_indexer.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)

def console_and_log(message, level="info"):
    """
    Prints `message` to the console and logs it
    with the specified logging level.
    """
    print(message)
    level = level.lower()
    if level == "info":
        logging.info(message)
    elif level == "warning":
        logging.warning(message)
    elif level == "error":
        logging.error(message)
    else:
        logging.debug(message)

class CLIPImageIndexer:
    def __init__(
        self,
        db_path="images.db",
        base_dir=None,
        dim=768,
        db_batch_size=4096,
        model_batch_size=128,
    ):
        self.db_path = db_path
        self.base_dir = base_dir if base_dir is not None else os.getcwd()
        self.dim = dim
        self.db_batch_size = db_batch_size
        self.model_batch_size = model_batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_model()
        self.conn = sqlite3.connect(self.db_path)

    def _setup_model(self):
        """
        Loads the ViT-L-14 model from OpenCLIP (OpenAI weights)
        and sets it to fp16 if possible.
        """
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name="ViT-L-14",
            pretrained="openai",
            precision="fp16",
            device=self.device,
            force_quick_gelu=True,
            jit=True,
        )
        self.model.eval()

    def _batch_image_to_vector(self, image_paths):
        images = []
        valid_paths = []
        for rel_path in image_paths:
            full_path = os.path.join(self.base_dir, rel_path)
            try:
                img = Image.open(full_path).convert("RGB")
                images.append(self.preprocess(img))
                valid_paths.append(rel_path)
            except Exception as e:
                console_and_log(f"⚠️ Error opening {full_path}: {e}", level="warning")
        image_tensor = torch.stack(images).to(self.device).half()
        with torch.no_grad():
            embeddings = self.model.encode_image(image_tensor, normalize=True)
        return embeddings, valid_paths

    def _generate_embeddings(self):
        cur = self.conn.cursor()
        cur.execute("SELECT id, filepath FROM images WHERE vector_blob IS NULL")
        rows = cur.fetchall()

        db_batch_ids = []
        db_batch_vecs = []
        db_batch_paths = []

        model_batch_paths = []
        model_batch_record_ids = []

        for rec_id, rel_path in rows:
            model_batch_paths.append(rel_path)
            model_batch_record_ids.append(rec_id)

            if len(model_batch_paths) == self.model_batch_size:
                try:
                    embeddings, valid_paths = self._batch_image_to_vector(
                        model_batch_paths
                    )
                    for i, embedding in enumerate(embeddings):
                        db_batch_ids.append(model_batch_record_ids[i])
                        db_batch_vecs.append(embedding)
                        db_batch_paths.append(valid_paths[i])
                        console_and_log(
                            f"✅ [{model_batch_record_ids[i]}] {os.path.basename(valid_paths[i])} indexed."
                        )
                except Exception as e:
                    console_and_log(f"⚠️ Error processing batch: {e}", level="error")

                model_batch_paths = []
                model_batch_record_ids = []

            if len(db_batch_ids) >= self.db_batch_size:
                yield db_batch_ids, db_batch_vecs
                db_batch_ids, db_batch_vecs, db_batch_paths = [], [], []

        if model_batch_paths:
            try:
                embeddings, valid_paths = self._batch_image_to_vector(model_batch_paths)
                for i, embedding in enumerate(embeddings):
                    db_batch_ids.append(model_batch_record_ids[i])
                    db_batch_vecs.append(embedding)
                    db_batch_paths.append(valid_paths[i])
                    console_and_log(
                        f"✅ [{model_batch_record_ids[i]}] {os.path.basename(valid_paths[i])} indexed."
                    )
            except Exception as e:
                console_and_log(f"⚠️ Error processing final batch: {e}", level="error")

        if db_batch_ids:
            yield db_batch_ids, db_batch_vecs

    def _update_db(self, batch_ids, batch_vecs):
        cur = self.conn.cursor()
        data_to_update = [
            (pickle.dumps(vec), rid) for vec, rid in zip(batch_vecs, batch_ids)
        ]
        cur.executemany("UPDATE images SET vector_blob=? WHERE id=?", data_to_update)
        self.conn.commit()

    def index_images(self):
        console_and_log(
            "🔍 Loading image paths from database and creating embeddings...",
            level="info",
        )
        total_processed = 0
        for batch_ids, batch_vecs in self._generate_embeddings():
            self._update_db(batch_ids, batch_vecs)
            total_processed += len(batch_ids)
            console_and_log(
                f"Batch processed: {len(batch_ids)} images. Total: {total_processed}",
                level="info",
            )
        console_and_log(
            f"\n🚀 A total of {total_processed} new images were processed and saved to the DB.",
            level="info",
        )


if __name__ == "__main__":
    base_dir = r"C:\Users\jfham\OneDrive\Dokumente\Workstation_Clones\image_recomender\image_recommender\images_v3"
    indexer = CLIPImageIndexer(
        db_path="images.db",
        base_dir=base_dir,
        dim=768,
        db_batch_size=8192,
        model_batch_size=128,
    )
    indexer.index_images()
    console_and_log(
        "✅ All images were successfully indexed and saved to the database.",
        level="info",
    )
