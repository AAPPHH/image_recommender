import os
import sqlite3
import pickle
import torch
from dreamsim import dreamsim
from PIL import Image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from logging_utils import setup_logging, console_and_log
setup_logging("dreamsim_indexer.log")

class DreamSimImageIndexer:
    def __init__(
        self,
        db_path: str = "images.db",
        base_dir: str = None,
        db_batch_size: int = 4096,
        model_batch_size: int = 128,
    ):
        self.db_path = db_path
        self.base_dir = base_dir if base_dir is not None else os.getcwd()
        self.db_batch_size = db_batch_size
        self.model_batch_size = model_batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_model()
        self.conn = sqlite3.connect(self.db_path)

    def _setup_model(self):
        """
        Loads the DreamSim model and the preprocessing pipeline.
        """
        self.model, self.preprocess = dreamsim(
            pretrained=True,
            device=self.device,
            normalize_embeds=True,
            dreamsim_type="open_clip_vitb32"
        )
        self.model.eval()
        # Warm up the model with a dummy tensor to initialize weights on device
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=self.device)
            emb = self.model.embed(dummy)
        self.dim = emb.shape[-1]

    def _batch_image_to_vector(self, image_paths):
        images = []
        valid_paths = []
        for rel_path in image_paths:
            full_path = os.path.join(self.base_dir, rel_path)
            try:
                img = Image.open(full_path).convert("RGB")
                tensor = self.preprocess(img)
                
                # If preprocess added a batch dimension, remove it
                if tensor.ndim == 4 and tensor.size(0) == 1:
                    tensor = tensor.squeeze(0)
                
                images.append(tensor)
                valid_paths.append(rel_path)
            
            except Exception as e:
                console_and_log(f"⚠️ Error opening {full_path}: {e}", level="warning")
        
        if not images:
            return torch.empty(0, self.dim), []
        
        # Stack and move to device, then compute and normalize embeddings
        image_tensor = torch.stack(images).to(self.device)
        with torch.no_grad():
            embeddings = self.model.embed(image_tensor)
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        return embeddings, valid_paths

    def _generate_embeddings(self):
        cur = self.conn.cursor()
        cur.execute("SELECT id, path FROM images WHERE dreamsim_vector_blob IS NULL")
        rows = cur.fetchall()

        db_batch_ids = []
        db_batch_vecs = []
        model_batch_paths = []
        model_batch_record_ids = []

        for rec_id, rel_path in rows:
            model_batch_paths.append(rel_path)
            model_batch_record_ids.append(rec_id)

            # Once enough images for the model batch, process them
            if len(model_batch_paths) >= self.model_batch_size:
                embeddings, valid_paths = self._batch_image_to_vector(model_batch_paths)
                for i, embedding in enumerate(embeddings):
                    db_batch_ids.append(model_batch_record_ids[i])
                    db_batch_vecs.append(embedding)
                    console_and_log(
                        f"✅ [{model_batch_record_ids[i]}] {os.path.basename(valid_paths[i])} indexed."
                    )
                model_batch_paths = []
                model_batch_record_ids = []

            # Once enough vectors for the DB batch, yield them
            if len(db_batch_ids) >= self.db_batch_size:
                yield db_batch_ids, db_batch_vecs
                db_batch_ids, db_batch_vecs = [], []

        # Process any remaining model batch
        if model_batch_paths:
            try:
                embeddings, valid_paths = self._batch_image_to_vector(model_batch_paths)
                for i, embedding in enumerate(embeddings):
                    db_batch_ids.append(model_batch_record_ids[i])
                    db_batch_vecs.append(embedding)
                    console_and_log(
                        f"✅ [{model_batch_record_ids[i]}] {os.path.basename(valid_paths[i])} indexed."
                    )
            except Exception as e:
                console_and_log(f"⚠️ Error processing final batch: {e}", level="error")
        
        # Yield any remaining DB batch
        if db_batch_ids:
            yield db_batch_ids, db_batch_vecs

    def _update_db(self, batch_ids, batch_vecs):
        cur = self.conn.cursor()
        data = [(pickle.dumps(vec.cpu()), rid) for vec, rid in zip(batch_vecs, batch_ids)]
        cur.executemany("UPDATE images SET dreamsim_vector_blob = ? WHERE id = ?", data)
        self.conn.commit()

    def index_images(self):
        console_and_log("🔍 Loading image paths and creating embeddings...", level="info")
        total = 0
        for batch_ids, batch_vecs in self._generate_embeddings():
            self._update_db(batch_ids, batch_vecs)
            total += len(batch_ids)
            console_and_log(f"Batch processed: {len(batch_ids)} images. Total: {total}", level="info")
        console_and_log(f"🚀 A total of {total} new images indexed and saved in DB.", level="info")

if __name__ == "__main__":
    DB_PATH = "images.db"
    BASE_DIR = r"C:\Users\jfham\OneDrive\Dokumente\Workstation_Clones\image_recomender\image_recommender\images_v3"
    DB_BATCH_SIZE = 8192
    MODEL_BATCH_SIZE = 128

    indexer = DreamSimImageIndexer(
        db_path=DB_PATH,
        base_dir=BASE_DIR,
        db_batch_size=DB_BATCH_SIZE,
        model_batch_size=MODEL_BATCH_SIZE,
    )
    indexer.index_images()
    console_and_log("✅ All images successfully indexed.", level="info")
