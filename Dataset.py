import os
import json
from PIL import Image
from torch.utils.data import Dataset

class HatefulMemesDataset(Dataset):
    def __init__(self, cfg, split="train", max_samples=None):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.image_size = cfg.image_size

        if split == "train":
            self.jsonl_path = cfg.train_jsonl
        elif split == "dev":
            self.jsonl_path = cfg.dev_jsonl
        else:
            self.jsonl_path = cfg.test_jsonl

        self.entries = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                self.entries.append(item)
                if max_samples is not None and max_samples >= len(self.entries):
                    break

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        item = self.entries[idx]
        text = item.get("text", "No Text")
        label = item["label"] if "label" in item else None  # 先检查是否存在
        if label is None:
            raise ValueError(f"数据缺失: {item}")
        image_path = item["img"]
        if image_path.startswith("img/"):
            image_path = image_path[4:]

        image_fn = os.path.join(self.cfg.img_folder, image_path)
        try:
            image = Image.open(image_fn).convert("RGB")
            image = image.resize((self.image_size, self.image_size))
        except Exception as e:
            raise ValueError(f"Error loading image {image_fn}: {e}")

        return {
            "image": image,
            "text": text,
            "label": label,
            "idx_meme": item["img"]
        }
