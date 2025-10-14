import json
import gzip
from pathlib import Path
from pymongo import MongoClient, UpdateOne

# ==== Cấu hình ====
INPUT_FILE = Path("data/traffic_clean.jsonl")  # đường dẫn tới file dữ liệu
MONGO_URI = "mongodb+srv://redwhite7769:JdOaHPpbjXIopN5E@tomtat.3ptutix.mongodb.net/?retryWrites=true&w=majority&appName=tomtat"
DB_NAME = "traffic"  # tên database muốn tạo
COLLECTION_NAME = "articles"  # tên collection muốn import
UPSERT_KEY = "_md5_clean"  # khoá nhận dạng duy nhất (nội dung đã làm sạch)
BATCH_SIZE = 2000

# ==== Kết nối MongoDB Atlas ====
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
col = db[COLLECTION_NAME]

# ==== Tạo index cần thiết ====
col.create_index([(UPSERT_KEY, 1)], unique=True)
col.create_index([("domain", 1), ("pub_date", 1)])
col.create_index([("title", "text"), ("content", "text")])


# ==== Hàm đọc file .jsonl hoặc .gz ====
def stream_jsonl(path: Path):
    if str(path).lower().endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    else:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


# ==== Thực hiện import ====
ops = []
count, inserted = 0, 0

for doc in stream_jsonl(INPUT_FILE):
    key = doc.get(UPSERT_KEY)
    if key:
        ops.append(UpdateOne({UPSERT_KEY: key}, {"$set": doc}, upsert=True))
    else:
        ops.append(UpdateOne({"url": doc.get("url")}, {"$set": doc}, upsert=True))

    if len(ops) >= BATCH_SIZE:
        result = col.bulk_write(ops, ordered=False)
        inserted += result.upserted_count + result.modified_count
        print(f"✅ Đã import {inserted} bản ghi...")
        ops.clear()

# flush cuối
if ops:
    result = col.bulk_write(ops, ordered=False)
    inserted += result.upserted_count + result.modified_count

print(f"🎉 Import hoàn tất! Tổng cộng: {inserted} bài viết.")