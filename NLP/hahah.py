import pandas as pd

# Đọc file JSON Lines
df = pd.read_json(r"D:/News-Classification/NLP/data_TL_check.jsonl", lines=True)


print(df['content'])
