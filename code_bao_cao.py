import re
import torch
import numpy as np
import pandas as pd
import networkx as nx
import pickle
from tqdm import tqdm
from underthesea import sent_tokenize, word_tokenize
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer

# ===============================
# 1. Preprocessing
# ===============================
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s\.\,\?\!\:\-\%]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def sentence_segmentation(text):
    if not text:
        return []
    sentences = sent_tokenize(text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def normalize_sentence_case(sentence):
    sentence = sentence.strip()
    if not sentence:
        return ""
    sentence = sentence.lower()
    sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
    return sentence

# ===============================
# 2. Embedding (PhoBERT)
# ===============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
model = AutoModel.from_pretrained("vinai/phobert-base-v2").to(DEVICE)
model.eval()

embedding_cache = {}

def embed_sentences(sentences, batch_size=32):
    embeddings = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]

        batch_tok = [word_tokenize(s, format="text") for s in batch]

        inputs = tokenizer(
            batch_tok,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        batch_emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.append(batch_emb)

    return np.vstack(embeddings)

def get_embeddings_cached(sentences):
    new_sentences = [s for s in sentences if s not in embedding_cache]
    if new_sentences:
        new_vecs = embed_sentences(new_sentences)
        for s, v in zip(new_sentences, new_vecs):
            embedding_cache[s] = v

        # Save cache
        with open("emb_cache.pkl", "wb") as f:
            pickle.dump(embedding_cache, f)

    return np.array([embedding_cache[s] for s in sentences])

# ===============================
# 3. Semantic TextRank
# ===============================
def semantic_textrank_summary(sentences, top_n=3):
    if len(sentences) == 0:
        return ""

    vectors = get_embeddings_cached(sentences)
    sim = cosine_similarity(vectors)
    sim[sim < 0.2] = 0  # Sparse graph

    nx_graph = nx.from_numpy_array(sim)
    try:
        scores = nx.pagerank(nx_graph)
    except:
        return " ".join(sentences[:top_n])

    ranked = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    return " ".join([s for _, s in ranked[:top_n]])

# ===============================
# 4. KMeans Extractive
# ===============================
def kmeans_summary(sentences, ratio=0.3):
    if len(sentences) == 0:
        return ""

    vectors = get_embeddings_cached(sentences)
    n_clusters = min(10, max(1, int(len(sentences) * ratio)))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(vectors)

    summary = []
    centers = kmeans.cluster_centers_
    for i in range(n_clusters):
        idxs = np.where(kmeans.labels_ == i)[0]
        if len(idxs) == 0:
            continue
        sims = cosine_similarity([centers[i]], vectors[idxs])[0]
        chosen = idxs[np.argmax(sims)]
        summary.append((chosen, sentences[chosen]))

    summary = sorted(summary, key=lambda x: x[0])
    return " ".join([s for _, s in summary])

# ===============================
# 5. Evaluation
# ===============================
semantic_model = SentenceTransformer("keepitreal/vietnamese-sbert", device=DEVICE)
rouge = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)

def evaluate_model(df, summary_column):
    r1_list, r2_list, rl_list, sem_list = [], [], [], []

    for ref_sents, summ in tqdm(zip(df['Text'], df[summary_column]), total=len(df)):
        ref_text = " ".join(ref_sents)
        summ_text = summ

        # ROUGE
        scores = rouge.score(ref_text, summ_text)
        r1_list.append(scores["rouge1"].fmeasure)
        r2_list.append(scores["rouge2"].fmeasure)
        rl_list.append(scores["rougeL"].fmeasure)

        # Semantic similarity
        ref_emb = semantic_model.encode(ref_text, convert_to_tensor=True)
        summ_emb = semantic_model.encode(summ_text, convert_to_tensor=True)
        sim_score = float(util.pytorch_cos_sim(ref_emb, summ_emb))
        sem_list.append(sim_score)

    return {
        "ROUGE-1": float(np.mean(r1_list)),
        "ROUGE-2": float(np.mean(r2_list)),
        "ROUGE-L": float(np.mean(rl_list)),
        "Semantic_Sim": float(np.mean(sem_list))
    }

# ===============================
# 6. Run pipeline
# ===============================
df = pd.read_excel(r"D:\projec_ds\data_cleaned.xlsx")

tqdm.pandas(desc="Processing")
df['Text'] = df['Text'].progress_apply(clean_text)
df['Text'] = df['Text'].progress_apply(sentence_segmentation)
df['Text'] = df['Text'].progress_apply(lambda s: [normalize_sentence_case(x) for x in s])

semantic_summaries = []
kmeans_summaries = []

for sentences in tqdm(df['Text'], desc="Generating summaries"):
    semantic_summaries.append(semantic_textrank_summary(sentences, top_n=3))
    kmeans_summaries.append(kmeans_summary(sentences, ratio=0.3))

df["summary_semantic_textrank"] = semantic_summaries
df["summary_kmeans"] = kmeans_summaries

# ===============================
# 7. Evaluate both models
# ===============================
res_semantic = evaluate_model(df, "summary_semantic_textrank")
res_kmeans = evaluate_model(df, "summary_kmeans")

results = pd.DataFrame([
    ["Semantic TextRank", *res_semantic.values()],
    ["KMeans Extractive", *res_kmeans.values()]
], columns=["Model", "ROUGE-1", "ROUGE-2", "ROUGE-L", "Semantic_Sim"])

print(results)

df.to_excel("df_with_summaries_and_evaluation.xlsx", index=False)
