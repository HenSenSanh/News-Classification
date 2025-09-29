
import argparse, os, json, random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report

from datasets import Dataset as HFDataset
from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding, EarlyStoppingCallback
)

# --------------------- Utils ---------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1m = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1m}

def class_weights(labels, num_labels: int):
    counts = np.bincount(labels, minlength=num_labels)
    freq = counts / max(counts.sum(), 1)
    inv = 1.0 / np.clip(freq, 1e-8, None)
    w = inv / inv.sum() * num_labels  # normalize to mean=1
    return torch.tensor(w, dtype=torch.float)

# --------------------- Data ---------------------

def load_data(csv_path: str, task: str, test_size: float, seed: int):
    df = pd.read_csv(csv_path)

    if "message_text" not in df.columns:
        raise ValueError("CSV must contain 'message_text' column")
    df = df.dropna(subset=["message_text"])
    df = df[df["message_text"].astype(str).str.len() > 0].copy()

    if task == "binary":
        if "label_binary" not in df.columns:
            raise ValueError("CSV must contain 'label_binary' for binary task")
        df["label_str"] = df["label_binary"].astype(int).astype(str)  # "0" / "1"
        target_col = "label_str"
    else:
        if "label" not in df.columns:
            raise ValueError("CSV must contain 'label' for multiclass task")
        df["label"] = df["label"].astype(str).str.strip().str.lower()
        target_col = "label"

    # Build mappings
    labels = sorted(df[target_col].unique().tolist())
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}

    # Map to ids for split stats if needed
    df["_yid"] = df[target_col].map(label2id).astype(int)

    # Group split by thread_id when available
    if "thread_id" in df.columns and df["thread_id"].notna().any():
        groups = df["thread_id"].astype(str)
        gss = GroupShuffleSplit(n_splits=1, train_size=1.0 - test_size, random_state=seed)
        train_idx, test_idx = next(gss.split(df, groups=groups))
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df  = df.iloc[test_idx ].reset_index(drop=True)
    else:
        # Fallback random split
        rng = np.random.RandomState(seed)
        mask = rng.rand(len(df)) < (1.0 - test_size)
        train_df = df[mask].reset_index(drop=True)
        test_df  = df[~mask].reset_index(drop=True)

    # Keep only text and target in final datasets
    return train_df, test_df, target_col, label2id, id2label

def build_hf_datasets(train_df, test_df, target_col, tokenizer, max_length: int):
    # Hugging Face Datasets expect a 'label' column of ints
    train_hf = HFDataset.from_pandas(
        train_df[["message_text", target_col]].rename(columns={target_col: "label"}),
        preserve_index=False
    )
    test_hf = HFDataset.from_pandas(
        test_df[["message_text", target_col]].rename(columns={target_col: "label"}),
        preserve_index=False
    )

    def tok_fn(batch):
        return tokenizer(batch["message_text"], truncation=True, max_length=max_length)

    train_tok = train_hf.map(tok_fn, batched=True, remove_columns=train_hf.column_names)
    test_tok  = test_hf.map(tok_fn, batched=True, remove_columns=test_hf.column_names)

    return train_tok, test_tok

# --------------------- Training ---------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a Transformer on crush_reply_v4.csv")
    parser.add_argument("--csv", type=str, required=True, help="Path to crush_reply_v4.csv")
    parser.add_argument("--task", type=str, default="multiclass", choices=["multiclass","binary"])
    parser.add_argument("--model_name", type=str, default="vinai/phobert-base",
                        help="HuggingFace model name, e.g. vinai/phobert-base or xlm-roberta-base")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train_bs", type=int, default=16)
    parser.add_argument("--eval_bs", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="./tfm_outputs")
    parser.add_argument("--use_class_weights", action="store_true", help="Enable inverse-frequency class weights")
    args = parser.parse_args()

    set_seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load & split
    train_df, test_df, target_col, label2id, id2label = load_data(args.csv, args.task, args.test_size, args.seed)
    num_labels = len(label2id)

    # Tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    config = AutoConfig.from_pretrained(args.model_name, num_labels=num_labels,
                                        id2label=id2label, label2id=label2id)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=config)

    # Build datasets
    train_tok, test_tok = build_hf_datasets(train_df, test_df, target_col, tokenizer, args.max_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Optional class weights
    if args.use_class_weights:
        # train_tok['label'] is a list of ints
        cw = class_weights(np.array(train_tok["label"], dtype=int), num_labels)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cw = cw.to(device)

        # Patch model.forward to apply weighted CE loss
        import types
        import torch.nn as nn
        orig_forward = model.forward
        def forward_with_weights(self, *f_args, **f_kwargs):
            outputs = orig_forward(*f_args, **f_kwargs)
            labels = f_kwargs.get("labels", None)
            if labels is None and len(f_args) > 0 and isinstance(f_args[0], dict):
                labels = f_args[0].get("labels", None)
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss(weight=cw)
                loss = loss_fct(outputs.logits, labels)
                outputs.loss = loss
            return outputs
        model.forward = types.MethodType(forward_with_weights, model)

    # Training
    training_args = TrainingArguments(
        output_dir=str(outdir),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        learning_rate=args.lr,
        weight_decay=args.wd,
        warmup_ratio=args.warmup_ratio,
        logging_steps=50,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        seed=args.seed,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=test_tok,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()

    # Evaluate & persist
    eval_res = trainer.evaluate()
    preds = np.argmax(trainer.predict(test_tok).predictions, axis=-1)

    metrics = {
        "task": args.task,
        "model_name": args.model_name,
        "label2id": label2id,
        "id2label": id2label,
        "eval": eval_res,
        "classification_report": classification_report(
            test_tok["label"], preds,
            target_names=[id2label[i] for i in range(len(id2label))]
        )
    }
    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Save best model
    best_dir = outdir / "best_model"
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    print("=== DONE ===")
    print("Saved model to:", best_dir)
    print("Saved metrics to:", outdir / "metrics.json")

if __name__ == "__main__":
    main()
