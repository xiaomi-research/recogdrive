"""LingoQA: answers each val question from its five frames and scores the answers with Lingo-Judge
(a prediction counts as correct when the judge logit against any reference answer is positive)."""

import os

from recogdrive.models.recogdrive.backbone import system_message as SYSTEM_PROMPT  # noqa: F401  (read by the CLI)

FRAMES = 5
SUFFIX = "\nAnswer the question in a single short sentence."
MAX_NEW_TOKENS, MIN_NEW_TOKENS, BATCH_SIZE, SESSION_LEN = 512, 1, 16, 32000
KEYS = ["question_id", "segment_id"]


def load(args):
    import pandas as pd

    questions = pd.read_parquet(args.data)[KEYS + ["question"]].drop_duplicates(KEYS)
    return [
        {
            "id": f"{row.question_id}_{row.segment_id}",
            "question_id": row.question_id,
            "segment_id": row.segment_id,
            "question": row.question,
            "prompt": "<image>\n" * FRAMES + row.question + SUFFIX,
            "images": [os.path.join(args.image_root, row.segment_id, f"{i}.jpg") for i in range(FRAMES)],
        }
        for row in questions.itertuples(index=False)
    ]


def clean(text) -> str:
    return str(text).lower().strip()


def finish(args, samples, answers):
    import pandas as pd
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    predictions = pd.DataFrame(
        [{"question_id": s["question_id"], "segment_id": s["segment_id"], "question": s["question"],
          "answer": answers.get(s["id"])} for s in samples]
    )
    predictions.to_csv(args.out / "predictions.csv", index=False)

    references = pd.read_parquet(args.data).groupby(KEYS)["answer"].agg(list)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.judge, use_fast=True)
    judge = AutoModelForSequenceClassification.from_pretrained(args.judge).eval().to(device)
    correct = []
    with torch.inference_mode():
        for row in predictions.itertuples(index=False):
            key = (row.question_id, row.segment_id)
            if key not in references.index:
                continue
            texts = [f"{tokenizer.cls_token}\nQuestion: {row.question}\nAnswer: {clean(reference)}\nStudent: {clean(row.answer)}"
                     for reference in references[key]]
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            correct.append(bool(judge(**inputs).logits.squeeze(-1).max() > 0))
    return {"score": sum(correct) / max(len(correct), 1), "matched": len(correct), "questions": len(samples)}
