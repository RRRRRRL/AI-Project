# 3-Minute Presentation Outline

1) Problem & Motivation (20–30s)
- Task: Binary sentiment classification on IMDB.
- Why it matters.

2) Method (60–75s)
- Model: DistilBERT fine-tuning (Transformer).
- Data: Tokenization, max length 256, 90/10 val split.
- Training: AdamW, warmup, early stopping, metrics.

3) Results (45–60s)
- Test metrics (Accuracy, F1, ROC-AUC).
- Curves (loss/F1).
- Confusion matrix and one or two error examples.

4) Takeaways & Future Work (20–30s)
- What worked, what didn’t.
- Next steps (bigger model, longer context, RAG, interpretability).

5) Demo (optional quick 10–15s)
- Run inference on a short review (positive/negative).

Tips
- Keep slides visual: one plot per slide, minimal text.
- Practice to fit within 3 minutes.
