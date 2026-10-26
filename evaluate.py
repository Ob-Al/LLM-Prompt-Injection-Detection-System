import argparse
import json

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from pidetect.classifier import InjectionClassifier
from pidetect.data import load_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="model")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--show-errors", type=int, default=10)
    args = ap.parse_args()

    clf = InjectionClassifier(args.model_dir, args.threshold)
    test = load_data(extra_csv=None)["test"]

    texts, labels = test["text"], test["label"]
    results = []
    for i in range(0, len(texts), 32):
        results += clf.predict_batch(texts[i : i + 32])

    preds = [int(r.is_injection) for r in results]
    scores = [r.score for r in results]

    print(classification_report(labels, preds, target_names=["benign", "injection"], digits=3))
    print("confusion matrix [[tn fp] [fn tp]]:", confusion_matrix(labels, preds).tolist())
    print("roc auc:", round(roc_auc_score(labels, scores), 4))

    print("\nthreshold sweep")
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        p = [int(s >= t) for s in scores]
        tp = sum(1 for a, b in zip(labels, p) if a == b == 1)
        fp = sum(1 for a, b in zip(labels, p) if a == 0 and b == 1)
        fn = sum(1 for a, b in zip(labels, p) if a == 1 and b == 0)
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        print(f"  t={t:.1f}  precision={prec:.3f}  recall={rec:.3f}  fp={fp}  fn={fn}")

    errors = [(t, l, r.score) for t, l, r in zip(texts, labels, results) if int(r.is_injection) != l]
    if errors and args.show_errors:
        print(f"\nmisclassified ({len(errors)} total)")
        for text, label, score in errors[: args.show_errors]:
            print(json.dumps({"label": label, "score": score, "text": text[:160]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
