import numpy as np

from evaluate import load
from bert_score import BERTScorer
from nltk.translate.bleu_score import corpus_bleu


def compute_bleu(references, hypothesis):
    # https://stackoverflow.com/questions/73938010/which-bleu-smoothing-function-is-commonly-used-for-image-captioning-evaluation
    references = [[reference.split()] for reference in references]
    hypothesis = [hypothesis.split() for hypothesis in hypothesis]
    bleu_scores = corpus_bleu(references, hypothesis)
    return np.mean(bleu_scores)


def compute_meteor(reference, hypothesis):
    metric = load('meteor')
    results = metric.compute(predictions=hypothesis, references=reference)
    return results['meteor']


def compute_rouge(reference, hypothesis):
    metric = load('rouge')
    results = metric.compute(predictions=hypothesis, references=reference, use_stemmer=False)
    return results


def compute_bertscore(reference, hypothesis):
    scorer = BERTScorer(lang='pt', rescale_with_baseline=True)
    precision, recall, f1_score = scorer.score(hypothesis, reference, verbose=True)
    return {
        'precision': precision.mean().item(),
        'recall': recall.mean().item(),
        'f1_score': f1_score.mean().item()
    }


def compute_eval_metrics(all_references: list, all_predictions: list) -> dict:
    bleu_score = compute_bleu(all_references, all_predictions)
    meteor_score = compute_meteor(all_references, all_predictions)
    rouge_scores = compute_rouge(all_references, all_predictions)
    bert_scores = compute_bertscore(all_references, all_predictions)
    return {
        'rouge1': rouge_scores['rouge1'],
        'rouge2': rouge_scores['rouge2'],
        'rougeL': rouge_scores['rougeL'],
        'bert_score_f1': bert_scores['f1_score'],
        'bleu': bleu_score,
        'meteor': meteor_score
    }
