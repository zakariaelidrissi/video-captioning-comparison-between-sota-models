import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')
nltk.download('wordnet')

def compute_bleu_at_n(references, hypotheses, n=4):
    """
        Compute BLEU@n scores for n=1,2,3,4
        
        Args:
            references: List of lists of reference captions
            hypotheses: List of generated captions
            n: Maximum n-gram level to evaluate (default: 4)
        
        Returns:
            Dictionary of BLEU scores at different n-gram levels
    """
    smoother = SmoothingFunction().method3
    bleu_scores = {}

    refs = [[ref.split() for ref in refs] for refs in references]
    hyps = [hyp.split() for hyp in hypotheses]
    
    if len(refs) != len(hyps):
        print(f"Warning: Mismatch in length. References: {len(refs)}, Hypotheses: {len(hyps)}")
    
    for i in range(1, n+1):
        weights = tuple([1.0/i if j < i else 0 for j in range(4)])
        
        score = 0
        for j in range(len(hyps)):
            score += sentence_bleu(
                refs[j], 
                hyps[j],
                weights=weights,
                smoothing_function=smoother
            )
        bleu_scores[f'BLEU@{i}'] = score / len(hyps) if hyps else 0
    return bleu_scores

def compute_meteor(references, hypotheses):
    scores = [meteor_score([word_tokenize(ref) for ref in references[vid]], word_tokenize(hypotheses[vid])) for vid in range(len(hypotheses))]
    return sum(scores) / len(scores) if scores else 0

def compute_rouge(references, hypotheses):
    from rouge_score import rouge_scorer
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for i in range(len(references)):
        if isinstance(references[i], list):
            ref = references[i][0]
        else:
            ref = references[i]
            
        hyp = hypotheses[i]
        score = scorer.score(ref, hyp)
        
        for metric in scores:
            scores[metric].append(score[metric].fmeasure)

    avg_scores = {metric: sum(values) / len(values) if values else 0 
                  for metric, values in scores.items()}
    return avg_scores

def compute_cider(references, hypotheses):
    refs = {}
    hyps = {}
    
    for i in range(len(references)):
        if isinstance(references[i], list):
            refs[i] = [" ".join(ref) if isinstance(ref, list) else ref for ref in references[i]]
        else:
            refs[i] = [references[i]]
        
        if isinstance(hypotheses[i], list):
            hyps[i] = hypotheses[i][0] if hypotheses[i] else ""
        else:
            hyps[i] = hypotheses[i]
    
    formatted_hyps = {k: [v] for k, v in hyps.items()}
    cider_scorer = Cider()
    score, _ = cider_scorer.compute_score(refs, formatted_hyps)

    return score

# Metric Functions To Evaluate Video Captioning Models
def bleu_metric(model, ground_truth, generated_captions):
    scores = []
    bleu_scores = compute_bleu_at_n(ground_truth, generated_captions)
    for metric, score in bleu_scores.items():
        print(f"{metric} Score: {score}")
        scores.append({
            'model': model,
            'Metric_name': metric,
            'Score': float(score)
        })

    return scores

def meteor_metric(model, ground_truth, generated_captions):
    scores = []    
    meteor_val = compute_meteor(ground_truth, generated_captions)
    print("meteor Score:", meteor_val)
    scores.append({
        'model': model,
        'Metric_name': "Meteor",
        'Score': float(meteor_val)
    })

    return scores

def rouge_metric(model, ground_truth, generated_captions):
    scores = []
    rouge_val = compute_rouge(ground_truth, generated_captions)
    print("Rouge Score:", rouge_val)
    for metric, score in rouge_val.items():
        scores.append({
            'model': model,
            'Metric_name': metric,
            'Score': float(score)
        })
    
    return scores

def cider_metric(model, ground_truth, generated_captions):
    scores = []    
    cider_val = compute_cider(ground_truth, generated_captions)
    print("Cider Score:", cider_val)
    scores.append({
        'model': model,
        'Metric_name': "Cider",
        'Score': float(cider_val.item())
    })

    return scores
