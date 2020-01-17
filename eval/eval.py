# from https://github.com/mtanti/coco-caption/blob/master/pycocoevalcap/bleu/bleu_scorer.py
from bleu_scorer import BleuScorer

true_sentences = []
pred_sentences = []


f = open("eval/preds.txt", "r")
for pred in f:
    pred = pred.split("[SEP]", 1)[0]
    pred_sentences.append(pred)
    #print(pred)
f.close() 

g = open("eval/golds.txt", "r")
for true in g:
    true = true.split("[SEP]", 1)[0]
    true_sentences.append([true])
    #print(true)
g.close()


bleu_scorer = BleuScorer(n=4) # up to 4 gram
for true, pred in zip(true_sentences, pred_sentences):
    bleu_scorer += (pred, true)

scores, instance_scores = bleu_scorer.compute_score(option='closest', verbose=0)
print("BLEU1 score: ", scores[0])
print("BLEU4 score: ", scores[3])