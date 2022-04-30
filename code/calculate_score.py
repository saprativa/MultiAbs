import rouge
import json
import os
import re

import nltk
nltk.download('punkt')

def prepare_results(p, r, f):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)


aggregator = 'Avg'
print('Evaluation with {}'.format(aggregator))
apply_avg = aggregator == 'Avg'
apply_best = aggregator == 'Best'

evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                        max_n=2,
                        limit_length=True,
                        length_limit=100,
                        length_limit_type='words',
                        apply_avg=apply_avg,
                        apply_best=apply_best,
                        alpha=0.5, # Default F1_score
                        weight_factor=1.2,
                        stemming=True)

def process(line):
    line = re.sub(r"#Person(\d+)#", r"Person\1", line)
    return line

with open("/home/tanik_1821cs08/sb/MultiAbs/data/ds/dialogsum.test.jsonl", 'r') as f: 
    golds = f.readlines()
golds = [json.loads(g) for g in golds]


all_hypothesis, all_references = [], []
for d in golds:
    all_references.append([process(d[f"summary{i}"]) for i in range(1, 4)])


with open("/tmp/sb-summarization1/generated_predictions.txt", 'r') as f:
    all_hypothesis = f.readlines()
all_hypothesis = [process(pred.strip()) for pred in all_hypothesis]

scores = evaluator.get_scores(all_hypothesis, all_references)
for metric, results in sorted(scores.items(), key=lambda x: x[0]):
    if not apply_avg and not apply_best: # value is a type of list as we evaluate each summary vs each reference
        for hypothesis_id, results_per_ref in enumerate(results):
            nb_references = len(results_per_ref['p'])
            for reference_id in range(nb_references):
                print('\tHypothesis #{} & Reference #{}: '.format(hypothesis_id, reference_id))
                print('\t' + prepare_results(results_per_ref['p'][reference_id], results_per_ref['r'][reference_id], results_per_ref['f'][reference_id]))
        print()
    else:
        print(prepare_results(results['p'], results['r'], results['f']))
print()