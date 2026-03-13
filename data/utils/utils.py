from pathlib  import Path
import re
import numpy as np
import evaluate

# read conll data
def read_conll_data(file_path, delim=" -X- _ "):
    file_path = Path(file_path)
    raw_text = file_path.read_text().strip()
    raw_docs = re.split(r'\n\s*\n', raw_text)
    
    tokens_list = []
    tags_list = []
    
    for doc in raw_docs:
        tokens = []
        tags = []
        for line in doc.split('\n'):
            if line.strip():
                parts = line.strip().split(delim)
                if len(parts) >= 2:
                    token, tag = parts[0], parts[1]
                    tokens.append(token)
                    tags.append(tag)
        
        if tokens:
            tokens_list.append(tokens)
            tags_list.append(tags)
    
    return tokens_list, tags_list

def compute_metrics(p, io_mode=False):
    predictions, labels = p
    
    seqeval = evaluate.load("seqeval")
    
    true_predictions = []
    true_labels = []
    
    for prediction, label in zip(predictions, labels):
        true_pred = []
        true_lab = []
        for p, l in zip(prediction, label):
            if l != -100:  
                if io_mode:
                    p = "O" if p == "O" else "I"
                    l = "O" if l == "O" else "I"
                true_pred.append(p)
                true_lab.append(l)
        true_predictions.append(true_pred)
        true_labels.append(true_lab)
    
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


