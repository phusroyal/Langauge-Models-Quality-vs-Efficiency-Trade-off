import os
import random
from typing import Iterable

import numpy as np
import torch, evaluate


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def limit_gpus(gpu_ids: Iterable[int]) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

def compute_metrics_trainer(eval_pred):
    # logits, labels = eval_pred
    # metric_acc = evaluate.load("accuracy")
    # metric_f1 = evaluate.load("f1")
    # predictions = np.argmax(logits, axis=-1)
    # return {'accuracy' : metric_acc.compute(predictions=predictions, references=labels), 
    #         'f1-macro' : metric_f1.compute(predictions=predictions, references=labels, average= 'macro')}
    metric_acc = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric_acc.compute(predictions=predictions, references=labels)

def compute_metrics(pred, true):
    metric_acc = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")
    return {'accuracy' : metric_acc.compute(predictions=pred, references=true), 
            'f1-macro' : metric_f1.compute(predictions=pred, references=true, average= 'macro')}
