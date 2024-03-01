from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification, EarlyStoppingCallback
from huggingface_hub import notebook_login, HfFolder, HfApi
from collections import Counter
import numpy as np

import argparse, torch, sys

from pathlib import Path
path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)

from utils.misc import seed_everything, compute_metrics_trainer, compute_metrics
from utils.data_loader import MyDataloader

from sklearn.metrics import classification_report

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_data(args) -> dict:
    """get data
    return:
        a dict of tokenized_datasets, data_collator, label2id, id2label, num_labels"""
    data_dict = {}
    my_data = MyDataloader(args.model)
    label2id, id2label, num_labels = my_data.label_id_converter()

    data_dict['tokenized_datasets'] = my_data.tokenize_data()
    data_dict['data_collator'] = my_data.data_collator()
    data_dict['tokenizer'] = my_data._tokenizer()
    data_dict['label2id'] = label2id
    data_dict['id2label'] = id2label
    data_dict['num_labels'] = num_labels
    

    return data_dict

def get_train_args(args, repo_name):
    return TrainingArguments(
                    output_dir=repo_name,
                    num_train_epochs=args.nepoch, ## Epochs
                    per_device_train_batch_size=args.bs,
                    per_device_eval_batch_size=args.bs,
                    fp16=True,
                    learning_rate=args.lr,
                    seed=args.seed,
                    # logging & evaluation strategies #
                    logging_dir=f"{repo_name}/logs",
                    logging_strategy="epoch",
                    evaluation_strategy="epoch",
                    save_strategy="epoch",
                    save_total_limit=2,
                    load_best_model_at_end=True,
                    metric_for_best_model="accuracy",
                    report_to="tensorboard",
                    # push to hub parameters #
                    push_to_hub=True,
                    hub_strategy="every_save",
                    hub_model_id=repo_name,
                    hub_token=HfFolder.get_token(),
                    )


def main():
    parser = argparse.ArgumentParser()

    # initialize
    parser.add_argument("--seed", type=int, default=46, help="seed to preproduce")

    # model
    parser.add_argument("--model", type=str, default="bert-base-cased", help="model's name")
    parser.add_argument("--nepoch", type=int, default=1, help="number of epochs")
    parser.add_argument("--bs", type=int, default=16, help="batch size")
    parser.add_argument("--lr", type=float, default=5e-6, help="learning rate")
    parser.add_argument("--es", type=int, default=3, help="early stopping")

    args = parser.parse_args()

    # seed everything
    seed_everything(args.seed)

    # save log
    args_dict = vars(args)
    log_name = ''
    for key, value in args_dict.items():
        n = str(key) +'_'+ str(value)+'-'
        log_name += n
    log_name = log_name[:-1]
    print('log_name: ', log_name)

    # model name to push to HuggingFace
    repo_name = 'phusroyal/'+args.model+'-massive_intent'
    print('repo_name: ', repo_name)
    
    # data loader
    print('load data...')
    data_dict = get_data(args)

    # training args
    train_args = get_train_args(args, repo_name)

    # get model
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=data_dict['num_labels'])
    # model = torch.compile(model) # using .compile from torch ver 2 to speed up training

    # trainer
    trainer = Trainer(
                    model,
                    train_args,
                    train_dataset=data_dict['tokenized_datasets']["train"],
                    eval_dataset=data_dict['tokenized_datasets']["validation"],
                    data_collator=data_dict['data_collator'],
                    tokenizer=data_dict['tokenizer'],
                    compute_metrics=compute_metrics_trainer,
                    callbacks = [EarlyStoppingCallback(early_stopping_patience = args.es)], ## For early stopping (patience = 3) ##
                )

    trainer.train()
    print('####EVALUATE####\n')
    print(trainer.evaluate())
    print('####TEST####\n')
    test_result = trainer.predict(data_dict['tokenized_datasets']["test"])
    print('\nTest results : \n\n', test_result.metrics)   
    
    predicted_values = np.argmax(test_result.predictions, axis=1)
    actual_values = test_result.label_ids
    metrics_dict = compute_metrics(pred=predicted_values, true=actual_values)

    labels = list(map(int, list(data_dict['id2label'].keys())))
    target_names = list(data_dict['label2id'].keys())
    clf_report = classification_report(actual_values, predicted_values, labels= labels, target_names= target_names)

    # write to file
    log = [metrics_dict, clf_report]
    with open(f'src/baseline/log/{log_name}.txt', 'w') as f:
        f.write(str(log))


    # save best model, metrics and create model card #
    trainer.create_model_card(model_name=train_args.hub_model_id)
    trainer.push_to_hub()

    ## Link for the model webpage ##
    whoami = HfApi().whoami()
    username = whoami['name']

    print(f"Model webpage link: https://huggingface.co/{repo_name}")

if __name__ == '__main__':
    main()