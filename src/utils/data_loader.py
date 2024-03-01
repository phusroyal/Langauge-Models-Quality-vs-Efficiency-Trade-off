from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding


class MyDataloader():
    def __init__(self, model_name) -> None:
        self.raw_data = load_dataset('AmazonScience/massive', 'en-US') ## Considering only the English dataset ##
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.tokenized_datasets = self.raw_data.map(self.tokenize_function, batched=True)
        self.tokenized_datasets = self.tokenized_datasets.remove_columns(['id', 'locale', 'partition','scenario','annot_utt', 'utt', 'worker_id', 'slot_method', 'judgments']) ## removing unwanted columns ##
        self.tokenized_datasets = self.tokenized_datasets.rename_column("intent", "labels")
        self.tokenized_datasets.set_format("torch")

    def tokenize_function(self, examples):
        # Tokenize the examples here using the desired method
        inputs = self.tokenizer(examples["utt"], truncation=True, padding="max_length", max_length=128, return_tensors="pt")
        return inputs

    def tokenize_data(self):      
        return self.tokenized_datasets
    
    def label_id_converter(self):
        labels = self.tokenized_datasets["train"].features["labels"].names
        num_labels = len(labels)
        label2id, id2label = dict(), dict()
        for i, label in enumerate(labels):
            label2id[label] = str(i)
            id2label[str(i)] = label
        
        return label2id, id2label, num_labels
    
    def _tokenizer(self):
        return self.tokenizer
    
    def data_collator(self):
        return DataCollatorWithPadding(tokenizer=self.tokenizer)

    def label_names(self):
        return self.raw_data['train'].features['intent'].names
    


    
