# Fine-tune a pre-trained model on a data/train_queries.csv which contains code snippets and their corresponding queries.
import pandas as pd
import argparse
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments


def load_data(queries_file, code_snippets_file):
    queries_df = pd.read_csv(queries_file)
    code_snippets_df = pd.read_csv(code_snippets_file,engine='python')
    return queries_df, code_snippets_df


def preprocess_data(queries_df, code_snippets_df, tokenizer):
    inputs = []
    labels = []
    for _, row in queries_df.iterrows():
        query = row['query']
        code_id = row['code_id']
        code_snippet = code_snippets_df.loc[code_snippets_df['id'] == code_id, 'code'].values[0]
        encoded_input = tokenizer(query, code_snippet, truncation=True, padding='max_length', max_length=512)
        inputs.append(encoded_input)
        labels.append(1)  # Assuming all pairs in training data are positive examples
    return inputs, labels

def main(args):
    # Load data
    queries_df, code_snippets_df = load_data(args.queries_file, args.code_snippets_file)

    # Load tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    model = RobertaForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    # Preprocess data
    inputs, labels = preprocess_data(queries_df, code_snippets_df, tokenizer)

    # Convert to torch dataset
    class CodeQueryDataset(torch.utils.data.Dataset):
        def __init__(self, inputs, labels):
            self.inputs = inputs
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.inputs[idx].items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

    dataset = CodeQueryDataset(inputs, labels)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        save_steps=10_000,
        save_total_limit=2,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    # Train model
    trainer.train()

    # Save model
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a pre-trained model on code-query pairs.")
    parser.add_argument("--queries_file", type=str, required=True, help="Path to the CSV file containing queries.")
    parser.add_argument("--code_snippets_file", type=str, required=True, help="Path to the CSV file containing code snippets.")
    parser.add_argument("--model_name", type=str, default="microsoft/codebert-base", help="Pre-trained model name or path.")
    parser.add_argument("--output_dir", type=str, default="./fine_tuned_model", help="Directory to save the fine-tuned model.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size.")
    args = parser.parse_args()

    main(args)