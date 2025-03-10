import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSeq2SeqLM

from my_tokenizer import my_tokenizer
from train_test_split import train_test_split, combine_dataset
from train import preprocess_function 

# Read csv
dataset = pd.read_csv('fixed_dataset.csv')

dataset_list = combine_dataset(dataset)

# Divide dataset
train_df, val_df, test_df = train_test_split(dataset)

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# Convert to hugging face datset
df = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset
})

# Create my custom tokenizer
hf_tokenizer = my_tokenizer(dataset_list)

# test tokinizer

tokens = hf_tokenizer.encode("ꯃꯥꯏ ꯑꯃꯥ ꯀꯨꯠꯂꯣꯟ ꯁꯦꯟꯁꯦ")  # Returns list of token IDs
print(hf_tokenizer.convert_ids_to_tokens(tokens))  # ✅ Correct way to get tokenized words

# ✅ Apply tokenizer to dataset
df_source = df.map(preprocess_function, batched=True, fn_kwargs={'hf_tokenizer': hf_tokenizer})


# Training configuration
training_args = TrainingArguments(
    output_dir='/content',
    per_device_train_batch_size=2,
    num_train_epochs=2,
    remove_unused_columns=True
)

model_name = "facebook/mbart-large-50"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = df_source['train'],
    eval_dataset = df_source['test']
)

if __name__ == "__main__":
    model_out_path = "./model_out"
    trainer.train()
    eval_results = trainer.evaluate()
    print(eval_results)
    model.save_pretrained(model_out_path)