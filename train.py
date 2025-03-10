def preprocess_function(batch, hf_tokenizer):
    source = batch["article"]
    target = batch["summary"]

    # ✅ Use hf_tokenizer, which supports direct calling
    source_ids = hf_tokenizer(source, truncation=True, padding="max_length", max_length=128)
    target_ids = hf_tokenizer(target, truncation=True, padding="max_length", max_length=128)

    # Replace pad token IDs with -100 for loss masking
    labels = target_ids["input_ids"]
    labels = [[(label if label != hf_tokenizer.pad_token_id else -100) for label in labels_example] for labels_example in labels]

    return {
        "input_ids": source_ids["input_ids"],
        "attention_mask": source_ids["attention_mask"],
        "labels": labels
    }

