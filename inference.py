# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer_path = './tokenizer_out' # provide upto the model parent dir
model_path = './model_out' # provide upto the model parent dir

# Load model
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

def summarize(article, max_length, min_length):
    """Summarize the article by providing the article

    Args:
        article (str): article to be summarize

    Returns:
        str: return the summarize article
    """
    inputs = tokenizer(article, max_length = 1024, truncation=True, return_tensors='pt')
    
    ## Generate the summary
    summary_ids = model.generate(inputs['input_ids'], max_length = max_length, min_length = min_length, length_penalty= 2, num_beams = 4, early_stopping= True)

    ## Decode the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens = True)

    return summary

