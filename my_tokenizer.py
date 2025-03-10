from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
from transformers import PreTrainedTokenizerFast


def my_tokenizer(dataset_list):
    """_summary_

    Args:
        dataset_list (list): list of all the text

    Returns:
        tokenizer: custom tokenizer
    """
    # Step 1: Create a Unigram Tokenizer
    tokenizer = Tokenizer(models.Unigram())

    # Step 2: Normalize Text
    tokenizer.normalizer = normalizers.NFKC()

    # Step 3: Use Pre-tokenizer
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()


    trainer = trainers.UnigramTrainer(
        vocab_size=10000,
        special_tokens=["<s>", "</s>", "<unk>", "<pad>", "<mask>"]
    )

    tokenizer.train_from_iterator(dataset_list, trainer)

    # ✅ Step 5: Convert to a Hugging Face Tokenizer the Correct Way
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, 
        unk_token="<unk>", 
        pad_token="<pad>", 
        mask_token="<mask>",
        bos_token="<s>",
        eos_token="</s>"
    )

    # Step 6: Save the Tokenizer
    # hf_tokenizer.save_pretrained("/content/model_directory")

    return hf_tokenizer