from pytorch_transformers import RobertaTokenizer as tokenization

text = 'Here is an example sentence'

tokenizer = tokenization.from_pretrained('roberta-large')
tokens = [tokenizer.tokenize(token) for token in text.split(' ')]
input_ids = tokenizer.convert_tokens_to_ids(tokens)

print(input_ids)