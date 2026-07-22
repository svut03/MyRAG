from transformers import AutoTokenizer, AutoModel

transformer = AutoModel.from_pretrained("google-bert/bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
topk = 3


text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors="pt")
output = transformer(**encoded_input)

print(output)
