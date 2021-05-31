from transformers import BertTokenizer, BertModel
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using {} device'.format(device))

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained("bert-base-multilingual-cased")
text = "Lijst Calimero behield de vijf zetels en Studenten Organisatie Groningen (SOG)" \
       " leverde een zetel in."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)

print(output)
