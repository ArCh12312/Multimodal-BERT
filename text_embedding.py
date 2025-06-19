import torch
from transformers import BertTokenizer, BertModel

class BERTFeatureExtractor:
    def __init__(self, model_name='bert-base-uncased'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pretrained tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
    
    def extract_features(self, texts):

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=125,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        with torch.no_grad():
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, 768)
            
            return last_hidden_state.cpu()  # (batch_size, seq_len, hidden_size)

def main():
    texts = [
        "This is the first example sentence.",
        "Here is another sentence to encode."
    ]
    
    extractor = BERTFeatureExtractor()
    features = extractor.extract_features(texts)
    print("Extracted features shape:", features.shape)  # Should be (2, 768)
    print(features)

if __name__ == "__main__":
    main()