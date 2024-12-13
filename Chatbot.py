import torch
import torch.nn as nn
import torch.optim as optim
from KalaTorch.datasets import create_data_loader
from KalaTorch.training import KaloTrainer
from KalaTorch.models.chatbotrnn import ChatbotRNN
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

# Step 1: Define Dataset
data = [
    {"intent": "greeting", "patterns": ["Hello", "Hi", "Hey there"], "responses": ["Hi!", "Hello!", "Hey!"]},
    {"intent": "goodbye", "patterns": ["Bye", "See you later", "Goodbye"], "responses": ["Goodbye!", "See you!", "Take care!"]},
    {"intent": "thanks", "patterns": ["Thanks", "Thank you", "I appreciate it"], "responses": ["You're welcome!", "No problem!", "Glad to help!"]},
]

# Preprocessing - Tokenize and Encode
all_patterns = []
all_labels = []
responses = {}

for intent in data:
    responses[intent["intent"]] = intent["responses"]
    for pattern in intent["patterns"]:
        all_patterns.append(pattern)
        all_labels.append(intent["intent"])

# Tokenization
all_tokens = [word_tokenize(pattern.lower()) for pattern in all_patterns]
vocab = list(set([word for tokens in all_tokens for word in tokens]))
word2idx = {word: idx for idx, word in enumerate(vocab)}

# Encode patterns
encoded_patterns = [[word2idx[word] for word in tokens] for tokens in all_tokens]
max_length = max(len(pattern) for pattern in encoded_patterns)
padded_patterns = [np.pad(pattern, (0, max_length - len(pattern))) for pattern in encoded_patterns]

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(all_labels)

# Step 2: Prepare Dataset
X = torch.tensor(padded_patterns, dtype=torch.long)
y = torch.tensor(encoded_labels, dtype=torch.long)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

train_loader = create_data_loader(X_train, y_train, batch_size=8)
val_loader = create_data_loader(X_val, y_val, batch_size=8)



vocab_size = len(vocab)
embed_size = 50
hidden_size = 128
output_size = len(label_encoder.classes_)
model = ChatbotRNN(vocab_size, embed_size, hidden_size, output_size)

# Step 4: Train the Model
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
trainer = KaloTrainer(model, optimizer, criterion)
trainer.train(train_loader, epochs=100)
trainer.evaluate(val_loader)

# Step 5: Chatbot Inference
def predict_intent(sentence):
    tokens = word_tokenize(sentence.lower())
    encoded = [word2idx.get(word, 0) for word in tokens]
    padded = np.pad(encoded, (0, max_length - len(encoded)))
    input_tensor = torch.tensor([padded], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        intent = label_encoder.inverse_transform(predicted.numpy())[0]
        return intent

def chatbot_response(sentence):
    intent = predict_intent(sentence)
    return np.random.choice(responses[intent])

# Chatbot Loop
print("Chatbot is ready! Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break
    response = chatbot_response(user_input)
    print(f"Bot: {response}")
