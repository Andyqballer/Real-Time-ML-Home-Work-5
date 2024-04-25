import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import numpy as np

# Dataset
sequence = """Next character prediction is a fundamental task in the field of natural language processing (NLP) ...
... possibilities for the future of text-based technology."""

# Tokenization
unique_chars = sorted(list(set(sequence)))
char_to_index = {ch: i for i, ch in enumerate(unique_chars)}
index_to_char = {i: ch for i, ch in enumerate(unique_chars)}
input_size = output_size = len(unique_chars)

def encode_sequence(seq):
    return [char_to_index[char] for char in seq]

def create_training_data(sequence, seq_length):
    sequences = []
    next_chars = []
    for i in range(len(sequence) - seq_length):
        sequences.append(encode_sequence(sequence[i: i + seq_length]))
        next_chars.append(encode_sequence(sequence[i + 1: i + seq_length + 1]))
    return torch.tensor(sequences, dtype=torch.long), torch.tensor(next_chars, dtype=torch.long)

# Preparing sequences
sequences_10, next_chars_10 = create_training_data(sequence, 10)
sequences_20, next_chars_20 = create_training_data(sequence, 20)
sequences_30, next_chars_30 = create_training_data(sequence, 30)

# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.Transformer(embed_dim, num_heads, num_layers, batch_first=True)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x, x)
        x = self.fc(x)
        return x

def train_model(model, train_data, next_chars, num_epochs=50, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        epoch_loss = 0.0  # Initialize epoch loss
        
        # Mini-batch training
        for i in range(0, len(train_data), batch_size):
            inputs = train_data[i:i+batch_size].to(device)
            targets = next_chars[i:i+batch_size].view(-1).to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, input_size), targets)
            epoch_loss += loss.item() * inputs.size(0)  # Accumulate loss for the epoch
            loss.backward()
            optimizer.step()
        
        avg_epoch_loss = epoch_loss / len(train_data)
        train_losses.append(avg_epoch_loss)

        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}")
    
    end_time = time.time()
    execution_time = end_time - start_time
    return train_losses, execution_time

# Training
transformer_model = TransformerModel(input_size, 128, 4, 256, 2)

train_losses_10, execution_time_10 = train_model(transformer_model, sequences_10, next_chars_10)
train_losses_20, execution_time_20 = train_model(transformer_model, sequences_20, next_chars_20)
train_losses_30, execution_time_30 = train_model(transformer_model, sequences_30, next_chars_30)

# Plotting Losses
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(train_losses_10)
plt.title("Sequence Length 10 Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")

plt.subplot(1, 3, 2)
plt.plot(train_losses_20)
plt.title("Sequence Length 20 Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")

plt.subplot(1, 3, 3)
plt.plot(train_losses_30)
plt.title("Sequence Length 30 Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")

plt.show()

print(f"Execution Time (Sequence Length 10): {execution_time_10:.2f} seconds")
print(f"Execution Time (Sequence Length 20): {execution_time_20:.2f} seconds")
print(f"Execution Time (Sequence Length 30): {execution_time_30:.2f} seconds")

# Report and Comparison
print("\nReport and Comparison:")
print("For Transformer Model:")
print(f"Training Loss for Sequence Length 10: {train_losses_10[-1]:.4f}")
print(f"Training Loss for Sequence Length 20: {train_losses_20[-1]:.4f}")
print(f"Training Loss for Sequence Length 30: {train_losses_30[-1]:.4f}")

# RNN-based Approach (For comparison, let's use LSTM)
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

# Training LSTM
lstm_model = LSTMModel(input_size, 128, 256, 2)
train_losses_lstm, execution_time_lstm = train_model(lstm_model, sequences_10, next_chars_10)

print("\nFor LSTM Model:")
print(f"Training Loss for Sequence Length 10: {train_losses_lstm[-1]:.4f}")

# Model size and computational complexity
num_params_transformer = sum(p.numel() for p in transformer_model.parameters())
num_params_lstm = sum(p.numel() for p in lstm_model.parameters())

print("\nModel Sizes:")
print(f"Transformer Model: {num_params_transformer} parameters")
print(f"LSTM Model: {num_params_lstm} parameters")
