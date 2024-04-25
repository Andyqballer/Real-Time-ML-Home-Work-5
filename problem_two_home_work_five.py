import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import matplotlib.pyplot as plt
import itertools

# Dataset and Preprocessing
english_to_french = [
    ("They are students", "Ils sont étudiants"),
    ("The cat is sleeping", "Le chat dort"),
    ("The sun is shining", "Le soleil brille"),
    ("We love music", "Nous aimons la musique"),
    ("She speaks French fluently", "Elle parle français couramment"),
    ("He enjoys reading books", "Il aime lire des livres"),
    ("They play soccer every weekend", "Ils jouent au football chaque week-end"),
    ("The movie starts at 7 PM", "Le film commence à 19 heures"),
    ("She wears a red dress", "Elle porte une robe rouge"),
    ("We cook dinner together", "Nous cuisinons le dîner ensemble"),
    ("He drives a blue car", "Il conduit une voiture bleue"),
    ("They visit museums often", "Ils visitent souvent des musées"),
]

# Extract input and target texts
input_texts = [pair[0] for pair in english_to_french]
target_texts = [pair[1] for pair in english_to_french]

# Create character sets
input_chars = sorted(set("".join(input_texts))) + ['\n']
target_chars = sorted(set("".join(target_texts))) + ['\n']

# Create dictionaries
input_char_to_index = {char: i for i, char in enumerate(input_chars)}
target_char_to_index = {char: i for i, char in enumerate(target_chars)}

# Encode sequences
def encode_sequence(sequence, char_to_index, seq_length):
    return [char_to_index[char] for char in sequence] + [char_to_index["\n"]] + [0] * (seq_length - len(sequence))

# Define maximum sequence lengths
max_input_length = max(len(seq) for seq in input_texts)
max_target_length = max(len(seq) for seq in target_texts)

# Encode input and target sequences
input_data = [encode_sequence(seq, input_char_to_index, max_input_length) for seq in input_texts]
target_data = [encode_sequence(seq, target_char_to_index, max_target_length) for seq in target_texts]

# Convert to tensors
input_tensor = torch.tensor(input_data, dtype=torch.long)
target_tensor = torch.tensor(target_data, dtype=torch.long)

# Create DataLoader
dataset = TensorDataset(input_tensor, target_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, target_dim, d_model, nhead, num_layers):
        super(TransformerModel, self).__init__()
        self.embedding_src = nn.Embedding(input_dim, d_model)
        self.embedding_tgt = nn.Embedding(target_dim, d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers, 
                                          num_decoder_layers=num_layers)
        self.fc_out = nn.Linear(d_model, target_dim)
        
    def forward(self, src, tgt):
        src = self.embedding_src(src)
        tgt = self.embedding_tgt(tgt)
        
        output = self.transformer(src, tgt)
        output = self.fc_out(output)
        
        return output

# Hyperparameter Tuning Function
def hyperparameter_tuning(d_model_values, nhead_values, num_layers_values):
    results = []

    for d_model, nhead, num_layers in itertools.product(d_model_values, nhead_values, num_layers_values):
        print(f"Training with d_model={d_model}, nhead={nhead}, num_layers={num_layers}")

        model = TransformerModel(len(input_chars), len(target_chars), d_model=d_model, nhead=nhead, num_layers=num_layers)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        train_losses = []
        val_accuracies = []

        start_time = time.time()

        for epoch in range(5):  # Reduced epochs for faster hyperparameter tuning
            model.train()
            total_loss = 0

            for input_batch, target_batch in dataloader:
                optimizer.zero_grad()
                output = model(input_batch.transpose(0, 1), target_batch.transpose(0, 1)[:-1, :])
                
                loss = criterion(output.reshape(-1, output.shape[-1]), target_batch.transpose(0, 1)[1:, :].reshape(-1))
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()

            avg_loss = total_loss/len(dataloader)
            train_losses.append(avg_loss)
            
            # Evaluation
            model.eval()
            correct = 0

            for input_batch, target_batch in dataloader:
                output = model(input_batch.transpose(0, 1), target_batch.transpose(0, 1)[:-1, :])
                pred = torch.argmax(output, dim=2)
                
                correct += torch.sum(pred == target_batch.transpose(0, 1)[1:, :]).item()

            accuracy = correct / (len(dataloader) * 64 * max_target_length)
            val_accuracies.append(accuracy)

            end_time = time.time()

            print(f"  Epoch {epoch+1}/{5} - Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.4f}")

        results.append({
            'd_model': d_model,
            'nhead': nhead,
            'num_layers': num_layers,
            'train_loss': train_losses[-1],
            'val_accuracy': val_accuracies[-1],
            'execution_time': end_time - start_time
        })

    return results

# Hyperparameters
d_model_values = [256, 512]
nhead_values = [4, 8]
num_layers_values = [2, 3]

results = hyperparameter_tuning(d_model_values, nhead_values, num_layers_values)

# Display results
for result in results:
    print(f"\nHyperparameters: d_model={result['d_model']}, nhead={result['nhead']}, num_layers={result['num_layers']}")
    print(f"  Train Loss: {result['train_loss']:.4f}, Validation Accuracy: {result['val_accuracy']:.4f}")
    print(f"  Execution Time: {result['execution_time']:.2f} seconds")

# Plotting
plt.figure(figsize=(15, 5))

# Losses for different hyperparameters
plt.subplot(1, 2, 1)
for result in results:
    label = f"d_model={result['d_model']}, nhead={result['nhead']}, layers={result['num_layers']}"
    plt.plot(range(1, 6), [loss for loss in [result['train_loss'] for _ in range(5)]], label=f"Train {label}")
    plt.plot(range(1, 6), [acc for acc in [result['val_accuracy'] for _ in range(5)]], label=f"Val {label}", linestyle='--')

plt.xlabel('Epochs')
plt.ylabel('Loss / Accuracy')
plt.title('Hyperparameter Tuning Results')
plt.legend()

plt.tight_layout()
plt.show()
