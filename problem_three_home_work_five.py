import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Data
english_to_french = [
    ("I am cold", "J'ai froid"),
    ("You are tired", "Tu es fatigué"),
    # ... (other sentences)
    ("He sings in the choir", "Il chante dans le chœur")
]

# Hyperparameters
EMBED_SIZE = 512
NUM_HEADS = 8
NUM_LAYERS = 3
DROPOUT = 0.1
LEARNING_RATE = 0.0005
BATCH_SIZE = 8
NUM_EPOCHS = 100

# Data preparation
def tokenize(text, lang='en'):
    return text.lower().split()

en_sentences = [pair[0] for pair in english_to_french]
fr_sentences = [pair[1] for pair in english_to_french]

en_vocab = set(token for sentence in en_sentences for token in tokenize(sentence, 'en'))
fr_vocab = set(token for sentence in fr_sentences for token in tokenize(sentence, 'fr'))

en_word2idx = {word: idx + 4 for idx, word in enumerate(en_vocab)}
en_word2idx['<pad>'] = 0
en_word2idx['<start>'] = 1
en_word2idx['<end>'] = 2
en_word2idx['<unk>'] = 3

fr_word2idx = {word: idx + 4 for idx, word in enumerate(fr_vocab)}
fr_word2idx['<pad>'] = 0
fr_word2idx['<start>'] = 1
fr_word2idx['<end>'] = 2
fr_word2idx['<unk>'] = 3

en_idx2word = {idx: word for word, idx in en_word2idx.items()}
fr_idx2word = {idx: word for word, idx in fr_word2idx.items()}

en_max_len = max(len(tokenize(sentence, 'en')) for sentence in en_sentences)
fr_max_len = max(len(tokenize(sentence, 'fr')) for sentence in fr_sentences)

# ... [Encoder, Decoder, PositionalEncoding, EncoderLayer, DecoderLayer, MultiHeadAttention, FeedForward classes]


# Encoder
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, dropout):
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pe = PositionalEncoding(embed_size, dropout)
        self.layers = nn.ModuleList([EncoderLayer(embed_size, heads, dropout) for _ in range(num_layers)])

    def forward(self, x):
        x = self.embed(x)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x)
        return x

# Decoder
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, dropout):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pe = PositionalEncoding(embed_size, dropout)
        self.layers = nn.ModuleList([DecoderLayer(embed_size, heads, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x, enc_out):
        x = self.embed(x)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, enc_out)
        x = self.fc_out(x)
        return x

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, embed_size, heads, dropout):
        super(EncoderLayer, self).__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.attn = MultiHeadAttention(embed_size, heads)
        self.ff = FeedForward(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x2 = self.attn(x, x, x)
        x = self.norm(x + self.dropout(x2))
        x2 = self.ff(x)
        x = self.norm(x + self.dropout(x2))
        return x

# Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self, embed_size, heads, dropout):
        super(DecoderLayer, self).__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.attn = MultiHeadAttention(embed_size, heads)
        self.attn_enc_dec = MultiHeadAttention(embed_size, heads)
        self.ff = FeedForward(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out):
        x2 = self.attn(x, x, x)
        x = self.norm(x + self.dropout(x2))
        x2 = self.attn_enc_dec(x, enc_out, enc_out)
        x = self.norm(x + self.dropout(x2))
        x2 = self.ff(x)
        x = self.norm(x + self.dropout(x2))
        return x

# Multi-head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.heads = heads
        self.d_k = embed_size // heads
        self.linear_layers = nn.ModuleList([nn.Linear(embed_size, embed_size) for _ in range(3)])
        self.output_linear = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, query, key, value):
        query, key, value = [l(x).view(x.size(0), -1, self.heads, self.d_k).transpose(1, 2) for l, x in zip(self.linear_layers, (query, key, value))]
        x = self.scaled_dot_product_attention(query, key, value)
        x = x.transpose(1, 2).contiguous().view(x.size(0), -1, self.heads * self.d_k)
        return self.output_linear(x)

    def scaled_dot_product_attention(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.d_k**0.5
        scores = nn.functional.softmax(scores, dim=-1)
        return torch.matmul(scores, value)

# Feed Forward
class FeedForward(nn.Module):
    def __init__(self, embed_size, d_ff=2048):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, d_ff)
        self.fc2 = nn.Linear(d_ff, embed_size)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# DataLoader
class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, en_sentences, fr_sentences, en_word2idx, fr_word2idx, en_max_len, fr_max_len):
        self.en_sentences = en_sentences
        self.fr_sentences = fr_sentences
        self.en_word2idx = en_word2idx
        self.fr_word2idx = fr_word2idx
        self.en_max_len = en_max_len
        self.fr_max_len = fr_max_len

    def __len__(self):
        return len(self.en_sentences)

    def __getitem__(self, idx):
        en_sentence = self.en_sentences[idx]
        fr_sentence = self.fr_sentences[idx]
        en_tokens = [en_word2idx.get(token, en_word2idx['<unk>']) for token in tokenize(en_sentence, 'en')]
        fr_tokens = [fr_word2idx.get(token, fr_word2idx['<unk>']) for token in tokenize(fr_sentence, 'fr')]
        en_pad_len = self.en_max_len - len(en_tokens)
        fr_pad_len = self.fr_max_len - len(fr_tokens)
        en_tokens.extend([en_word2idx['<pad>']] * en_pad_len)
        fr_tokens.extend([fr_word2idx['<pad>']] * fr_pad_len)
        return {
            'en_tokens': torch.tensor(en_tokens, dtype=torch.long),
            'fr_tokens': torch.tensor(fr_tokens, dtype=torch.long)
        }


# DataLoader
class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, en_sentences, fr_sentences, en_word2idx, fr_word2idx, en_max_len, fr_max_len):
        self.en_sentences = en_sentences
        self.fr_sentences = fr_sentences
        self.en_word2idx = en_word2idx
        self.fr_word2idx = fr_word2idx
        self.en_max_len = en_max_len
        self.fr_max_len = fr_max_len

    def __len__(self):
        return len(self.en_sentences)

    def __getitem__(self, idx):
        en_sentence = self.en_sentences[idx]
        fr_sentence = self.fr_sentences[idx]
        en_tokens = [en_word2idx.get(token, en_word2idx['<unk>']) for token in tokenize(en_sentence, 'en')]
        fr_tokens = [fr_word2idx.get(token, fr_word2idx['<unk>']) for token in tokenize(fr_sentence, 'fr')]
        en_pad_len = self.en_max_len - len(en_tokens)
        fr_pad_len = self.fr_max_len - len(fr_tokens)
        en_tokens.extend([en_word2idx['<pad>']] * en_pad_len)
        fr_tokens.extend([fr_word2idx['<pad>']] * fr_pad_len)
        return {
            'en_tokens': torch.tensor(en_tokens, dtype=torch.long),
            'fr_tokens': torch.tensor(fr_tokens, dtype=torch.long)
        }

dataset = TranslationDataset(en_sentences, fr_sentences, en_word2idx, fr_word2idx, en_max_len, fr_max_len)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize model, loss, optimizer
encoder = Encoder(len(en_word2idx), EMBED_SIZE, NUM_LAYERS, NUM_HEADS, DROPOUT)
decoder = Decoder(len(fr_word2idx), EMBED_SIZE, NUM_LAYERS, NUM_HEADS, DROPOUT)
criterion = nn.CrossEntropyLoss(ignore_index=fr_word2idx['<pad>'])
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LEARNING_RATE)

# Lists to store losses
train_losses = []
val_losses = []

# Training
encoder.train()
decoder.train()
for epoch in range(NUM_EPOCHS):
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        en_tokens = batch['en_tokens']
        fr_tokens = batch['fr_tokens']
        
        enc_out = encoder(en_tokens)
        
        output = decoder(fr_tokens[:, :-1], enc_out)
        output = output.contiguous().view(-1, output.shape[-1])
        fr_tokens = fr_tokens[:, 1:].contiguous().view(-1)
        
        loss = criterion(output, fr_tokens)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    epoch_loss = total_loss / len(dataloader)
    train_losses.append(epoch_loss)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss}")

    # Evaluation
    encoder.eval()
    decoder.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            en_tokens = batch['en_tokens']
            fr_tokens = batch['fr_tokens']
            
            enc_out = encoder(en_tokens)
            
            output = decoder(fr_tokens[:, :-1], enc_out)
            output = output.contiguous().view(-1, output.shape[-1])
            fr_tokens = fr_tokens[:, 1:].contiguous().view(-1)
            
            loss = criterion(output, fr_tokens)
            total_loss += loss.item()
            
            _, predicted = output.max(1)
            correct = (predicted == fr_tokens).sum().item()
            total_correct += correct
            total_samples += fr_tokens.size(0)

    epoch_val_loss = total_loss / len(dataloader)
    val_losses.append(epoch_val_loss)

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Report
print(f"Final Training Loss: {train_losses[-1]}")
print(f"Final Validation Loss: {val_losses[-1]}")
print(f"Validation Accuracy: {(total_correct / total_samples) * 100:.2f}%")

# Qualitative Validation
def translate_sentence(sentence, encoder, decoder, max_length=50):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        tokens = [en_word2idx.get(token, en_word2idx['<unk>']) for token in tokenize(sentence, 'en')]
        input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        
        enc_out = encoder(input_tensor)
        
        decoder_input = torch.tensor([fr_word2idx['<start>']], dtype=torch.long)
        translated_sentence = []
        
        for _ in range(max_length):
            output = decoder(decoder_input.unsqueeze(0), enc_out)
            _, top_token = output.topk(1)
            decoder_input = torch.tensor([top_token.squeeze().item()], dtype=torch.long)
            
            if top_token.squeeze().item() == fr_word2idx['<end>']:
                break
            
            translated_sentence.append(fr_idx2word[top_token.squeeze().item()])
        
        return ' '.join(translated_sentence)

sample_sentences = [
    "I am cold",
    "You are tired",
    "She speaks French fluently",
    "We watch movies on Fridays"
]

for sentence in sample_sentences:
    translated = translate_sentence(sentence, encoder, decoder)
    print(f"English: {sentence}, French: {translated}")
