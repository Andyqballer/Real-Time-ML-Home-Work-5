import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

# Define the dataset
english_to_french = [
    ("They visit museums often", "Ils visitent souvent les musées"),
    ("The restaurant serves delicious food", "Le restaurant sert une délicieuse nourriture"),
    ("She studies mathematics at university", "Elle étudie les mathématiques à l'université"),
    ("We watch movies on Fridays", "Nous regardons des films le vendredi"),
    ("He listens to music while jogging", "Il écoute de la musique en faisant du jogging"),
    ("They travel around the world", "Ils voyagent autour du monde"),
    ("The book is on the table", "Le livre est sur la table"),
    ("She dances gracefully", "Elle danse avec grâce"),
    ("We celebrate birthdays with cake", "Nous fêtons les anniversaires avec un gâteau"),
    ("He works hard every day", "Il travaille dur tous les jours"),
    ("They speak different languages", "Ils parlent différentes langues"),
    ("The flowers bloom in spring", "Les fleurs éclosent au printemps"),
    ("She writes poetry in her free time", "Elle écrit de la poésie pendant son temps libre"),
    ("We learn something new every day", "Nous apprenons quelque chose de nouveau chaque jour"),
    ("The dog barks loudly", "Le chien aboie bruyamment"),
]

# Shuffle the dataset
random.shuffle(english_to_french)

# Split dataset into train and validation sets
split = int(0.8 * len(english_to_french))
train_data = english_to_french[:split]
val_data = english_to_french[split:]

# Define vocabulary
english_vocab = set()
french_vocab = set()
for pair in english_to_french:
    english_sentence, french_sentence = pair
    english_vocab.update(english_sentence.split())
    french_vocab.update(french_sentence.split())

# Add special tokens for padding, start, and end of sentence
PAD_token = 0
SOS_token = 1
EOS_token = 2
english_vocab.add('<PAD>')
english_vocab.add('<SOS>')
english_vocab.add('<EOS>')
french_vocab.add('<PAD>')
french_vocab.add('<SOS>')
french_vocab.add('<EOS>')

# Create word to index dictionaries
english_word_to_index = {word: i for i, word in enumerate(english_vocab)}
french_word_to_index = {word: i for i, word in enumerate(french_vocab)}

# Convert sentences to tensors of word indices
def sentence_to_tensor(sentence, vocab, word_to_index):
    indexes = [word_to_index[word] for word in sentence.split()]
    indexes.append(EOS_token)  # Append <EOS> token
    return torch.tensor(indexes, dtype=torch.long).view(-1, 1)

# Define the training function
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=20):
    encoder_hidden = torch.zeros(1, 1, encoder.hidden_size)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    loss = 0

    _, encoder_hidden = encoder(input_tensor)

    decoder_input = torch.tensor([[SOS_token]])

    decoder_hidden = encoder_hidden

    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()

        loss += criterion(decoder_output, target_tensor[di])
        if decoder_input.item() == EOS_token:
            break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

# Define the evaluation loss function for validation
def evaluate_loss(input_tensor, target_tensor, encoder, decoder, criterion):
    encoder_hidden = torch.zeros(1, 1, encoder.hidden_size)

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    loss = 0

    _, encoder_hidden = encoder(input_tensor)

    decoder_input = torch.tensor([[SOS_token]])

    decoder_hidden = encoder_hidden

    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()

        loss += criterion(decoder_output, target_tensor[di])
        if decoder_input.item() == EOS_token:
            break

    return loss.item() / target_length

# Define the evaluation function
def evaluate(encoder, decoder, sentence, max_length=20):
    with torch.no_grad():
        input_tensor = sentence_to_tensor(sentence, french_vocab, french_word_to_index)
        input_length = input_tensor.size()[0]
        encoder_hidden = torch.zeros(1, 1, encoder.hidden_size)

        _, encoder_hidden = encoder(input_tensor)

        decoder_input = torch.tensor([[SOS_token]])
        decoder_hidden = encoder_hidden

        decoded_words = []

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(english_index_to_word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return ' '.join(decoded_words)

class DecoderGRU(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = nn.functional.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

class EncoderGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input):
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded)
        return output, hidden

# Initialize the models
encoder = EncoderGRU(len(french_vocab), 256)
decoder = DecoderGRU(256, len(english_vocab))

# Define the optimizers and criterion
encoder_optimizer = optim.SGD(encoder.parameters(), lr=0.01)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=0.01)
criterion = nn.NLLLoss()

# Training the model
n_iters = 10000
print_every = 1000
plot_every = 100
all_losses = []
train_losses = []
val_losses = []

for iter in range(1, n_iters + 1):
    training_pair = random.choice(train_data)
    input_tensor = sentence_to_tensor(training_pair[1], french_vocab, french_word_to_index)
    target_tensor = sentence_to_tensor(training_pair[0], english_vocab, english_word_to_index)

    loss = train(input_tensor, target_tensor, encoder,
                 decoder, encoder_optimizer, decoder_optimizer, criterion)
    all_losses.append(loss)
    train_losses.append(loss)

    if iter % print_every == 0:
        avg_loss = sum(train_losses[-print_every:]) / print_every
        print('%d %d%% %.4f' % (iter, iter / n_iters * 100, avg_loss))

    if iter % plot_every == 0:
        avg_loss = sum(all_losses[-plot_every:]) / plot_every
        val_loss = sum(evaluate_loss(sentence_to_tensor(pair[1], french_vocab, french_word_to_index),
                                     sentence_to_tensor(pair[0], english_vocab, english_word_to_index),
                                     encoder, decoder, criterion) for pair in val_data) / len(val_data)
        val_losses.append(val_loss)
        print("Validation Loss:", val_loss)

# Plotting the losses
plt.figure()
plt.plot(all_losses, label='Training Loss')
plt.plot(range(0, n_iters, plot_every), val_losses, label='Validation Loss', marker='o')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Evaluation on validation set
def evaluate_accuracy(encoder, decoder, val_data):
    correct = 0
    total = len(val_data)
    
    for pair in val_data:
        input_sentence = pair[1]
        target_sentence = pair[0]
        
        output_sentence = evaluate(encoder, decoder, input_sentence)
        
        if output_sentence == target_sentence:
            correct += 1
            
    return correct / total

val_accuracy = evaluate_accuracy(encoder, decoder, val_data)
print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')

# Evaluation on some French sentences
def evaluate_randomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(val_data)
        print('French:', pair[1])
        print('Ground truth English:', pair[0])
        output_sentence = evaluate(encoder, decoder, pair[1])
        print('Generated English:', output_sentence)
        print('')

evaluate_randomly(encoder, decoder)
