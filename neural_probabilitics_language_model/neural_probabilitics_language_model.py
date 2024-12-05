import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Define the Neural Probabilistic Language Model


class NPLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, hidden_dim):
        super(NPLM, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_layer = nn.Linear(
            (context_size - 1) * embedding_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, context_words):
        # Embedding lookup
        embeddings = self.embeddings(context_words).view(
            context_words.size(0), -1)
        # Hidden layer
        hidden = torch.tanh(self.hidden_layer(embeddings))
        # Output layer
        logits = self.output_layer(hidden)
        return logits

# Define a Dataset


class TextDataset(Dataset):
    def __init__(self, sequences, vocab, context_size):
        self.data = []
        self.vocab = vocab
        self.context_size = context_size
        for sequence in sequences:
            for i in range(len(sequence) - context_size + 1):
                context = sequence[i:i+context_size-1]
                target = sequence[i+context_size-1]
                self.data.append((context, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        return torch.tensor(context), torch.tensor(target)


# Hyperparameters
vocab_size = 5000  # Example vocabulary size
embedding_dim = 100
context_size = 5   # n-1 (n is the total window size including the target)
hidden_dim = 128
batch_size = 64
epochs = 10

# Define model, loss, and optimizer
model = NPLM(vocab_size, embedding_dim, context_size, hidden_dim)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Example data (toy dataset)
# Replace this with a real dataset
sequences = [
    [1, 2, 3, 4, 5, 6],  # Toy example sequence (use real word indices here)
    [3, 4, 5, 6, 7, 8],
]
dataloader = DataLoader(TextDataset(
    sequences, vocab_size, context_size), batch_size=batch_size, shuffle=True)

# Training Loop


def train(model, dataloader, loss_fn, optimizer, epochs=10):
    for epoch in range(epochs):
        total_loss = 0
        for context, target in dataloader:
            optimizer.zero_grad()
            # Forward pass
            predictions = model(context)
            # Compute loss
            loss = loss_fn(predictions, target)
            # Backward pass
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss}")


# Start Training
train(model, dataloader, loss_fn, optimizer, epochs)
