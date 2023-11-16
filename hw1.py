'''
 - Average score of generated sequence = 0.9997
 - The problem with training the model as a GAN is that the critic (discriminator) is trained to give a low score to randomly generated sequences and
   a high score to anything non-random. The critic is simply classifying random from non-random, but this may not directly correlate with musical quality.
   As long as the composer (generator) produces something that is non-random the critic will give it a high score and the composer will learn nothing.
 - A possible solution would be to diversify the training data for critic and composer. Diversify the training data to include not only good (score = 1)
   and bad (score = 0) music, but also music with varying scores (0.1 - 0.9). This will help the network to learn a broader range of musical patterns. 
'''

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from random import choice

from model_base import ComposerBase, CriticBase
from midi2seq import process_midi_seq, piano2seq, seq2piano, random_piano, dim

from tqdm import tqdm
import gdown

device = "cuda" if torch.cuda.is_available() else "cpu"

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, X):
        out = self.embedding(X)
        out, _ = self.lstm(out)
        out = self.output_layer(out)
        return out

class Composer(ComposerBase):
  def __init__(self, load_trained=False):
    '''
    :param load_trained
        If load_trained is True, load a trained model from a file.
        Should include code to download the file from Google drive if necessary.
        else, construct the model
    '''
    self.vocab_size = dim # Define the size of dictionary of embeddings
    self.embedding_dim = 512 # Define the size of each embedding vector
    self.hidden_size = 1024  # Define the LSTM hidden size
    self.num_layers = 3  # Define the number of LSTM layers
    self.num_class = dim # Define the number of output classifications

    self.learning_rate = 0.001 # Define the learing rate
    self.model = LSTM(self.vocab_size, self.embedding_dim, self.hidden_size, self.num_layers,  self.num_class)
    self.loss_func = nn.CrossEntropyLoss()
    self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    if load_trained:
        url = "https://drive.google.com/uc?id=1ntBxEm45MHCMwgQ95-wSCWNuVDFs9_a9"
        output = "composer.pt"

        gdown.download(url, output, quiet=True)

        url = "https://drive.google.com/uc?id=1cNmvk4wdXUPH4QLJNMAMOqv1hpvJLhS-"
        output = "initial_tokens.npy"

        gdown.download(url, output, quiet=True)
        
        self.model.load_state_dict(torch.load("composer.pt"))
        self.initial_tokens = np.load("initial_tokens.npy")

  def save(self):
    torch.save(self.model.state_dict(), "composer.pt")

  def train(self, x):
    '''
    Train the model on one batch of data
    :param x: train data. For composer training, a single torch tensor will be given
    and for critic training, x will be a tuple of two tensors (data, label)
    :return: (mean) loss of the model on the batch
    '''
    self.model.to(device)
    self.model.train()

    batch_size = x.shape[0]
    seq_len = x.shape[1]

    x = x.to(device).long()

    output = self.model(x)
    output  = output.to(device)

    output = output.reshape(batch_size * seq_len, -1)
    output = output[:-1, :]

    x = x.reshape(-1)
    x = x[1:]

    # Compute the loss
    loss = self.loss_func(output, x)

    # Backpropagation and optimization
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    return loss.item()

  def compose(self, n):
    '''
    Generate a music sequence
    :param n: length of the sequence to be generated
    :return: the generated sequence
    '''
    self.model.to(device)
    self.model.eval()  # Set the model to evaluation mode

    input_sequence = torch.tensor([[choice(self.initial_tokens)]]).to(device).long()

    generated_sequence = []

    with torch.no_grad():
        # Generate the sequence token by token
        for _ in range(n):
            # Generate the next token using the model
            output = self.model(input_sequence)
            prediction = torch.softmax(output[:, -1], -1)
            prediction = torch.argmax(prediction)

            input_sequence = torch.cat((input_sequence, prediction.view(1,1)), 1)

            prediction = prediction.item()
            generated_sequence.append(prediction)

    return np.stack(generated_sequence)

class Critic(CriticBase):
  def __init__(self, load_trained=False):
    '''
    :param load_trained
        If load_trained is True, load a trained model from a file.
        Should include code to download the file from Google drive if necessary.
        else, construct the model
    '''
    self.vocab_size = dim # Define the size of dictionary of embeddings
    self.embedding_dim = 512 # Define the size of each embedding vector
    self.hidden_size = 1024  # Define the LSTM hidden size
    self.num_layers = 3  # Define the number of LSTM layers
    self.num_class = 1 # Define the number of output classifications

    self.learning_rate = 0.001 # Define the learing rate
    self.model = LSTM(self.vocab_size, self.embedding_dim, self.hidden_size, self.num_layers,  self.num_class)
    self.loss_func = nn.MSELoss()
    self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    if load_trained:
      url = "https://drive.google.com/uc?id=1JGqDvcRKJJ_aE5mLG3uQ0ScmUDG6CL5p"
      output = "critic.pt"

      gdown.download(url, output, quiet=True)

      self.model.load_state_dict(torch.load("critic.pt"))    

  def save(self):
    torch.save(self.model.state_dict(), "critic.pt")

  def train(self, x):
    '''
    Train the model on one batch of data
    :param x: train data. For composer training, a single torch tensor will be given
    and for critic training, x will be a tuple of two tensors (data, label)
    :return: (mean) loss of the model on the batch
    '''
    self.model.to(device)
    self.model.train()

    data, label = x
    data, label = data.to(device).long(), label.to(device)

    # # Forward pass
    output = self.model(data).to(device)
    output = output[:,-1].reshape(-1)

    # # Compute the loss
    loss = self.loss_func(output, label)

    # # Backpropagation and optimization
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    return loss.item()

  def score(self, x):
    """
    Compute the score of a music sequence.
    :param x: a music sequence as a torch tensor
    :return: the score between 0 and 1 that reflects the quality of the music; the closer to 1, the better
    """
    self.model.to(device)
    self.model.eval()

    x = x.unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
      output = self.model(x)[:,-1].reshape(-1)

    return torch.clamp(output, torch.zeros_like(output), torch.ones_like(output)).item()

if __name__ == '__main__':
    try:
      load_trained = True
      if load_trained:
        print("Loading models...")
      else:
        print("Training models...")
        
      composer = Composer(load_trained=load_trained)
      critic = Critic(load_trained=load_trained)

      if not load_trained:
        epoch = 100
        bsz = 32
        maxlen = 150

        good_seq = torch.from_numpy(process_midi_seq(maxlen=maxlen))
        good_labels = torch.ones(good_seq.shape[0])

        bad_seq = []
        
        for i in range(len(good_seq)):
          bad_seq.append(random_piano())

        bad_seq = torch.from_numpy(process_midi_seq(bad_seq, maxlen=maxlen))
        bad_labels = torch.zeros(bad_seq.shape[0])

        data = torch.cat((good_seq, bad_seq))
        label = torch.cat((good_labels, bad_labels))

        composer_loader = DataLoader(TensorDataset(good_seq), shuffle=True, batch_size=bsz, num_workers=4)
        critic_loader = DataLoader(TensorDataset(data,label), shuffle=True, batch_size=bsz, num_workers=4)

        print("Training Composer")
        for i in tqdm(range(epoch)):
            for batch_ndx, sample in enumerate(tqdm(composer_loader, leave=False)):
                loss = composer.train(sample[0].long())
        
        print("Training Critic")
        for i in tqdm(range(epoch)):
            for batch_ndx, sample in enumerate(tqdm(critic_loader, leave=False)):
                loss = critic.train(sample)

        print("Saving models...")

        composer.save()
        critic.save()

      print("Generating sequences...")

      generated_seq_count = 1
      generated_seq_len = 500
      avg_score = 0

      for i in tqdm(range(generated_seq_count)):
        midi = composer.compose(generated_seq_len)
        sequence = torch.from_numpy(midi)
        score = critic.score(sequence)
        avg_score += score
        midi = seq2piano(midi)
        midi.write(f'generated/piano{i}.midi')
      
      print(f"Average score: {avg_score/generated_seq_count}")
      print(f"Score: {critic.score(torch.from_numpy(piano2seq(random_piano(500))))}")

    except KeyboardInterrupt:
      if not load_trained:
        input("Press Enter to save models...")
        print("Saving models...")

        composer.save()
        critic.save()

      input("Press Enter to generate sequences...")
      print("Generating sequences...")

      avg_score = 0

      for i in tqdm(range(generated_seq_count)):
        midi = composer.compose(generated_seq_len)
        sequence = torch.from_numpy(midi)
        score = critic.score(sequence)
        avg_score += score
        midi = seq2piano(midi)
        midi.write(f'generated/piano{i}.midi')
      
      print(f"Average score: {avg_score/generated_seq_count}")