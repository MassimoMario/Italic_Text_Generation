import torch
import torch.nn as nn
from torch.nn import functional as F



class RNNcVAE(nn.Module):
    ''' Class of a VAE where both Encoder and Decoder are RNNs, inherting from the nn.RNN module

    Attributes
    ----------
    embedding_matrix : 2d torch tensor matrix from word2vec embedding
    idx2word : vocabulary to pass from indices to word
    num_classes : int, number of target classes
    hidden_dim : int, dimension of RNNs hidden state
    latent_dim : int, dimension of the VAE latent space
    num_layers : int, number of RNNs layers
    sos_token : torch tensor of the 'start of the sequence' token
    vocab_size : int, number of unique tokens in the dataset

    
    Methods
    -------
    forward(x) : perform the forward pass of the VAE
    reparametrization(mu, log_var) : perform the reparametrization trick
    condition_on_label(z, label) : add the conditioning label to the latent variable
    sample(label, len_sample, temperature) : inference for generating sequences
    number_parameters() : returns the number of model parameters
    '''

    def __init__(self, embedding_matrix, idx2word, num_classes, hidden_dim, latent_dim, num_layers, sos_token, vocab_size):
        super(RNNcVAE, self).__init__()

        self.embedding_dim = embedding_matrix.shape[1]
        self.idx2word = idx2word
        self.sos_token = sos_token
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze = True)
        self.layer_norm = nn.LayerNorm(self.embedding_dim)

        self.encoder = nn.RNN(self.embedding_dim, hidden_dim, num_layers, batch_first = True)
        self.decoder = nn.RNN(self.embedding_dim, hidden_dim, num_layers, batch_first = True)

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)  
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc_hidden = nn.Linear(latent_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)

        self.fc_condition = nn.Linear(num_classes, latent_dim)

    
    def forward(self, x, label):
        ''' Performs the VAE forward pass 
        Input
        -------
        x : torch tensor with shape [Batch_size, Sequence_length], input sequence
        label : torch tensor with shape [Batch_size, 3], conditioning label
        

        Returns
        -------
        reconstructed_sequence : torch tensor with shape [Batch_size, Sequence_length, Embedding_dimension] 
        mu : torch tensor with shape [num_layers, batch_size, latent_dim]
        logvar : torch tensor with shape [num_layers, batch_size, latent_dim]
        '''

        # embedding
        embedded_input = self.embedding(x)
        embedded_input = self.layer_norm(embedded_input)

        # encoder pass
        _, hn = self.encoder(embedded_input)

        # computing mu and logvar 
        mu = self.fc_mu(hn)
        logvar = self.fc_logvar(hn)
        z = self.reparametrization(mu, logvar)

        # conditioning with label
        z = self.condition_on_label(z, label)
        z = self.fc_hidden(z)

        # prepare sos_token for the decoder
        sos_token = self.sos_token.repeat(x.size(0),1)
        sos_token = self.embedding(sos_token)
        sos_token = self.layer_norm(sos_token)

        # preparing decoder inputs
        decoder_input = torch.cat((sos_token, embedded_input), dim = 1)
        decoder_input = decoder_input[:,:-1,:]

        # decoder pass and Linear layer to vocab_size dimensions
        reconstructed_sequence, _ = self.decoder(decoder_input, z)
        reconstructed_sequence = self.fc(reconstructed_sequence)
        
        return reconstructed_sequence, mu, logvar
    


    def reparametrization(self, mu, log_var):
        ''' Reparametrization trick
        
        Inputs
        -------
        mu : torch tensor
        log_var : torch tensor
            
        
        Returns
        -------
        mu + eps*std : torch tensor with the same shape as mu and log_var '''
        
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)

        return mu + eps*std



    def condition_on_label(self, z, label):
        ''' Perform the conditioning of the VAE

        Inputs
        -------
        z : torch tensor
        label : torch tensor


        Returns:
        --------
        z + projected_label : torch tensor
        '''

        # Linear layer to latent space dimensions
        projected_label = self.fc_condition(label)
        projected_label = projected_label.repeat((self.num_layers,1,1))

        return z + projected_label

    

    def sample(self, label, len_sample = 25, temperature = 1):
        ''' Function for inference and generating sentences

        Inputs:
        --------
        label : torch tensor with shape [1,3]
        len_sample : int, length of the generated sentence
        temperature : int, temperature factor for sampling from softmax

        Returns:
        --------
        ' '.join(sampled_text) : string of the generated sentence
        perplexity : float, perplexity score
        '''

        # taking z from a Gaussian distribution
        z = torch.randn((self.num_layers, 1, self.latent_dim))
        perplexity = 0.0

        self.eval()
        with torch.no_grad():

            # conditioning on label
            z = self.condition_on_label(z, label)
            z = self.fc_hidden(z)

            # prepare sos_token for the decoder
            sos_token = self.sos_token.repeat(1,1)
            sos_token = self.embedding(sos_token)
            sos_token = self.layer_norm(sos_token)



            sampled_text = []

            # decoder pass where the input is the previous output
            output = sos_token
            for _ in range(len_sample):
                outputs, _ = self.decoder(output, z)
                outputs = self.fc(outputs)
               
               # taking next token from the last RNN output
                next_token = torch.multinomial(F.softmax(outputs[:,-1,:] / temperature, dim = -1), 1)

                # computing log of probabilities
                perplexity += torch.log(F.softmax(outputs[:,-1,:] / temperature, dim = -1)[0][next_token])

                # appending next token to the sampled text
                sampled_text.append(next_token)
                next_token = self.embedding(next_token)
                next_token = self.layer_norm(next_token)

                output = torch.cat((output, next_token), dim=1)
       

        # bringing sampled text to string with idx2word
        sampled_text = [self.idx2word[w.item()] for w in sampled_text]

        # computing perplexity
        perplexity = torch.exp(- perplexity / len_sample).item()
        
        return ' '.join(sampled_text), perplexity
    

    

    def number_parameters(self):
        ''' Function returning number of model parameters '''

        model_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print('Total number of model parameters: ', model_params)

        return None
    

# --------------------------------------------------------------------------------------------------------- #


class GRUcVAE(RNNcVAE, nn.Module):
    ''' Class of a VAE where both Encoder and Decoder are RNNs with GRU units, inherting from the nn.GRU module and RNNcVAE class '''

    def __init__(self, embedding_matrix, idx2word, num_classes, hidden_dim, latent_dim, num_layers, sos_token, vocab_size):
        super().__init__(embedding_matrix, idx2word, num_classes, hidden_dim, latent_dim, num_layers, sos_token, vocab_size)

        self.embedding_dim = embedding_matrix.shape[1]
        self.idx2word = idx2word
        self.sos_token = sos_token
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze = True)
        self.layer_norm = nn.LayerNorm(self.embedding_dim)

        self.encoder = nn.GRU(self.embedding_dim, hidden_dim, num_layers, batch_first = True)
        self.decoder = nn.GRU(self.embedding_dim, hidden_dim, num_layers, batch_first = True)

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)  
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc_hidden = nn.Linear(latent_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)



# --------------------------------------------------------------------------------------------------------- #



class LSTMcVAE(nn.Module):
    ''' Class of a VAE where Encoder is a LSTM and the Decoder is a GRU, inherting from the nn.LSTM and nn.GRU module

    Attributes
    ----------
    embedding_matrix : 2d torch tensor matrix from word2vec embedding
    idx2word : vocabulary to pass from indices to word
    num_classes : int, number of target classes
    hidden_dim : int, dimension of RNNs hidden state
    latent_dim : int, dimension of the VAE latent space
    num_layers : int, number of RNNs layers
    sos_token : torch tensor of the 'start of the sequence' token
    vocab_size : int, number of unique tokens in the dataset

    
    Methods
    -------
    forward(x) : perform the forward pass of the VAE
    reparametrization(mu, log_var) : perform the reparametrization trick
    condition_on_label(z, label) : add the conditioning label to the latent variable
    sample(label, len_sample, temperature) : inference for generating sequences
    number_parameters() : returns the number of model parameters
    '''

    def __init__(self, embedding_matrix, idx2word, num_classes, hidden_dim, latent_dim, num_layers, sos_token, vocab_size):
        super(LSTMcVAE, self).__init__()

        self.embedding_dim = embedding_matrix.shape[1]
        self.idx2word = idx2word
        self.sos_token = sos_token
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze = True)
        self.layer_norm = nn.LayerNorm(self.embedding_dim)

        self.encoder = nn.LSTM(self.embedding_dim, hidden_dim, num_layers, batch_first = True)
        self.decoder = nn.GRU(self.embedding_dim, hidden_dim, num_layers, batch_first = True)

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)  
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc_hidden = nn.Linear(latent_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)

        self.fc_condition = nn.Linear(num_classes, latent_dim)

    
    def forward(self, x, label):
        ''' Performs the VAE forward pass 
        Input
        -------
        x : torch tensor with shape [Batch_size, Sequence_length], input sequence
        label : torch tensor with shape [Batch_size, 3], conditioning label
        

        Returns
        -------
        reconstructed_sequence : torch tensor with shape [Batch_size, Sequence_length, Embedding_dimension] 
        mu : torch tensor with shape [num_layers, batch_size, latent_dim]
        logvar : torch tensor with shape [num_layers, batch_size, latent_dim]
        '''

        # embedding
        embedded_input = self.embedding(x)
        embedded_input = self.layer_norm(embedded_input)

        # encoder pass
        _, (hn, cn) = self.encoder(embedded_input)

        # computing mu and logvar
        mu = self.fc_mu(hn)
        logvar = self.fc_logvar(hn)
        z = self.reparametrization(mu, logvar)

        # conditioning on label
        z = self.condition_on_label(z, label)
        z = self.fc_hidden(z)

        # prepare sos_token for the decoder
        sos_token = self.sos_token.repeat(x.size(0),1)
        sos_token = self.embedding(sos_token)
        sos_token = self.layer_norm(sos_token)

        # preparing decoder inputs
        decoder_input = torch.cat((sos_token, embedded_input), dim = 1)
        decoder_input = decoder_input[:,:-1,:]

        # decoder pass and Linear layer to vocab_size dimensions
        reconstructed_sequence, _ = self.decoder(decoder_input, z)
        reconstructed_sequence = self.fc(reconstructed_sequence)
        
        return reconstructed_sequence, mu, logvar
    


    def reparametrization(self, mu, log_var):
        ''' Reparametrization trick
        
        Inputs
        -------
        mu : torch tensor
        log_var : torch tensor
            
        
        Returns
        -------
        mu + eps*std : torch tensor with the same shape as mu and log_var'''
        
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)

        return mu + eps*std



    def condition_on_label(self, z, label):
        ''' Perform the conditioning of the VAE

        Inputs
        -------
        z : torch tensor
        label : torch tensor


        Returns:
        --------
        z + projected_label : torch tensor
        '''

        # Linear layer to latent space dimensions
        projected_label = self.fc_condition(label)
        projected_label = projected_label.repeat((self.num_layers,1,1))

        return z + projected_label

    

    def sample(self, label, len_sample = 25, temperature = 1):
        ''' Function for inference and generating sentences

        Inputs:
        --------
        label : torch tensor with shape [1,3]
        len_sample : int, length of the generated sentence
        temperature : int, temperature factor for sampling from softmax

        Returns:
        --------
        ' '.join(sampled_text) : string of the generated sentence
        perplexity : float, perplexity score
        '''

        # z from Gaussian distribution
        z = torch.randn((self.num_layers, 1, self.latent_dim))
        perplexity = 0.0

        self.eval()
        with torch.no_grad():

            # conditioning on label
            z = self.condition_on_label(z, label)
            z = self.fc_hidden(z)

            # prepare sos_token for the decoder
            sos_token = self.sos_token.repeat(1,1)
            sos_token = self.embedding(sos_token)
            sos_token = self.layer_norm(sos_token)



            sampled_text = []

            # decoder pass where the input is the previous output
            output = sos_token
            for _ in range(len_sample):
                outputs, _ = self.decoder(output, z)
                outputs = self.fc(outputs)
        
                # taking next token from last output
                next_token = torch.multinomial(F.softmax(outputs[:,-1,:] / temperature, dim = -1), 1)

                # computing log of probabilities
                perplexity += torch.log(F.softmax(outputs[:,-1,:] / temperature, dim = -1)[0][next_token])

                # appending next token to the sampled text
                sampled_text.append(next_token)
                next_token = self.embedding(next_token)
                next_token = self.layer_norm(next_token)

                output = torch.cat((output, next_token), dim=1)
       

        # bringing sampled text to string with idx2word
        sampled_text = [self.idx2word[w.item()] for w in sampled_text]

        # computing average perplexity
        perplexity = torch.exp(- perplexity / len_sample).item()

        return ' '.join(sampled_text), perplexity
    

    

    def number_parameters(self):
        ''' Function returning number of model parameters '''

        model_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print('Total number of model parameters: ', model_params)

        return None
    


# --------------------------------------------------------------------------------------------------------- #



class CNNClassifier(nn.Module):
    '''  Class of a CNN Classifier for text, made up of Conv2d layers

    Attributes
    ----------
    embedding_matrix : 2d torch tensor matrix from word2vec embedding
    num_classes : int, number of classes
    num_filters : int, number of filters in the Conv2d layer
    kernel_sizes : list of int, sizes of kernels in Conv2d layers

    Methods
    ----------
    forward(x) : forward pass of the Classifier'''

    def __init__(self, embedding_matrix, num_classes, num_filters, kernel_sizes):
        super(CNNClassifier, self).__init__()
        self.embedding_dim = embedding_matrix.shape[1]

        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze = True)
        self.conv_layers = nn.ModuleList([nn.Conv2d(1, num_filters, (k, self.embedding_dim)) for k in kernel_sizes])
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)
        
    def forward(self,x):
        ''' Forward pass function
        
        Input
        ----------
        x : 2D torch tensor tensor, input sentence with shape [Batch size, Sequence length]
        
        Returns
        ----------
        out : 2D torch tensor with probabilities for every class'''

        # Word Embedding
        x = self.embedding(x)
        x = x.unsqueeze(1)

         # Convolutional layers and Max pool
        conv_results = [F.relu(conv(x)).squeeze(3) for conv in self.conv_layers]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conv_results]

        # Concatenation of pooled output and Linear layer to num_class dimensions
        cat = torch.cat(pooled, dim = 1)
        out = self.fc(cat)
        
        return F.softmax(out, dim=-1)