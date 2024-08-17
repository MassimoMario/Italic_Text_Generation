import torch
import torch.nn as nn
from torch.nn import functional as F



class RNNcVAE(nn.Module):
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
        embedded_input = self.embedding(x)
        embedded_input = self.layer_norm(embedded_input)

        _, hn = self.encoder(embedded_input)

        mu = self.fc_mu(hn)
        logvar = self.fc_logvar(hn)
        z = self.reparametrization(mu, logvar)

        z = self.condition_on_label(z, label)
        z = self.fc_hidden(z)

        # prepare sos_token for the decoder
        sos_token = self.sos_token.repeat(x.size(0),1)
        sos_token = self.embedding(sos_token)
        sos_token = self.layer_norm(sos_token)

        decoder_input = torch.cat((sos_token, embedded_input), dim = 1)
        decoder_input = decoder_input[:,:-1,:]

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
        projected_label = self.fc_condition(label)
        projected_label = projected_label.repeat((self.num_layers,1,1))

        return z + projected_label

    

    def sample(self, label, len_sample = 25, temperature = 1, sample_type = 'multinomial'):
        z = torch.randn((self.num_layers, 1, self.latent_dim))
        perplexity = 0.0

        self.eval()
        with torch.no_grad():
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
               
                next_token = torch.multinomial(F.softmax(outputs[:,-1,:] / temperature, dim = -1), 1)
                perplexity += torch.log(F.softmax(outputs[:,-1,:] / temperature, dim = -1)[0][next_token])
                sampled_text.append(next_token)
                next_token = self.embedding(next_token)
                next_token = self.layer_norm(next_token)

                output = torch.cat((output, next_token), dim=1)
       


        sampled_text = [self.idx2word[w.item()] for w in sampled_text]

        perplexity = torch.exp(- perplexity / len_sample).item()
        
        return ' '.join(sampled_text), perplexity
    

    

    def number_parameters(self):

        model_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print('Total number of model parameters: ', model_params)

        return None
    

# --------------------------------------------------------------------------------------------------------- #


class GRUcVAE(RNNcVAE, nn.Module):
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
        embedded_input = self.embedding(x)
        embedded_input = self.layer_norm(embedded_input)

        _, (hn, cn) = self.encoder(embedded_input)

        mu = self.fc_mu(hn)
        logvar = self.fc_logvar(hn)
        z = self.reparametrization(mu, logvar)

        z = self.condition_on_label(z, label)
        z = self.fc_hidden(z)

        # prepare sos_token for the decoder
        sos_token = self.sos_token.repeat(x.size(0),1)
        sos_token = self.embedding(sos_token)
        sos_token = self.layer_norm(sos_token)

        decoder_input = torch.cat((sos_token, embedded_input), dim = 1)
        decoder_input = decoder_input[:,:-1,:]

        reconstructed_sequence, _ = self.decoder(decoder_input, z)
        '''# reconstructing sequence through the decoder giving z as hidden state for each time step
        reconstructed_sequence = []
        for t in range(x.shape[1]):
            outputs, _ = self.decoder(decoder_input[:,:t+1,:], z)
            reconstructed_sequence.append(outputs[:,-1,:].unsqueeze(1))

        # concatenating reconstructed words and push them into vocab_size dimensions
        reconstructed_sequence = torch.cat(reconstructed_sequence, dim=1)'''
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
        projected_label = self.fc_condition(label)
        projected_label = projected_label.repeat((self.num_layers,1,1))

        return z + projected_label

    

    def sample(self, label, len_sample = 25, temperature = 1, sample_type = 'multinomial'):
        z = torch.randn((self.num_layers, 1, self.latent_dim))
        perplexity = 0.0

        self.eval()
        with torch.no_grad():
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
        
                next_token = torch.multinomial(F.softmax(outputs[:,-1,:] / temperature, dim = -1), 1)
                perplexity += torch.log(F.softmax(outputs[:,-1,:] / temperature, dim = -1)[0][next_token])
                sampled_text.append(next_token)
                next_token = self.embedding(next_token)
                next_token = self.layer_norm(next_token)

                output = torch.cat((output, next_token), dim=1)
       

      
        

        sampled_text = [self.idx2word[w.item()] for w in sampled_text]


        perplexity = torch.exp(- perplexity / len_sample).item()
        return ' '.join(sampled_text), perplexity
    

    

    def number_parameters(self):

        model_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print('Total number of model parameters: ', model_params)

        return None
    


# --------------------------------------------------------------------------------------------------------- #



class CNNClassifier(nn.Module):
    def __init__(self, embedding_matrix, num_classes, num_filters, kernel_sizes):
        super(CNNClassifier, self).__init__()
        self.embedding_dim = embedding_matrix.shape[1]

        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze = True)
        self.conv_layers = nn.ModuleList([nn.Conv2d(1, num_filters, (k, self.embedding_dim)) for k in kernel_sizes])
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)
        
    def forward(self,x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        conv_results = [F.relu(conv(x)).squeeze(3) for conv in self.conv_layers]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conv_results]
        cat = torch.cat(pooled, dim = 1)
        out = self.fc(cat)
        return F.softmax(out, dim=-1)