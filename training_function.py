import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def vae_loss(recon_x, x, mu, logvar, l_kl = 0.05, loss_fn = nn.CrossEntropyLoss()):
    ''' Function computing loss function for the VAE
    
    Inputs
    ---------
    recon_x : torch tensor with shape [Batch_size, Sequence_length, Embedding_dim]
    x : torch tensor with shape [Batch_size, Sequence_length]
    mu : torch tensor with shape [num_layers, batch_size, latent_dim]
    logvar : torch tensor with shape [num_layers, batch_size, latent_dim]
    l_kl : float, factor for KL Divergence
    loss_fn : loss function, initialized with nn.CrossEntropyLoss()
    
    Returns
    ---------
    L + l_kl*KLD: float, reconstruction loss + KL Divergence'''

    L = loss_fn(recon_x.reshape((recon_x.size(0)*recon_x.size(1),recon_x.size(2))), x.view(-1))
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return L + l_kl * KLD



# ------------------------------------------------------------------------------------------------------------ #


def training(model, train_loader, val_loader, num_epochs, lr = 4e-4, title = 'Training'):
    ''' Training function
    
    Input
    --------
    model : istance of a CNNClassifier, RNNClassifier, GRUClassifier, LSTMClassifier or TClassifier
    train_loader : istance of torch Dataloader with training data and labels
    val_loader : istance of torch Dataloader with validation data and labels
    num_epochs : int, number of epochs
    lr : float, learning rate for Adam optimizer
    title : str, Title of the matplot figure
    
    Returns
    --------
    None '''

    params = list(model.parameters())

    # Optimizer
    optimizer = torch.optim.Adam(params, lr = lr)

    
    train_losses = []
    val_losses = []
    

    l_kl = 0.05

    # For loop over epochs
    for epoch in tqdm(range(num_epochs)):
    
        train_loss = 0.0
        average_loss = 0.0
        val_loss = 0.0
        average_val_loss = 0.0


        # For loop for every batch
        for  i, (inputs, label) in enumerate(train_loader):
            inputs = inputs.to(device)
            label = label.to(device)
            label = label.type(torch.FloatTensor)

            optimizer.zero_grad()
            

            # forward pass through classifier
            recon_x, mu, logvar = model(inputs, label)
    
            # comuting training loss
            loss = vae_loss(recon_x.to(device),
                            inputs.to(device),
                            mu.to(device),
                            logvar.to(device),
                            l_kl)
            
            loss.backward()
            train_loss += loss.item()


            optimizer.step()

            
            
        
        
        # Validation
        with torch.no_grad():
            for i, (inputs, label) in enumerate(val_loader):
                inputs = inputs.to(device)
                label = label.to(device)
                label = label.type(torch.FloatTensor)
    

                # forward pass through classifier
                recon_x, mu, logvar = model(inputs, label)
                
                
                # comuting validation loss
                val_loss_tot = vae_loss(recon_x.to(device),
                                        inputs.to(device),
                                        mu.to(device),
                                        logvar.to(device),
                                        l_kl)
                
                val_loss += val_loss_tot.item()



            
        
        # Computing average training and validation loss
        average_loss = train_loss / len(train_loader.dataset)
        train_losses.append(average_loss)

        average_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(average_val_loss)

        

        # printing average training and validation losses
        print(f'====> Epoch: {epoch+1} Average train loss: {average_loss:.4f}, Average val loss: {average_val_loss:.4f}')
    

    # Plotting training and validation curve at the end of the for loop 
    plt.plot(np.linspace(1,num_epochs,len(train_losses)), train_losses, c = 'darkcyan',label = 'train')
    plt.plot(np.linspace(1,num_epochs,len(val_losses)), val_losses, c = 'orange',label = 'val')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title + ' cVAE Training')
    plt.show()


    return None



# ------------------------------------------------------------------------------------------------------------ #



def CNN_loss(y_s, labels, loss_fn=nn.CrossEntropyLoss()):
    ''' Function computing loss for CNN Classifier
    
    Inputs:
    --------
    y_s : torch tensor with shape [Batch_size, 3], predicted labels
    labels : torch tensor with shape [Batch_size, 3], ground truth labels 
    
    Returns:
    --------
    L_classif : float, loss value'''

    L_classif = loss_fn(y_s, labels)

    return L_classif




# ------------------------------------------------------------------------------------------------------------ #




def train_CNN(style_classif, train_loader, val_loader, num_epochs, lr = 4e-4):
    ''' Function for training the CNN Classifier
    
    Inputs:
    --------
    style_classif : istance of CNNClassifier
    train_loader : istance of torch Dataloader with training data and labels
    val_loader : istance of torch Dataloader with validation data and labels
    num_epochs : int, number of epochs
    lr : float, learning rate for Adam optimizer

    Returns:
    average_losses : list of training loss values
    '''
    params = list(style_classif.parameters())

    optimizer = torch.optim.Adam(params, lr = lr)

    average_losses = []
    val_losses = []
    
    for epoch in tqdm(range(num_epochs)):
        train_loss = 0.0
        average_loss = 0.0
        val_loss = 0.0
        average_val_loss = 0.0
        
       
        for  i, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.type(torch.FloatTensor)

            optimizer.zero_grad()


            pred_style = style_classif(data)
            
    
            loss_tot = CNN_loss(pred_style, labels)
            loss_tot.backward()
            train_loss += loss_tot.item()


            optimizer.step()
            
        
        
        average_loss = train_loss / len(train_loader.dataset)
        
        print(f'====> Epoch: {epoch+1} Average loss: {average_loss:.4f}')
        average_losses.append(average_loss)

        with torch.no_grad():
            for i, (data, labels) in enumerate(val_loader):
                data = data.to(device)
                labels = labels.type(torch.FloatTensor)
                

                pred_style = style_classif(data)

                
                
                val_loss_tot = CNN_loss(pred_style, labels)
                val_loss += val_loss_tot.item()


                
            
            
        average_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(average_val_loss)

    
    plt.plot(np.linspace(1,num_epochs,len(average_losses)), average_losses, c = 'darkcyan',label = 'train')
    plt.plot(np.linspace(1,num_epochs,len(val_losses)), val_losses, c = 'orange',label = 'val')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()
    return average_losses