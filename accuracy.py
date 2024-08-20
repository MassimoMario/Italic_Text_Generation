import numpy as np
from tqdm import tqdm
import torch





def style_accuracy(cvae, classifier, word2idx, n_samples = 1000, seq_length = 30, temperature = 0.7, name = 'RNN'):
    ''' Function computing classifier accuracy and perplexity
    
    Inputs
    ---------
    cvae : istance of RNNcVAE, GRUcVAE or LSTMcVAE
    classifier : istance of CNNClassifier
    word2idx : vocabulary passing from word to int indices
    n_samples : int, number of samples on which accuracies is computed
    seq_length : int, sequence length of the generated sentences
    temperature : float, temperature factor for generating
    name : str, name of the model, RNN, GRU or LSTM
    
    Returns
    ---------
    None '''

    # Vocabulary defining three styles
    style_label = {'Dante' : torch.FloatTensor([1,0,0]), 'Italian' : torch.FloatTensor([0,1,0]), 'Neapolitan' : torch.FloatTensor([0,0,1])}


    label = style_label['Dante']
    average_perplexity = []

    dante_accuracy = 0.0

    # Computing Dante accuracy
    for _ in tqdm(range(n_samples)):
        sentence, perplexity = cvae.sample(label, seq_length, temperature)
        average_perplexity.append(perplexity)

        sentence = [word2idx[w] for w in sentence.split()]
        pred_label = torch.argmax(classifier(torch.tensor(sentence).unsqueeze(0)))
        if pred_label == 0.0:
            dante_accuracy += 1

    dante_accuracy = dante_accuracy / n_samples


    label = style_label['Italian']

    italian_accuracy = 0.0

    # Computing Italian accuracy
    for _ in tqdm(range(n_samples)):
        sentence, perplexity = cvae.sample(label, seq_length, temperature)
        average_perplexity.append(perplexity)

        sentence = [word2idx[w] for w in sentence.split()]
        pred_label = torch.argmax(classifier(torch.tensor(sentence).unsqueeze(0)))
        if pred_label == 1.0:
            italian_accuracy += 1

    italian_accuracy = italian_accuracy / n_samples



    label = style_label['Neapolitan']

    neapolitan_accuracy = 0.0

    # Computing Neapolitan accuracy
    for _ in tqdm(range(n_samples)):
        sentence, perplexity = cvae.sample(label, seq_length, temperature)
        average_perplexity.append(perplexity)

        sentence = [word2idx[w] for w in sentence.split()]
        pred_label = torch.argmax(classifier(torch.tensor(sentence).unsqueeze(0)))
        if pred_label == 2.0:
            neapolitan_accuracy += 1

    neapolitan_accuracy = neapolitan_accuracy / n_samples


    # Computing overall accuracy and average Perplexity
    overall_accuracy = (dante_accuracy + italian_accuracy + neapolitan_accuracy) / 3
    average_perplexity = np.mean(average_perplexity)


    # Print results
    print(name, 'cVAE \n')

    print('Dante accuracy: ', dante_accuracy, '\n')

    print('Italian accuracy: ', italian_accuracy, '\n')
    
    print('Neapolitan accuracy: ', neapolitan_accuracy, '\n')

    print('Overall ', name, ' accuracy: ', overall_accuracy, '\n')

    print('Average ', name, ' Perplexity: ', average_perplexity)