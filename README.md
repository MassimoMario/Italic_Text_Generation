# Generating text in three italic language styles: _Dante, Italian_ and _Neapolitan_, using a cVAE based on RNNs  🖋️📖
Pytorch implementation of Text Generation.

The goal of this project is to generate text in _Dante, Italian_, and _Neapolitan_ language style, using a Conditional Variational Autoencoder where both Encoder and Decoder are Recurrent Neural Networks.
The VAE is conditioned using style labels, as follows:


The Word Embedding layer has been initialized using a Word2vec model.

The three text corpus taken in consideration for training are:
* Dante: _Divina Commedia_
* Italian: _Uno, nessuno e centomila_ by Luigi Pirandello, and _I Malavoglia_ by Giovanni Verga
* Neapolitan: _Lo cunto de li cunti_ by Giambattista Basile

I followed this idea to condition a VAE:

IMMAGINE

Taken from . . .

Basically a one-hot encoded label is projected into _latent space_ dimensions and added to the latent variable in order to decode an output belonging to that style label.

During inference phase the model decodes a sentence starting from  $z \sim \mathcal{N} _ {0, 1}$ + projected (given) label.

I took in consideration 3 models:
* **RNN cVAE**: both Encoder and Decoder are RNN
* **GRU cVAE**: both Encoder and Decoder are RNN with GRU units
* **LSTM cVAE**: the Encoder is a LSTM and the Decoder a GRU

# Table of Contents
- [Structure](#Structure)
- [Requirements](#Requirements)
- [Usage](#Usage)
- [Results](#Results)
  
# Structure
* [RNN_cVAE.ipynb](RNN_cVAE.ipynb) is a notebook where all the models are defined and trained, with examples of generated text in those three styles
* [text_corpus](text_corpus) repository contains the three corpus used for training
* [pretrained](pretrained) repository contains the pre-trained models
  
# Requirements
* Numpy
* Matplotlib
* Pytorch
* Gensim

# Usage
First clone the repository:

```bash
git clone git@github.com:MassimoMario/Italic_Text_Style_Classification.git
```

Make sure to have Pytorch and Gensim installed:
```bash
pip install torch
```

```bash
pip install gensim
```
# Results
Here's the training curves for the three models:

TRAINING IMAGE

An independent CNN Classifier has been trained to detect the generated style from the models, and so to quantify the accuracy in generating sentence with a given style.

The **SA**, Style Accuracy, and **PPL** $= 2^{- \frac{1}{N} \sum_i log_2 \left( P(w_i)\right)}$, Perplexity, are here reported for the three models:

| Model | SA | Average PPL |
| --- | --- | --- |
| RNN cVAE | 91.96 % | 37.41 | 
| GRU cVAE | 96.00 % | 23.43 | 
| LSTM cVAE | 99.43 % | 37.53 | 
