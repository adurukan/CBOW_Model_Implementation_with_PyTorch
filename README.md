# CBOW_Model_Implementation_with_PyTorch

The Continuous Bag-of-Words model (CBOW) is frequently used in NLP deep learning. It is a model that tries to predict words given
the context of a few words before and a few words after the target word. This is distinct from language modeling, since CBOW is not
sequential and does not have to be probabilistic. Typically, CBOW is used to quickly train word embeddings, and these embeddings are
used to initialize the embeddings of some more complicated model. Usually, this is referred to as pretraining embeddings. It almost
always helps performance a couple of percent.

The implementation is based on the [Word Embeddings: Encoding Lexical Semantics](https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html) tutorial. 

The losses are shown in the below for 10 epochs:
[227.83707904815674, 222.54315400123596, 217.38093447685242, 212.34736227989197, 207.43812465667725, 202.64905071258545, 197.97451376914978, 193.4090849161148, 188.94702577590942, 184.5824944972992]
As can be seen, they are constantly decreasing which means that the semantic information is being learned.
