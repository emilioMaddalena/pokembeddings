# pokembeddings
Gotta embed 'em all! :fire: :ocean: :seedling: :zap:

Learning a word embedding for the 151 gen1 pokemons. 

![alt text](https://github.com/emilioMaddalena/pokembeddings/blob/main/data/embedding_projection.png)

**Project structure:**
- `train_data`: data to learn the embeddings from
- `evaluation_data`: data to see how well the model performs
- `notebooks`: the actual code

**Explanation**

The model is a [classical word2vec network](https://www.tensorflow.org/text/tutorials/word2vec) implemented in tensorflow Keras.

The architecture is rather simple: a one-hot encoder of the input words, followed by an embedding matrix $W_{v,d}$ for the center word and another one $W'_{v,d}$ for the context words, and finally an output layer. 

Here, $v$ is the size of the vocabulary and $d$ is the dimension of the embedding space. The former is fixed by the training corpus, and the latter is a hyperparameter.
