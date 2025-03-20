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

The architecture is rather simple. It accepts `center words` and `context words`, both of which are one-hot encoded. 

Each of the two is transformed by a separate matrix of the same size: $W_{v,d}$ and $W'_{v,d}$, where $v$ is the vocabulary size and $d$ the embedding space dimension. 

Finally, the transformed vectors pass through an output transformation to yield probabilities. The dot product paired with a softmax would be a valid choice. Notice that there are no trainable parameters here.

Once the network is trained, the context matrix $W'_{v,d}$ is ignored and only $W_{v,d}$ is used to get new embeddings.

TODO: Discuss loss functions.

**Advanced details**

- Replacing the softmax
- Negative sampling
- CBOW vs skip-gram

