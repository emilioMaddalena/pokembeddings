# pokembeddings
Gotta embed 'em all! :fire: :ocean: :seedling: :zap:

Learning a word embedding for the 151 gen1 pokemons. The training corpus is composed of pokedex entries.

The goal is to arrive at a well-separated, explainable embedding space of minimum dimension.

**Follow these steps:**
- Train a `Word2Vec` model
- Take a look at the resulting embedding using `Word2Vec.visualize_embeddings()` 
- Save it `Word2Vec.save()` 
- See how well the model performs via
`uv run pytest --model-path path evaluation_tests`