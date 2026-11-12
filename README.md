# Capitulo 1 - Evolucion y Fundamentos Cognitivos de los Modelos de Mundo


## Transfomers

### Tokenizacion: words -> tokens

The tokenization is the process of converting text into computer representations

1. word level -> split text by whitespace or punctuations with the out of vocabulary problem
2. Character level -> loses sematic meaning
3. sub word -> breaks frequent words into whole units and rare words into meaningful chunks

### Embeddings: tokens -> semantic vectors

Transforms the process of converting tokens into vector representations, each token has its own vectors, but we use a model here to create the table. The point is to generate similar vectors for similar tokens (semantically)

### Positional Encoding

Investigate deeply

### Attention Mechanism semantic vectors -> sematic weighted matrix

Calculates Query, Key, Value for each vector. (investigate more deeply)
