# who-is-wayne
A silly experiment with word vectors and cosine similarity to find out whether Bruce Wayne is more related to Batman or 
Joker.

## what it does
From some example text, it creates `target-context` vectors using `batman`, `wayne` and `joker` as targets and checks
which of batman or joker is closer to wayne based on the context of the text. The intuition is that, similar words 
occur together, and for similar words the contexts are similar as well. So if `batman` and `wayne` are similar in meaning they
should be sharing similar context. Same goes for `joker` and `wayne` too.

In the process it uses cosine similarity as a metric to check how similar two word vectors can be.
If `u` and `v` are two word vectors, then the cosine sim between them would be. The closer the cosine similarity of 
two vectors is to 1, the more similar they are.

<a href="https://www.codecogs.com/eqnedit.php?latex=sim\left&space;(&space;u,&space;v&space;\right&space;)&space;=&space;\frac{u\cdot&space;v}{\left&space;\|&space;u&space;\right&space;\|\left&space;\|&space;v&space;\right&space;\|}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?sim\left&space;(&space;u,&space;v&space;\right&space;)&space;=&space;\frac{u\cdot&space;v}{\left&space;\|&space;u&space;\right&space;\|\left&space;\|&space;v&space;\right&space;\|}" title="sim\left ( u, v \right ) = \frac{u\cdot v}{\left \| u \right \|\left \| v \right \|}" /></a>

There's a catch however, building the vectors would require `bag of words` approach and the window size chosen for this
method will have significant effect on the cosine similarity score.

__check the jupyter notebook for more info__

## running on your computer
Create a `conda` env or `virtualenv` or just use the base python env you have in your system. Then install the dependencies.
```bash
pip install -r requirements.txt
```

Run the script with the window size as an arg
```bash
python wayneid.py <window_size>
```

### Examples:
```bash
python wayneid.py 3
```
```bash
| result          |   sim(batman, wayne) |   sim(joker, wayne) |   window_size |
|-----------------+----------------------+---------------------+---------------|
| Wayne is Batman |             0.377964 |            0.204124 |             3 |
```

```bash
python wayneid.py 5
```

```bash
| result          |   sim(batman, wayne) |   sim(joker, wayne) |   window_size |
|-----------------+----------------------+---------------------+---------------|
| Wayne is Batman |             0.408248 |            0.258199 |             5 |
```

## Recommended reading
- [Bag-of-words model - Wikipedia](https://www.wikiwand.com/en/Bag-of-words_model)
- [A Gentle Introduction to the Bag-of-Words Model](https://machinelearningmastery.com/gentle-introduction-bag-words-model/)
- [3 basic approaches in Bag of Words which are better than Word Embeddings](https://towardsdatascience.com/3-basic-approaches-in-bag-of-words-which-are-better-than-word-embeddings-c2cbc7398016)
- [Naive   Bayes   and   SentimentClassification, Jurafsky Martin Ch 4](https://web.stanford.edu/~jurafsky/slp3/4.pdf)
- [Lecture by Laura Kallmeyer at Heinrich-Heine-Universität Düsseldorf](https://user.phil-fak.uni-duesseldorf.de/~kallmeyer/MachineLearning/vector-semantics.pdf)

