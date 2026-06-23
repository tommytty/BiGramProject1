\# Bigram Language Model



A character-level bigram language model that generates new name-like strings,

built from scratch as part of learning Andrej Karpathy's "Neural Networks: 

Zero to Hero" series.



\## What it does

\- Reads 32,000 names from `names.txt`

\- Builds a 27×27 matrix counting character bigram frequencies

\- Normalizes counts into a probability distribution

\- Samples new names from the resulting distribution

\- Evaluates the model using negative log-likelihood



\## Tools

Python, PyTorch, NumPy, Matplotlib, Jupyter



\## Status

Built statistical bigram version. Next step: rebuild as a neural network 

and verify it converges to the same loss.

