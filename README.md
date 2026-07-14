# Bigram Language Model

A character-level bigram language model that generates new name-like strings,
built from scratch as part of learning Andrej Karpathy's "Neural Networks:
Zero to Hero" series. Implements the same model two different ways — direct
counting and gradient-based neural network training — to demonstrate that
they are mathematically equivalent approaches to the same problem.

## What it does

- Reads 32,000 names from `names.txt`
- Builds a character-to-index vocabulary (27 tokens: 26 letters + 1 start/end marker)
- **Approach 1 — Counting:** builds a 27×27 matrix counting character bigram
  frequencies, then normalizes into a probability distribution
- **Approach 2 — Neural network:** one-hot encodes each character, trains a
  single linear layer via gradient descent to predict the next character,
  using softmax and negative log-likelihood as the loss
- Samples new names from either model's resulting distribution
- Evaluates both models using negative log-likelihood (NLL)

## Results

| Model                     | Avg. NLL per bigram |
|---------------------------|---------------------|
| Uniform random baseline   | ~3.30 (log(27))      |
| Counting model            | ~2.45                |
| Neural network model      | ~2.45                |

The neural network, trained from random initialization using only gradient
descent, converges to the **same loss** as the direct counting approach.
This demonstrates a core idea in deep learning: training a model with
gradient descent on negative log-likelihood is mathematically equivalent to
directly counting and normalizing frequencies, for problems simple enough
that the optimal solution has a closed form. The neural network formulation
is the one that scales to problems where no closed-form solution exists.

## Tools

Python, PyTorch, NumPy, Matplotlib, Jupyter

## Status

Both the counting-based and neural-network-based bigram models are complete
and verified to converge to the same loss. Next step: move to a multi-layer
neural network with richer context (more than a single preceding character)
to reduce loss further.