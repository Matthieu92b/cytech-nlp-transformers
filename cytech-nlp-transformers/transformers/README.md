# Transformer from scratch — 4 TP

Reconstruction pas à pas d'un mini-Transformer (style GPT, niveau caractère) entraîné sur **Tiny Shakespeare**, en PyTorch.
Chaque TP isole une brique ; le dernier les assemble et génère du texte.

| TP | Thème | Notebook | |
|---|---|---|---|
| 1 | Données, tokenisation, modèle bigramme | [TP1_bigramme](notebooks/TP1_bigramme.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Matthieu92b/cytech-nlp-transformers/blob/main/transformers/notebooks/TP1_bigramme.ipynb) |
| 2 | Auto-attention (moyenne causale → Q/K/V) | [TP2_auto_attention](notebooks/TP2_auto_attention.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Matthieu92b/cytech-nlp-transformers/blob/main/transformers/notebooks/TP2_auto_attention.ipynb) |
| 3 | LayerNorm, MLP, connexion résiduelle | [TP3_layernorm_mlp](notebooks/TP3_layernorm_mlp.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Matthieu92b/cytech-nlp-transformers/blob/main/transformers/notebooks/TP3_layernorm_mlp.ipynb) |
| 4 | Modèle complet, entraînement, génération | [TP4_mini_transformer](notebooks/TP4_mini_transformer.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Matthieu92b/cytech-nlp-transformers/blob/main/transformers/notebooks/TP4_mini_transformer.ipynb) |

Chaque notebook contient le code complété, les sorties d'exécution et les réponses aux questions d'interprétation.

## Résultats (TP 4, CPU, ~4 min)

- 0,21 M paramètres — `n_embd=64`, `n_head=4`, `n_layer=4`, `block_size=32`
- Loss : 4,40 → **train 1,66 / val 1,82** après 5 000 itérations (le bigramme du TP 1 plafonne à ~2,45)

## Architecture

```
idx (B,T) → token emb + position emb (B,T,64)
          → 4 × [ x + MHA(LN(x)) ; x + FFN(LN(x)) ]
          → LN → lm_head → logits (B,T,65) → softmax → multinomial
```

## Lancer en local

```bash
pip install -r ../requirements.txt
jupyter lab notebooks/
```

Le corpus (`input.txt`) est téléchargé automatiquement au premier lancement.

Inspiré de *Let's build GPT: from scratch* (A. Karpathy). TP réalisés dans le cadre du cours IA — CY Tech.
