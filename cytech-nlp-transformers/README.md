# NLP & Transformers — TP

TP du cours *NLP et Transformers* — CY Tech, ING 2026-2027. Tout est en PyTorch sur le corpus **Tiny Shakespeare**.

## Structure

```
nlp/            séances 1-2 : pipeline NLP, ML vs DL, prompting
transformers/   séances 3-4 : mini-Transformer (GPT caractère) from scratch
```

## NLP — [détails](nlp/README.md)

Tâche : deviner **quel personnage parle** (7 classes) à partir d'une réplique.

| TP | Thème | |
|---|---|---|
| 2 | Deep Learning : ML (NB, LogReg, SVM) vs DL (MLP, fastText, BiLSTM, CNN) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Matthieu92b/cytech-nlp-transformers/blob/main/nlp/notebooks/TP2_deep_learning.ipynb) |
| 3 | Prompting : la même tâche avec un LLM (zero/few-shot, JSON, température) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Matthieu92b/cytech-nlp-transformers/blob/main/nlp/notebooks/TP3_prompting.ipynb) |

## Transformers — [détails](transformers/README.md)

| TP | Thème | |
|---|---|---|
| 1 | Données, tokenisation, modèle bigramme | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Matthieu92b/cytech-nlp-transformers/blob/main/transformers/notebooks/TP1_bigramme.ipynb) |
| 2 | Auto-attention (moyenne causale → Q/K/V) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Matthieu92b/cytech-nlp-transformers/blob/main/transformers/notebooks/TP2_auto_attention.ipynb) |
| 3 | LayerNorm, MLP, connexion résiduelle | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Matthieu92b/cytech-nlp-transformers/blob/main/transformers/notebooks/TP3_layernorm_mlp.ipynb) |
| 4 | Modèle complet, entraînement, génération | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Matthieu92b/cytech-nlp-transformers/blob/main/transformers/notebooks/TP4_mini_transformer.ipynb) |

## Résultats clés

- **NLP TP 2** : F1 macro test 0,48 (SVM TF-IDF caractères) / 0,50 (MLP TF-IDF) contre 0,05 pour la baseline
- **Transformers TP 4** : 0,21 M paramètres, loss val 1,82 (le bigramme plafonne à ~2,45)

## Lancer en local

```bash
pip install -r requirements.txt
jupyter lab
```
Le corpus est téléchargé automatiquement au premier lancement de chaque notebook. Le TP 3 NLP nécessite un GPU (Colab T4).
