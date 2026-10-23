# NLP sur Tiny Shakespeare — TP 2 et TP 3

Cours *NLP et Transformers* — CY Tech 2026-2027.

**Tâche commune** : attribution de réplique — à partir du texte d'une réplique, prédire **quel personnage parle** parmi 7 personnages (un par pièce) : Gloucester, Duke Vincentio, Menenius, Romeo, Petruchio, Leontes, King Richard II.
Les labels viennent du corpus lui-même (`NOM:` avant chaque réplique). 1 066 répliques, split stratifié 70/15/15, **même test set** dans les deux TP.

| TP | Thème | Notebook | |
|---|---|---|---|
| 2 | Deep Learning : pipeline complet, ML vs DL | [TP2_deep_learning](notebooks/TP2_deep_learning.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Matthieu92b/cytech-nlp-transformers/blob/main/nlp/notebooks/TP2_deep_learning.ipynb) |
| 3 | Prompting : la même tâche avec un LLM | [TP3_prompting](notebooks/TP3_prompting.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Matthieu92b/cytech-nlp-transformers/blob/main/nlp/notebooks/TP3_prompting.ipynb) |

## TP 2 — ML vs DL

1. **Prétraitement** : parsing des répliques, filtrage (< 3 mots), déduplication avant split, normalisation (stopwords conservés : *thou/thee* sont des marqueurs de style)
2. **Représentations** : BoW, TF-IDF mots (1-2 grammes), TF-IDF caractères (2-5 grammes), embeddings appris
3. **Modèles** : Naive Bayes, régression logistique, SVM linéaire · MLP sur TF-IDF, fastText, BiLSTM, CNN 1D (PyTorch, loss pondérée, early stopping)
4. **Évaluation** : accuracy, précision/rappel/F1 macro, balanced accuracy, matrices de confusion, mots les plus discriminants, analyse d'erreurs par longueur

| Test (160 répliques) | F1 macro |
|---|---|
| Baseline classe majoritaire | 0,05 |
| SVM linéaire + TF-IDF caractères | 0,48 |
| MLP sur TF-IDF | 0,50 |

Le DL n'apporte rien de significatif ici : trop peu de données pour apprendre des embeddings (BiLSTM ≈ 0,35 en val), le signal est lexical.

## TP 3 — Prompting

1. **Prompting** : rôles system/user, zero-shot, zero-shot + contexte (pièces), few-shot (exemples du train), indice avant réponse ; sortie JSON
2. **Modèle** : `Qwen2.5-1.5B-Instruct` via 🤗 transformers (ou n'importe quelle API compatible OpenAI)
3. **Sortie générée** : parsing robuste, taux de sorties invalides, effet température / top-p, génération « à la manière de » évaluée par un classifieur
4. **Évaluation** : mêmes métriques que le TP 2, biais de prédiction, latence, tableau classique vs LLM

⚠️ À exécuter sur **Colab GPU (T4)**, ~10-15 min.

## Lancer en local

```bash
pip install -r ../requirements.txt
jupyter lab notebooks/
```
Le corpus est téléchargé automatiquement au premier lancement.
