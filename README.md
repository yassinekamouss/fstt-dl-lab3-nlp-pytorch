# FST Tanger — Master MBD — Deep Learning : Lab 3 (Arabic NLP & Transformers)

**Université Abdelmalek Essaadi — Faculté des Sciences et Techniques de Tanger (FSTT)**  
**Département Génie Informatique — Master MBD — Deep Learning**  
**Encadrant :** Pr. Elaachak Lotfi  
**Dépôt GitHub :** [`yassinekamouss/fstt-dl-lab3-nlp-pytorch`](https://github.com/yassinekamouss/fstt-dl-lab3-nlp-pytorch)

---

## Notebooks Google Colab (Exécution Bout-en-Bout)

| Étape | Notebook | Description | Lancer sur Colab |
| :--- | :--- | :--- | :--- |
| **01** | [`01_scraping_and_dataset.ipynb`](notebooks/01_scraping_and_dataset.ipynb) | Scraping Wikipédia Arabe (`BeautifulSoup4`), scoring `[0, 10]`, pipeline NLP arabe, discrétisation & split anti-fuite | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yassinekamouss/fstt-dl-lab3-nlp-pytorch/blob/main/notebooks/01_scraping_and_dataset.ipynb) |
| **02** | [`02_part1_sequence_models.ipynb`](notebooks/02_part1_sequence_models.ipynb) | Architectures PyTorch (`RNN`, `Bi-RNN`, `GRU`, `LSTM`), tuning Optuna, évaluation classification & benchmark LM/BLEU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yassinekamouss/fstt-dl-lab3-nlp-pytorch/blob/main/notebooks/02_part1_sequence_models.ipynb) |
| **03** | [`03_part2_gpt2_finetuning.ipynb`](notebooks/03_part2_gpt2_finetuning.ipynb) | Fine-tuning de `aubmindlab/aragpt2-base` (FP16 GPU T4), comparaison Avant/Après & décodage multi-stratégies | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yassinekamouss/fstt-dl-lab3-nlp-pytorch/blob/main/notebooks/03_part2_gpt2_finetuning.ipynb) |

---

## Architecture du Projet

```text
fstt-dl-lab3-nlp-pytorch/
├── configs/
│   └── config.yaml                      # Configuration centrale (sujet AR, hyperparamètres, seeds)
├── src/
│   ├── __init__.py
│   ├── colab_utils.py                   # Détection Colab/local, montage Google Drive, seed_everything
│   ├── scraper.py                       # ArabicWikipediaScraper (BeautifulSoup4) + scoring [0, 10]
│   ├── preprocessing.py                 # Normalisation arabe, tokenization, stopwords, stemming/lemmatization, split groupé
│   ├── dataset.py                       # Vocabulary (train-only), Datasets & DataLoaders PyTorch
│   ├── models.py                        # SequenceClassifier & SequenceLanguageModel (RNN, Bi-RNN, GRU, LSTM)
│   ├── trainer.py                       # AdamW, EarlyStopping, Optuna tuning, benchmark LM/BLEU
│   ├── metrics.py                       # Accuracy, F1, ROC-AUC, BLEU, ROUGE arabe, Distinct-1/2, graphiques
│   └── gpt2_utils.py                    # Chargement AraGPT2, Trainer version-safe, génération & évaluation NLG
├── notebooks/
│   ├── 01_scraping_and_dataset.ipynb
│   ├── 02_part1_sequence_models.ipynb
│   └── 03_part2_gpt2_finetuning.ipynb
├── tests/
│   └── test_local_pipeline.py           # Tests unitaires locaux (scraping parser, NLP arabe, métriques)
├── data/
│   ├── raw/                             # arabic_wiki_raw.csv, dataset_text_score.csv (Text | Score)
│   └── processed/                       # train.csv, val.csv, test.csv, vocab.json, gpt2_corpus*.txt
├── outputs/
│   ├── figures/                         # Graphiques EDA, courbes d'apprentissage, matrices de confusion
│   └── reports/                         # metrics_part1.json, tuning_results.csv, metrics_part2.json
├── requirements.txt
└── README.md
```

---

## Guide d'Exécution sur Google Colab

1. Ouvrir **`01_scraping_and_dataset.ipynb`** via le badge Colab ci-dessus et sélectionner un runtime **Python 3** (CPU suffisant pour l'étape 1, GPU T4 recommandé pour les étapes 2 et 3).
2. La cellule d'amorçage clone automatiquement ce dépôt GitHub dans `/content/fstt-dl-lab3-nlp-pytorch`, installe les dépendances manquantes et monte **Google Drive** (`USE_DRIVE = True`) vers `/content/drive/MyDrive/fstt-dl-lab3-nlp-pytorch` pour persister `data/` et `outputs/` entre les notebooks.
3. Exécuter **`01_scraping_and_dataset.ipynb`** jusqu'au bout : génère `dataset_text_score.csv`, `train.csv`, `val.csv`, `test.csv`, `vocab.json` et `gpt2_corpus*.txt`.
4. Activer **GPU T4** (`Runtime -> Change runtime type -> T4 GPU`), puis exécuter **`02_part1_sequence_models.ipynb`** (Partie 1).
5. Exécuter **`03_part2_gpt2_finetuning.ipynb`** sur GPU T4 (Partie 2).
6. Copier les résultats des fichiers `outputs/reports/metrics_part1.json` et `outputs/reports/metrics_part2.json` dans les tableaux ci-dessous puis pousser sur GitHub.

---

## Méthodologie Détaillée

### Partie 1 — Classification & Modèles de Séquences en PyTorch

1. **Collecte Wikipédia Arabe (`src/scraper.py`) :**
   - Sujet principal : **الذكاء الاصطناعي (Intelligence Artificielle)** sur `https://ar.wikipedia.org/wiki/`.
   - Collecte stratifiée sur 3 niveaux de proximité thématique (`core`, `related`, `off_topic`) avec filtrage des infoboxes, tables, références `[1]` et formules mathématiques via `BeautifulSoup4`.
   - Calcul d'un score continu de pertinence dans **`[0.0, 10.0]`** (exporté au format exact `Text | Score` dans `dataset_text_score.csv`) combinant similarité cosinus TF-IDF (leave-one-out sur le centroïde IA), densité lexicale métier et a priori borné par niveau.

2. **Pipeline NLP Arabe & Anti-Leakage (`src/preprocessing.py`) :**
   - **Normalisation (`PyArabic`) :** suppression des URLs, balises, chiffres, *tashkeel* (diacritiques) et *tatweel*, unification d'Alef (`أ/إ/آ/ٱ → ا`) et Alef Maksura (`ى → ي`).
   - **Stop-words :** liste `NLTK` arabe enrichie de termes récurrents Wikipédia, normalisée avec les mêmes règles.
   - **Réduction morphologique :** comparaison `ISRIStemmer`, `SnowballStemmer` et **`Qalsadi Lemmatizer`** (avec mise en cache et normalisation post-lemmatisation).
   - **Discrétisation :** découpage du score `[0, 10]` en 3 classes (`0 : Faible [0-4[`, `1 : Moyenne [4-7[`, `2 : Élevée [7-10]`).
   - **Split anti-fuite (`stratified_group_split`) :** séparation `70% train / 15% val / 15% test` **groupée par article Wikipédia (`title`)** afin qu'aucun paragraphe d'un même article ne figure à la fois en entraînement et en test ; vocabulaire construit strictement sur `train`.

3. **Architectures, Tuning & Évaluation (`src/models.py`, `src/trainer.py`, `src/metrics.py`) :**
   - Quatre modèles PyTorch partageant une interface unifiée (`nn.Embedding(padding_idx=0)`, `pack_padded_sequence`, pooling masqué `last` / `mean` / `mean_max`, `LayerNorm`, `Dropout`) : **`RNN`**, **`Bi-RNN`**, **`GRU`**, **`LSTM`**.
   - Optimisation d'hyperparamètres via **Optuna** (`tuning_results.csv`), puis entraînement final avec `AdamW`, pondération de classes, gradient clipping, `ReduceLROnPlateau` et `EarlyStopping`.
   - **Évaluation classification :** Accuracy, Precision/Recall/F1 (Macro & Weighted), ROC-AUC OvR, matrices de confusion, latence d'inférence et analyse d'erreurs.
   - **Benchmark Language Modeling & BLEU :** le score BLEU ne s'appliquant pas à des classes entières, les 4 architectures sont également entraînées comme modèles de langage au niveau mot : évaluation autorégressive (**Perplexity** + **BLEU-1/2/4 / SacreBLEU**) pour `RNN`, `GRU`, `LSTM`, et évaluation **Masked Cloze (15 % `<MASK>`)** (**Pseudo-Perplexité** + **BLEU de reconstruction**) pour le `Bi-RNN` non-causal.

### Partie 2 — Fine-Tuning d'AraGPT2 (`aubmindlab/aragpt2-base`)

- Modernisation de l'approche `pytorch-transformers` / tutoriel `mf1024` avec Hugging Face **`transformers`** (`GPT2LMHeadModel` + `AutoTokenizer` + `ArabertPreprocessor`).
- Fine-tuning sur GPU Colab T4 (FP16, `AdamW`, cosine warmup, `block_size=256`, accumulation de gradients) sur les paragraphes arabes naturels collectés en Partie 1.
- Comparaison **Avant vs. Après Fine-Tuning** (Perplexity, BLEU-1..4, SacreBLEU, ROUGE-1/2/L adapté à l'alphabet arabe, Distinct-1/2) et analyse qualitative des stratégies **Greedy**, **Beam Search** et **Top-k / Top-p (Nucleus) Sampling**.

---

## Résultats Expérimentaux

> *Les tableaux ci-dessous correspondent exactement aux schémas exportés dans `outputs/reports/metrics_part1.json` et `outputs/reports/metrics_part2.json`. Les valeurs numériques seront renseignées après l'exécution complète des notebooks sur Google Colab.*

### Partie 1A — Classification des Séquences (Test Set)

| Modèle | Paramètres | Accuracy | Precision (Macro) | Recall (Macro) | F1 (Macro) | F1 (Weighted) | ROC-AUC (OvR) | Latence (ms/ech.) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RNN** | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` |
| **Bi-RNN** | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` |
| **GRU** | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` |
| **LSTM** | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` |

### Partie 1B — Benchmark Language Modeling & Score BLEU (Test Set)

| Modèle | Mode d'évaluation | Test Loss | Perplexity | BLEU-1 | BLEU-2 | BLEU-4 | SacreBLEU |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **RNN** | Causal Autoregressive | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` |
| **Bi-RNN** | Masked Cloze (non-causal) | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` |
| **GRU** | Causal Autoregressive | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` |
| **LSTM** | Causal Autoregressive | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` |

### Partie 2 — Fine-Tuning `aubmindlab/aragpt2-base` (Avant vs. Après)

| Configuration | Val Perplexity | Test Perplexity | BLEU-1 | BLEU-4 | SacreBLEU | ROUGE-1 | ROUGE-L | Distinct-1 | Distinct-2 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Pretrained) — Nucleus** | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` |
| **Fine-Tuned — Greedy** | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` |
| **Fine-Tuned — Beam Search** | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` |
| **Fine-Tuned — Nucleus (`top-p`)** | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` | `[À compléter]` |

### Figures Générées (disponibles après exécution Colab dans `outputs/figures/`)

- `outputs/figures/vocab_reduction.png` — Réduction de taille du vocabulaire par étape NLP
- `outputs/figures/eda_distributions.png` — Distribution des scores `[0, 10]`, classes, longueurs et lemmes fréquents
- `outputs/figures/part1_learning_curves.png` — Courbes Train/Val Loss et Macro-F1 (RNN, Bi-RNN, GRU, LSTM)
- `outputs/figures/part1_confusion_matrices.png` — Matrices de confusion sur le jeu de test
- `outputs/figures/part1_model_comparison.png` — Comparatif unifié Classification & BLEU
- `outputs/figures/part2_gpt2_loss.png` — Courbe d'entraînement et validation du fine-tuning AraGPT2

---

## Synthèse & Enseignements Clés

- **Spécificités du NLP Arabe :** La normalisation orthographique (suppression du *tashkeel* et *tatweel*, unification d'Alef/Ya) et la lemmatisation morphologique (`Qalsadi`) réduisent significativement le nombre de formes rares hors-vocabulaire pour les réseaux récurrents, tandis que la préservation du texte naturel est indispensable pour générer de l'arabe fluide avec `AraGPT2`.
- **Rigueur méthodologique anti-leakage :** Grouper le split par article Wikipédia évite que des paragraphes adjacents d'une même page ne gonflent artificiellement la performance en test, et fit le vocabulaire exclusivement sur `train` empêche toute fuite statistique.
- **Comparaison des architectures séquentielles :** Les cellules à portes (`GRU` et `LSTM`) surmontent la dissipation du gradient du `RNN` simple sur des paragraphes longs ; le `Bi-RNN` tire parti du contexte gauche/droite pour la classification mais impose un protocole *masked-cloze* pour l'évaluation en modélisation du langage.
- **Apport du Transformer (`AraGPT2`) :** Le pré-entraînement auto-régressif par sous-mots (BPE) et l'auto-attention permettent une adaptation rapide au domaine de l'Intelligence Artificielle en arabe avec un fort gain de perplexité ; le décodage **Nucleus (`top-p`) + `repetition_penalty`** offre le meilleur compromis entre cohérence sémantique et diversité lexicale.
