# Fake News Classification: Shortcut Learning and Out-of-Distribution Generalization

This project is a two-notebook NLP analysis on the Kaggle [Fake and Real News dataset][https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset].  
The goal is not only to classify fake news, but to investigate whether high benchmark performance reflects real generalization or shortcut learning.

The project was developed as an individual machine learning/NLP portfolio project, with a focus on dataset bias, classical ML baselines, transformer fine-tuning, and robustness evaluation.

## Project Motivation

Fake news classification is often reported as a high-accuracy task on benchmark datasets. However, during exploratory analysis, this dataset shows recurring artifacts and source-specific patterns, especially Reuters-style formatting in real news and formatting or lexical patterns in fake news.

For this reason, the project asks a more critical question:

> Are models learning to detect fake news, or are they learning shortcuts specific to this dataset?

To answer this, the project compares:

1. Classical machine learning models trained on titles plus article bodies.
2. A fine-tuned DistilBERT model trained only on headlines.
3. A custom out-of-distribution evaluation set with real Reuters/AP News headlines and synthetic fake headlines.

## Repository Structure

```text
.
├── 01_eda_ml_models_comparison_portfolio.ipynb
├── 02_distilbert_generalization_portfolio_final.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

The original Kaggle CSV files are not included in the repository. To reproduce the notebooks, place them locally as:

```text
dataset/Fake.csv
dataset/True.csv
```

For the Colab version of the DistilBERT notebook, the files can also be loaded from Google Drive.

## Notebook 1/2: EDA and Classical ML Baselines

The first notebook builds a complete classical NLP pipeline and uses it to expose the limits of benchmark accuracy.

Main steps:

- Loaded and merged the Kaggle fake and real news files with binary labels: `0 = Fake`, `1 = Real`.
- Performed EDA on class distribution, punctuation patterns, word frequency, article length, and title/body structure.
- Identified recurring dataset artifacts, including Reuters-related source patterns and formatting traces.
- Applied preprocessing with NLTK lemmatization, stopword removal, TF-IDF vectorization, and targeted regex cleaning.
- Trained and tuned XGBoost, LinearSVC, and MLP models with 3-fold cross-validation.

### Classical ML Setup

The classical models were trained on combined title and article body text.

Key preprocessing and vectorization choices:

- `train_test_split(test_size=0.25, random_state=42, stratify=y)`
- WordNet lemmatization with NLTK
- English stopword removal
- TF-IDF vectorization with:
  - `max_features=1000`
  - `max_df=0.5`
  - `ngram_range=(1, 2)`
  - English stopwords

Hyperparameter tuning:

- `GridSearchCV`
- `cv=3`
- `scoring="accuracy"`

Best classical model configurations:

| Model | Best parameters |
|---|---|
| XGBoost | `learning_rate=0.2`, `max_depth=5`, `n_estimators=100`, `subsample=0.8` |
| LinearSVC | `C=1`, `penalty="l1"` |
| MLP | `hidden_layer_sizes=(32,)`, `alpha=0.01`, `learning_rate_init=0.01` |

### Classical ML Benchmark Results

All three models achieved very high in-distribution performance on the Kaggle test split.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| XGBoost | 98.45% | 98.32% | 98.42% | 98.37% |
| LinearSVC | 98.23% | 97.88% | 98.40% | 98.14% |
| MLP | 98.30% | 98.24% | 98.17% | 98.21% |

At first sight, these results look excellent. However, the EDA and feature analysis showed that the models were likely exploiting dataset-specific signals, such as Reuters-related tokens, formatting patterns, URLs, image references, and recurring political phrasing.

### Custom Robustness Test

To evaluate generalization, I created a small custom benchmark with 40 headlines:

- 20 synthetic fake headlines generated to imitate realistic news style.
- 20 real headlines from Reuters and AP News.
- Topics included both political news and climate/sustainability/energy news.

On this custom out-of-distribution set, the classical ML models dropped sharply:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| XGBoost | 0.350 | 0.000 | 0.000 | 0.000 |
| LinearSVC | 0.350 | 0.312 | 0.250 | 0.278 |
| MLP | 0.275 | 0.235 | 0.200 | 0.216 |

This confirmed the main finding of Notebook 1: high benchmark accuracy was not enough to demonstrate real generalization. The classical models learned useful patterns for the Kaggle distribution, but failed when evaluated on more diverse headlines.

## Notebook 2/2: DistilBERT Fine-Tuning and Generalization

The second notebook tests whether a transformer-based model can generalize better than TF-IDF-based classical ML models.

DistilBERT was chosen because it is lighter than BERT while still providing contextual token representations. To reduce computation and make training feasible on Google Colab, the model was trained only on news headlines.

### DistilBERT Setup

Base model:

- `distilbert-base-uncased`

Input setup:

- Headlines only
- `max_length=256`
- Hugging Face `Dataset` format
- PyTorch tensors with `input_ids`, `attention_mask`, and `label`

Train/test split:

- `test_size=0.25`
- `random_state=42`
- `stratify=y`

Regularization and fine-tuning choices:

- `dropout=0.3`
- `attention_dropout=0.3`
- `weight_decay=0.1`
- AdamW optimizer through `optim="adamw_torch"`
- `learning_rate=2e-5`
- `num_train_epochs=3`
- `per_device_train_batch_size=16`
- `per_device_eval_batch_size=16`
- cosine learning-rate scheduler
- `warmup_ratio=0.1`
- early stopping with `early_stopping_patience=1`
- `load_best_model_at_end=True`
- best model selected by `eval_loss`

The notebook also saves the fine-tuned model and tokenizer to Google Drive, so the GPU training step only needs to be run once. After that, the model can be reloaded quickly for evaluation or inference, including on CPU.

### DistilBERT Benchmark Results

On the Kaggle test split, DistilBERT achieved strong and balanced in-distribution performance:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Fake | 0.99 | 0.97 | 0.98 | 4477 |
| Real | 0.98 | 0.99 | 0.98 | 5300 |

Overall benchmark results:

| Metric | Score |
|---|---:|
| Accuracy | 0.98 |
| Macro F1 | 0.98 |
| Weighted F1 | 0.98 |

These results show that the model learned the in-distribution classification task effectively. The more important test, however, was the same custom benchmark used in Notebook 1.

### DistilBERT Custom Benchmark Results

On the custom out-of-distribution benchmark, DistilBERT achieved:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Fake | 0.75 | 0.60 | 0.67 | 20 |
| Real | 0.67 | 0.80 | 0.73 | 20 |

Overall custom results:

| Metric | Score |
|---|---:|
| Accuracy | 0.70 |
| Macro F1 | 0.70 |
| Weighted F1 | 0.70 |

Compared with the classical ML baselines, DistilBERT generalized much better on the custom set.

The strongest improvement was on real-news detection. The model achieved high recall for the real class and identified Reuters-style real headlines very reliably, even when they came from examples outside the original train/test split.

The model still showed limitations:

- It had more difficulty with AP News headlines, likely because their style and structure differ more from the Reuters-heavy patterns in the Kaggle dataset.
- It struggled with some synthetic fake headlines written in a formal journalistic style.
- Fake-news recall on the custom set was lower than real-news recall, suggesting that realistic formatting can still influence the model.

## Main Findings

1. Classical ML models reached about 98% benchmark performance, but failed on out-of-distribution examples.
2. EDA and feature analysis revealed strong shortcut-learning risks in the Kaggle dataset.
3. A custom benchmark was necessary to expose the gap between benchmark accuracy and real generalization.
4. DistilBERT, fine-tuned with regularization, generalized substantially better than TF-IDF-based models.
5. The transformer model was especially stronger at recognizing real Reuters-style headlines outside the original distribution, although AP News and realistic fake headlines remained challenging.

## Future Work: Notebook 3 with LLM Prompting

A natural next step would be to add a third notebook focused on prompting with large language models.

This would make the project more aligned with the current state of the art in NLP and allow a comparison between:

- Classical ML with TF-IDF features
- Fine-tuned transformer classification with DistilBERT
- Prompt-based classification using instruction-tuned LLMs

Possible experiments for Notebook 3:

- Zero-shot prompting
- Few-shot prompting
- Prompt templates with explicit reasoning criteria
- Comparison between direct label prediction and explanation-based prompting
- Evaluation on a larger and more diverse custom benchmark

The custom dataset should also be expanded with more sources, topics, and writing styles. This would make the robustness evaluation more reliable and help distinguish between factual reasoning, source-style recognition, and shortcut learning.

## Skills Demonstrated

- Exploratory data analysis for NLP datasets
- Text preprocessing with NLTK and regular expressions
- TF-IDF feature engineering
- Classical ML model training and hyperparameter tuning
- Grid search and cross-validation
- Model evaluation with precision, recall, F1-score, and confusion matrices
- Shortcut-learning and dataset-bias analysis
- Transformer fine-tuning with Hugging Face and PyTorch
- Regularization strategies for NLP fine-tuning
- Out-of-distribution evaluation design

## Requirements

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

The DistilBERT notebook can be run on Google Colab with GPU acceleration. Once the fine-tuned model is saved to Google Drive, later evaluation runs can reload the checkpoint without retraining.

## Dataset

Dataset: [Kaggle Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)  
Expected files:

```text
dataset/Fake.csv
dataset/True.csv
```

The dataset files are excluded from version control to keep the repository lightweight and to avoid redistributing external data.