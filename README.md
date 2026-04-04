# NLP Project on MIMIC-CXR: NER, Classification, and Generative Modeling

This repository contains our course project work on clinical NLP using radiology reports from a sampled MIMIC-CXR-style dataset. The project began with **Named Entity Recognition (NER)** using spaCy and later expanded to **document-level classification** with encoder models, **classical machine learning baselines**, and **decoder / LLM-style generative approaches** for clinical label prediction.

The main goal of the project is to explore how different NLP paradigms can extract structured information from chest X-ray reports, ranging from entity recognition at sentence level to multi-label prediction at report level.

## Project Scope

Clinical reports contain rich but highly compressed information, including findings, anatomical references, relations, uncertainty, and negation. In this project, we investigated multiple ways of modeling that information:

1. **spaCy-based NER**
   - Sentence-level annotation and entity extraction
   - Conversion of manually annotated sentence chunks into spaCy-compatible training data
   - Training and evaluation of a baseline NER model

2. **Encoder-based models**
   - BERT / ClinicalBERT-style models for report-level prediction
   - Multi-label or multi-class classification of radiology findings

3. **Classical baselines**
   - TF-IDF + LinearSVC / Logistic Regression
   - Multi-label classification using one-vs-rest strategies

4. **Decoder / LLM-style models**
   - Prompt-based and generative prediction of clinical labels
   - Experiments that reformulate classification as text generation

## Repository Structure

### Notebooks

- `Data_investigation.ipynb`  
  Exploratory data analysis of the sampled dataset.

- `ner_spacy.ipynb`  
  spaCy-based NER training and evaluation. This is the main notebook for the **entity recognition** part of the project.

- `Encoder_model.ipynb`  
  Experiments with encoder-based deep learning models for report-level classification.

- `classical-and-decoder.ipynb`  
  Classical ML baselines and decoder/generative experiments for clinical label prediction.

- `llm_ner_classifier_notebook.ipynb`  
  Experiments with LLM-based prompting and classification / entity-related tasks.

### Python Scripts

- `Training_BERT.py`  
  Training pipeline for a BERT-based model.

- `retrain_all_layers_Bert.py`  
  Variant of BERT training with all layers unfrozen.

- `add_sentence_column.py`  
  Utility script to enrich the dataset with sentence-level information.

- `split_sentences_to_txt.py`  
  Script for splitting reports into sentence chunks for annotation.

### spaCy Artifacts

- `train.spacy`  
  spaCy training data / config artifact.

- `eval.spacy`  
  spaCy evaluation artifact.

- `spacy_standard_predictions.json`  
  Predictions from a baseline spaCy model.

- `ner_model/`  
  Directory containing saved NER model outputs.

### Data Files

- `sampled_1000_data.csv`  
  Main sampled dataset used in the experiments.

- `sampled_1000_data_with_sentence.csv`  
  Dataset version with sentence segmentation.

### Annotation Files

#### Sentence text files
- `sentence_chunk_Dawood.txt`
- `sentence_chunk_Lucas.txt`
- `sentence_chunk_Zahra.txt`
- `sentence_chunk_haozhe.txt`
- `sentence_chunk_combined.txt`

#### Annotated JSON files
- `sentence_chunk_Dawood_Annotated.json`
- `sentence_chunk_Lucas_Annotated.json`
- `sentence_chunk_Zahra_Annotated.json`
- `sentence_chunk_haozhe_Annotated.json`
- `sentence_chunk_combined_Annotated.json`

These files were used in the NER stage of the project to create manually annotated sentence-level data.

## Task Breakdown

### 1. Named Entity Recognition (spaCy stage)

The first stage of the project focused on identifying relevant clinical entities from sentence chunks. This included:

- preparing sentence-level text for annotation
- manually annotating entities in JSON files
- training a spaCy NER pipeline
- evaluating baseline predictions

This stage answers the question:  
**“Which important clinical entities appear in the text?”**

### 2. Report-Level Classification (encoder and classical stage)

The next stage moved from span extraction to report-level prediction. Instead of detecting exact entity spans, these models predict whether a report expresses certain clinical findings or labels.

Typical pipeline:
- extract report text
- preprocess / clean text
- build label vectors
- train either:
  - TF-IDF + SVM / Logistic Regression
  - BERT / ClinicalBERT-style encoder model

This stage answers the question:  
**“Which clinical labels or findings are present in the report?”**

### 3. Generative Modeling (decoder / LLM stage)

In the final stage, we explored generative approaches where the model predicts labels as text rather than only as classifier outputs.

Instead of:
- returning a binary or multi-class vector

the model is prompted to generate something like:
- a list of findings
- a structured response
- a label sequence

This stage answers the question:  
**“Can a language model generate the relevant structured output directly from the report?”**

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Roypic/NLP_project_mimiccxr.git
cd NLP_project_mimiccxr
