# Named Entity Recognition Project

This repository contains code, data, and experiments for a Named Entity Recognition (NER) project developed for an NLP course. It includes spaCy-based models, BERT-based models, and exploratory experiments with large language models.

## Files and Directories

### Notebooks

* `Data_investigation.ipynb`: Exploratory analysis of the dataset.
* `Encoder_model.ipynb`: Experiments with encoder/BERT-based NER models.
* `ner_spacy.ipynb`: Training and evaluation of spaCy NER models.
* `llm_ner_classifier_notebook.ipynb`: NER or entity classification using large language models.
* `classical_and_decoder.ipynb`: Experiments with classical baseline (LinearSVC) and Decoder models.

### Python Scripts

* `Training_BERT.py`: Trains a BERT-based NER model.
* `retrain_all_layers_Bert.py`: Retrains a BERT model with all layers unfrozen.
* `add_sentence_column.py`: Adds sentence-level information to a dataset.
* `split_sentences_to_txt.py`: Splits raw text into sentence chunks for annotation.

### spaCy Configuration and Output

* `train.spacy`: spaCy training configuration.
* `eval.spacy`: spaCy evaluation configuration.
* `spacy_standard_predictions.json`: Predictions from a baseline spaCy model.

### Data Files

* `sampled_1000_data.csv`: Sampled dataset used for experiments.
* `sampled_1000_data_with_sentence.csv`: Dataset with sentence segmentation added.

### Sentence Chunks (Text)

* `sentence_chunk_Dawood.txt`
* `sentence_chunk_Lucas.txt`
* `sentence_chunk_Zahra.txt`
* `sentence_chunk_haozhe.txt`
* `sentence_chunk_combined.txt`

These files contain sentence-level text used for annotation and training.

### Annotated Data (JSON)

* `sentence_chunk_Dawood_Annotated.json`
* `sentence_chunk_Lucas_Annotated.json`
* `sentence_chunk_Zahra_Annotated.json`
* `sentence_chunk_haozhe_Annotated.json`
* `sentence_chunk_combined_Annotated.json`

These files contain annotated NER labels corresponding to the sentence chunk text files.

### Other

* `ner_model/`: Directory for storing trained NER models.
* `NLP_project_2.pdf`: Project description or final report.

## Contributors

Haozhe Luo,
Daunibe,
Lucas Vizzotto de Castro,
Bazgh

