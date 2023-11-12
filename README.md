# NLP_project 
### Final NLP project EPF fifth year
##### Multilabel classification problem 

## Table of Contents
- [Table of Contents](#-table-of-contents)
- [Overview](#-overview)
- [Multilabel Classification](#-multilabel-classification)
- [Features](#-features)
- [Repository Structure](#-repository-structure)
- [Project review](#-project-review)

---

## Overview
This project is a final assessment on NLP at EPF engineering school, deep learning major. This was a first autonomous exercise of implementing basic NLP models, as well as training a simple sequence model using pytorch neural network module. The training was done on a dataset of news articles containing the title and topic. 

---

## Multilabel Classification
The objective here is to train a model that can predict with accuracy the topic of an article based on it's title. We have 8 differents topic from the dataset.

---

## Features 
This repository contains the following:
- `Exploratory Data Analysis`
- `Preprocessor function and baseline model class`
- `Basic model training and tuning`
- `Simple deep learning model`

---

## Repository Structure

```
└── NLP_project/
    ├── EDA.ipynb
    ├── Sequence_model.ipynp
    ├── Simple_machine_learning.ipynb
    ├── labelled_newscatcher_dataset.csv
    ├── model_class.py
    ├── preprocess.py

```

---

## Project review
This project was a first step in autonomy in a NLP machine learning problem. This was also personaly a first in using pytorch in an unguided exercise. I also decided to take one a multilabel classifying problem, which turned out to be quite challenging. I was able to reach a decent accuracy on basic models without much tuning thanks to the fact that the training dataset was quite big with a close to perfectly uniform label distribution. <br>
However pushing the performance was much more difficult. It took too much time to test different types of models. In future project i would probably makes some changes on the way I defined my model baseline and preprocessing.<br>
As for the Sequence model, the result is far from perfect. Altough I think my understanding of the pytorch neural network module has improved a lot, it was very time confusing for me to arrived at this point. I would have liked to test out different types of model such as RNN for example, and optimize more parameter to obtain a better model. <br>
It was still a very usefull project, as I learned a lot overall, and it also highlighted the aspects that I needed to work on.

---

[**Return**](#Top)

---

