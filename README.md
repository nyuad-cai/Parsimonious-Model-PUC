# A parsimonious prediction model for positive urine cultures in the outpatient setting

Table of contents
=================

<!--ts-->
   * [About the repository](#About-the-repository)
      * [Abstract](#Abstract)
      * [Overview](#Overview)
   * [How to run the code](#How-to-run-the-code)
      * [Prerequisites](#Requirements)
      * [Data Structure](#Data-Structure)
      * [Code](#Code)
      * [Pretrained models](#Pretrained_models)
   * [Citation](#Citation)
   
<!--te-->

About the repository
============
This repository is an implementation of the model developed for predicting positive urine cultures as described in our paper  "A parsimonious prediction model for positive urine cultures in the outpatient setting", developed in a collaboration between Cleaveland Clinic Abu Dhabi and New York University Abu Dhabi. This imlementation includes scripts and jupyter notebooks that allow for training and evaluation of the models on different datasets, as well as the pre-trained models. 

## Abstract
 Urine culture is often considered the gold standard for detecting the presence of an infection. Since culture is expensive and often requires 24-48 hours, clinicians may prescribe medication based on presenting symptoms and urine dipstick test, which is considerably cheaper than culture and provides instant results. Despite its ease of use, urine dipstick test is not entirely reliable and increases the workload on urine culture. In this paper, we use a real-world dataset consisting of 17,579 outpatient encounters collected between 2015 and 2021 at a large multi-specialty hospital in Abu Dhabi, United Arab Emirates. We develop and evaluate a simple parsimonious prediction model for positive urine cultures based on a minimal input set of ten features selected from the patient's presenting vital signs, history, and dipstick results.  In a test set of 5,236 encounters, the parsimonious model achieves an area under the receiver operating characteristic curve (AUROC) of 0.813 (95\% CI: 0.797-0.830) for predicting a bacterial count $\geq10^5$ CFU/ml, which is the typical cut-off used to assess for urinary tract infections. It outperforms a model that uses dipstick features only that achieves an AUROC of 0.776 (95\% CI: 0.758-0.796). In our retrospective analysis, we find that the model can potentially lead to an increase of 4.6\% in specificity and 30.4\% in positive predictive value at the same level of clinician sensitivity in prescribing antibiotics. Our proposed model can be easily deployed at point-of-care, highlighting its value in aiding clinical decision-making.

## Overview
![fullModel](https://github.com/nyuad-cai/Parsimonious-Model-PUC/blob/main/full_figure.png)

**Overview of the proposed model**(a) We illustrate an example of an outpatient encounter. Upon evaluating the patient's symptoms, a clinician performs a urine dipstick while they wait for the urine culture results. Our proposed parsimonious model can make a prediction ahead of the culture results to inform the decision-making process. (b) We summarize the model development process. We first extract the features, preprocess the data, and then develop three prediction models with all the features (original model), with the top ten predictive features (parsimonious model), and with the dipstick features only (dipstick model).

# How to run the code

## Prerequisites
- numpy==1.22.4
- pandas==1.4.1
- shap==0.41.0
- matplotlib==3.1.2

## Data Structure
For data confidentiality and privacy restrictions, we are unable to release the dataset samples used to train and evaluate the models. However, we provide empty dataframes with the preprocessed dataset structure with the required input format used to train the "original", "parsimonious", and "dipstick" models referred to in our paper. The dataset columns and their corrosponsing types are shown in the [figure](#Overview) above.

## The Code 
We provide the source code used to reproduce the plots and results shown in our paper. Specifically, `parsimonious_model.ipynb`includes the training and evaluation of the three evaluated model and shap analysis to select the feature set for the parsimonious model. 

## Pretrained models
To allow for the reproducability of the results on external datasets, we exported our pre-trained models using the two bacterial thresholds, 10^5 CFU/m and 10^4 CFU/ml, respectivly. The trained models can be accessed in the [trained_models](https://github.com/nyuad-cai/Parsimonious-Model-PUC/tree/main/trained_models) folder.

## Citation

If you found this code useful, please cite our paper: ...... TO ADD

