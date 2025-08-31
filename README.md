# Insurance-Claim-Classification-with-DistilBERT

This project trains a machine learning model to classify insurance claims into three categories: **auto, home, and health**.  
It uses **DistilBERT** from Hugging Face Transformers with **PyTorch** for training and evaluation.  

## Features
- Preprocess claim descriptions with **tokenization**  
- Train a **DistilBERT sequence classification model**  
- Evaluate with a **confusion matrix**  
- Visualize claims with **word clouds**  
- Predict categories for new claims  

## Setup
Install dependencies (skip if using Google Colab):
```bash
pip install wordcloud pandas torch transformers scikit-learn matplotlib numpy accelerate
