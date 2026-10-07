# Grammar Scoring Engine for Spoken Audio

An AI-powered system for automatically predicting the grammar quality of spoken English from short speech recordings.

The project was developed for the **SHL Hiring Assessment 2026** and predicts a continuous grammar score from spoken audio using a combination of **Automatic Speech Recognition (ASR), acoustic features, linguistic features, transformer embeddings, and ensemble machine learning**.

---

## Overview

Evaluating spoken grammar manually can be time-consuming and subjective. This project attempts to automate the process by analyzing both:

- **What the speaker says**
- **How the speaker speaks**

Each input is a 45–60 second WAV speech recording, and the system predicts a grammar score on a **0–5 scale**.

The model is evaluated using:

- **RMSE (Root Mean Squared Error)**
- **Pearson Correlation**

---

## Pipeline

```text
              Speech Audio
                   |
                   v
          Audio Preprocessing
                   |
          +--------+--------+
          |                 |
          v                 v
       Whisper ASR      Audio Analysis
          |                 |
          v                 v
      Transcript      Acoustic Features
          |                 |
          +--------+--------+
                   |
                   v
           Feature Engineering
                   |
      +------------+-------------+
      |            |             |
      v            v             v
   Text       RoBERTa        wav2vec2
 Features     Embeddings     Embeddings
      |            |             |
      +------------+-------------+
                   |
                   v
             ML Models
                   |
      +------------+-------------+
      |            |             |
      v            v             v
 ExtraTrees     XGBoost       LightGBM
                   |
             Ridge Models
                   |
                   v
          Non-Negative Stacking
                   |
                   v
          Grammar Score (0–5)
