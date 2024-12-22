# Introduction

This repository contains the training code for the 8th-place solution in the Kaggle competition **2024 Eedi Mining Misconceptions in Mathematics**. The competition focuses on developing an NLP-driven machine learning model to accurately predict the affinity between misconceptions and incorrect answers (distractors) in multiple-choice questions.  
(Competition overview: [Eedi Mining Misconceptions in Mathematics](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/overview))

This training code implements an **iterative hard negative mining pipeline** for a **retriever-reranker** framework.  

### Inference Workflow
The primary inference pipeline the training code is based upon is as follows:  
retrievers (4 Qwen2.5-14b models ensembled) → 40 misconceptions → reranker (single Qwen2.5-32b model)→ 25 misconceptions

### Additional Resources
For a detailed explanation and the inference notebook, please refer to the competition solution write-up:  
[Solution Write-up](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551412)
