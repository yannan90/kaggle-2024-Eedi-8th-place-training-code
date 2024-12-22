# Introduction

This repository contains the training code for the 8th-place solution in the Kaggle competition **2024 Eedi Mining Misconceptions in Mathematics**. The competition focuses on developing an NLP-driven machine learning model to accurately predict the affinity between misconceptions and incorrect answers (distractors) in multiple-choice questions.  
(Competition overview: [Eedi Mining Misconceptions in Mathematics](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/overview))

This training code implements an **iterative hard negative mining pipeline** for a **retriever-reranker** framework.  

### Inference Workflow
The primary inference pipeline the training code is based upon is as follows:  
1. **Retriever**: An ensemble of four Qwen2.5-14B models combined via weighted sum generates a shortlist of **40 misconceptions**.  
2. **Reranker**: A single Qwen2.5-32B model refines the list to the final **25 misconceptions**.  

### Additional Resources
For a detailed explanation and the inference notebook, please refer to the competition discussion:  
[Discussion Link](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551412)
