# Introduction

This repository contains the training code and quantization code (AWQ) for the 8th-place solution in the Kaggle competition **2024 Eedi Mining Misconceptions in Mathematics**. The competition focuses on developing an NLP-driven machine learning model to accurately predict the affinity between misconceptions and incorrect answers (distractors) in multiple-choice questions. <br>
(Competition overview: [Eedi Mining Misconceptions in Mathematics](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/overview))

This code implements a **hard negative mining pipeline** for a **retriever-reranker** framework, and evetually perform **quantization (AWQ 4bit format)** to the reranker, for inference on limited GPU resources. 

### Training Flow
**Retriever Training**:  
  1. The retriever generates hard negative training data.  
  2. Train the retriever.  
  3. Repeat the previous steps iteratively.  

**Reranker Training**:  
  1. The retriever generates hard negative training data.  
  2. Train the reranker.  
  3. Repeat the previous steps iteratively.  
  4. Apply reranker quantization.

### How to use the code
The main implementation is organized within the Jupyter Notebook file **`Eedi_run.ipynb`**. Each code block in the notebook corresponds to a specific functionality within the overall pipeline, invoking relevant Python files to handle the underlying processes.<br>
The hard negative mining & retriever/reranker trainings should be performed multiple time (iteratively) to improve final inference performance.

### Additional Resources
For a detailed explanation and the inference notebook, please refer to the competition solution write-up: [Eedi 8th Place Solution Write-up](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551412)

