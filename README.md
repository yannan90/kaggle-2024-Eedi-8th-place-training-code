# Introduction
This repo contains the training code for the 8th place solution of kaggle competition - 2024 Eedi Mining Misconceptions in Mathematics, which develops an NLP model driven by ML to accurately predict the affinity between misconceptions and incorrect answers (distractors) in multiple-choice questions. (https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/overview)

It emplements a iterative hard negative mining pipeline for a retriever-reranker scheme.

The basic inference scheme derived from this training pipeline is:
retrievers (4 Qwen2.5-14b models ensembled by weighted sum) → 40 misconceptions → reranker (a single Qwen2.5-32b model)→ 25 misconceptions.

For more detailed explanation and inference notebook can be found:
https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551412
