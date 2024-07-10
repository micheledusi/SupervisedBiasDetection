# Weakly-Supervised Bias Detection
This repository contains the code to perform the experimentation of Bias Detection on different pre-trained Transformers-based Language Models. The code includes multiple experiments, investigating different features of the problem and implementing various techniques. Further information can be found in our paper *"Discrimination Bias Detection through Categorical Association in Pre-trained Language Models"* by Dusi, Arici, Gerevini, Serina, and Putelli (currently under evaluation).

## Repository Structure
- `data`: folder containing all the **input data for the experiments**. Documents are divided in two subfolders: `properties`, for datasets referring to a single property, and `crossed-evaluation`, with datasets referring to pairs of properties.
- `src`: folder containing all the **code for executing the experiments**. Code is divided in different software modules; the code is implemented in *Python 3.x*.
- `cache`: folder currently absent. It will be created when the code is executed, and will contain the **cached data for executing the experiments** (typically, pre-computed embeddings).
- `results`: folder currently absent. It will be created when the code is executed, and will contain the **results data of the experiments**.

## Experiments 
*[currently under update]*
