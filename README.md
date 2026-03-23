# Knowledge Restoration-driven Prompt Optimization

This repository contains the code for the paper **"Knowledge Restoration-driven Prompt Optimization: Unlocking LLM Potential on Open-Domain Relational triplet Extraction"**.
The original code has been extended and adapted for domain-specific applications in Land Use and Land Cover (LULC) knowledge extraction.

## Overview

The project focuses on unlocking the performance of large language models (LLMs) in extracting triplets in the open domain by leveraging LLMs to iteratively optimize prompts for open domain relational triplet extraction (ORTE). The core method integrates knowledge-restoration-driven feedback to enhance prompt optimization, enabling LLM to better adapt and extract relational triplets.

## Project Structure

The project structure is as follows:

```
datasets/
    example.txt              # Example data file
evaluate/
    references/
        example.txt          # Reference for evaluation
        evaluation_script.py  # Evaluation script
    README.md                # Evaluation documentation
model_utils/
    llms.py                  # LLM configuration and setup
prompts/
    -----                  # Auxiliary prompt definitions
schemas/
    example_schema.csv       # Example schema for the datasets
tools/
README.md                # General tools documentation
requirements.txt         # Required dependencies for the project
run_tg_batch.py          # Script to execute the process
```

### Key Directories and Files

1. **datasets**: Contains dataset files for the project. The datasets such as WebNLG, REBEL, and Wiki-NRE are used for relational triplett extraction tasks and are based on EDC : *"Extract, Define, Canonicalize: An LLM-based Framework for Knowledge Graph Construction"*.https://doi.org/10.18653/v1/2024.emnlp-main.548
2. **evaluate/references**: Includes evaluation scripts and reference files for assessing the performance of relational triplett extraction.
3. **schemas**: Contains schema definitions for various datasets used in the project, including `example_schema.csv` for organizing dataset entries.
4. **prompts**: Stores prompt configuration files used for optimizing and improving the extraction process through LLM-based approaches.
5. Datasets, references, schemas all come from EDC *"Extract, Define, Canonicalize: An LLM-based Framework for Knowledge Graph Construction"*.https://doi.org/10.18653/v1/2024.emnlp-main.548

## Setup and Configuration

1. **Install Dependencies**:
    First, ensure that all required dependencies are installed. You can install them using:

   ```
   pip install -r requirements.txt
   ```

2. **Configure Models**:
    LLM configurations and settings are located in the `llms.py` file. Customize this file to suit your desired model and configuration.

3. **Run the Extraction**:
    After setting up the models, data, and configuration files, run the `run_tg_batch.py` script to execute the extraction process:

   ```
   python run_tg_batch.py
   ```

