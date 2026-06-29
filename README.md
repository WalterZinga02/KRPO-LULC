# Knowledge Restoration-driven Prompt Optimization for LULC Triplet Extraction

This repository builds on **Knowledge Restoration-driven Prompt Optimization: Unlocking LLM Potential on Open-Domain Relational Triplet Extraction** and adapts the original KRPO workflow to domain-specific **Land Use and Land Cover (LULC)** knowledge extraction.

The current project contains:

- the original prompt optimization and triplet extraction utilities;
- LULC datasets, schemas, and few-shot examples;
- a dedicated LULC inference script supporting OpenAI, Gemini, and Ollama-compatible local models;
- post-processing, benchmarking, and evaluation utilities.

## Repository Structure

```text
datasets/
  lulc_dataset.txt                         Main LULC input dataset
  lulc_sample.txt                          Smaller LULC sample
  lulc_test.txt                            Test dataset
  example.txt                              Example dataset

schemas/
  lulc_dataset_schema.csv                  Relation schema for the main LULC dataset
  lulc_sample_schema.csv                   Relation schema for the sample dataset
  lulc_test_schema.csv                     Relation schema for the test dataset
  example_schema.csv                       Example relation schema

prompts/
  main_prompt/                             Base prompts used by the KRPO pipeline
  few_shot_examples/                       Few-shot examples grouped by dataset

evaluate/
  evaluation_script.py                     Triplet evaluation script
  references/                              Reference files for evaluation

metrics/
  input/                                   Ignored metric input files
  output/                                  Ignored metric outputs
  agreement_processor.py                   Annotation agreement metrics
  barplot.py                               Benchmark metric plots
  benchmarks.py                            Benchmark comparison metrics
  fuzzy_eval.py                            Fuzzy matching utilities
  heatmap_plotter.py                       Model comparison heatmaps
  match_checker.py                         Match inspection utilities
  recurring_patterns.py                    Pattern analysis utilities
  statistics.py                            Per-sentence annotation metrics

tools/
  input/                                   Ignored tool input files
  output/                                  Ignored tool outputs
  annotation.py                            Annotation helper
  replace_h_t_4reldef.py                   Relation definition helper
  subset_extractor.py                      Random subset generation from datasets

model_utils/
  llms.py                                  Legacy LLM wrapper used by KRPO scripts

run_lulc_inference.py                      Main LULC triplet extraction script
post_processing.py                         Triplet validation and schema-aware post-processing
run_tg_batch.py                            Original KRPO batch extraction script
textcleaner.py                             Text preprocessing utilities

all_corpus_processed.xlsx                  Text extracted from source documents
cleaned_dataset.xlsx                       Processed data for evaluation
requirements.txt                           Python dependencies
```

## Installation

Create and activate a virtual environment, then install the required dependencies.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

The main API client dependency is already included in `requirements.txt`:

```text
openai
```

Energy tracking dependencies are optional. Install them only if you want emissions or energy estimates:

```powershell
pip install ecologits[openai]
pip install codecarbon
```

## API Keys

API keys are read from local environment variables. Do not hard-code keys in the repository.

Required variables:

```text
OPENAI_API_KEY   Required for OpenAI models
GEMINI_API_KEY   Required for Gemini models
```

On Windows PowerShell, save them permanently for your user with:

```powershell
[Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "YOUR_OPENAI_KEY", "User")
[Environment]::SetEnvironmentVariable("GEMINI_API_KEY", "YOUR_GEMINI_KEY", "User")
```

Close and reopen VS Code or the terminal after setting them.

## Running LULC Inference

The main script is:

```powershell
python run_lulc_inference.py
```

It reads:

- input sentences from `datasets/lulc_dataset.txt`;
- few-shot examples from `prompts/few_shot_examples/lulc_dataset/oie_few_shot_examples.txt`;
- schema relations from `schemas/lulc_dataset_schema.csv`.

It writes outputs to:

```text
outputs/lulc_dataset/
  raw_triplets.txt
  final_triplets.txt
  parsing_errors.txt
  benchmark.json
```

### OpenAI

```powershell
$env:LLM_PROVIDER="openai"
$env:MODEL_NAME="gpt-4o-mini"
python run_lulc_inference.py
```

### Gemini

Gemini is called through Google's OpenAI-compatible endpoint.

```powershell
$env:LLM_PROVIDER="gemini"
$env:MODEL_NAME="gemini-2.5-flash"
python run_lulc_inference.py
```

### Ollama

Make sure Ollama is installed and the local model is available.

```powershell
ollama pull llama3
```

Start the Ollama server if it is not already running:

```powershell
ollama serve
```

Then run inference:

```powershell
$env:LLM_PROVIDER="ollama"
$env:MODEL_NAME="llama3"
python run_lulc_inference.py
```

By default, the Ollama endpoint is:

```text
http://localhost:11434/v1
```

You can override it with:

```powershell
$env:OLLAMA_BASE_URL="http://localhost:11434/v1"
```

## Energy Tracking

`ENERGY_TRACKER` defaults to `auto`, so you normally do not need to set it.

Default behavior:

```text
openai  -> ecologits
gemini  -> ecologits
ollama  -> codecarbon
```

Available values:

```text
auto         Use EcoLogits for API models and CodeCarbon for local Ollama runs
ecologits    Force EcoLogits
codecarbon   Force CodeCarbon
none         Disable energy tracking
```

Example:

```powershell
$env:ENERGY_TRACKER="none"
python run_lulc_inference.py
```

## Outputs and Benchmarking

`run_lulc_inference.py` produces:

- `raw_triplets.txt`: raw parsed triplets from the model;
- `final_triplets.txt`: post-processed triplets after schema-aware validation;
- `parsing_errors.txt`: failed responses or parsing issues;
- `benchmark.json`: runtime, token, energy, CO2, and extraction statistics for the latest run.

Important note: `raw_triplets.txt`, `final_triplets.txt`, and `parsing_errors.txt` are reset at the start of each run. If you run OpenAI, Gemini, and Ollama experiments sequentially, copy or rename the output files between runs if you need to preserve each model's triplets.

The benchmark file is overwritten at the end of each run and includes the provider, model, energy tracker, runtime, token counts, and triplet counts.

## Evaluation

The evaluation script is located in `evaluate/evaluation_script.py`.

Example:

```powershell
python evaluate/evaluation_script.py --edc_output outputs/lulc_dataset/final_triplets.txt --reference evaluate/references/example.txt --max_length_diff 5
```

See `evaluate/README.md` for additional notes.

## Main Configuration Variables

```text
LLM_PROVIDER      openai, gemini, or ollama
MODEL_NAME        Model identifier passed to the selected provider
ENERGY_TRACKER    auto, ecologits, codecarbon, or none
OPENAI_API_KEY    OpenAI API key
GEMINI_API_KEY    Gemini API key
OLLAMA_BASE_URL   Optional custom Ollama endpoint
```

## Notes

- Keep API keys outside the repository.
- The `.gitignore` excludes generated outputs, logs, virtual environments, and `.env` files.
- The current LULC inference path is centered on `run_lulc_inference.py`; `run_tg_batch.py` and `model_utils/llms.py` are retained for the original KRPO workflow.
