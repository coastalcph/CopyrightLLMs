

## Installation

```bash
conda create -y -n copyright-llm python=3.10 
conda activate copyright-llm

pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## How to run

See available model configurations in [`llm_wrapper.py`](./llm_wrapper.py) under `MODEL_CONFIGS` and available runtime parameters in [`run.py`](./run.py) under `RunConfig`.

Example to sequentially run OPT (default 6.7B) and Falcon (default 7B) models and save outputs to custom path:

```bash
python run.py \
    --multirun \
    model_name=opt,falcon \
    dataset_name=books \
    dataset_path=./books \
    save_path=./hf-outputs
```

Example to run GPT-3.5-Turbo and save results locally to custom path:
```bash
export OPENAI_API_KEY="my-key-123"

python run.py \
    model_name=gpt-3.5 \
    dataset_name=books \
    dataset_path=./books \
    save_strategy=LOCAL \
    save_path=./gpt-outputs
```

The codebase also supports saving the results to WandB by adding 'save_strategy=WANDB' as an argument. Before running you also need to export your wandb key or be logged in to wandb.

## How to Cite

```bibtex
@inproceedings{karamolegkou-etal-2023-copyright,
    title = "Copyright Violations and Large Language Models",
    author = "Karamolegkou, Antonia  and
      Li, Jiaang  and
      Zhou, Li  and
      S{\o}gaard, Anders",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.458",
    doi = "10.18653/v1/2023.emnlp-main.458",
    pages = "7403--7412",
}
```
