# GLIME: Generalizable Lifelong Model Editing via Preference Optimization

Official implementation of **"Generalizable Lifelong Model Editing via Preference Optimization" (EMNLP 2026)**.

## Installation

```bash
conda create -n glime python=3.10 -y
conda activate glime

pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate
pip install numpy scipy scikit-learn nltk regex openai pyyaml tqdm
```

## Usage

```bash
bash run.sh
```

or directly:

```bash
python run.py \
    --alg_name=GLIME \
    --hparams_fname=./hparams/GLIME/llama3.1-8b.yaml \
    --ds_name=./data/glime_mquake.jsonl \
    --sequential_edit true \
    --save_path=./results/mquake_llama31_8b.json
```

| Argument | Description |
| --- | --- |
| `--alg_name` | Editing algorithm. Only `GLIME` is registered in `ALG_DICT`. |
| `--hparams_fname` | Path to the YAML hyperparameter file under `hparams/GLIME/`. |
| `--ds_name` | Path to the benchmark JSONL file. |
| `--sequential_edit` | Run edits as one continuous lifelong stream without restoring weights. |
| `--save_path` | Where to dump the per-case metrics JSON. Omit to skip saving. |

## Evaluation

| Metric | Question asked | What it measures |
| --- | --- | --- |
| `prompt_acc` | the original edit prompt | **Reliability** — did the edit take? |
| `generality_acc` | a paraphrase of the edit prompt | **Generality** — does it survive rephrasing? |
| `multiple_choice_acc` | the edit prompt as a 4-way A/B/C/D choice | **Consistency** under a different output format |
| `multi_hop_acc` | a question requiring the edited fact plus one more hop | **Portability** — is the new fact reasoned *with*? |
| `locality_acc` | an unrelated question | **Locality** — is untouched knowledge preserved? |

## Citation

```bibtex

```

## Acknowledgements

The editing framework in `easyeditor/` is adapted from
[EasyEdit](https://github.com/zjunlp/EasyEdit). We thank the authors for
releasing it.
