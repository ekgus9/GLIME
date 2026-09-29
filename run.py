import json
import torch
import argparse
from datasets import load_dataset

from easyeditor import BaseEditor, GLIMEHyperParams


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "t", "yes", "y", "1"):
        return True
    if v.lower() in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v}")


def main(alg_name, hparams_fname, ds_name, sequential_edit, save_path):

    with open(ds_name, 'r', encoding='utf-8') as f:
        test_ds = [json.loads(line) for line in f if line.strip()]
    
    prompts = []
    ground_truth = []
    target_new = []
    subject = []
    rephrase_prompts = []

    for i in test_ds:
        prompts.append(i['prompt'])
        ground_truth.append(i['target_true'])
        target_new.append(i["target_new"])
        subject.append(i['subject'])
        rephrase_prompts.append(i)

    hparams=GLIMEHyperParams.from_hparams(hparams_fname)
    editor=BaseEditor.from_hparams(hparams)

    dpo_ds = load_dataset("arcee-ai/general-dpo-datasets", "OpenHermesPreferences")
    dpo = dpo_ds['train']

    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        target_new=target_new,
        ground_truth=ground_truth,
        subject=subject,
        rephrase_prompts=rephrase_prompts,
        keep_original_weight=True,
        sequential_edit=sequential_edit,
        dpo=dpo,
        )

    if save_path is not None:
        json.dump(metrics, open(save_path, 'w', encoding='utf-8'), indent=4, ensure_ascii = False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        default="GLIME",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. "
        "If continuing from previous run, specify the run_id in --continue_from_run.",
        required=False,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=False,
    )
    parser.add_argument(
        "--ds_name",
    )
    parser.add_argument(
        "--sequential_edit",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        )
    parser.add_argument(
        "--save_path",
        default=None,
        )
        
    parser.set_defaults(skip_generation_tests=False, conserve_memory=False)
    args = parser.parse_args()

    main(
        args.alg_name,
        args.hparams_fname,
        args.ds_name,
        args.sequential_edit,
        args.save_path,
    )
