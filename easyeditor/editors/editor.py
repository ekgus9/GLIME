from typing import Optional, Union, List, Tuple, Dict
from time import time
from tqdm import tqdm
import torch
import numpy as np
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, BitsAndBytesConfig
from transformers import LlamaTokenizer,PreTrainedTokenizerFast, LlamaTokenizerFast
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import GPT2TokenizerFast, GPT2Tokenizer
from ..util.globals import *
from .utils import _chunks, _prepare_requests, summary_metrics
from ..evaluate import compute_edit_quality, compute_icl_edit_quality, compute_glime_edit_quality
from ..util import nethook
from ..util.hparams import HyperParams
from ..util.alg_dict import *
from ..evaluate.evaluate_utils import test_generation_quality

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

LOG = logging.getLogger(__name__)
def make_logs():

    f_h, s_h = get_handler('logs', log_name='run.log')
    LOG.addHandler(f_h)
    LOG.addHandler(s_h)

class BaseEditor:
    """Base editor for all methods"""

    @classmethod
    def from_hparams(cls, hparams: HyperParams):
        return cls(hparams)

    def __init__(self, hparams: HyperParams):
        assert hparams is not None, 'Error: hparams is None.'
        self.model_name = hparams.model_name
        self.apply_algo = ALG_DICT[hparams.alg_name]
        self.alg_name = hparams.alg_name
        make_logs()
        LOG.info("Instantiating model")

        if type(self.model_name) is str:
            device_map = 'auto' if hparams.model_parallel else None
            torch_dtype = torch.bfloat16
            
            # QLoRA configuration
            if hparams.alg_name == 'QLoRA':
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=(hparams.quantization_bit == 4),
                    bnb_4bit_use_double_quant=hparams.double_quant,
                    bnb_4bit_quant_type=hparams.quant_type,
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
                model_kwargs = {
                    "quantization_config": bnb_config,
                    "torch_dtype": torch_dtype,
                    "device_map": {'': hparams.device}
                }
            else:
                model_kwargs = {
                    "torch_dtype": torch_dtype,
                    "device_map": device_map
                }

            if 't5' in self.model_name.lower():
                self.model = T5ForConditionalGeneration.from_pretrained(self.model_name, **model_kwargs)
                self.tok = T5Tokenizer.from_pretrained(self.model_name)
            elif 'chatglm-api' in self.model_name.lower():
                self.model, self.tok = None, None
                self.hparams = hparams
                return
            elif 'gpt-3.5' in self.model_name.lower():
                self.model, self.tok = None, None
            elif 'gpt' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
                self.tok = GPT2Tokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'llama' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'baichuan' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs, trust_remote_code=True)
                self.tok = AutoTokenizer.from_pretrained(self.model_name,trust_remote_code=True)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'chatglm' in self.model_name.lower():
                self.model = AutoModel.from_pretrained(self.model_name,trust_remote_code=True, **model_kwargs)
                self.tok = AutoTokenizer.from_pretrained(self.model_name,trust_remote_code=True)
                if 'chatglm2'in self.model_name.lower(): 
                    self.tok.unk_token_id = 64787
                else: 
                    self.tok.pad_token_id = self.tok.eos_token_id
            elif 'internlm' in self.model_name.lower():
                self.model = AutoModel.from_pretrained(self.model_name,trust_remote_code=True, **model_kwargs)
                self.tok = AutoTokenizer.from_pretrained(self.model_name,trust_remote_code=True)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'qwen2' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name,trust_remote_code=True, torch_dtype=torch_dtype if hparams.alg_name not in ['MEND'] else torch.bfloat16, device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name, eos_token='<|endoftext|>', pad_token='<|endoftext|>',unk_token='<|endoftext|>', trust_remote_code=True)
            elif 'qwen' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name,fp32=False,trust_remote_code=True, **model_kwargs)
                self.tok = AutoTokenizer.from_pretrained(self.model_name, eos_token='<|endoftext|>', pad_token='<|endoftext|>',unk_token='<|endoftext|>', trust_remote_code=True)
            elif 'mistral' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            else:
                raise NotImplementedError

            if self.tok is not None and (isinstance(self.tok, GPT2Tokenizer) or isinstance(self.tok, GPT2TokenizerFast) or isinstance(self.tok, LlamaTokenizer) or isinstance(self.tok, LlamaTokenizerFast) or isinstance(self.tok, PreTrainedTokenizerFast)) and (hparams.alg_name not in ['ROME', 'MEMIT', 'EMMET', 'R-ROME','AlphaEdit','CORE']):
                LOG.info('AutoRegressive Model detected, set the padding side of Tokenizer to left...')
                self.tok.padding_side = 'left'
            if self.tok is not None and ('mistral' in self.model_name.lower() or 'llama' in self.model_name.lower() or 'qwen' in self.model_name.lower()) and (hparams.alg_name in ['ROME', 'MEMIT', 'EMMET', 'R-ROME','AlphaEdit', 'CORE']):
                LOG.info('AutoRegressive Model detected, set the padding side of Tokenizer to right...')
                self.tok.padding_side = 'right'
        else:
            self.model, self.tok = self.model_name

        if hparams.model_parallel: 
            hparams.device = str(self.model.device).split(":")[1]
        if not hparams.model_parallel and hasattr(hparams, 'device') and hparams.alg_name != 'QLoRA':
            self.model.to(f'cuda:{hparams.device}')

    def edit(self,
                   prompts: List[str],
                   target_new: List[str],
                   target_neg: Optional[List[str]] = None,
                   ground_truth: Optional[List[str]] = None,
                   rephrase_prompts: Optional[List[str]] = None,
                   locality_inputs: Optional[Dict] = None,
                   portability_inputs: Optional[Dict] = None,
                   sequential_edit=False,
                   verbose=True,
                   **kwargs
                   ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        """
        assert len(prompts) == len(target_new)
        if ground_truth is not None:
            if isinstance(ground_truth, str):
                ground_truth = [ground_truth,]
            else:
                assert len(ground_truth) == len(prompts)
        else: # Default ground truth is <|endoftext|>
            ground_truth = ['<|endoftext|>' for _ in range(len(prompts))]

        requests = _prepare_requests(prompts, target_new, ground_truth, target_neg, rephrase_prompts, locality_inputs, portability_inputs, **kwargs)

        all_metrics = []
        for record_chunks in tqdm(_chunks(requests, 1)):
            start = time()
            edited_model, weights_copy = self.apply_algo(
                self.model,
                self.tok,
                record_chunks,
                self.hparams,
                copy=False,
                return_orig_weights=True,
                keep_original_weight=False
            )
            exec_time = time() - start
            LOG.info(f"Execution editing took {exec_time}")

        start = time()
        chunk_metrics = []
        for i, request in enumerate(tqdm(requests)):
            metrics = {
                'case_id': i,
                "requested_rewrite": request,
                "time": exec_time,
                "post": compute_glime_edit_quality(edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device),
            }
            chunk_metrics.append(metrics)
        LOG.info(f"Evaluation took {time() - start}")
        all_metrics.extend(chunk_metrics)

        return all_metrics, edited_model, weights_copy
