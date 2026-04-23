import numpy as np
import torch
import pandas as pd
import os
import sys
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from huggingface_hub import login
token_path = '/workspaces/mpath/oanoufa/data/ibenstoken.txt'
with open(token_path, 'r') as f:
    token = f.read().strip()
login(token=token)

workspace_path = "/workspaces/mpath/oanoufa"
constants_path = f"{workspace_path}/python_scripts/utils/constants.py"
sys.path.append(os.path.dirname(constants_path))
import constants
import model_config

pure_st_to_id_dict = constants.ST_TO_ID_DICT
num_subtypes       = len(pure_st_to_id_dict)
config             = model_config.config
max_length         = config["sequence_length"]
pad_multiple_of    = config["pad_multiple_of"]

from hiv_dataset_class import HIVSequenceDataset, open_memmaps, visualize_sample
from hiv_nt_training_class import HFModelForHIVSubtyping, train_step, validation_step
from hiv_nt_metrics_class import HIVSubtypingMetrics


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=True)

    device = torch.device(config["device"])
    print(f"Using device: {device}")
    print(f"Torch CPU threads: {torch.get_num_threads()}")

    model = HFModelForHIVSubtyping(model_name=config["model_name"], num_subtypes=num_subtypes)
    model = model.to(device)

    print(f"Model loaded: {config['model_name']}")
    print(f"Number of subtypes: {num_subtypes}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ------------------------------------------------------------------
    # Data loading — open memmaps (no RAM allocation for the full dataset)
    # ------------------------------------------------------------------
    seq_mm, lbl_mm, mask_mm = open_memmaps(config["sequences_path"], config["labels_path"], config["loss_masks_path"])
    metadata = pd.read_csv(config["metadata_path"], sep='\t')

    assert seq_mm.shape[0] == len(metadata), "Mismatch between sequences and metadata row counts"
    print(f"Sequences memmap : {seq_mm.shape}  dtype={seq_mm.dtype}")
    print(f"Labels memmap    : {lbl_mm.shape}  dtype={lbl_mm.dtype}")
    print(f"Loss masks memmap: {mask_mm.shape}  dtype={mask_mm.dtype}")

    # ------------------------------------------------------------------
    # Datasets — each split views its subset of rows via stored indices
    # ------------------------------------------------------------------
    test_dataset = HIVSequenceDataset(
        seq_mm=seq_mm, lbl_mm=lbl_mm, mask_mm=mask_mm, metadata=metadata,
        tokenizer=tokenizer, n_subtypes=num_subtypes,
        max_length=max_length, pad_multiple_of=pad_multiple_of, split="test",
    )

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    test_loader = DataLoader(
        test_dataset, batch_size=1,
        shuffle=False, num_workers=config["num_workers"],
    )
    print(f"Test samples  : {len(test_dataset)}")

    id_to_st   = {v: k for k, v in pure_st_to_id_dict.items()}
    n_token_id = tokenizer.encode("N", add_special_tokens=False)[0]

    test_metrics  = HIVSubtypingMetrics(num_subtypes, "test",
        output_path=os.path.join(config["data_cache_dir"], f"test_metrics_v{config['model_version']}.tsv"),
        id_to_st=id_to_st, n_token_id=n_token_id)


    # ------------------------------------------------------------------
    # Test evaluation with best checkpoint
    # ------------------------------------------------------------------
    print(f"\nLoading best model from checkpoint for testing...")
    checkpoint = torch.load(
        os.path.join(config["data_cache_dir"], config["checkpoint_name"]),
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"],
                          strict=False)
    model.eval()

    print(f"\nRunning test evaluation...")
    sample_vis_dir = f"{workspace_path}/figs/sample_vis_v{config['model_version']}/"
    os.makedirs(sample_vis_dir, exist_ok=True)
    for i, test_batch in tqdm(enumerate(test_loader),
                              mininterval=30,
                              desc="Running test batches...",
                              total=config["max_val_batches"] * config["batch_size"]):
        sample_name = metadata[metadata['split'] == 'test'].iloc[i]['sequence_name']
        validation_step(model, test_batch, test_metrics)

        preds  = model(test_batch["input_ids"].to(device))["subtype_logits"]
        sample_pred = {
            "input_ids":      test_batch["input_ids"][0].cpu().detach(),
            "attention_mask": test_batch["attention_mask"][0].cpu().detach(),
            "labels":         torch.sigmoid(preds).squeeze(0).cpu().detach(),
        }
        sample_true = {
            "input_ids":      test_batch["input_ids"][0].cpu().detach(),
            "attention_mask": test_batch["attention_mask"][0].cpu().detach(),
            "labels":         test_batch["labels"][0].cpu().detach(),
        }
        if i < 5:
            out_path = f"{sample_vis_dir}/test_sample_{i}_{sample_name}.png"
            visualize_sample(sample=sample_pred,
                             pure_st_to_id_dict=pure_st_to_id_dict,
                             idx='test_' + str(i) + '_' + sample_name, path=out_path)
            visualize_sample(sample=sample_true,
                             pure_st_to_id_dict=pure_st_to_id_dict,
                             idx='test_' + str(i) + '_' + sample_name, path=out_path.replace('.png', '_true.png'))
        if i >= config["max_val_batches"] * config["batch_size"]:
            break
    test_metrics.print_detailed()
    test_metrics.save_metrics()
