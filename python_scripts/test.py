import numpy as np
import torch
import pandas as pd
import os
from Bio import SeqIO
import sys
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from collections import defaultdict
from tqdm import tqdm
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import config

from huggingface_hub import login
TOKEN_PATH = config.TOKEN_PATH
with open(TOKEN_PATH, 'r') as f:
    token = f.read().strip()
login(token=token)

WORKSPACE_PATH = config.WORKSPACE_PATH

ST_TO_ID_DICT = config.ST_TO_ID_DICT
NUM_SUBTYPES       = len(ST_TO_ID_DICT)
MODEL_CONFIG       = config.MODEL_CONFIG
MAX_LENGTH         = config.SEQ_LEN_AFTER_PAD
PAD_MULTIPLE_OF    = config.PAD_LEN
PURE_REF_PATH      = config.PURE_REF_PATH

from figs import visualize_sample, visualize_confusion_matrix
from dataset_class import HIVSequenceDataset, open_memmaps
from model_class import HFModelForHIVSubtyping, train_step, validation_step
from metrics_class import HIVSubtypingMetrics
from utils import build_hxb2_ata_maps


if __name__ == "__main__":

    hxb2_ata_seq = None
    for i, rec in enumerate(SeqIO.parse(PURE_REF_PATH, "fasta")):
        if i == 0:
            hxb2_ata_seq = str(rec.seq).upper()
            print(f"  HXB2 record id : {rec.id}")
            break
    if hxb2_ata_seq is None:
        sys.exit("ERROR: pure_ref FASTA is empty.")

    ata_to_hxb2, hxb2_to_ata = build_hxb2_ata_maps(hxb2_ata_seq)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG["model_name"], trust_remote_code=True)

    device = torch.device(MODEL_CONFIG["device"])
    print(f"Using device: {device}")
    print(f"Torch CPU threads: {torch.get_num_threads()}")

    model = HFModelForHIVSubtyping(model_name=MODEL_CONFIG["model_name"], num_subtypes=NUM_SUBTYPES)
    model = model.to(device)

    print(f"Model loaded: {MODEL_CONFIG['model_name']}")
    print(f"Number of subtypes: {NUM_SUBTYPES}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ------------------------------------------------------------------
    # Data loading — open memmaps (no RAM allocation for the full dataset)
    # ------------------------------------------------------------------
    seq_mm, lbl_mm, mask_mm = open_memmaps(MODEL_CONFIG["sequences_path"], MODEL_CONFIG["labels_path"], MODEL_CONFIG["loss_masks_path"])
    metadata = pd.read_csv(MODEL_CONFIG["metadata_path"], sep='\t')

    assert seq_mm.shape[0] == len(metadata), "Mismatch between sequences and metadata row counts"
    print(f"Sequences memmap : {seq_mm.shape}  dtype={seq_mm.dtype}")
    print(f"Labels memmap    : {lbl_mm.shape}  dtype={lbl_mm.dtype}")
    print(f"Loss masks memmap: {mask_mm.shape}  dtype={mask_mm.dtype}")

    # ------------------------------------------------------------------
    # Datasets — each split views its subset of rows via stored indices
    # ------------------------------------------------------------------
    test_dataset = HIVSequenceDataset(
        seq_mm=seq_mm, lbl_mm=lbl_mm, mask_mm=mask_mm, metadata=metadata,
        tokenizer=tokenizer, n_subtypes=NUM_SUBTYPES,
        max_length=MAX_LENGTH, pad_multiple_of=PAD_MULTIPLE_OF, split="test",
    )

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    test_loader = DataLoader(
        test_dataset, batch_size=1,
        shuffle=False, num_workers=MODEL_CONFIG["num_workers"],
    )
    print(f"Test samples  : {len(test_dataset)}")

    id_to_st   = {v: k for k, v in ST_TO_ID_DICT.items()}
    n_token_id = tokenizer.encode("N", add_special_tokens=False)[0]

    test_metrics  = HIVSubtypingMetrics(NUM_SUBTYPES, "test",
        output_path=os.path.join(MODEL_CONFIG["metrics_dir"], f"test_metrics_v{MODEL_CONFIG['model_version']}.tsv"),
        id_to_st=id_to_st, n_token_id=n_token_id)


    # ------------------------------------------------------------------
    # Test evaluation with best checkpoint
    # ------------------------------------------------------------------
    print(f"\nLoading best model from checkpoint for testing...")
    checkpoint = torch.load(
        os.path.join(MODEL_CONFIG["checkpoint_dir"], MODEL_CONFIG["checkpoint_name"]),
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"],
                        strict=False)
    model.eval()

    print(f"\nRunning test evaluation...")
    sample_vis_dir = f"{WORKSPACE_PATH}/figs/sample_vis_v{MODEL_CONFIG['model_version']}/"
    os.makedirs(sample_vis_dir, exist_ok=True)
    for i, test_batch in tqdm(enumerate(test_loader),
                              mininterval=30,
                              desc="Running test batches...",
                              total=MODEL_CONFIG["max_val_batches"] * MODEL_CONFIG["batch_size"]):
        sample_name = metadata[metadata['split'] == 'test'].iloc[i]['sequence_name']
        validation_step(model, test_batch, test_metrics)

        preds  = model(test_batch["input_ids"].to(device))["subtype_logits"]
        sample_pred = {
            "input_ids":      test_batch["input_ids"][0].cpu().detach(),
            "loss_mask": test_batch["loss_mask"][0].cpu().detach(),
            "attention_mask": test_batch["attention_mask"][0].cpu().detach(),
            "labels":         torch.sigmoid(preds).squeeze(0).cpu().detach(),
        }
        sample_true = {
            "input_ids":      test_batch["input_ids"][0].cpu().detach(),
            "loss_mask": test_batch["loss_mask"][0].cpu().detach(),
            "attention_mask": test_batch["attention_mask"][0].cpu().detach(),
            "labels":         test_batch["labels"][0].cpu().detach(),
        }
        if i < 10:
            out_path = f"{sample_vis_dir}/test_sample_{i}_{sample_name}.png"
            visualize_sample(sample=sample_pred,
                             hxb2_to_ata=hxb2_to_ata,
                             pure_st_to_id_dict=ST_TO_ID_DICT,
                             idx='test_' + str(i) + '_' + sample_name, path=out_path)
            visualize_sample(sample=sample_true,
                             hxb2_to_ata=hxb2_to_ata,
                             pure_st_to_id_dict=ST_TO_ID_DICT,
                             idx='test_' + str(i) + '_' + sample_name, path=out_path.replace('.png', '_true.png'))
        if i >= MODEL_CONFIG["max_val_batches"] * MODEL_CONFIG["batch_size"]:
            break
    test_metrics.print_detailed()
    test_metrics.save_metrics()
    conf_matrix_path = f"{WORKSPACE_PATH}/figs/confusion_matrix.html"
    visualize_confusion_matrix(
        metrics=test_metrics,
        save_path = conf_matrix_path,
    )

