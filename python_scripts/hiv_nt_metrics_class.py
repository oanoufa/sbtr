from torchmetrics.classification import MultilabelF1Score, MultilabelPrecision, MultilabelRecall
import numpy as np
import torch
import pandas as pd
import os
import sys
import config

workspace_path = config.WORKSPACE_PATH
pure_st_to_id_dict = config.ST_TO_ID_DICT
num_subtypes = len(pure_st_to_id_dict)
MODEL_CONFIG = config.MODEL_CONFIG
max_length = MODEL_CONFIG["sequence_length"]
device = torch.device(MODEL_CONFIG["device"])

# High-level clade groups to calculate true biological performance
CLADE_GROUPS = {
    "A_Clade": ["A1", "A2", "A3", "A4", "A6", "A7", "A8"],
    "F_Clade": ["F1", "F2"],
    "B/D": ["B", "D"]
}

class HIVSubtypingMetrics:
    """Token-level multi-label metrics with per-subtype and base/gap diagnostics."""
    def __init__(self, num_subtypes: int, split: str, output_path: str = None,
                 id_to_st: dict = None, n_token_id: int = None):
        self.num_subtypes = num_subtypes
        self.split = split
        self.output_path = output_path
        self.id_to_st = id_to_st or {i: str(i) for i in range(num_subtypes)}
        self.n_token_id = n_token_id

        if output_path is not None and os.path.isfile(output_path):
            os.remove(output_path)

        # --- Micro metrics ---
        self.f1_micro = MultilabelF1Score(num_labels=num_subtypes, average="micro").to(device)
        self.precision_micro = MultilabelPrecision(num_labels=num_subtypes, average="micro").to(device)
        self.recall_micro = MultilabelRecall(num_labels=num_subtypes, average="micro").to(device)

        # --- Clade-Level Metrics Setup ---
        clade_names = []
        st_to_clade_idx = {}
        
        for st, id_ in pure_st_to_id_dict.items():
            found = False
            for c_name, members in CLADE_GROUPS.items():
                if st in members:
                    if c_name not in clade_names: clade_names.append(c_name)
                    st_to_clade_idx[id_] = clade_names.index(c_name)
                    found = True
                    break
            if not found:
                clade_names.append(st) # Keep others separate
                st_to_clade_idx[id_] = clade_names.index(st)
                
        self.num_clades = len(clade_names)
        self.clade_names = clade_names
        
        # Projection matrix from (num_subtypes -> num_clades)
        proj = torch.zeros(num_subtypes, self.num_clades)
        for i in range(num_subtypes):
            proj[i, st_to_clade_idx[i]] = 1.0
        self.clade_proj = proj.to(device)

        self.f1_clade = MultilabelF1Score(num_labels=self.num_clades, average="micro").to(device)

        # --- Per-subtype metrics ---
        self.f1_per_st = MultilabelF1Score(num_labels=num_subtypes, average="none").to(device)
        self.precision_per_st = MultilabelPrecision(num_labels=num_subtypes, average="none").to(device)
        self.recall_per_st = MultilabelRecall(num_labels=num_subtypes, average="none").to(device)

        self.losses = []
        self.fp_confusion = torch.zeros(num_subtypes, num_subtypes)

    def reset(self):
        self.f1_micro.reset()
        self.precision_micro.reset()
        self.recall_micro.reset()
        self.f1_clade.reset()
        self.f1_per_st.reset()
        self.precision_per_st.reset()
        self.recall_per_st.reset()
        self.losses = []
        self.fp_confusion.zero_()

    def update(self, preds, targets, loss_mask, loss, input_ids=None, loss_unreduced=None):
        mask_bool = loss_mask.bool()

        preds_flat   = preds[mask_bool.expand_as(preds)].view(-1, self.num_subtypes)
        targets_flat = targets[mask_bool.expand_as(targets)].view(-1, self.num_subtypes)

        self.f1_micro.update(preds_flat, targets_flat)
        self.precision_micro.update(preds_flat, targets_flat)
        self.recall_micro.update(preds_flat, targets_flat)
        
        # --- Clade Projection ---
        preds_clade = (preds_flat @ self.clade_proj > 0).float()
        targets_clade = (targets_flat @ self.clade_proj > 0).float()
        self.f1_clade.update(preds_clade, targets_clade)

        self.f1_per_st.update(preds_flat, targets_flat)
        self.precision_per_st.update(preds_flat, targets_flat)
        self.recall_per_st.update(preds_flat, targets_flat)

        with torch.no_grad():
            fp = (preds_flat == 1) & (targets_flat == 0)
            self.fp_confusion += (fp.float().T @ targets_flat.float()).cpu()

        self.losses.append(loss)

    def compute(self):
        result = {
            "f1/micro": self.f1_micro.compute().cpu().item(),
            "f1/clade_micro": self.f1_clade.compute().cpu().item(),
            "precision/micro": self.precision_micro.compute().cpu().item(),
            "recall/micro": self.recall_micro.compute().cpu().item(),
            "loss": np.mean(self.losses),
        }
        return result

    def compute_detailed(self):
        result = self.compute()
        f1_per = self.f1_per_st.compute().cpu()
        p_per  = self.precision_per_st.compute().cpu()
        r_per  = self.recall_per_st.compute().cpu()

        result["per_subtype"] = {
            self.id_to_st[i]: {
                "f1": f1_per[i].item(),
                "precision": p_per[i].item(),
                "recall": r_per[i].item(),
            }
            for i in range(self.num_subtypes)
        }
        result["fp_confusion"] = self.fp_confusion.clone()
        return result

    def print_metrics(self):
        m = self.compute()
        line = (f"[{self.split.upper()}] "
                f"Loss: {m['loss']:.4f} | "
                f"Raw F1: {m['f1/micro']:.4f} | "
                f"Clade F1: {m['f1/clade_micro']:.4f} | " # Highlighted Clade Score
                f"P: {m['precision/micro']:.4f} | "
                f"R: {m['recall/micro']:.4f}")
        print(line)

    def print_detailed(self, top_n=10):
        m = self.compute_detailed()
        self.print_metrics()

        print(f"\n  Per-subtype metrics (worst F1 first):")
        print(f"  {'Subtype':>8s}  {'F1':>6s}  {'P':>6s}  {'R':>6s}")
        print(f"  {'-' * 30}")
        for st, v in sorted(m["per_subtype"].items(), key=lambda x: x[1]["f1"]):
            print(f"  {st:>8s}  {v['f1']:.4f}  {v['precision']:.4f}  {v['recall']:.4f}")

        fp = m["fp_confusion"].clone()
        fp.fill_diagonal_(0)
        k = min(top_n, (fp > 0).sum().item())
        if k > 0:
            vals, idxs = torch.topk(fp.flatten(), k)
            print(f"\n  Top false positives (predicted X, true was Y):")
            for val, flat in zip(vals, idxs):
                if val.item() == 0: break
                i, j = flat.item() // self.num_subtypes, flat.item() % self.num_subtypes
                print(f"    Predicted {self.id_to_st[i]:>6s}, "
                      f"true was {self.id_to_st[j]:>6s}: {int(val):>8d} positions")

    def save_metrics(self, step=None):
        m = self.compute()
        if step is not None:
            m['step'] = step
        df = pd.DataFrame([m])
        file_exists = os.path.isfile(self.output_path)
        df.to_csv(self.output_path, mode='a', index=False, sep='\t',
                  header=not file_exists)