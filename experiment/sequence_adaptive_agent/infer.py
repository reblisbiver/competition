import os
import json
import torch
import numpy as np

# Attempt flexible import of models module so infer.py works when imported
# as a package or loaded from file path by other scripts.
try:
    from experiment.sequence_adaptive_agent import models
except Exception:
    try:
        from sequence_adaptive_agent import models
    except Exception:
        # Fallback: load models.py by filesystem path relative to this file
        import importlib.util
        this_dir = os.path.dirname(__file__)
        models_path = os.path.join(this_dir, 'models.py')
        spec = importlib.util.spec_from_file_location('sequence_adaptive_agent.models', models_path)
        mod_models = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod_models)
        models = mod_models


ROOT_DIR = os.path.dirname(__file__)
OUTDIR = os.path.join(ROOT_DIR, 'output_overnight')


def _find_checkpoint():
    # Find the most recent checkpoint across any output_overnight* directories.
    root = os.path.dirname(__file__)
    cand_files = []
    for name in os.listdir(root):
        if name.startswith('output_overnight') and os.path.isdir(os.path.join(root, name)):
            d = os.path.join(root, name)
            for f in os.listdir(d):
                if f.endswith('.pt'):
                    cand_files.append(os.path.join(d, f))
    # also check default OUTDIR
    if os.path.isdir(OUTDIR):
        for f in os.listdir(OUTDIR):
            if f.endswith('.pt'):
                cand_files.append(os.path.join(OUTDIR, f))

    if not cand_files:
        return None
    # return newest by modification time
    cand_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cand_files[0]


def _load_model(device='cpu'):
    ckpt = _find_checkpoint()
    if ckpt is None:
        return None
    # try to load checkpoint and infer architecture if needed
    data = torch.load(ckpt, map_location=device)
    state = data.get('state_dict', data) if isinstance(data, dict) else data

    # helper to strip module. prefix
    def _strip_state(s):
        if not isinstance(s, dict):
            return s
        new = {}
        for k, v in s.items():
            nk = k.replace('module.', '')
            new[nk] = v
        return new

    state = _strip_state(state)

    # attempt default construction first
    try:
        policy = models.BiLSTMPolicy(input_dim=7, hidden_dim=128, num_layers=1)
        policy.load_state_dict(state)
        policy.to(device)
        policy.eval()
        return policy
    except Exception:
        # try to infer hidden_dim from fc weight shape if available
        try:
            if 'fc.weight' in state:
                fc_w = state['fc.weight']
            elif 'fc.weight' in state:
                fc_w = state['fc.weight']
            else:
                # find any weight name for fc
                fc_w = None
                for k in state.keys():
                    if k.endswith('fc.weight') or k.endswith('.fc.weight'):
                        fc_w = state[k]
                        break
            if fc_w is not None:
                in_features = fc_w.shape[1]
                # bilstm is bidirectional so hidden_dim*2 == in_features
                inferred_h = max(1, int(in_features // 2))
            else:
                inferred_h = 128
        except Exception:
            inferred_h = 128

    # final attempt with inferred hidden dim
    policy = models.BiLSTMPolicy(input_dim=7, hidden_dim=inferred_h, num_layers=1)
    try:
        policy.load_state_dict(state)
    except Exception:
        # try stripped keys again
        try:
            policy.load_state_dict(_strip_state(state))
        except Exception:
            # as a last resort, load partial matching keys (ignore missing)
            sd = policy.state_dict()
            for k, v in state.items():
                if k in sd and sd[k].shape == v.shape:
                    sd[k] = v
            policy.load_state_dict(sd)
    policy.to(device)
    policy.eval()
    return policy


# cached model
_MODEL = None


def bilstm_infer(target_allocations, anti_target_allocations, is_target_choices):
    """
    Expected inputs: lists of 0/1 values. Return (target_alloc, anti_target_alloc) as ints 0/1.
    This function loads the trained BiLSTM policy and uses it to produce a
    probability for allocating reward to the target side. We then threshold
    at 0.5 and map to a (target, anti) pair. Minimal budget checks applied.
    """
    global _MODEL
    if _MODEL is None:
        _MODEL = _load_model(device='cpu')
        if _MODEL is None:
            raise RuntimeError('No bilstm checkpoint found in output_overnight')

    # sanitize inputs
    targ = list(map(int, target_allocations)) if target_allocations is not None else []
    anti = list(map(int, anti_target_allocations)) if anti_target_allocations is not None else []
    choices = list(map(int, is_target_choices)) if is_target_choices is not None else []

    trial = max(len(targ), len(anti), len(choices))

    # compute remaining budgets based on allocations so far
    TOTAL_REWARDS = 25
    allocated_t = sum(targ)
    allocated_a = sum(anti)
    rem_t = max(0, TOTAL_REWARDS - allocated_t)
    rem_a = max(0, TOTAL_REWARDS - allocated_a)

    # prepare a single-step input representing next trial state
    t_norm = float(trial) / 100.0
    rem_t_norm = float(rem_t) / TOTAL_REWARDS
    rem_a_norm = float(rem_a) / TOTAL_REWARDS
    # placeholder zeros for the three middle features
    zeros = [0.0, 0.0, 0.0]
    last_choice = float(choices[-1]) if len(choices) > 0 else 0.0

    feat = np.array([t_norm, rem_t_norm, rem_a_norm] + zeros + [last_choice], dtype=np.float32)
    x = torch.from_numpy(feat).unsqueeze(0).unsqueeze(0)  # (1,1,7)

    with torch.no_grad():
        out = _MODEL(x)
        if isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out
        prob = torch.sigmoid(logits).item()

    # decide allocation for target and anti-target
    target_alloc = 1 if prob > 0.5 and rem_t > 0 else 0
    anti_alloc = 1 if prob <= 0.5 and rem_a > 0 else 0

    return int(target_alloc), int(anti_alloc)
