import os
import sys
import importlib.util
import random
from typing import List, Tuple, Optional

import numpy as np
import torch

def _ensure_repo_root_on_syspath():
	# infer.py lives in: <repo_root>/experiment/sequence_adaptive_agent/infer.py
	# To import `experiment.*`, sys.path must include <repo_root>.
	here = os.path.dirname(__file__)
	repo_root = os.path.abspath(os.path.join(here, '..', '..'))
	if repo_root not in sys.path:
		sys.path.insert(0, repo_root)


def _load_models_module_fallback():
	"""Load models.py by path as a last resort (no package import needed)."""
	models_path = os.path.join(os.path.dirname(__file__), 'models.py')
	spec = importlib.util.spec_from_file_location('sequence_adaptive_agent.models', models_path)
	if spec is None or spec.loader is None:
		raise ImportError(f'Failed to create spec for models.py at {models_path}')
	mod = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(mod)
	return mod


_ensure_repo_root_on_syspath()

# Import models in a way that works both in-repo and in the competition runner.
try:
	from experiment.sequence_adaptive_agent import models  # type: ignore
except Exception:
	models = _load_models_module_fallback()


ROOT_DIR = os.path.dirname(__file__)

# Checkpoint lookup configuration
_PREFERRED_OUTDIR_NAME = 'output_overnight_resume_20251216'
_CKPT_SEARCH_LOCATIONS: List[str] = []

TOTAL_TRIALS = 100
PER_SIDE = 25
INPUT_DIM = 7


def _find_checkpoint() -> Optional[str]:
	"""Find the bilstm checkpoint under a specific output directory.

	To reduce ambiguity, we only look under:
	- experiment/sequence_adaptive_agent/output_overnight_resume_20251216/

	Prefers bilstm_best.pt if present, else newest *.pt by mtime.
	"""
	global _CKPT_SEARCH_LOCATIONS
	root = os.path.dirname(__file__)
	preferred_dir = os.path.join(root, _PREFERRED_OUTDIR_NAME)
	local_best = os.path.join(root, 'bilstm_best.pt')

	searched: List[str] = []
	# 1) Explicit override (useful in remote runners)
	env_ckpt = os.environ.get('BILSTM_CHECKPOINT', '').strip()
	if env_ckpt:
		searched.append(f'ENV:BILSTM_CHECKPOINT={env_ckpt}')
		if os.path.isfile(env_ckpt):
			_CKPT_SEARCH_LOCATIONS = searched
			return env_ckpt

	# 2) Preferred output directory
	searched.append(preferred_dir)
	cand_files: List[str] = []
	if os.path.isdir(preferred_dir):
		for f in os.listdir(preferred_dir):
			if f.endswith('.pt'):
				cand_files.append(os.path.join(preferred_dir, f))

	# 3) Fallback: local directory (same folder as infer.py)
	searched.append(root)
	if os.path.isfile(local_best):
		_CKPT_SEARCH_LOCATIONS = searched
		return local_best
	try:
		for f in os.listdir(root):
			if f.endswith('.pt'):
				cand_files.append(os.path.join(root, f))
	except Exception:
		pass

	# 4) Legacy fallback: any output_overnight* dir under this folder
	try:
		for name in os.listdir(root):
			if name.startswith('output_overnight') and os.path.isdir(os.path.join(root, name)):
				d = os.path.join(root, name)
				searched.append(d)
				best_in_dir = os.path.join(d, 'bilstm_best.pt')
				if os.path.isfile(best_in_dir):
					_CKPT_SEARCH_LOCATIONS = searched
					return best_in_dir
				for f in os.listdir(d):
					if f.endswith('.pt'):
						cand_files.append(os.path.join(d, f))
	except Exception:
		pass

	_CKPT_SEARCH_LOCATIONS = searched
	if not cand_files:
		return None

	# Prefer an explicit best checkpoint if present.
	best = [p for p in cand_files if os.path.basename(p) == 'bilstm_best.pt']
	if best:
		return best[0]

	# Otherwise choose newest by mtime.
	cand_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
	return cand_files[0]


def _strip_state_dict(sd: dict) -> dict:
	return {k.replace('module.', ''): v for k, v in sd.items()}


def _load_model(device: str = 'cpu') -> Optional[torch.nn.Module]:
	ckpt = _find_checkpoint()
	if ckpt is None:
		return None

	data = torch.load(ckpt, map_location=device, weights_only=False)
	if isinstance(data, dict):
		state = data.get('state_dict', data)
		meta = data.get('meta', {})
	else:
		state = data
		meta = {}

	state = _strip_state_dict(state)
	hidden_dim = int(meta.get('hidden_dim', 128))
	num_layers = int(meta.get('num_layers', 1))
	dropout = float(meta.get('dropout', 0.0))

	policy = models.BiLSTMPolicy(input_dim=INPUT_DIM, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
	policy.load_state_dict(state)
	policy.to(device)
	policy.eval()
	return policy


_MODEL: Optional[torch.nn.Module] = None


def _build_feature_seq(
	target_allocations: List[int],
	anti_target_allocations: List[int],
	is_target_choices: List[int],
) -> torch.Tensor:
	"""Build (1, seq_len, 7) features for the policy, up to current trial."""
	targ = list(target_allocations)
	anti = list(anti_target_allocations)
	choices = list(is_target_choices)

	trial = max(len(targ), len(anti), len(choices))

	# pad histories to same length (previous trials)
	if len(targ) < trial:
		targ.extend([0] * (trial - len(targ)))
	if len(anti) < trial:
		anti.extend([0] * (trial - len(anti)))
	if len(choices) < trial:
		choices.extend([0] * (trial - len(choices)))

	feats = []
	for i in range(trial + 1):
		rem_t = max(0, PER_SIDE - sum(targ[:i]))
		rem_a = max(0, PER_SIDE - sum(anti[:i]))

		choice_rate = float(np.mean(choices[:i])) if i > 0 else 0.5
		last_choice = float(choices[i - 1]) if i > 0 else 0.0

		if i > 0:
			last_reward = 1.0 if ((choices[i - 1] == 1 and targ[i - 1] == 1) or (choices[i - 1] == 0 and anti[i - 1] == 1)) else 0.0
			last_alloc_total = float(targ[i - 1] + anti[i - 1]) / 2.0
		else:
			last_reward = 0.0
			last_alloc_total = 0.0

		feat = np.array(
			[
				float(i) / float(TOTAL_TRIALS),
				float(rem_t) / float(PER_SIDE),
				float(rem_a) / float(PER_SIDE),
				choice_rate,
				last_choice,
				last_reward,
				last_alloc_total,
			],
			dtype=np.float32,
		)
		feats.append(feat)

	x = torch.from_numpy(np.stack(feats, axis=0)).unsqueeze(0)  # (1, seq_len, 7)
	return x


def _decide_with_forcing(p: float, rem: int, trials_left_including_current: int) -> int:
	"""Decide 0/1 with constraints to guarantee exact PER_SIDE by end."""
	if rem <= 0:
		return 0

	# If we must allocate on every remaining trial to reach PER_SIDE, force it.
	if rem >= trials_left_including_current:
		return 1

	action = 1 if random.random() < p else 0

	# After taking action, ensure feasibility: rem_after <= trials_left_after
	trials_left_after = trials_left_including_current - 1
	rem_after = rem - action
	if rem_after > trials_left_after:
		action = 1

	return int(action)


def bilstm_infer(
	target_allocations: List[int],
	anti_target_allocations: List[int],
	is_target_choices: List[int],
) -> Tuple[int, int]:
	"""Infer next allocations.

	IMPORTANT: Signature and return structure must not change.
	Inputs are lists of 0/1 (or bool-ish). Returns two ints (0/1):
	(target_alloc, anti_target_alloc).

	Constraint: over TOTAL_TRIALS, each side must have exactly PER_SIDE allocations.
	Overlap is allowed (both 1 in the same trial).
	"""
	global _MODEL
	if _MODEL is None:
		_MODEL = _load_model(device='cpu')
		if _MODEL is None:
			preferred = os.path.join(os.path.dirname(__file__), _PREFERRED_OUTDIR_NAME, 'bilstm_best.pt')
			searched = '\n  - '.join(_CKPT_SEARCH_LOCATIONS) if _CKPT_SEARCH_LOCATIONS else '(none)'
			raise RuntimeError(
				'No bilstm checkpoint found.\n'
				f'Expected (recommended): {preferred}\n'
				'Also supports env var BILSTM_CHECKPOINT pointing to a .pt file.\n'
				f'Searched:\n  - {searched}'
			)

	targ = list(map(int, target_allocations)) if target_allocations is not None else []
	anti = list(map(int, anti_target_allocations)) if anti_target_allocations is not None else []
	choices = list(map(bool, is_target_choices)) if is_target_choices is not None else []

	trial = max(len(targ), len(anti), len(choices))
	if trial >= TOTAL_TRIALS:
		return 0, 0

	allocated_t = sum(targ)
	allocated_a = sum(anti)
	rem_t = max(0, PER_SIDE - allocated_t)
	rem_a = max(0, PER_SIDE - allocated_a)

	trials_left_including_current = TOTAL_TRIALS - trial

	x = _build_feature_seq(targ, anti, choices)

	with torch.no_grad():
		out = _MODEL(x)
		logits = out[0] if isinstance(out, tuple) else out
		logits_last = logits.squeeze(0)  # (2,)
		probs = torch.sigmoid(logits_last)
		p_t = float(probs[0].item())
		p_a = float(probs[1].item())

	act_t = _decide_with_forcing(p_t, rem_t, trials_left_including_current)
	act_a = _decide_with_forcing(p_a, rem_a, trials_left_including_current)

	# Hard safety clamps
	if rem_t <= 0:
		act_t = 0
	if rem_a <= 0:
		act_a = 0

	return int(act_t), int(act_a)