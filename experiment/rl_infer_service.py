from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import os
import importlib.util
from typing import List, Optional

app = FastAPI()


class PredictRequest(BaseModel):
    bias_rewards: List
    unbias_rewards: List
    is_bias_choice: List
    model: Optional[str] = None
    user_id: Optional[str] = None


def load_infer_module(model_name: Optional[str] = None):
    """Load an inference implementation based on `model_name`.

    Behavior:
    - If `model_name` corresponds to a file under `sequences/dynamic/<model>.py`, load it and expect `allocate()`.
    - Else try to load `sequences/dynamic/lstm_dynamic.py` and use `allocate()` if present.
    - Else fallback to `sequence_adaptive_agent.infer` and use `bilstm_infer()`.
    - Returns a tuple `(module, mode)` where mode is 'dynamic' or 'bilstm'.
    """
    base = os.path.dirname(__file__)

    # If explicit dynamic model name provided, check file existence
    if model_name:
        dyn_candidate = os.path.join(base, 'sequences', 'dynamic',
                                     model_name + '.py')
        if os.path.exists(dyn_candidate):
            spec = importlib.util.spec_from_file_location(
                f'sequences.dynamic.{model_name}', dyn_candidate)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, 'allocate'):
                return mod, 'dynamic'

    # Try generic dynamic allocator file (lstm_dynamic.py) if present
    dyn_path = os.path.join(base, 'sequences', 'dynamic', 'lstm_dynamic.py')
    if os.path.exists(dyn_path):
        spec = importlib.util.spec_from_file_location(
            'sequences.dynamic.lstm_dynamic', dyn_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if hasattr(mod, 'allocate'):
            return mod, 'dynamic'

    # Fallback: sequence_adaptive_agent.infer (bilstm)
    try:
        import sequence_adaptive_agent.infer as infer_mod
        if hasattr(infer_mod, 'bilstm_infer'):
            return infer_mod, 'bilstm'
    except Exception:
        pass

    # filesystem fallback for infer.py
    candidate = os.path.join(base, 'sequence_adaptive_agent', 'infer.py')
    if os.path.exists(candidate):
        spec = importlib.util.spec_from_file_location(
            'sequence_adaptive_agent.infer', candidate)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if hasattr(mod, 'bilstm_infer'):
            return mod, 'bilstm'

    raise ImportError('no suitable infer/allocate module found')


@app.post('/predict')
def predict(req: PredictRequest):
    try:
        infer_mod, mode = load_infer_module(req.model)
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f'infer module load error: {e}')

    try:
        # dynamic allocate: expect allocate(bias_rewards, unbias_rewards, is_bias_choice)
        if mode == 'dynamic' and hasattr(infer_mod, 'allocate'):
            t, a = infer_mod.allocate(req.bias_rewards, req.unbias_rewards,
                                      req.is_bias_choice)
            # return JSON keys expected by backend.php: 'biased' and 'unbiased'
            return {'status': 'ok', 'biased': int(t), 'unbiased': int(a)}

        # bilstm inference
        if mode == 'bilstm' and hasattr(infer_mod, 'bilstm_infer'):
            res = infer_mod.bilstm_infer(req.bias_rewards, req.unbias_rewards,
                                         req.is_bias_choice)
            if not res or len(res) < 2:
                raise ValueError('unexpected infer output')
            return {
                'status': 'ok',
                'biased': int(res[0]),
                'unbiased': int(res[1])
            }

        # As a final fallback, attempt to use allocate or bilstm_infer if present
        if hasattr(infer_mod, 'allocate'):
            t, a = infer_mod.allocate(req.bias_rewards, req.unbias_rewards,
                                      req.is_bias_choice)
            return {'status': 'ok', 'biased': int(t), 'unbiased': int(a)}
        if hasattr(infer_mod, 'bilstm_infer'):
            res = infer_mod.bilstm_infer(req.bias_rewards, req.unbias_rewards,
                                         req.is_bias_choice)
            return {
                'status': 'ok',
                'biased': int(res[0]),
                'unbiased': int(res[1])
            }

        raise HTTPException(status_code=500,
                            detail='no valid inference function found')
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'inference error: {e}')


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8001)
