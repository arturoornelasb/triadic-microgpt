"""
ModelWorker — QThread workers for async model operations.

All model inference runs in background threads to keep the UI responsive.
Results are communicated back to the main thread via Qt signals.
"""
import traceback
from pathlib import Path

from PySide6.QtCore import QThread, Signal


class ModelLoadWorker(QThread):
    """Loads a model checkpoint asynchronously."""
    loaded = Signal(object)    # emits ModelInterface on success
    progress = Signal(str)     # emits status messages
    error = Signal(str)        # emits error message on failure

    def __init__(self, backend: str, checkpoint_path: str, tokenizer_path: str,
                 n_bits: int = 64, align_mode: str = 'infonce', hf_model_name: str = ''):
        super().__init__()
        self.backend = backend
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path
        self.n_bits = n_bits
        self.align_mode = align_mode
        self.hf_model_name = hf_model_name

    def run(self):
        try:
            import sys
            from pathlib import Path
            PROJECT_ROOT = Path(__file__).parent.parent.parent
            sys.path.insert(0, str(PROJECT_ROOT))
            sys.path.insert(0, str(PROJECT_ROOT / 'triadic-head'))

            import torch
            from src.triadic import PrimeMapper, TriadicValidator

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            mapper = PrimeMapper(self.n_bits)
            validator = TriadicValidator()

            if self.backend == 'native':
                self.progress.emit(f'Loading native model from {Path(self.checkpoint_path).name}...')
                from src.evaluate import load_model
                model, tokenizer, config = load_model(
                    self.checkpoint_path, self.tokenizer_path, device
                )
                mapper = PrimeMapper(config.n_triadic_bits)

                from ui.model_interface import ModelInterface
                iface = ModelInterface(
                    backend='native',
                    model=model,
                    tokenizer=tokenizer,
                    mapper=mapper,
                    validator=validator,
                    device=device,
                    config=config,
                )
                self.progress.emit('Native model loaded.')
                self.loaded.emit(iface)

            elif self.backend == 'transfer':
                self.progress.emit('Loading GPT-2 base model...')
                from transformers import GPT2LMHeadModel, GPT2Tokenizer
                gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')

                self.progress.emit('Building GPT2TriadicModel...')
                sys.path.insert(0, str(PROJECT_ROOT / 'experiment10' / 'src'))
                from model import GPT2TriadicModel
                model = GPT2TriadicModel(gpt2_model, n_triadic_bits=self.n_bits)

                self.progress.emit(f'Loading transfer weights from {Path(self.checkpoint_path).name}...')
                state = torch.load(self.checkpoint_path, map_location=device, weights_only=False)
                if 'model_state_dict' in state:
                    model.load_state_dict(state['model_state_dict'], strict=False)
                    n_bits = state.get('n_triadic_bits', self.n_bits)
                    if n_bits != self.n_bits:
                        self.n_bits = n_bits
                        mapper = PrimeMapper(n_bits)
                else:
                    model.load_state_dict(state, strict=False)
                model.to(device)
                model.eval()

                tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
                mapper = PrimeMapper(self.n_bits)

                from ui.model_interface import ModelInterface
                iface = ModelInterface(
                    backend='transfer',
                    model=model,
                    tokenizer=tokenizer,
                    mapper=mapper,
                    validator=validator,
                    device=device,
                )
                self.progress.emit('GPT-2 Transfer model loaded.')
                self.loaded.emit(iface)

            else:  # 'hf'
                self.progress.emit(f'Loading TriadicWrapper on {self.hf_model_name}...')
                from triadic_head import TriadicWrapper
                wrapper = TriadicWrapper(
                    self.hf_model_name,
                    n_bits=self.n_bits,
                    align_mode=self.align_mode,
                    device=str(device),
                )
                if self.checkpoint_path and Path(self.checkpoint_path).exists():
                    self.progress.emit('Loading trained wrapper weights...')
                    state = torch.load(self.checkpoint_path, map_location=device)
                    wrapper.triadic_head.load_state_dict(state)
                wrapper.model.eval()

                # HF wrapper doesn't use our tokenizer; pass None
                from ui.model_interface import ModelInterface
                iface = ModelInterface(
                    backend='hf',
                    model=wrapper.model,
                    tokenizer=None,
                    mapper=mapper,
                    validator=validator,
                    device=device,
                    hf_wrapper=wrapper,
                )
                self.progress.emit('TriadicWrapper loaded.')
                self.loaded.emit(iface)

        except Exception as e:
            self.error.emit(f'{type(e).__name__}: {e}\n{traceback.format_exc()}')


class TaskWorker(QThread):
    """Generic worker for running any callable in a background thread."""
    result_ready = Signal(object)   # object avoids PySide6 int64 overflow on big primes
    error_occurred = Signal(str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def run(self):
        try:
            result = self._fn(*self._args, **self._kwargs)
            self.result_ready.emit(result if isinstance(result, dict) else {'value': result})
        except Exception as e:
            self.error_occurred.emit(f'{type(e).__name__}: {e}\n{traceback.format_exc()}')
