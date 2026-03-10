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
                self.progress.emit(f'Cargando modelo nativo desde {Path(self.checkpoint_path).name}...')
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
                self.progress.emit('Modelo nativo cargado.')
                self.loaded.emit(iface)

            else:  # 'hf'
                self.progress.emit(f'Cargando TriadicWrapper sobre {self.hf_model_name}...')
                from triadic_head import TriadicWrapper
                wrapper = TriadicWrapper(
                    self.hf_model_name,
                    n_bits=self.n_bits,
                    align_mode=self.align_mode,
                    device=str(device),
                )
                if self.checkpoint_path and Path(self.checkpoint_path).exists():
                    self.progress.emit('Cargando pesos entrenados del wrapper...')
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
                self.progress.emit('TriadicWrapper cargado.')
                self.loaded.emit(iface)

        except Exception as e:
            self.error.emit(f'{type(e).__name__}: {e}\n{traceback.format_exc()}')


class TaskWorker(QThread):
    """Generic worker for running any callable in a background thread."""
    result_ready = Signal(dict)
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
