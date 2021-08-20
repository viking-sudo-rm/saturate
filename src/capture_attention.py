from typing import Callable, List, Tuple, Iterator
from torch import Tensor
from torch.nn import Module

from .language_model import LanguageModel
from .si_transformer import SiEncoder, SiSelfAttention


class _CaptureAttentionContextManager:
    def __init__(self,
                 capture: "CaptureAttention",
                 consume: Callable[[Tensor], Tensor],
                ):
        capture.cm = self
        self.capture = capture
        self.consume = consume
        self.attns = []

    def __enter__(self) -> List[Tuple[str, Tensor]]:
        """Return a pointer to the list where attention will be stored."""
        return self.attns
    
    def __exit__(self, type, value, traceback):
        self.capture.cm = None
    
    def add(self, name: str, probs: Tensor) -> None:
        if self.consume is None:
            attn = self.consume(probs)
        else:
            attn = probs
        self.attns.append((name, attn))

class CaptureAttention(Module):
    """Capture attention patterns in my custom transformer."""

    def __init__(self, model, return_attns: bool = False):
        super().__init__()
        assert isinstance(model, LanguageModel)
        assert isinstance(model.encoder, SiEncoder), f"Invalid encoder type: {type(model.encoder)}"
        self.model = model
        self.return_attns = return_attns
        self.hash_to_name = {}
        self.cm = None

        for name, module in model.named_modules():
            if isinstance(module, SiSelfAttention):
                module.register_forward_hook(self._callback)
                self.hash_to_name[hash(module)] = name

    def forward(self, inputs: Tensor) -> Tensor:
        return self.model(inputs)

    def _callback(self, module: Module, _: Tensor, outputs: Tensor):
        """Save the output attention distribution, if we are in a `capture_attention` context."""
        if self.cm is None:
            return
        name = self.hash_to_name[hash(module)]
        _, attns = outputs
        self.cm.add(name, attns)

    def capture_attention(self, reg) -> Iterator[None]:
        return _CaptureAttentionContextManager(self, reg)
