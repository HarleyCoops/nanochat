"""
Hugging Face-compatible configuration for the nanochat Transformer.

This mirrors the hyperparameters used by nanochat's GPT implementation while
exposing a standard `PretrainedConfig` interface so that AutoConfig can locate
and instantiate the model from Hub checkpoints.
"""

from transformers.configuration_utils import PretrainedConfig


class NanoChatConfig(PretrainedConfig):
    # Match the identifier used when registering with AutoConfig / AutoTokenizer
    model_type = "nanochat"
    attribute_map = {
        "num_attention_heads": "n_head",
        "hidden_size": "n_embd",
        "num_hidden_layers": "n_layer",
    }
    tokenizer_class = "NanoChatTokenizer"

    def __init__(
        self,
        vocab_size=65536,
        sequence_len=2048,
        n_layer=20,
        n_head=10,
        n_kv_head=10,
        n_embd=1280,
        rotary_dim=None,
        activation_function="relu_squared",
        use_rope=True,
        use_qk_norm=True,
        tie_word_embeddings=False,
        softcap=15.0,
        bos_token_id=1,
        eos_token_id=1,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.sequence_len = sequence_len
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_embd = n_embd
        self.rotary_dim = rotary_dim or (n_embd // n_head)
        self.activation_function = activation_function
        self.use_rope = use_rope
        self.use_qk_norm = use_qk_norm
        self.tie_word_embeddings = tie_word_embeddings
        self.softcap = softcap

