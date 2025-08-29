from gemma3_text270m.config import Gemma3TextConfig
from gemma3_text270m.model import Gemma3ForCausalLM
from gemma3_text270m.tokenizer import Gemma3Tokenizer
from gemma3_text270m.generate import Gemma3Generator


def test_generator_constructs():
    cfg = Gemma3TextConfig()
    model = Gemma3ForCausalLM(cfg)
    tok = Gemma3Tokenizer()
    gen = Gemma3Generator(model, tok)
    assert gen.model is model
    assert gen.tokenizer is tok

