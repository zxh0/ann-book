import pytest
import torch

from toyllm.model import Block, SmolLM2
from toyllm.tokenizer import BPETokenizer



@pytest.fixture(scope="module")
def model(model_dir):
    return SmolLM2.from_pretrained(model_dir)



def test_parameter_count_matches_checkpoint(model):
    assert model.num_parameters() == 134_515_008


def test_lm_head_shares_the_embedding_parameter(model):
    """Tied, not copied — sharing the object is what keeps the count at 134.5M."""
    assert model.lm_head.weight is model.embed_tokens.weight


def test_shapes(model):
    ids = torch.tensor([[1, 2, 3, 4]])
    logits = model(ids)
    assert logits.shape == (1, 4, model.cfg.vocab_size)


def test_hidden_states_follow_hf_convention(model):
    ids = torch.tensor([[1, 2, 3]])
    logits, hs = model(ids, output_hidden_states=True)
    assert len(hs) == model.cfg.num_hidden_layers + 1
    # The last entry is normalized, the others are not — so it is far smaller.
    assert hs[-1].abs().max() < hs[-2].abs().max()


def test_logits_match_reference(model, reference):
    with torch.no_grad():
        logits = model(reference["input_ids"])
    rel = (logits - reference["logits"]).abs().max() / reference["logits"].abs().max()
    assert rel < 1e-5, rel


def test_predicted_tokens_match_reference_exactly(model, reference):
    """Tolerances are for floats; the argmax is discrete and must agree."""
    with torch.no_grad():
        logits = model(reference["input_ids"])
    assert torch.equal(logits.argmax(-1), reference["logits"].argmax(-1))


def test_every_layer_matches_reference(model, reference):
    """Layer-by-layer, so a divergence points at the block that caused it."""
    with torch.no_grad():
        _, hs = model(reference["input_ids"], output_hidden_states=True)
    ref = reference["hidden_states"]
    assert len(hs) == ref.shape[0]
    for i, (ours, theirs) in enumerate(zip(hs, ref)):
        rel = (ours - theirs).abs().max() / theirs.abs().max()
        assert rel < 1e-5, f"layer {i}: relative {rel:.3e}"


def test_greedy_decode_matches_reference(model, reference):
    with torch.no_grad():
        out = model.generate(reference["input_ids"], max_new_tokens=20)
    assert torch.equal(out, reference["greedy_20"])


def test_generates_expected_text(model, model_dir, reference):
    tok = BPETokenizer.from_file(model_dir)
    ids = torch.tensor([tok.encode("The capital of France is")])
    assert torch.equal(ids, reference["input_ids"])
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=6)
    assert tok.decode(out[0].tolist()).startswith("The capital of France is the capital of")


def test_block_second_norm_reads_post_attention_residual(model):
    """A block must normalize `h`, not `x`, before the MLP. Both run; one is wrong."""
    cfg = model.cfg
    block = model.layers[0]
    torch.manual_seed(0)
    x = torch.randn(1, 4, cfg.hidden_size)
    cos, sin = model.rotary(torch.arange(4))
    from toyllm.layers import causal_mask

    mask = causal_mask(4)

    with torch.no_grad():
        correct = block(x, cos, sin, mask)
        h = x + block.self_attn(block.input_layernorm(x), cos, sin, mask)
        wrong = h + block.mlp(block.post_attention_layernorm(x))  # normalizes x, not h
    assert torch.allclose(correct, h + block.mlp(block.post_attention_layernorm(h)), atol=1e-6)
    assert not torch.allclose(correct, wrong, atol=1e-3)


def test_residual_stream_is_never_normalized_in_place(model):
    """Pre-norm: block output = input + something, so the stream passes through."""
    cfg = model.cfg
    block = Block(cfg)
    with torch.no_grad():  # zero both sublayers' output projections
        block.self_attn.o_proj.weight.zero_()
        block.mlp.down_proj.weight.zero_()
        x = torch.randn(1, 3, cfg.hidden_size)
        cos, sin = model.rotary(torch.arange(3))
        out = block(x, cos, sin, None)
    assert torch.allclose(out, x, atol=1e-6)
