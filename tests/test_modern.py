import pytest
import torch

from experiment import VARIANTS, main
from modern_gpt import ModernConfig, ModernGPT, RMSNorm, Rotary
from tinygpt.model import GPT, GPTConfig

BASE = dict(vocab_size=20, block_size=16, n_layer=2, n_head=4, n_embd=32, dropout=0.0)


def build(seed=0, **switches):
    torch.manual_seed(seed)
    return ModernGPT(ModernConfig(**BASE, **switches)).eval()


def test_switches_off_has_same_parameter_count_as_project_4():
    tiny = GPT(GPTConfig(**BASE))
    assert build().num_parameters() == tiny.num_parameters()


@pytest.mark.parametrize("name", VARIANTS)
def test_future_tokens_cannot_affect_past_predictions(name):
    model = build(**VARIANTS[name])
    ids = torch.randint(20, (1, 12))
    changed = ids.clone()
    changed[0, 8:] = (changed[0, 8:] + 1) % 20
    with torch.no_grad():
        first, _ = model(ids)
        second, _ = model(changed)
    assert torch.allclose(first[0, :8], second[0, :8], atol=1e-5)
    assert not torch.allclose(first[0, 8:], second[0, 8:], atol=1e-5)


@pytest.mark.parametrize("name", VARIANTS)
def test_kv_cache_gives_same_logits_as_full_run(name):
    model = build(**VARIANTS[name])
    ids = torch.randint(20, (2, 10))
    with torch.no_grad():
        full, _ = model(ids)
        logits, past = model.forward_cached(ids[:, :4])
        pieces = [logits]
        for i in range(4, 10):
            logits, past = model.forward_cached(ids[:, i : i + 1], past)
            pieces.append(logits)
    assert torch.allclose(torch.cat(pieces, dim=1), full, atol=1e-4)


@pytest.mark.parametrize("name", VARIANTS)
def test_cached_generation_matches_uncached_past_the_block_size(name):
    model = build(**VARIANTS[name])
    prompt = torch.randint(20, (1, 3))
    # 40 new tokens forces cache rebuilds. Near-zero temperature makes both methods
    # choose the most likely token, so their outputs should be identical.
    uncached = model.generate(prompt, 40, temperature=1e-6)
    cached = model.generate_cached(prompt, 40, temperature=1e-6)
    assert torch.equal(uncached, cached)
    assert cached.shape == (1, 43)


def test_flash_attention_matches_manual_attention():
    manual = build()
    flash = ModernGPT(ModernConfig(**BASE, use_flash=True)).eval()
    flash.load_state_dict(manual.state_dict())
    ids = torch.randint(20, (2, 12))
    with torch.no_grad():
        assert torch.allclose(manual(ids)[0], flash(ids)[0], atol=1e-5)


def test_grouped_query_attention_uses_fewer_parameters():
    assert build(n_kv_head=2).num_parameters() < build().num_parameters()


def test_n_kv_head_equal_to_n_head_is_ordinary_attention():
    assert build(n_kv_head=4).num_parameters() == build().num_parameters()


def test_invalid_kv_head_count_is_rejected():
    with pytest.raises(ValueError, match="divisible by n_kv_head"):
        build(n_kv_head=3)


def test_rope_removes_position_table():
    assert build(use_rope=True).position_embedding is None
    assert build().position_embedding is not None


def test_rotary_keeps_length_and_makes_dot_product_depend_on_distance():
    rotary = Rotary(8, 20)
    x = torch.randn(1, 1, 1, 8).repeat(1, 1, 20, 1)
    rotated = rotary.apply(x)
    assert torch.allclose(rotated.norm(dim=-1), x.norm(dim=-1), atol=1e-5)
    # Pairs the same distance apart have the same dot product, wherever they are.
    a = (rotated[0, 0, 2] * rotated[0, 0, 5]).sum()
    b = (rotated[0, 0, 11] * rotated[0, 0, 14]).sum()
    assert a.item() == pytest.approx(b.item(), abs=1e-4)
    # Position 0 isn't rotated at all.
    assert torch.allclose(rotated[0, 0, 0], x[0, 0, 0])


def test_rotary_requires_even_head_size():
    with pytest.raises(ValueError, match="even"):
        Rotary(7, 10)


def test_rmsnorm_gives_unit_root_mean_square():
    out = RMSNorm(16)(torch.randn(4, 16) * 50)
    assert out.pow(2).mean(dim=-1).sqrt().mean().item() == pytest.approx(1.0, abs=1e-3)


def test_input_past_block_size_is_rejected():
    with pytest.raises(ValueError, match="exceeds block_size"):
        build()(torch.zeros(1, 17, dtype=torch.long))


def test_experiment_end_to_end(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.txt").write_text("ANNA: hello there\nBEN: hi\n\n" * 50)
    (data / "val.txt").write_text("BEN: hello\nANNA: hi there\n\n" * 10)
    out = tmp_path / "out"
    main(["--data", str(data), "--out", str(out), "--iters", "4", "--seeds", "1", "--layers", "1", "--embd", "16", "--block", "16"])
    text = (out / "results.md").read_text(encoding="utf-8")
    for name in VARIANTS:
        assert name in text
    assert (out / "comparison.png").exists()
