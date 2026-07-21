"""Unit tests for the model runner's hook mechanics on a tiny ungated model.

These mirror the §8 known-answer tests at toy scale so hook correctness is
CI-checkable without a large download (tiny-random-LlamaForCausalLM ~ a few MB).
Skipped cleanly when the model can't be fetched (offline CI).
"""
import numpy as np
import pytest

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def subject():
    try:
        from models import HFSubject

        return HFSubject(TINY, "main", device="cpu")
    except Exception as e:  # offline / hub unavailable
        pytest.skip(f"tiny model unavailable: {e}")


def test_generation_logs_decoding_and_is_seed_deterministic(subject):
    a = subject.generate("Hello", seed=7, max_new_tokens=8)
    b = subject.generate("Hello", seed=7, max_new_tokens=8)
    c = subject.generate("Hello", seed=8, max_new_tokens=8)
    assert a.meta["decoding"]["seed"] == 7
    assert a.meta["decoding"]["temperature"] == 1.0
    assert a.output_text == b.output_text  # same seed, same output
    # different seed *may* coincide on a tiny vocab, but decoding meta must differ
    assert c.meta["decoding"]["seed"] == 8


def test_rescore_shapes_and_layers(subject):
    last = subject.n_layers - 1
    cap = subject.rescore("one two three", read_layers=(0, last))
    assert set(cap.activations) == {0, last}
    d_model = subject.model.config.hidden_size
    for acts in cap.activations.values():
        assert acts.ndim == 2 and acts.shape[1] == d_model


def test_same_site_steering_shift_is_lambda_w_dot_v(subject):
    rng = np.random.default_rng(0)
    d = subject.model.config.hidden_size
    w = rng.normal(size=d)
    v = rng.normal(size=d).astype(np.float32)
    lam = 0.7
    from models.runner import SteeringHook

    read = subject.n_layers - 1
    clean = subject.rescore("a b c d", read_layers=(read,)).activations[read]
    steered = subject.rescore(
        "a b c d", read_layers=(read,),
        steering=[SteeringHook(layer=read, vector=v, coefficient=lam)],
    ).activations[read]
    shift = (steered.mean(0) - clean.mean(0)) @ w
    predicted = lam * (w @ v.astype(np.float64))
    assert shift == pytest.approx(predicted, rel=1e-3)


def test_above_read_injection_is_bit_identical_at_read_layer(subject):
    from models.runner import SteeringHook

    rng = np.random.default_rng(1)
    v = rng.normal(size=subject.model.config.hidden_size).astype(np.float32)
    read, above_layer = 0, subject.n_layers - 1
    clean = subject.rescore("x y z", read_layers=(read,)).activations[read]
    above = subject.rescore(
        "x y z", read_layers=(read,),
        steering=[SteeringHook(layer=above_layer, vector=v, coefficient=100.0)],
    ).activations[read]
    assert np.array_equal(clean, above)  # Pre-Reg §4 [FIX2-1] layer-ordering test


def test_below_read_injection_reaches_read_layer(subject):
    from models.runner import SteeringHook

    rng = np.random.default_rng(2)
    v = rng.normal(size=subject.model.config.hidden_size).astype(np.float32)
    read = subject.n_layers - 1
    clean = subject.rescore("x y z", read_layers=(read,)).activations[read]
    below = subject.rescore(
        "x y z", read_layers=(read,),
        steering=[SteeringHook(layer=0, vector=v, coefficient=0.5)],
    ).activations[read]
    assert not np.array_equal(clean, below)  # additive carry


def test_activation_cache_roundtrip(tmp_path):
    from models.runner import ActivationCache

    cache = ActivationCache(tmp_path)
    key = cache.key("subj", "item", "cond", (1, 2))
    assert cache.get(key) is None
    acts = {1: np.random.default_rng(0).normal(size=(5, 8)).astype(np.float32)}
    cache.put(key, acts)
    back = cache.get(key)
    assert np.array_equal(back[1], acts[1])
    # key must be sensitive to every component (extraction-code version is in subject id)
    assert cache.key("subj2", "item", "cond", (1, 2)) != key
    assert cache.key("subj", "item", "cond", (1, 3)) != key
