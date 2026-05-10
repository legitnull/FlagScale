"""Test that chunked cross-entropy matches standard F.cross_entropy."""

import torch
import torch.nn.functional as F

from flagscale.train.utils.chunked_cross_entropy import chunked_cross_entropy_loss


def test_chunked_ce_matches_standard():
    torch.manual_seed(42)
    B, S, V = 2, 512, 151936
    logits = torch.randn(B, S, V, dtype=torch.bfloat16, device="cuda")
    labels = torch.randint(0, V, (B, S), device="cuda")
    labels[labels % 10 == 0] = -100

    ref = F.cross_entropy(logits.reshape(-1, V).float(), labels.reshape(-1), ignore_index=-100)
    chunked = chunked_cross_entropy_loss(logits, labels, chunk_tokens=128)
    assert torch.allclose(ref, chunked, atol=1e-4), (
        f"ref={ref.item():.6f}, chunked={chunked.item():.6f}"
    )


def test_chunked_ce_different_chunk_sizes():
    torch.manual_seed(123)
    B, S, V = 4, 256, 32000
    logits = torch.randn(B, S, V, dtype=torch.bfloat16, device="cuda")
    labels = torch.randint(0, V, (B, S), device="cuda")
    labels[:, :10] = -100

    ref = F.cross_entropy(logits.reshape(-1, V).float(), labels.reshape(-1), ignore_index=-100)
    for chunk_tokens in [32, 64, 128, 256, 512]:
        chunked = chunked_cross_entropy_loss(logits, labels, chunk_tokens=chunk_tokens)
        assert torch.allclose(ref, chunked, atol=1e-4), (
            f"chunk_tokens={chunk_tokens}: ref={ref.item():.6f}, chunked={chunked.item():.6f}"
        )


def test_chunked_ce_all_masked():
    B, S, V = 2, 64, 1000
    logits = torch.randn(B, S, V, dtype=torch.bfloat16, device="cuda")
    labels = torch.full((B, S), -100, device="cuda", dtype=torch.long)
    loss = chunked_cross_entropy_loss(logits, labels, chunk_tokens=16)
    assert loss.item() == 0.0


def test_no_chunking_fallback():
    torch.manual_seed(7)
    B, S, V = 2, 128, 32000
    logits = torch.randn(B, S, V, dtype=torch.bfloat16, device="cuda")
    labels = torch.randint(0, V, (B, S), device="cuda")

    ref = F.cross_entropy(logits.reshape(-1, V).float(), labels.reshape(-1), ignore_index=-100)
    result = chunked_cross_entropy_loss(logits, labels, chunk_tokens=0)
    assert torch.allclose(ref, result, atol=1e-5), (
        f"ref={ref.item():.6f}, result={result.item():.6f}"
    )


if __name__ == "__main__":
    test_chunked_ce_matches_standard()
    print("PASSED: test_chunked_ce_matches_standard")
    test_chunked_ce_different_chunk_sizes()
    print("PASSED: test_chunked_ce_different_chunk_sizes")
    test_chunked_ce_all_masked()
    print("PASSED: test_chunked_ce_all_masked")
    test_no_chunking_fallback()
    print("PASSED: test_no_chunking_fallback")
    print("All tests passed!")
