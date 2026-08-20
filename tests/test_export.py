import math

import pytest
import torch
from torch import nn

from splade_poc.export_winner import SpladeDocumentWrapper


class FakeMaskedLanguageModel(nn.Module):
    def forward(self, input_ids, attention_mask, token_type_ids):
        del input_ids, attention_mask, token_type_ids
        return (
            torch.tensor(
                [[[0.0, 1.0, -1.0, 3.0], [100.0, 100.0, 100.0, 100.0]]],
                dtype=torch.float32,
            ),
        )


def test_document_wrapper_masks_then_applies_splade_pooling_and_topk() -> None:
    wrapper = SpladeDocumentWrapper(FakeMaskedLanguageModel(), top_k=2)

    values, indices = wrapper(
        torch.tensor([[10, 20]], dtype=torch.int64),
        torch.tensor([[1, 0]], dtype=torch.int64),
        torch.tensor([[0, 0]], dtype=torch.int64),
    )

    assert indices.tolist() == [[3, 1]]
    assert values.tolist()[0][0] == pytest.approx(math.log(4.0))
    assert values.tolist()[0][1] == pytest.approx(math.log(2.0))
