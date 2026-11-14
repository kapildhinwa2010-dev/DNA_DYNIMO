import torch

from dna_dynimo.fourier_dna import FourierDNA
from dna_dynimo.model_generator import (
    Frozen1BModel,
    estimate_model_parameters,
    generate_1b_model,
)


def test_small_checkpoint_round_trip(tmp_path):
    dna = FourierDNA(hidden=8, num_freqs=4, chunk_size=32)
    output = tmp_path / "nested" / "small-model.pt"
    result = generate_1b_model(dna, output, num_layers=2, layer_size=8)

    checkpoint = torch.load(output, map_location="cpu", weights_only=True)
    model = Frozen1BModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.inference_mode():
        prediction = model(torch.ones(2, 1))

    assert result["output_path"] == str(output.resolve())
    assert checkpoint["metadata"]["total_parameters"] == estimate_model_parameters(2, 8)
    assert prediction.shape == (2, 1)


def test_parameter_estimate_matches_model():
    model = Frozen1BModel(d=11, num_layers=3)
    actual = sum(parameter.numel() for parameter in model.parameters())
    assert estimate_model_parameters(3, 11) == actual
