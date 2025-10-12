import torch


class ProjectionHead(torch.nn.Module):
    def __init__(self,
        input_size: int=768,
        hidden_size: int=512,
        output_size: int=256
    ) -> None:
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
