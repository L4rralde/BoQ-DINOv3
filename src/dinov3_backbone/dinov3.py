import os

import torch

from .urls import dinov3_vitb16


class DinoV3(torch.nn.Module):
    AVAILABLE_MODELS = {
        'dinov3_vitb16': dinov3_vitb16
    }

    def __init__(
        self,
        dinov3_repo_path: os.PathLike,
        backbone_name: str="dinov3_vitb16",
        unfreeze_n_blocks: int=2,
        reshape_output: bool=True,
        norm_layer: bool=False
    ):
        super().__init__()
        
        print("Using DINOv3 with:")
        print(f" - backbone_name: {backbone_name}")
        print(f" - unfreeze_n_blocks: {unfreeze_n_blocks}")
        print(f" - reshape_output: {reshape_output}")
        print(f" - norm_layer: {norm_layer}")

        self.backbone_name = backbone_name
        self.unfreeze_n_blocks = unfreeze_n_blocks
        self.reshape_output = reshape_output
        self.norm_layer = norm_layer

        # make sure the backbone_name is in the available models
        if self.backbone_name not in self.AVAILABLE_MODELS:
            print(f"Backbone {self.backbone_name} is not recognized!, using dinov3_vitb16")
            self.backbone_name = "dinov3_vitb16"

        self.dino = torch.hub.load(
            dinov3_repo_path,
            self.backbone_name,
            source = 'local',
            weights = self.AVAILABLE_MODELS[self.backbone_name]
        )

        # freeze all parameters
        for param in self.dino.parameters():
            param.requires_grad = False


        if self.unfreeze_n_blocks == 0:
            self.frozen_blocks = self.dino.blocks
            self.trainable_blocks = []
        else:
            self.frozen_blocks = self.dino.blocks[:-self.unfreeze_n_blocks]
            self.trainable_blocks = self.dino.blocks[-unfreeze_n_blocks:]
    
        # unfreeze the last few blocks
        for block in self.trainable_blocks:
            for param in block.parameters():
                param.requires_grad = True
        
        self.out_channels = self.dino.num_features
        print("Out channels", self.out_channels)

    @property
    def patch_size(self) -> int:
        return self.dino.patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, h, w = x.shape

        with torch.no_grad():
            x, (H, W) = self.dino.prepare_tokens_with_masks(x)
            #Rope sincos positional embedding
            rope_sincos = self.dino.rope_embed(H=H, W=W)
            for blk in self.frozen_blocks:
                x = blk(x, rope_sincos)

        # Last blocks are trained
        for blk in self.trainable_blocks:
            x = blk(x, rope_sincos)

        norm_x = self.dino.norm(x)
        if self.norm_layer:
            x = norm_x

        class_token = norm_x[:, 0]
        register_token = x[:, 1: self.dino.n_storage_tokens + 1] #Probably it adds nothing for inference

        features = x[:, self.dino.n_storage_tokens + 1 :]
        if self.reshape_output:
            _, _, C = features.shape # or C = self.embed_dim
            patch_size = self.patch_size
            features = features.permute(0, 2, 1).view(B, C, h // patch_size, w // patch_size)

        return {
            'features': features,
            'cls': class_token,
            'registers': register_token
        }
