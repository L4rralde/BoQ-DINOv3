dependencies = ['torch', 'torchvision']

import sys
import os

# Add BoQ's src directory directly to path
boq_root = os.path.dirname(__file__)  # Root of the cloned repo
sys.path.append(os.path.join(boq_root, "src"))  

import torch
from backbones import ResNet, DinoV2
from boq import BoQ
from dinov3_backbone import DinoV3

    

class VPRModel(torch.nn.Module):
    def __init__(self, 
                 backbone,
                 aggregator):
        super().__init__()
        self.backbone = backbone
        self.aggregator = aggregator
        
    def forward(self, x):
        x = self.backbone(x)
        x, attns = self.aggregator(x)
        return x, attns


AVAILABLE_BACKBONES = {
    # this list will be extended
    # "resnet18": [8192 , 4096],
    "resnet50": [16384],
    "dinov2": [12288],
}

MODEL_URLS = {
    "resnet50_16384": "https://github.com/amaralibey/Bag-of-Queries/releases/download/v1.0/resnet50_16384.pth",
    "dinov2_12288": "https://github.com/amaralibey/Bag-of-Queries/releases/download/v1.0/dinov2_12288.pth",
    # "resnet50_4096": "",
}

def get_trained_boq(backbone_name="resnet50", output_dim=16384):
    if backbone_name not in AVAILABLE_BACKBONES:
        raise ValueError(f"backbone_name should be one of {list(AVAILABLE_BACKBONES.keys())}")
    try:
        output_dim = int(output_dim)
    except:
        raise ValueError(f"output_dim should be an integer, not a {type(output_dim)}")
    if output_dim not in AVAILABLE_BACKBONES[backbone_name]:
        raise ValueError(f"output_dim should be one of {AVAILABLE_BACKBONES[backbone_name]}")
    
    if "dinov2" in backbone_name:
        # load the backbone
        backbone = DinoV2()
        # load the aggregator
        aggregator = BoQ(
            in_channels=backbone.out_channels,  # make sure the backbone has out_channels attribute
            proj_channels=384,
            num_queries=64,
            num_layers=2,
            row_dim=output_dim//384, # 32 for dinov2
        )
        
    elif "resnet" in backbone_name:
        backbone = ResNet(
                backbone_name=backbone_name,
                crop_last_block=True,
            )
        aggregator = BoQ(
                in_channels=backbone.out_channels,  # make sure the backbone has out_channels attribute
                proj_channels=512,
                num_queries=64,
                num_layers=2,
                row_dim=output_dim//512, # 32 for resnet
            )

    vpr_model = VPRModel(
            backbone=backbone,
            aggregator=aggregator
        )
    
    vpr_model.load_state_dict(
        torch.hub.load_state_dict_from_url(
            MODEL_URLS[f"{backbone_name}_{output_dim}"],
            map_location=torch.device('cpu')
        )
    )
    return vpr_model



def get_dinov3_boq(model_name="dinov3"):
    MODEL_URLS = {
        "dinov2": "https://github.com/L4rralde/BoQ-DINOv3/releases/download/dinov3_exp1/dinov2.ckpt",
        "dinov3": "https://github.com/L4rralde/BoQ-DINOv3/releases/download/dinov3_exp1/dinov3.ckpt",
        "dinov3_norm": "https://github.com/L4rralde/BoQ-DINOv3/releases/download/dinov3_exp1/dinov3_norm.ckpt"
    }
    if model_name not in MODEL_URLS:
        raise ValueError(f"backbone_name should be one of {list(MODEL_URLS.keys())}")

    if "dinov3" in model_name:
        dinov3_repo_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'submodules', 'dinov3'
        )
        backbone = DinoV3(
            dinov3_repo_path,
            backbone_name='dinov3_vitb16',
            unfreeze_n_blocks=0,
            norm_layer=(model_name == "dinov3_norm")
        )

    # Instantiate the backbone and define the image size for training and validation
    elif "dinov2" in model_name:
        backbone = DinoV2(
            backbone_name='dinov2_vitb14',
            unfreeze_n_blocks=0,
            norm_layer=False
        )

    else:
        pass #FUTURE

    aggregator = BoQ(
        in_channels=backbone.out_channels,
        proj_channels=512,
        num_queries=64,
        num_layers=2,
        row_dim=16,
    )

    vpr_model = VPRModel(
        backbone=backbone,
        aggregator=aggregator
    )

    checkpoint = torch.hub.load_state_dict_from_url(
        MODEL_URLS[model_name],
        map_location=torch.device('cpu')
    )
    vpr_model.load_state_dict(checkpoint['state_dict'])

    return vpr_model
