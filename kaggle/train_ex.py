from models.biformer import BiFormer
from torchinfo import summary
import torch
import torch.nn as nn
model_urls = {
    "biformer_tiny_in1k": "https://matix.li/e36fe9fb086c",
    "biformer_small_in1k": "https://api.onedrive.com/v1.0/shares/s!AkBbczdRlZvChHDyM-x9KWRBZ832/root/content",
    "biformer_base_in1k": "https://api.onedrive.com/v1.0/shares/s!AkBbczdRlZvChHI_XPhoadjaNxtO/root/content",
}

model = BiFormer(
        depth=[2, 2, 8, 2],
        embed_dim=[64, 128, 256, 512], mlp_ratios=[3, 3, 3, 3],
        num_classes=2,
        #------------------------------
        n_win=7,
        kv_downsample_mode='identity',
        kv_per_wins=[-1, -1, -1, -1],
        topks=[1, 4, 16, -2],
        side_dwconv=5,
        before_attn_dwconv=3,
        layer_scale_init_value=-1,
        qk_dims=[64, 128, 256, 512],
        head_dim=32,
        param_routing=False, diff_routing=False, soft_routing=False,
        pre_norm=True,
        pe=None
    #-------------------------------
    )
# Assuming 'downsample_layers' is a Sequential module
downsample_layers = model.downsample_layers

# Print the downsample_layers architecture
print(downsample_layers)

#model.default_cfg = _cfg()

model_key = 'biformer_tiny_in1k'
url = model_urls[model_key]
checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", file_name=f"{model_key}.pth")

for k in list(checkpoint["model"].keys()):
    if "head" in k:
        del checkpoint["model"][k]
model.load_state_dict(checkpoint["model"],strict=False)
summary(model,input_size=(1,3, 224,224))