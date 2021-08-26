import torch
import torch.nn as nn
import math
from transfer import transfer_loader

class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class GlobalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8, fp32fft=True):
        super().__init__()

        # (14, 8, 768, 2)
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h
        self.fp32fft = fp32fft

    def forward(self, x, spatial_size=None):
        B, N, C = x.shape   # (B, 196, 768)
        a = b = int(math.sqrt(N))  # sqrt(196) == 14

        x = x.view(B, a, b, C) # (B, 14, 14, 768)

        if self.fp32fft:
            dtype = x.dtype
            x = x.to(torch.float32)

        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho') # (B, 14, 8, 768)
        weight = torch.view_as_complex(self.complex_weight) # complex  (14, 8, 768)
        x = x * weight 
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')
        # normalize by 1/sqrt(n) (making the real FFT orthonormal)
        # (B, 14, 8, 768) -> # (B, 14, 14, 768)

        if self.fp32fft:
            x = x.to(dtype)

        x = x.reshape(B, N, C) # (B, 196, 768)

        return x

class FilterBlock(nn.Module):
    # dim = 768
    def __init__(self, dim, mlp_ratio=4., drop=0., h=14, w=8, fp32fft=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim) 
        self.filter = GlobalFilter(dim, h=h, w=w, fp32fft=fp32fft)
        self.norm2 = nn.LayerNorm(dim)
        ffn_hidden_dim = int(dim * mlp_ratio)
        self.ffn = FFN(in_features=dim, hidden_features=ffn_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.ffn(self.norm2(self.filter(self.norm1(x)))) # skip-connection
        return x # -> # (B, 196, 768)

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size)  # 224, 224
        patch_size = (patch_size , patch_size)  # 16, 16
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
   
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2) # (B, 768, 14, 14) -> (B, 768, 196) -> (B, 196, 768)
        return x

class GFNet(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=9, embed_dim=768, depth=12,
                 mlp_ratio=4., drop_rate=0.5, fp32fft=True):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim 

        self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches 

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))   # (1, 196, 768)
        self.pos_drop = nn.Dropout(p=drop_rate)

        h = img_size // patch_size  # 14 
        w = h // 2 + 1  # 8

        self.blocks = nn.ModuleList([
            FilterBlock(
                dim=embed_dim, mlp_ratio=mlp_ratio,
                drop=drop_rate, h=h, w=w, fp32fft=fp32fft)
            for i in range(depth)])   

        self.norm = nn.LayerNorm(embed_dim)

        # Classifier head
        self.head = nn.Linear(self.num_features, num_classes) 

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x).mean(1) # pooling
        x = self.head(x)
        return x    

if __name__ == "__main__":
    x = torch.rand((10, 3, 224, 224))
    model = GFNet()
    ypred = model(x)
    print(ypred.shape)
    torch.save(model.state_dict(), './weight.pth')
    print(model.state_dict()["head.bias"])
    model = transfer_loader(model, "./weight.pth")
    print(model.state_dict()["head.bias"])