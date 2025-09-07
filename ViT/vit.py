# Made by Ryan Geisen
import torch, math
import torch.nn as nn
import torch.nn.functional as F
import einops


# Takes (batch size, num frames, height, width, num channels) as input -> returns (batch size, number of tubes, 1D representation of tube)
class VidInputEmbedding(nn.Module):
    def __init__(self, patch_size=16, num_frames=4, n_channels=3, device='cuda', latent_size=768, batch_size=8, image_size=128):
        super(VidInputEmbedding, self).__init__()
        self.latent_size = latent_size      # D: dimension of embeddings
        self.patch_size = patch_size        # P: size of each patch 
        self.num_frames = num_frames        # F: number of frames
        self.n_channels = n_channels        # C: number of channels
        self.device = device
        self.batch_size = batch_size        # B: Batch size
        self.num_patch = (image_size//patch_size)**2    # N: number of patches per frame
        self.input_size = self.patch_size * self.patch_size * self.n_channels   # Size of flattened patch
        self.pos_scale = math.sqrt(self.latent_size)
        # Linear projection from patch size to latent size
        # in: (P * P * C) -> out: D
        self.linearProjection = nn.Linear(self.input_size, self.latent_size)
        
        # Class token to be prepended to each frame
        # Shape: [1, 1, D]
        self.class_token = nn.Parameter(torch.randn(1, 1, self.latent_size)).to(device)
        
        # Positional embedding added to patches + cls token
        # Shape: [1, 1, N+1, D] where N+1 accounts for cls token
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patch + 1, 1)
            ).to(device)
    
    def forward(self, input_data):
        # input data shape: [B, H, W, C]
        # where H=W=image_size
        input_data = input_data.to(self.device)

        if self.n_channels == 3:
            # Rearrange into patches
            # From: [B, H, W, C]
            # To: [B, (H/P)*(W/P), P*P*C]
            patches = einops.rearrange(
                input_data, 'b (h h1) (w w1) c -> b (h w) (h1 w1 c)',
                h1=self.patch_size, w1=self.patch_size
            )
        else:
            # Rearrange into patches
            # From: [B, H, W]
            # To: [B, (H/P)*(W/P), P*P]
            patches = einops.rearrange(
                input_data, 'b (h h1) (w w1) -> b (h w) (h1 w1)',
                h1=self.patch_size, w1=self.patch_size
            )

        # Project patches to latent dimension
        # From: [B, N, P*P*C] -> [B, N, D]
        linear_projection = self.linearProjection(patches)
        
        batch_size, num_patches,_= linear_projection.shape
        
        # Create and append class tokens
        # class_tokens shape: [B, 1, D]
        class_tokens = self.class_token.expand(batch_size, 1, self.latent_size)

        
        # Concatenate to get [B, N+1, D] (adds one D dimensional token per image)
        linear_projection = torch.cat((class_tokens, linear_projection), dim=1)

        
        # Add positional embedding (Pos embeddings are an offset of existing values not additional tokens)
        # pos_embedding expands from [1, N+1, D] to [B, N+1, D] (add same offset to patch in same spatial location irrelevant of frame)
        linear_projection += self.pos_embedding.expand(batch_size, -1, self.latent_size)/self.pos_scale

        
        return linear_projection

class SpatialTransformerEncoder(nn.Module):
    def __init__(self, latent_size=768, num_heads=12, dropout=0.1):
        super(SpatialTransformerEncoder, self).__init__()
        self.latent_size = latent_size
        
        # For processing patches within a single frame
        self.norm1 = nn.LayerNorm(latent_size)
        self.attn = nn.MultiheadAttention(latent_size, num_heads, dropout=dropout, batch_first=True)
        
        # Feed-forward network
        self.norm2 = nn.LayerNorm(latent_size)
        self.ffn = nn.Sequential(
            nn.Linear(latent_size, latent_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_size * 4, latent_size),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # x shape: [p (b*f) d] where:
        # b*f = batch_size * num_frames (treating each frame independently)
        # p = num_patches + 1 (patches + class token)
        # d = latent_size
        
        # Self-attention among patches
        norm_x = self.norm1(x)
        attn_outputs, attn_weights = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_outputs
        
        # FFN
        x = x + self.ffn(self.norm2(x))
        
        return x, attn_weights


class Vit(nn.Module):
    def __init__(self, num_spatial_encoders=12, num_temporal_encoders=4, latent_size=768, 
                 device='cuda', num_heads=12, num_class=2, dropout=0.1, patch_size=16, 
                 num_frames=4, n_channels=3, batch_size=8, image_size = 128):
        super(ViVit, self).__init__()
        self.num_spatial_encoders = num_spatial_encoders        # Number of encoders for spatial encoding
        self.latent_size = latent_size                          # Dimension of qkv
        self.device = device                                    # Device model is on
        self.num_class = num_class                              # Numer of Classes (Binary so only 2)
        self.dropout = dropout                                  # Amount of dropout to regulate overfitting
        self.batch_size = batch_size                            # Batch Size for reference
        self.num_patches = (image_size//patch_size)**2          # Total number of patches
        
        # Embeds Video Spatially (i.e. vectorizes frames and adds positional embedding and classification tokens)
        self.spatial_embedding = VidInputEmbedding(patch_size, num_frames, n_channels, device, latent_size, batch_size,image_size)
        
        # Creates the encoder stacks
        self.spatial_encStack = nn.ModuleList([SpatialTransformerEncoder(latent_size, num_heads, dropout) for _ in range(self.num_spatial_encoders)])
        
        # Final classification
        self.MLP_head = nn.Sequential(
            nn.LayerNorm(latent_size), # should be latent_size * num_frames if doing all class tokens + patches
            nn.Dropout(dropout),
            nn.Linear(latent_size, latent_size// 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_size // 2, num_class)
        )
        
    def forward(self, input):
        '''
        B = Batch Size
        F = Number of Frames in one sample (Every sample has the same number of frames)
        H = Height of one Frame (all frames same height in all samples)
        W = Width of Frames in one sample (all frames are same width in all samples)
        C = Number of Channels (If only 1 channel, this dimension is squeezed)
        P = Number of patches + 1 class token
        D = Latent Dimension
        '''
        # Spatial Processing

        embedded_input = self.spatial_embedding(input) #[B H W C?] -> [B,P,D]
        batch_size, num_patches, latent_dim = embedded_input.shape
        
        # Ensure each frame is processed independently for spatial relationships within a frame (VIT essentially)
        # Basically makes frames * batch size the effective batch size for processing images rather than videos
        spatial_input = embedded_input
        
        # Spatial attention - maintains all patch information
        for enc_layer in self.spatial_encStack:
            spatial_input, _ = enc_layer(spatial_input)
        
        
        spatial_cls_token = spatial_input[:,0,:]    # [B P D] -> [B D] (only take class token)
        
        # Classification
        output = self.MLP_head(spatial_cls_token)

        return output