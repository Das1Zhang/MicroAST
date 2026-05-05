import torch.nn as nn

from function import adaptive_instance_normalization as featMod
from function import calc_mean_std

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groupnum):
        super(ConvLayer, self).__init__()
        # Padding Layer
        padding_size = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding_size)

        # Convolution Layer
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, groups=groupnum)

    def forward(self, x):
        x = self.reflection_pad(x)
        x = self.conv_layer(x)
        return x

class ResidualLayer(nn.Module):
    def __init__(self, channels=128, kernel_size=3, groupnum=1):
        super(ResidualLayer, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size, stride=1, groupnum=groupnum)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size, stride=1, groupnum=groupnum)

    def forward(self, x, weight=None, bias=None, filterMod=False):
        if filterMod:
            x1 = self.conv1(x)
            x2 = weight * x1 + bias * x
            
            x3 = self.relu(x2)
            x4 = self.conv2(x3)
            x5 = weight * x4 + bias * x3
            return x + x5
        else: 
            return x + self.conv2(self.relu(self.conv1(x)))

class SELayer(nn.Module):
    """Squeeze-and-Excitation channel attention module.

    Lightweight mechanism that adaptively recalibrates channel-wise feature
    responses by explicitly modeling interdependencies between channels.
    Computation overhead is negligible (< 1% FLOPs increase).

    Squeeze: Global average pooling compresses spatial info into channel descriptors.
    Excitation: Two FC layers learn channel-wise dependencies (C -> C/r -> C).
    Scale: Attention weights are broadcast back to the spatial dimensions.

    With reduction=4 and C=64, adds only ~2K parameters per SE block.
    """
    def __init__(self, channels, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze: H x W -> 1 x 1
        y = self.avg_pool(x).view(b, c)
        # Excitation: learn per-channel importance weights
        y = self.fc(y).view(b, c, 1, 1)
        # Scale: re-weight each channel
        return x * y.expand_as(x)


class StructureAwarenessBranch(nn.Module):
    """Spatial structure awareness branch — generates structure preservation maps.

    This extremely lightweight branch takes content encoder features and produces
    a per-pixel structure preservation map. The map is used to spatially weight
    the AdaIN style modulation in the Decoder:
      - High map value → strong content structure (edges, textures, object boundaries)
                         → reduce style intensity to preserve content
      - Low map value  → smooth region (sky, background)
                         → apply full style intensity

    Architecture: two 3x3 convs with a C//8 bottleneck (~4.5K params each).
    The 3x3 kernel provides local spatial pattern awareness beyond per-pixel
    channel statistics, which is essential for detecting edges and textures.

    No detach — gradients flow back to the content encoder, encouraging it to
    produce features that are more structure-aware.
    """
    def __init__(self, channels):
        super(StructureAwarenessBranch, self).__init__()
        bottleneck = max(channels // 8, 4)  # ensure at least 4 channels
        self.conv = nn.Sequential(
            nn.Conv2d(channels, bottleneck, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # outputs [0, 1] structure preservation weights
        )

    def forward(self, feat):
        # feat: [B, C, H, W] → structure_map: [B, 1, H, W]
        return self.conv(feat)


# Control the number of channels
slim_factor = 1

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.enc1 = nn.Sequential(
            ConvLayer(3, int(16*slim_factor), kernel_size=9, stride=1, groupnum=1),
            nn.ReLU(inplace=True),
            ConvLayer(int(16*slim_factor), int(32*slim_factor), kernel_size=3, stride=2, groupnum=int(16*slim_factor)),
            nn.ReLU(inplace=True),
            ConvLayer(int(32*slim_factor), int(32*slim_factor), kernel_size=1, stride=1, groupnum=1),
            nn.ReLU(inplace=True),
            ConvLayer(int(32*slim_factor), int(64*slim_factor), kernel_size=3, stride=2, groupnum=int(32*slim_factor)),
            nn.ReLU(inplace=True),
            ConvLayer(int(64*slim_factor), int(64*slim_factor), kernel_size=1, stride=1, groupnum=1),
            nn.ReLU(inplace=True),
            ResidualLayer(int(64*slim_factor), kernel_size=3),
            )
        self.enc2 = nn.Sequential(
            ResidualLayer(int(64*slim_factor), kernel_size=3)
            )
        # SE 通道注意力：在 enc1 和 enc2 输出后各加一个 SE 块，
        # 让编码器在有限深度下自动强化最具代表性的特征通道
        self.se1 = SELayer(int(64*slim_factor), reduction=4)
        self.se2 = SELayer(int(64*slim_factor), reduction=4)

    def forward(self, x):
        x1 = self.enc1(x)
        x1 = self.se1(x1)   # 通道注意力精炼浅层特征
        x2 = self.enc2(x1)
        x2 = self.se2(x2)   # 通道注意力精炼深层特征
        out = [x1, x2]
        return out


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.dec1 = ResidualLayer(int(64*slim_factor), kernel_size=3)
        self.dec2 = ResidualLayer(int(64*slim_factor), kernel_size=3)
        self.dec3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvLayer(int(64*slim_factor), int(32*slim_factor), kernel_size=3, stride=1, groupnum=int(32*slim_factor)),
            nn.ReLU(inplace=True),
            ConvLayer(int(32*slim_factor), int(32*slim_factor), kernel_size=1, stride=1, groupnum=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvLayer(int(32*slim_factor), int(16*slim_factor), kernel_size=3, stride=1, groupnum=int(16*slim_factor)),
            nn.ReLU(inplace=True),
            ConvLayer(int(16*slim_factor), int(16*slim_factor), kernel_size=1, stride=1, groupnum=1),
            nn.ReLU(inplace=True),
            ConvLayer(int(16*slim_factor), 3, kernel_size=9, stride=1, groupnum=1)
            )
        # SE 通道注意力：在 dec1 和 dec2 残差块输出后各加一个 SE 块，
        # 帮助解码器在上采样重建过程中聚焦关键特征通道
        self.se1 = SELayer(int(64*slim_factor), reduction=4)
        self.se2 = SELayer(int(64*slim_factor), reduction=4)
        # 空间结构感知分支：从内容特征生成逐像素结构保持图，
        # 用于对 AdaIN 风格注入做空间加权的调制
        self.structure_branch_1 = StructureAwarenessBranch(int(64*slim_factor))
        self.structure_branch_2 = StructureAwarenessBranch(int(64*slim_factor))

    def forward(self, x, s, w, b, alpha):
        # --- 生成空间结构保持图 ---
        # structure_map_1: 深层语义结构（物体轮廓、主体边界）
        # structure_map_2: 浅层纹理结构（局部边缘、细节）
        structure_map_1 = self.structure_branch_1(x[1])  # [B, 1, H, W]
        structure_map_2 = self.structure_branch_2(x[0])  # [B, 1, H, W]

        # --- 第一层风格注入（深层 AdaIN）：结构感知加权 ---
        x1 = featMod(x[1], s[1])
        # 结构越强的区域，风格化越弱（保留内容）；平滑区域充分风格化
        spatial_alpha_1 = alpha * (1 - structure_map_1)
        x1 = spatial_alpha_1 * x1 + (1 - spatial_alpha_1) * x[1]

        x2 = self.dec1(x1, w[1], b[1], filterMod=True)
        x2 = self.se1(x2)  # 通道注意力精炼第一层解码特征

        # --- 第二层风格注入（浅层 AdaIN）：结构感知加权 ---
        x3 = featMod(x2, s[0])
        spatial_alpha_2 = alpha * (1 - structure_map_2)
        x3 = spatial_alpha_2 * x3 + (1 - spatial_alpha_2) * x2

        x4 = self.dec2(x3, w[0], b[0], filterMod=True)
        x4 = self.se2(x4)  # 通道注意力精炼第二层解码特征

        out = self.dec3(x4)
        return out



class Modulator(nn.Module):
    def __init__(self):
        super(Modulator, self).__init__()
        self.weight1 = nn.Sequential(
            ConvLayer(int(64*slim_factor), int(64*slim_factor), kernel_size=3, stride=1, groupnum=int(64*slim_factor)),
            nn.AdaptiveAvgPool2d((1,1))
            )  
        self.bias1 = nn.Sequential(
            ConvLayer(int(64*slim_factor), int(64*slim_factor), kernel_size=3, stride=1, groupnum=int(64*slim_factor)),
            nn.AdaptiveAvgPool2d((1,1))
            )
        self.weight2 = nn.Sequential(
            ConvLayer(int(64*slim_factor), int(64*slim_factor), kernel_size=3, stride=1, groupnum=int(64*slim_factor)),
            nn.AdaptiveAvgPool2d((1,1))
            )  
        self.bias2 = nn.Sequential(
            ConvLayer(int(64*slim_factor), int(64*slim_factor), kernel_size=3, stride=1, groupnum=int(64*slim_factor)),
            nn.AdaptiveAvgPool2d((1,1))
            )

    def forward(self, x):
        w1 = self.weight1(x[0])
        b1 = self.bias1(x[0])
        
        w2 = self.weight2(x[1])
        b2 = self.bias2(x[1])
        
        return [w1,w2], [b1,b2]


vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)


class Net(nn.Module):
    def __init__(self, vgg, content_encoder, style_encoder, modulator, decoder):
        super(Net, self).__init__()
        vgg_enc_layers = list(vgg.children())
        self.vgg_enc_1 = nn.Sequential(*vgg_enc_layers[:4])  # input -> relu1_1
        self.vgg_enc_2 = nn.Sequential(*vgg_enc_layers[4:11])  # relu1_1 -> relu2_1
        self.vgg_enc_3 = nn.Sequential(*vgg_enc_layers[11:18])  # relu2_1 -> relu3_1
        self.vgg_enc_4 = nn.Sequential(*vgg_enc_layers[18:31])  # relu3_1 -> relu4_1
        
        self.content_encoder = content_encoder
        self.style_encoder = style_encoder
        self.modulator = modulator
        self.decoder = decoder
        self.mse_loss = nn.MSELoss()

        # fix the encoder
        for name in ['vgg_enc_1', 'vgg_enc_2', 'vgg_enc_3', 'vgg_enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_vgg_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'vgg_enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]
   

    # extract relu4_1 from input image
    def encode_vgg_content(self, input):
        for i in range(4):
            input = getattr(self, 'vgg_enc_{:d}'.format(i + 1))(input)
        return input
    
    
    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def _pairwise_style_loss(self, A, B):
        """Compute [B, B] pairwise style loss matrix on GPU.

        Replaces the nested for-loop: for each (i, j), computes
        calc_style_loss(A[i], B[j]) via broadcasting.

        A, B: [B, C, H, W]
        Returns: [B, B] where entry [i, j] = style_loss(A[i], B[j])
        """
        A_mean, A_std = calc_mean_std(A)  # [B, C, 1, 1]
        B_mean, B_std = calc_mean_std(B)  # [B, C, 1, 1]
        # Broadcasting: [B, 1, C, 1, 1] - [1, B, C, 1, 1] = [B, B, C, 1, 1]
        mean_diff = (A_mean.unsqueeze(1) - B_mean.unsqueeze(0)).pow(2)
        std_diff  = (A_std.unsqueeze(1)  - B_std.unsqueeze(0)).pow(2)
        return (mean_diff + std_diff).mean(dim=2).view(A.size(0), B.size(0))

    def _pairwise_content_loss(self, A, B):
        """Compute [B, B] pairwise content loss matrix on GPU.

        A, B: [B, C, 1, 1] (modulation signals)
        Returns: [B, B] where entry [i, j] = mse(A[i], B[j])
        """
        diff = (A.unsqueeze(1) - B.unsqueeze(0)).pow(2)  # [B, B, C, 1, 1]
        return diff.mean(dim=2).view(A.size(0), B.size(0))
    
    

    def forward(self, content, style, alpha=1.0):
        assert 0 <= alpha <= 1
        
        # extract style modulation signals
        style_feats = self.style_encoder(style)
        filter_weights, filter_biases = self.modulator(style_feats)

        # extract content features
        content_feats = self.content_encoder(content)

        # generate results  
        res = self.decoder(content_feats, style_feats, filter_weights, filter_biases, alpha)
        
        # vgg content and style loss
        res_feats_vgg = self.encode_with_vgg_intermediate(res)
        
        style_feats_vgg = self.encode_with_vgg_intermediate(style)
        content_feats_vgg = self.encode_vgg_content(content)

        loss_c = self.calc_content_loss(res_feats_vgg[-1], content_feats_vgg)
        loss_s = self.calc_style_loss(res_feats_vgg[0], style_feats_vgg[0])
        for i in range(1, 4):
            loss_s = loss_s + self.calc_style_loss(res_feats_vgg[i], style_feats_vgg[i])

        res_style_feats = self.style_encoder(res)
        res_filter_weights, res_filter_biases = self.modulator(res_style_feats)
        
        # --- style signal contrastive loss (vectorized) ---
        # Replaces the original O(B^2) nested Python loop with GPU-parallel
        # broadcasting.  Python-level iteration count drops from B^2 to zero;
        # all pairwise distances are computed simultaneously on GPU.

        # FeatMod pairwise matrices
        # pos:  res vs style  (both levels)
        # neg:  res vs RES    (level 0), res vs style (level 1)
        fm_pos_0 = self._pairwise_style_loss(res_style_feats[0], style_feats[0])
        fm_pos_1 = self._pairwise_style_loss(res_style_feats[1], style_feats[1])
        fm_neg_0 = self._pairwise_style_loss(res_style_feats[0], res_style_feats[0])
        fm_neg_1 = fm_pos_1  # identical: res vs style

        featmod_pos = fm_pos_0 + fm_pos_1  # [B, B]
        featmod_neg = fm_neg_0 + fm_neg_1  # [B, B]

        # FilterMod pairwise matrix (same for pos and neg: all res vs style)
        filtermod = (self._pairwise_content_loss(res_filter_weights[0], filter_weights[0]) +
                     self._pairwise_content_loss(res_filter_weights[1], filter_weights[1]) +
                     self._pairwise_content_loss(res_filter_biases[0],  filter_biases[0]) +
                     self._pairwise_content_loss(res_filter_biases[1],  filter_biases[1]))

        # pos[i] = diagonal, neg[i] = sum of each row excluding diagonal
        pos = featmod_pos.diag() + filtermod.diag()                  # [B]
        neg = ((featmod_neg + filtermod).sum(dim=1)
               - (featmod_neg.diag() + filtermod.diag()))           # [B]

        loss_contrastive = (pos / neg).sum()

        return res, loss_c, loss_s, loss_contrastive
    

    
class TestNet(nn.Module):
    def __init__(self, content_encoder, style_encoder, modulator, decoder):
        super(TestNet, self).__init__()

        self.content_encoder = content_encoder
        self.style_encoder = style_encoder
        self.modulator = modulator
        self.decoder = decoder


    def forward(self, content, style, alpha=1.0):
        assert 0 <= alpha <= 1

        style_feats = self.style_encoder(style)
        filter_weights, filter_biases = self.modulator(style_feats)

        content_feats = self.content_encoder(content)

        res = self.decoder(content_feats, style_feats, filter_weights, filter_biases, alpha)

        return res


class VideoTestNet(nn.Module):
    """Temporally consistent TestNet for video style transfer.

    Wraps the standard TestNet with a TemporalSmoother that applies EMA-based
    smoothing to content features, style features, and modulation signals
    across consecutive video frames.

    The smoothing is done in latent space (not pixel space), which prevents
    ghosting artifacts while effectively suppressing flickering caused by
    frame-to-frame feature jitter in the content encoder.

    Usage:
        net = VideoTestNet(content_enc, style_enc, modulator, decoder, momentum=0.7)
        for frame in video_frames:
            output = net(frame, style_image, alpha=1.0)
            save(output)
        net.reset_smoother()  # between videos or scene cuts
    """
    def __init__(self, content_encoder, style_encoder, modulator, decoder,
                 momentum=0.7):
        super(VideoTestNet, self).__init__()
        # Import here to avoid circular dependency
        from temporal_smoother import TemporalSmoother

        self.content_encoder = content_encoder
        self.style_encoder = style_encoder
        self.modulator = modulator
        self.decoder = decoder
        self.smoother = TemporalSmoother(momentum=momentum)

    def forward(self, content, style, alpha=1.0):
        assert 0 <= alpha <= 1

        # 1. Extract and temporally smooth style modulation signals
        style_feats = self.style_encoder(style)
        style_feats = self.smoother.smooth_style(style_feats)

        filter_weights, filter_biases = self.modulator(style_feats)
        filter_weights, filter_biases = self.smoother.smooth_modulation(
            filter_weights, filter_biases)

        # 2. Extract and temporally smooth content features
        #    THIS IS THE KEY STEP: content features vary slightly between
        #    adjacent frames, and the decoder amplifies these differences
        #    into visible flicker. EMA smoothing in latent space suppresses
        #    this jitter without causing ghosting.
        content_feats = self.content_encoder(content)
        content_feats = self.smoother.smooth_content(content_feats)

        # 3. Decode with smoothed signals
        res = self.decoder(content_feats, style_feats,
                           filter_weights, filter_biases, alpha)
        return res

    def reset_smoother(self):
        """Reset temporal buffers.

        Call between different videos, after scene cuts, or when seeking.
        """
        self.smoother.reset()
