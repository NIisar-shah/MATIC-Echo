import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2,
        )
        self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)

        x = F.pad(
            x,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
            ],
        )

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)



class EchoMambaTemporalBlock(nn.Module):
    """
    EchoMamba-T temporal selective state-space block.

    This is the first/simple EchoMamba version.
    It applies a diagonal input-dependent selective SSM over compact
    bottleneck tokens extracted from each video frame.

    Input:
        x: [B, C, T, H, W]

    Output:
        y: [B, C, T, H, W]
    """

    def __init__(self, channels: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or channels

        self.channels = channels
        self.hidden_dim = hidden_dim

        # Token projection
        self.norm = nn.LayerNorm(channels)
        self.in_proj = nn.Linear(channels, hidden_dim)

        # Selective SSM parameter projections
        self.delta_proj = nn.Linear(hidden_dim, hidden_dim)
        self.B_proj = nn.Linear(hidden_dim, hidden_dim)
        self.C_proj = nn.Linear(hidden_dim, hidden_dim)

        # Diagonal stable state matrix parameterization:
        # A = -exp(A_log), so A is always negative/stable.
        self.A_log = nn.Parameter(torch.zeros(hidden_dim))
        self.D = nn.Parameter(torch.ones(hidden_dim))

        # Back to feature channels
        self.out_proj = nn.Linear(hidden_dim, channels)

        # Lightweight temporal mixing after SSM reinjection
        self.temporal_mix = nn.Sequential(
            nn.Conv3d(
                channels,
                channels,
                kernel_size=(3, 1, 1),
                padding=(1, 0, 0),
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(channels),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        x: [B, C, T, H, W]
        """
        b, c, t, h, w = x.shape

        # Compact temporal tokens from bottleneck features
        pooled = x.mean(dim=(-1, -2))         # [B, C, T]
        pooled = pooled.permute(0, 2, 1)      # [B, T, C]

        tokens = self.norm(pooled)
        u = self.in_proj(tokens)              # [B, T, hidden_dim]

        # Input-dependent selective parameters
        delta = torch.nn.functional.softplus(self.delta_proj(u)) + 1e-4  # [B, T, Hdim]
        B_t = self.B_proj(u)                                            # [B, T, Hdim]
        C_t = self.C_proj(u)                                            # [B, T, Hdim]

        # Stable diagonal A
        A = -torch.exp(self.A_log).view(1, -1)                            # [1, Hdim]
        D = self.D.view(1, -1)                                            # [1, Hdim]

        state = torch.zeros(
            b,
            self.hidden_dim,
            device=x.device,
            dtype=x.dtype,
        )

        outputs = []
        for i in range(t):
            u_i = u[:, i, :]          # [B, Hdim]
            dt_i = delta[:, i, :]     # [B, Hdim]
            B_i = B_t[:, i, :]        # [B, Hdim]
            C_i = C_t[:, i, :]        # [B, Hdim]

            # Simplified diagonal selective SSM recurrence
            # A_bar = exp(delta * A)
            A_bar = torch.exp(dt_i * A)
            B_bar = dt_i * B_i

            state = A_bar * state + B_bar * u_i
            y_i = C_i * state + D * u_i

            outputs.append(y_i)

        y = torch.stack(outputs, dim=1)          # [B, T, Hdim]
        mod = self.out_proj(y)                   # [B, T, C]
        mod = mod.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)  # [B, C, T, 1, 1]

        # Residual temporal feature reinjection
        out = x + mod
        out = out + self.temporal_mix(out)
        out = self.relu(out)

        return out


class EchoMambaTUNet(nn.Module):
    """
    EchoMamba-T temporal selective SSM U-Net.

    Input:
        [B, 3, T, H, W]

    Output:
        [B, T, 1, H, W]
    """

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        self.inc = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)

        self.temporal_block1 = EchoMambaTemporalBlock(256, hidden_dim=256)
        self.temporal_block2 = EchoMambaTemporalBlock(256, hidden_dim=256)

        self.up1 = Up(256, 128, 128)
        self.up2 = Up(128, 64, 64)
        self.up3 = Up(64, 32, 32)

        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, video):
        """
        video: [B, 3, T, H, W]
        """
        b, c, t, h, w = video.shape

        x = video.permute(0, 2, 1, 3, 4).contiguous()   # [B, T, C, H, W]
        x = x.view(b * t, c, h, w)                      # [B*T, C, H, W]

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        _, c4, h4, w4 = x4.shape
        x4 = x4.view(b, t, c4, h4, w4).permute(0, 2, 1, 3, 4).contiguous()  # [B, C, T, H, W]

        x4 = self.temporal_block1(x4)
        x4 = self.temporal_block2(x4)

        x4 = x4.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c4, h4, w4)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.out_conv(x)

        logits = logits.view(b, t, 1, h, w)
        return logits

class EchoMambaSpatialBlock(nn.Module):
    """
    EchoMamba-ST spatial selective SSM block.

    This block applies four-directional spatial selective scanning
    on bottleneck feature maps for each frame.

    Input:
        x: [B, C, T, H, W]

    Output:
        y: [B, C, T, H, W]
    """

    def __init__(self, channels: int, hidden_dim: int = 128):
        super().__init__()

        self.channels = channels
        self.hidden_dim = hidden_dim

        self.norm = nn.LayerNorm(channels)
        self.in_proj = nn.Linear(channels, hidden_dim)

        self.delta_proj = nn.Linear(hidden_dim, hidden_dim)
        self.B_proj = nn.Linear(hidden_dim, hidden_dim)
        self.C_proj = nn.Linear(hidden_dim, hidden_dim)

        # Stable diagonal state matrix: A = -exp(A_log)
        self.A_log = nn.Parameter(torch.zeros(hidden_dim))
        self.D = nn.Parameter(torch.ones(hidden_dim))

        self.out_proj = nn.Linear(hidden_dim, channels)

        self.spatial_mix = nn.Sequential(
            nn.Conv3d(
                channels,
                channels,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(channels),
        )

        self.relu = nn.ReLU(inplace=True)

    def _selective_scan(self, seq):
        """
        seq: [N, L, C]
        returns: [N, L, C]
        """
        n, l, c = seq.shape

        seq = self.norm(seq)
        u = self.in_proj(seq)  # [N, L, hidden_dim]

        delta = torch.nn.functional.softplus(self.delta_proj(u)) + 1e-4
        B_t = self.B_proj(u)
        C_t = self.C_proj(u)

        A = -torch.exp(self.A_log).view(1, -1)
        D = self.D.view(1, -1)

        state = torch.zeros(
            n,
            self.hidden_dim,
            device=seq.device,
            dtype=seq.dtype,
        )

        outputs = []
        for i in range(l):
            u_i = u[:, i, :]
            dt_i = delta[:, i, :]
            B_i = B_t[:, i, :]
            C_i = C_t[:, i, :]

            A_bar = torch.exp(dt_i * A)
            B_bar = dt_i * B_i

            state = A_bar * state + B_bar * u_i
            y_i = C_i * state + D * u_i
            outputs.append(y_i)

        y = torch.stack(outputs, dim=1)  # [N, L, hidden_dim]
        y = self.out_proj(y)             # [N, L, C]
        return y

    def forward(self, x):
        """
        x: [B, C, T, H, W]
        """
        b, c, t, h, w = x.shape

        # [B, C, T, H, W] -> [B*T, H, W, C]
        grid = x.permute(0, 2, 3, 4, 1).contiguous()
        grid = grid.view(b * t, h, w, c)

        # Direction 1: row-major
        seq_hw = grid.view(b * t, h * w, c)

        # Direction 2: reverse row-major
        seq_hw_rev = torch.flip(seq_hw, dims=[1])

        # Direction 3: column-major
        seq_wh = grid.transpose(1, 2).contiguous().view(b * t, h * w, c)

        # Direction 4: reverse column-major
        seq_wh_rev = torch.flip(seq_wh, dims=[1])

        y_hw = self._selective_scan(seq_hw)
        y_hw_rev = self._selective_scan(seq_hw_rev)
        y_wh = self._selective_scan(seq_wh)
        y_wh_rev = self._selective_scan(seq_wh_rev)

        # Restore direction outputs to [B*T, H, W, C]
        out_hw = y_hw.view(b * t, h, w, c)

        out_hw_rev = torch.flip(y_hw_rev, dims=[1])
        out_hw_rev = out_hw_rev.view(b * t, h, w, c)

        out_wh = y_wh.view(b * t, w, h, c).transpose(1, 2).contiguous()

        out_wh_rev = torch.flip(y_wh_rev, dims=[1])
        out_wh_rev = out_wh_rev.view(b * t, w, h, c).transpose(1, 2).contiguous()

        out = (out_hw + out_hw_rev + out_wh + out_wh_rev) / 4.0

        # [B*T, H, W, C] -> [B, C, T, H, W]
        out = out.view(b, t, h, w, c).permute(0, 4, 1, 2, 3).contiguous()

        out = x + out
        out = out + self.spatial_mix(out)
        out = self.relu(out)

        return out


class EchoMambaSTUNet(nn.Module):
    """
    EchoMamba-ST: spatial-temporal selective SSM U-Net.

    This model extends EchoMamba-T by adding a spatial selective
    SSM branch at the bottleneck.

    Input:
        [B, 3, T, H, W]

    Output:
        [B, T, 1, H, W]
    """

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        self.inc = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)

        self.temporal_block1 = EchoMambaTemporalBlock(256, hidden_dim=256)
        self.spatial_block1 = EchoMambaSpatialBlock(256, hidden_dim=128)

        self.fuse_st = nn.Sequential(
            nn.Conv3d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
        )

        self.temporal_block2 = EchoMambaTemporalBlock(256, hidden_dim=256)

        self.up1 = Up(256, 128, 128)
        self.up2 = Up(128, 64, 64)
        self.up3 = Up(64, 32, 32)

        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, video):
        """
        video: [B, 3, T, H, W]
        """
        b, c, t, h, w = video.shape

        x = video.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(b * t, c, h, w)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        _, c4, h4, w4 = x4.shape
        x4 = x4.view(b, t, c4, h4, w4).permute(0, 2, 1, 3, 4).contiguous()

        x4_temp = self.temporal_block1(x4)
        x4_spat = self.spatial_block1(x4)

        x4 = torch.cat([x4_temp, x4_spat], dim=1)
        x4 = self.fuse_st(x4)

        x4 = self.temporal_block2(x4)

        x4 = x4.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c4, h4, w4)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        logits = self.out_conv(x)
        logits = logits.view(b, t, 1, h, w)

        return logits



class EchoMambaSpatialLiteBlock(nn.Module):
    """
    EchoMamba-ST-v2 lightweight spatial selective SSM block.

    Differences from ST-v1:
    - hidden_dim is smaller
    - only two scan directions are used
    - residual update is controlled by a small learnable scale
    - no heavy spatial-temporal concatenation fusion is used

    Input:
        x: [B, C, T, H, W]

    Output:
        y: [B, C, T, H, W]
    """

    def __init__(self, channels: int, hidden_dim: int = 64, init_scale: float = 0.01):
        super().__init__()

        self.channels = channels
        self.hidden_dim = hidden_dim

        self.norm = nn.LayerNorm(channels)
        self.in_proj = nn.Linear(channels, hidden_dim)

        self.delta_proj = nn.Linear(hidden_dim, hidden_dim)
        self.B_proj = nn.Linear(hidden_dim, hidden_dim)
        self.C_proj = nn.Linear(hidden_dim, hidden_dim)

        # Stable diagonal state matrix: A = -exp(A_log)
        self.A_log = nn.Parameter(torch.zeros(hidden_dim))
        self.D = nn.Parameter(torch.ones(hidden_dim))

        self.out_proj = nn.Linear(hidden_dim, channels)

        # Small learnable residual scale.
        # Initialized small so this block starts close to identity.
        self.res_scale = nn.Parameter(torch.tensor(float(init_scale)))

        # Very lightweight local spatial refinement
        self.local_refine = nn.Sequential(
            nn.Conv3d(
                channels,
                channels,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(channels),
        )

        self.relu = nn.ReLU(inplace=True)

    def _selective_scan(self, seq):
        """
        seq: [N, L, C]
        returns: [N, L, C]
        """
        n, l, c = seq.shape

        seq = self.norm(seq)
        u = self.in_proj(seq)  # [N, L, hidden_dim]

        delta = torch.nn.functional.softplus(self.delta_proj(u)) + 1e-4
        B_t = self.B_proj(u)
        C_t = self.C_proj(u)

        A = -torch.exp(self.A_log).view(1, -1)
        D = self.D.view(1, -1)

        state = torch.zeros(
            n,
            self.hidden_dim,
            device=seq.device,
            dtype=seq.dtype,
        )

        outputs = []
        for i in range(l):
            u_i = u[:, i, :]
            dt_i = delta[:, i, :]
            B_i = B_t[:, i, :]
            C_i = C_t[:, i, :]

            A_bar = torch.exp(dt_i * A)
            B_bar = dt_i * B_i

            state = A_bar * state + B_bar * u_i
            y_i = C_i * state + D * u_i
            outputs.append(y_i)

        y = torch.stack(outputs, dim=1)
        y = self.out_proj(y)
        return y

    def forward(self, x):
        """
        x: [B, C, T, H, W]
        """
        b, c, t, h, w = x.shape

        # [B, C, T, H, W] -> [B*T, H, W, C]
        grid = x.permute(0, 2, 3, 4, 1).contiguous()
        grid = grid.view(b * t, h, w, c)

        # Direction 1: row-major scan
        seq_hw = grid.view(b * t, h * w, c)

        # Direction 2: column-major scan
        seq_wh = grid.transpose(1, 2).contiguous().view(b * t, h * w, c)

        y_hw = self._selective_scan(seq_hw)
        y_wh = self._selective_scan(seq_wh)

        # Restore to grid
        out_hw = y_hw.view(b * t, h, w, c)
        out_wh = y_wh.view(b * t, w, h, c).transpose(1, 2).contiguous()

        out = (out_hw + out_wh) / 2.0

        # [B*T, H, W, C] -> [B, C, T, H, W]
        out = out.view(b, t, h, w, c).permute(0, 4, 1, 2, 3).contiguous()

        # Small residual spatial correction
        out = x + self.res_scale * out

        # Local refinement is also residual-scaled to avoid early disruption
        out = out + self.res_scale * self.local_refine(out)
        out = self.relu(out)

        return out


class EchoMambaSTv2UNet(nn.Module):
    """
    EchoMamba-ST-v2: lightweight spatial-temporal selective SSM U-Net.

    This version keeps the stable EchoMamba-T temporal path and adds a
    lightweight spatial selective scan as a small residual correction.

    Input:
        [B, 3, T, H, W]

    Output:
        [B, T, 1, H, W]
    """

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        self.inc = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)

        # Keep temporal branch unchanged from EchoMamba-T
        self.temporal_block1 = EchoMambaTemporalBlock(256, hidden_dim=256)

        # Lightweight spatial correction
        self.spatial_lite = EchoMambaSpatialLiteBlock(
            channels=256,
            hidden_dim=64,
            init_scale=0.01,
        )

        self.temporal_block2 = EchoMambaTemporalBlock(256, hidden_dim=256)

        self.up1 = Up(256, 128, 128)
        self.up2 = Up(128, 64, 64)
        self.up3 = Up(64, 32, 32)

        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, video):
        """
        video: [B, 3, T, H, W]
        """
        b, c, t, h, w = video.shape

        x = video.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(b * t, c, h, w)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        _, c4, h4, w4 = x4.shape
        x4 = x4.view(b, t, c4, h4, w4).permute(0, 2, 1, 3, 4).contiguous()

        x4 = self.temporal_block1(x4)
        x4 = self.spatial_lite(x4)
        x4 = self.temporal_block2(x4)

        x4 = x4.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c4, h4, w4)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        logits = self.out_conv(x)
        logits = logits.view(b, t, 1, h, w)

        return logits



class EchoMambaCycleTemporalBlock(nn.Module):
    """
    EchoMamba-Cycle temporal selective SSM block.

    This block extends EchoMamba-T by adding cardiac-cycle-aware conditioning.
    The conditioning is derived from:
    - learned phase probability
    - learned LV area proxy
    - area-change direction
    - normalized temporal position

    Input:
        x: [B, C, T, H, W]

    Output:
        y: [B, C, T, H, W]
    """

    def __init__(self, channels: int, hidden_dim: int = None, rhythm_dim: int = 64):
        super().__init__()
        hidden_dim = hidden_dim or channels

        self.channels = channels
        self.hidden_dim = hidden_dim
        self.rhythm_dim = rhythm_dim

        # Token projection
        self.norm = nn.LayerNorm(channels)
        self.in_proj = nn.Linear(channels, hidden_dim)

        # Base selective SSM projections
        self.delta_proj = nn.Linear(hidden_dim, hidden_dim)
        self.B_proj = nn.Linear(hidden_dim, hidden_dim)
        self.C_proj = nn.Linear(hidden_dim, hidden_dim)

        # Stable diagonal state matrix
        self.A_log = nn.Parameter(torch.zeros(hidden_dim))
        self.D = nn.Parameter(torch.ones(hidden_dim))

        # Learned rhythm descriptors from temporal tokens
        self.phase_head = nn.Linear(hidden_dim, 1)
        self.area_head = nn.Linear(hidden_dim, 1)

        # Rhythm embedding from [phase, area, delta_area, normalized_time]
        self.rhythm_mlp = nn.Sequential(
            nn.Linear(4, rhythm_dim),
            nn.SiLU(inplace=True),
            nn.Linear(rhythm_dim, rhythm_dim),
            nn.SiLU(inplace=True),
        )

        # Rhythm-conditioned FiLM modulation for SSM parameters.
        # These are zero-initialized so the block starts close to EchoMamba-T.
        self.delta_gamma = nn.Linear(rhythm_dim, hidden_dim)
        self.delta_beta = nn.Linear(rhythm_dim, hidden_dim)

        self.B_gamma = nn.Linear(rhythm_dim, hidden_dim)
        self.B_beta = nn.Linear(rhythm_dim, hidden_dim)

        self.C_gamma = nn.Linear(rhythm_dim, hidden_dim)
        self.C_beta = nn.Linear(rhythm_dim, hidden_dim)

        for layer in [
            self.delta_gamma,
            self.delta_beta,
            self.B_gamma,
            self.B_beta,
            self.C_gamma,
            self.C_beta,
        ]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

        self.out_proj = nn.Linear(hidden_dim, channels)

        self.temporal_mix = nn.Sequential(
            nn.Conv3d(
                channels,
                channels,
                kernel_size=(3, 1, 1),
                padding=(1, 0, 0),
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(channels),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        x: [B, C, T, H, W]
        """
        b, c, t, h, w = x.shape

        # Compact temporal tokens
        pooled = x.mean(dim=(-1, -2))       # [B, C, T]
        pooled = pooled.permute(0, 2, 1)    # [B, T, C]

        tokens = self.norm(pooled)
        u = self.in_proj(tokens)            # [B, T, hidden_dim]

        # Learned soft phase and area proxies
        phase = torch.sigmoid(self.phase_head(u))  # [B, T, 1]
        area = torch.sigmoid(self.area_head(u))    # [B, T, 1]

        # Area-change direction/proxy
        delta_area = torch.zeros_like(area)
        if t > 1:
            delta_area[:, 1:, :] = area[:, 1:, :] - area[:, :-1, :]

        # Normalized temporal position
        tau = torch.linspace(
            0.0,
            1.0,
            steps=t,
            device=x.device,
            dtype=x.dtype,
        ).view(1, t, 1).expand(b, t, 1)

        rhythm_in = torch.cat([phase, area, delta_area, tau], dim=-1)  # [B, T, 4]
        rhythm = self.rhythm_mlp(rhythm_in)                            # [B, T, rhythm_dim]

        # Base selective parameters
        delta_base = self.delta_proj(u)
        B_base = self.B_proj(u)
        C_base = self.C_proj(u)

        # Rhythm-conditioned parameter modulation
        delta_mod = delta_base * (1.0 + self.delta_gamma(rhythm)) + self.delta_beta(rhythm)
        B_mod = B_base * (1.0 + self.B_gamma(rhythm)) + self.B_beta(rhythm)
        C_mod = C_base * (1.0 + self.C_gamma(rhythm)) + self.C_beta(rhythm)

        delta = torch.nn.functional.softplus(delta_mod) + 1e-4
        B_t = B_mod
        C_t = C_mod

        A = -torch.exp(self.A_log).view(1, -1)
        D = self.D.view(1, -1)

        state = torch.zeros(
            b,
            self.hidden_dim,
            device=x.device,
            dtype=x.dtype,
        )

        outputs = []
        for i in range(t):
            u_i = u[:, i, :]
            dt_i = delta[:, i, :]
            B_i = B_t[:, i, :]
            C_i = C_t[:, i, :]

            A_bar = torch.exp(dt_i * A)
            B_bar = dt_i * B_i

            state = A_bar * state + B_bar * u_i
            y_i = C_i * state + D * u_i

            outputs.append(y_i)

        y = torch.stack(outputs, dim=1)  # [B, T, hidden_dim]
        mod = self.out_proj(y)           # [B, T, C]
        mod = mod.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)

        out = x + mod
        out = out + self.temporal_mix(out)
        out = self.relu(out)

        return out


class EchoMambaCycleUNet(nn.Module):
    """
    EchoMamba-Cycle: cardiac-cycle-aware temporal selective SSM U-Net.

    This model is built on EchoMamba-T but replaces the generic temporal
    Mamba block with a cardiac-cycle-conditioned temporal block.

    Input:
        [B, 3, T, H, W]

    Output:
        [B, T, 1, H, W]
    """

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        self.inc = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)

        self.cycle_block1 = EchoMambaCycleTemporalBlock(256, hidden_dim=256, rhythm_dim=64)
        self.cycle_block2 = EchoMambaCycleTemporalBlock(256, hidden_dim=256, rhythm_dim=64)

        self.up1 = Up(256, 128, 128)
        self.up2 = Up(128, 64, 64)
        self.up3 = Up(64, 32, 32)

        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, video):
        """
        video: [B, 3, T, H, W]
        """
        b, c, t, h, w = video.shape

        x = video.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(b * t, c, h, w)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        _, c4, h4, w4 = x4.shape
        x4 = x4.view(b, t, c4, h4, w4).permute(0, 2, 1, 3, 4).contiguous()

        x4 = self.cycle_block1(x4)
        x4 = self.cycle_block2(x4)

        x4 = x4.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c4, h4, w4)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        logits = self.out_conv(x)
        logits = logits.view(b, t, 1, h, w)

        return logits
