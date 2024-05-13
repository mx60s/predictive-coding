import torch
from torch import nn, optim
import torch.nn.functional as F

device = 'cuda:1'

# arch from here: https://github.com/julianstastny/VAE-ResNet18-PyTorch/tree/master
# with transposed convolutions subbed out -- may need to change that later

class ResizeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x    

class BasicBlockEnc(nn.Module):
    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = in_planes*stride

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.bn1(F.relu(self.conv1(x)))
        out = self.bn2(F.relu(self.conv2(out)))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BasicBlockDec(nn.Module):
    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes/stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.bn2(F.relu(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18Enc(nn.Module):
    def __init__(self, num_Blocks=[2,2,2,2], nc=3):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(nc, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlockEnc, 64, num_Blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlockEnc, 128, num_Blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, 256, num_Blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, 256, num_Blocks[3], stride=1)

        self.final = nn.Linear(256, 128)

    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = F.adaptive_avg_pool2d(x, 1) # this sets it to [channel, 1, 1]
        x = x.view(x.size(0), -1)
        return self.final(x) 


class ResNet18Dec(nn.Module):
    def __init__(self, z_dim=128, num_Blocks=[2,2,2,2], nc=3):
        super().__init__()
        self.in_planes = 512

        # this is set so high so you can have the first stride=2, so it expands the H,W dims more
        # accurate to the referenced ResNet
        self.linear = nn.Linear(z_dim, 512)

        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=2)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    # feels bad to interpolate up so much. Should I try to just run this on 64x64 imgs instead?
    def forward(self, x):
        #print('start dec', torch.cuda.memory_allocated(device))
        x = self.linear(x)
        x = x.view(x.size(0), 512, 1, 1)
        x = F.interpolate(x, scale_factor=4)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = torch.sigmoid(self.conv1(x))
        x = x.view(x.size(0), 3, 64, 64)
        #print('end dec', torch.cuda.memory_allocated(device))
        return x

class MySelfAttention(nn.Module):
    def __init__(self, embed_dim=144, heads=8):
        super().__init__()
        sequence_length = 7
        self.mask = torch.nn.Transformer.generate_square_subsequent_mask(sz=sequence_length).to(device)
        
        self.attn = nn.MultiheadAttention(embed_dim, heads, batch_first=True)

    def forward(self, x):
        out_attn, attn_weights = self.attn(x, x, x, need_weights=True, is_causal=True, attn_mask=self.mask)
        return out_attn[:, -1, :], attn_weights

class PredictiveCoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNet18Enc()
        self.attn = MySelfAttention(embed_dim=128)
        self.decoder = ResNet18Dec(z_dim=128)

    def forward(self, x):
        batch_size, sequence_length, c, h, w = x.size()
        x = x.view(batch_size * sequence_length, c, h, w)

        encoded_frames = self.encoder(x)
        encoded_frames = encoded_frames.view(batch_size, sequence_length, -1)
        z, weights = self.attn(encoded_frames)
        
        pred = self.decoder(z)
        
        return pred#, weights

    def get_latents(self, x):
        return self.encoder(x)
    
    def get_latent_preds(self, x):
        batch_size, sequence_length, c, h, w = x.size()
        x = x.view(batch_size * sequence_length, c, h, w)

        encoded_frames = self.encoder(x)
        encoded_frames = encoded_frames.view(batch_size, sequence_length, -1)
        z, weights = self.attn(encoded_frames)

        return z

class TurnScaler(nn.Module):
    def __init__(self, input_dim=1, output_dim=16):
        super(TurnScaler, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return F.tanh(self.fc(x))

class PredictiveCoderWithHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNet18Enc()
        #self.norm = nn.LayerNorm(144)
        self.attn = MySelfAttention(embed_dim=144) # with the expanded loc population of 168 and head dir pop of 16
        self.decoder = ResNet18Dec(z_dim=144)
 
        self.scale = TurnScaler()
        #self.freeze_scale() # I'm going to try to not freeze the scale for next time

    def freeze_scale(self):
        for param in self.scale.parameters():
            param.requires_grad = False

    def forward(self, x, d):
        batch_size, sequence_length = d.shape
        d = d.view(batch_size * sequence_length, 1)
        displacements_scaled = self.scale(d)
        displacements_scaled = displacements_scaled.view(batch_size, sequence_length, -1)

        batch_size, sequence_length, c, h, w = x.size()
        x = x.view(batch_size * sequence_length, c, h, w)
        
        encoded_frames = self.encoder(x)
        encoded_frames = encoded_frames.view(batch_size, sequence_length, -1)

        concatenated_vector = torch.cat((encoded_frames, displacements_scaled), dim=2)

        # Apply Layer Normalization
        #normalized_sequence = concatenated_vector#self.norm(concatenated_vector)
        
        z, weights = self.attn(concatenated_vector)

        pred = self.decoder(z)
        return pred

    def get_latents(self, x):
        # this presumes the use of a different data set than what you would train it with
        # aka no sequences
        return self.encoder(x)
        
# plot response curve for the units
    
    def get_latent_preds(self, x, d, ablate=-1):
        """
        Return the predicted latents instead of the image
        """
        batch_size, sequence_length = d.shape
        d = d.view(batch_size * sequence_length, 1)
        displacements_scaled = self.scale(d)
        displacements_scaled = displacements_scaled.view(batch_size, sequence_length, -1)

        batch_size, sequence_length, c, h, w = x.size()
        x = x.view(batch_size * sequence_length, c, h, w)
        
        encoded_frames = self.encoder(x)
        encoded_frames = encoded_frames.view(batch_size, sequence_length, -1)

        concatenated_vector = torch.cat((encoded_frames, displacements_scaled), dim=2)

        # Apply Layer Normalization
        #normalized_sequence = self.norm(concatenated_vector)
        
        z, weights = self.attn(concatenated_vector)

        return z #, weights

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNet18Enc()
        self.decoder = ResNet18Dec(z_dim=128)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def get_latent_preds(self, x):
        return self.encoder(x)


class PostPredictionHDPredictor(nn.Module):
    """
    A simple feedforward network which predicts the yaw of the agent from a set of latent variables
    """
    def __init__(self, latent_model: PredictiveCoder , input_dim=144, hidden_dim=256):
        super().__init__()
        self.latent = latent_model
        
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, 2)
        
    def forward(self, x, d, ablate=-1):
        with torch.no_grad():
            z = self.latent.get_latent_preds(x, d, ablate=ablate)
        
        out = F.relu(self.layer1(z))
        out = self.layer2(out)
        
        return out

class PostPredictionLocationPredictor(nn.Module):
    """
    A simple feedforward network which predicts the location of the agent from a set of latent variables
    """
    def __init__(self, latent_model: PredictiveCoder, input_dim=144, hidden_dim=256):
        super().__init__()
        self.latent = latent_model
        
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, 2)
        
    def forward(self, x, d=None):
        with torch.no_grad():
            if torch.is_tensor(d):
                z = self.latent.get_latent_preds(x, d)
            else:
                z = self.latent.get_latent_preds(x)
        
        out = F.relu(self.layer1(z))
        out = self.layer2(out)
        
        return out

class HeadDirectionPredictor(nn.Module):
    """
    A simple feedforward network which predicts the yaw of the agent from a set of latent variables
    """
    def __init__(self, latent_model: PredictiveCoder , input_dim=128, hidden_dim=256):
        super().__init__()
        self.encoder = latent_model.encoder
        
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, 2)
        
    def forward(self, x):
        with torch.no_grad():
            z = self.encoder(x)
        
        out = F.relu(self.layer1(z))
        out = self.layer2(out)
        
        return out

class LocationPredictor(nn.Module):
    """
    A simple feedforward network which predicts the position of the agent from a set of latent variables
    """
    def __init__(self, latent_model: PredictiveCoder , input_dim=128, hidden_dim=256):
        super().__init__()
        self.encoder = latent_model.encoder
        
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, 2)
        
    def forward(self, x):
        with torch.no_grad():
            z = self.encoder(x)
        
        out = F.relu(self.layer1(z))
        out = self.layer2(out)
        
        return out