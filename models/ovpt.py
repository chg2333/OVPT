import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from .model import Model


def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1) - 1,
                                                                 -1, -1), ('cpu', 'cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                              proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x



class OVPT_S(Model):
    def __init__(self, name, nclasses=40, pretraining=True, cnn_name='densenet121'):
        super(OVPT_S, self).__init__(name)

        # m40
        self.classnames = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
                           'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
                           'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
                           'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
                           'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']

        # M10
        # self.classnames = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']

        self.nclasses = nclasses
        self.pretraining = pretraining
        self.cnn_name = cnn_name
        self.use_resnet = cnn_name.startswith('resnet')
        self.use_densenet = cnn_name.startswith('densenet')

        self.mean = Variable(torch.FloatTensor([0.0142, 0.0142, 0.0142]), requires_grad=False).cuda()
        self.std = Variable(torch.FloatTensor([0.0818, 0.0818, 0.0818]), requires_grad=False).cuda()

        if self.use_resnet:
            if self.cnn_name == 'resnet18':
                self.net = models.resnet18(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512, 40)
            elif self.cnn_name == 'resnet34':
                self.net = models.resnet34(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512, 10)
            elif self.cnn_name == 'resnet50':
                self.net = models.resnet50(pretrained=self.pretraining)
                self.net.fc = nn.Linear(2048, 40)
        elif self.use_densenet:
            if self.cnn_name =='densenet121':
                self.net = models.densenet121(pretrained=self.pretraining)
                self.net.classifier = nn.Linear(1024, 40)

    def forward(self, x):

            return self.net(x)


class OVPT_M(Model):
    def __init__(self, name, model, pool_mode='PT', nclasses=40, cnn_name='densenet121', num_views=6):
        super(OVPT_M, self).__init__(name)

        # m40
        self.classnames = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
                            'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
                            'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
                            'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
                            'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']

        # M10
        #self.classnames = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']

        self.nclasses = nclasses
        self.num_views = num_views

        self.cnn_name = cnn_name
        self.use_resnet = cnn_name.startswith('resnet')
        self.use_densenet = cnn_name.startswith('densenet')

        if self.use_resnet:
            if cnn_name == 'resnet18':
                self.net_1 = nn.Sequential(*list(model.net.children())[:-1])

            elif self.cnn_name == 'resnet34':
                self.net_1 = nn.Sequential(*list(model.net.children())[:-1])

            elif self.cnn_name == 'resnet50':
                self.net_1 = nn.Sequential(*list(model.net.children())[:-1])

        elif self.use_densenet:
            if self.cnn_name =='densenet121':
                self.net_1 = nn.Sequential(*list(model.net.children())[:-1])

        self.pool_mode = pool_mode

        # tiny small base
        PT_size = 'tiny'

        # tiny
        if PT_size == 'tiny':
            embed_dim = 192
            num_heads = 3
        # small
        elif PT_size == 'small':
            embed_dim = 384
            num_heads = 6
        # base
        elif PT_size == 'base':
            embed_dim = 768
            num_heads = 12
        else:
            embed_dim = 768
            num_heads = 2

        self.GAP = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.proj = nn.Conv2d(1024, embed_dim, kernel_size=(1, 1), stride=(1, 1))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_views + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=0.)
        norm_layer = nn.LayerNorm
        act_layer = nn.GELU
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4, qkv_bias=False, drop=0.,
                attn_drop=0, drop_path=0, norm_layer=norm_layer, act_layer=act_layer)])
        self.norm = nn.LayerNorm(embed_dim)
        self.pre_logits = nn.Identity()
        self.head = nn.Linear(embed_dim, 40)

    def forward(self, x):

        fi = self.net_1(x)
        fi = self.GAP(fi)

        xi = self.proj(fi).flatten(2).transpose(1, 2)

        xi = xi.view((int(xi.shape[0] / self.num_views), self.num_views, xi.shape[2]))

        cls_token = self.cls_token.expand(xi.shape[0], -1, -1)

        x0 = self.pos_drop(torch.cat((cls_token, xi), dim=1) + self.pos_embed)

        x1 = self.blocks(x0)
        x1 = self.norm(x1)

        if self.pool_mode == 'PT':
            x_class = self.pre_logits(x1[:, 0])

            x_PT = x1[:, 1:]
            x_PT = torch.max(x_PT, 1)[0]

            y = x_PT + x_class

        elif self.pool_mode == 'T':
            y = self.pre_logits(x1[:, 0])

        y = self.head(y)

        return y





