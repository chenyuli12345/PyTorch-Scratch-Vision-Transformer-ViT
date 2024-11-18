import torch
import torch.nn as nn


# B -> 小批量大小
# C -> 输入图像数据的通道数
# IH -> 图像高度
# IW -> 图像宽度
# P -> 图像块/patch的大小
# E -> 嵌入维度
# N -> 图像块/patch的数量 = IH/P * IW/P = IH * IW / P^2
# S -> 序列的长度 = IH * IW / P^2 + 1 or N + 1 (多出来的1是分类Token/Classification Token) = 图像块/patch的数量 + 1
# Q -> 查询的序列长度(等于自注意力的S)
# K -> 键的序列长度(等于自注意力的S)
# V -> 值的序列长度(等于自注意力的S)
# H -> 注意力头的数量
# HE -> 头嵌入的维度 嵌入维度/注意力头的数量 = E/H
# CL -> 分类的数量

class EmbedLayer(nn.Module):
    """### 用于图像嵌入/Patch Embedding的类。
    - 将图像划分为小块（patch），并通过 Conv2D 操作对小块进行嵌入（其功能与线性层相同）。  
    - 然后，为所有小块的嵌入添加一个可学习的位置嵌入向量，以提供空间位置信息。  
    - 最后，添加一个分类标记（classification token），用于对图像进行分类。  
    
    构造参数:
        n_channels (int) : 输入图像的通道数C
        embed_dim  (int) : 嵌入维度E
        image_size (int) : 图像的尺寸
        patch_size (int) : 图像块/patch的尺寸
        dropout  (float) : dropout的概率
    输入:
        x (tensor): 形状为(B, C, IW, IH)=(小批量大小, 输入图像数据的通道数, 图像高度, 图像宽度)
    返回:
        Tensor: 嵌入之后的图像，形状为(B, S, E) = (小批量大小, 序列的长度, 嵌入维度)，其中序列的长度等于图像块（patch）的数量 + 1，这里图像块的数量等于图像高度*图像宽度/patch的大小的平方
    """    
    def __init__(self, n_channels, embed_dim, image_size, patch_size, dropout=0.0): #构造函数，接受参数为输入图像的通道数，嵌入维度，图像尺寸，图像块/patch的尺寸，dropout的概率
        super().__init__()
        self.conv1         = nn.Conv2d(n_channels, embed_dim, kernel_size=patch_size, stride=patch_size)  # Patch Encoding，一个卷积层，输入通道数为n_channels（图像数据通道数C），输出通道数为embed_dim（嵌入维度E），卷积核大小为patch_size，步长为patch_size
        self.pos_embedding = nn.Parameter(torch.zeros(1, (image_size // patch_size) ** 2, embed_dim), requires_grad=True) #可学习的位置嵌入，一个张量，初始化为(1, (图像尺寸/图像块尺寸)^2, 嵌入维度embed_dim)的全零张量
        self.cls_token     = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True) # Classification Token/分类token，初始化为为(1, 1, 嵌入维度)的全零张量
        self.dropout       = nn.Dropout(dropout) #dropout层，概率为dropout

    def forward(self, x): #前向传播函数，接受参数为输入图像的张量x
        B = x.shape[0] #获取小批量大小为输入数据x的第0维的大小
        x = self.conv1(x) #图像先经过卷积层，目的是将图像划分为小块（patch）并对小块进行嵌入
        #从输入数据x形状为(B,C,IH,IW)=(小批量大小,输入图像数据的通道数,图像高度,图像宽度)，经过卷积层后变为(B,E,IH/P,IW/P)=(小批量大小,嵌入维度,图像高度/patch的大小或高度,图像宽度/patch的大小或宽度)
        x = x.reshape([B, x.shape[1], -1]) #将图像块/patch展平
        #将经过卷积层的形状为(B,E,IH/P,IW/P)=(小批量大小,嵌入维度,图像高度/patch的大小或高度,图像宽度/patch的大小或宽度)的x重塑形状为(B,E,(IH/P*IW/P))=(小批量大小,嵌入维度,图像高度*图像宽度/patch的大小的平方)，也就是(B,E,N)
        x = x.permute(0, 2, 1)  #交换x的第二维和第三维，即重新排列以将序列维度放在中间
        #将形状为(B,E,(IH/P*IW/P))=(小批量大小,嵌入维度,图像高度*图像宽度/patch的大小的平方)，也就是(B,E,N)的x重塑为(B,(IH/P*IW/P),E)=(小批量大小,图像高度*图像宽度/patch的大小的平方,嵌入维度)，即(B,N,E)
        x = x + self.pos_embedding  #添加位置嵌入，形状不变，仍为(B,(IH/P*IW/P),E)=(小批量大小,图像高度*图像宽度/patch的大小的平方,嵌入维度)，即(B,N,E)
        x = torch.cat((torch.repeat_interleave(self.cls_token, B, 0), x), dim=1) #在每个序列的开头添加分类标记（classification token）
        #形状从上一步的(B,N,E)变为(B,(N+1),E)，即(B,S,E)，这里B为小批量大小，S为序列的长度（即图像高度*图像宽度/patch的大小的平方+1），E为嵌入维度
        x = self.dropout(x) #将处理过的x输入到dropout层
        return x #返回处理过的x


class SelfAttention(nn.Module):
    """
    用于计算自注意力（Self-Attention）的类。
    构造参数:
        embed_dim (int)        : 嵌入维度E
        n_attention_heads (int): 用于执行多头注意力（MultiHeadAttention）的注意力头数量H
    输入:
        x(tensor): 形状为(B,S,E)=(小批量大小,序列的长度,嵌入维度)，其中序列的长度等于图像块（patch）的数量 + 1，这里图像块的数量等于图像高度*图像宽度/patch的大小的平方
    返回:
        张量: 自注意力的输出形状仍然为(B,S,E)=(小批量大小,序列的长度,嵌入维度)，其中序列的长度等于图像块（patch）的数量 + 1。形状与输入一样
    """    
    def __init__(self, embed_dim, n_attention_heads): #构造函数，接受参数为嵌入维度E和注意力头的数量H
        super().__init__()
        self.embed_dim          = embed_dim
        self.n_attention_heads  = n_attention_heads
        self.head_embed_dim     = embed_dim // n_attention_heads #头嵌入的维度HE，等于嵌入维度除以注意力头的数量，也就是HE = E/H

        self.queries            = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads)   #查询投影，本质上是一个全连接层，输入维度为嵌入维度E，输出维度为头嵌入的维度HE乘以注意力头的数量H，也就是(H*HE)
        #（这里改quadratic）
        self.keys               = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads)   #键投影，本质上是一个全连接层，输入维度为嵌入维度E，输出维度为头嵌入的维度HE乘以注意力头的数量H，也就是(H*HE)
        #（这里删了）
        self.values             = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads)   #值投影，本质上是一个全连接层，输入维度为嵌入维度E，输出维度为头嵌入的维度HE乘以注意力头的数量H，也就是(H*HE)
        self.out_projection     = nn.Linear(self.head_embed_dim * self.n_attention_heads, self.embed_dim)   #输出投影，本质上是一个全连接层，输入维度为头嵌入的维度HE乘以注意力头的数量H，即H*HE，输出维度为嵌入维度E

    def forward(self, x): #前向传播函数，接受参数为输入数据x
        b, s, e = x.shape  #获取输入张量的形状，b为小批量大小，s为序列长度（图像块patch的数量+1），e为嵌入维度。注意再自注意力的情况下查询、键、值的长度均等于序列长度S

        xq = self.queries(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)     # B, Q, E      ->  B, Q, (H*HE)  ->  B, Q, H, HE
        xq = xq.permute(0, 2, 1, 3)                                                         # B, Q, H, HE  ->  B, H, Q, HE
        xk = self.keys(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)        # B, K, E      ->  B, K, (H*HE)  ->  B, K, H, HE
        xk = xk.permute(0, 2, 1, 3)                                                         # B, K, H, HE  ->  B, H, K, HE
        xv = self.values(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)      # B, V, E      ->  B, V, (H*HE)  ->  B, V, H, HE
        xv = xv.permute(0, 2, 1, 3)                                                         # B, V, H, HE  ->  B, H, V, HE


        # Compute Attention presoftmax values
        xk = xk.permute(0, 1, 3, 2)                                                         # B, H, K, HE  ->  B, H, HE, K
        x_attention = torch.matmul(xq, xk)                                                  # B, H, Q, HE  *   B, H, HE, K   ->  B, H, Q, K    (Matmul tutorial eg: A, B, C, D  *  A, B, E, F  ->  A, B, C, F   if D==E)

        x_attention /= float(self.head_embed_dim) ** 0.5                                    # Scale presoftmax values for stability

        x_attention = torch.softmax(x_attention, dim=-1)                                    # Compute Attention Matrix

        x = torch.matmul(x_attention, xv)                                                   # B, H, Q, K  *  B, H, V, HE  ->  B, H, Q, HE     Compute Attention product with Values

        # Format the output
        x = x.permute(0, 2, 1, 3)                                                           # B, H, Q, HE -> B, Q, H, HE
        x = x.reshape(b, s, e)                                                              # B, Q, H, HE -> B, Q, (H*HE)

        x = self.out_projection(x)                                                          # B, Q,(H*HE) -> B, Q, E
        return x


class Encoder(nn.Module):
    """
    Class for creating an encoder layer

    Parameters:
        embed_dim (int)         : Embedding dimension
        n_attention_heads (int) : Number of attention heads to use for performing MultiHeadAttention
        forward_mul (float)     : Used to calculate dimension of the hidden fc layer = embed_dim * forward_mul
        dropout (float)         : Dropout parameter
    
    Input:
        x (tensor): Tensor of shape B, S, E

    Returns:
        Tensor: Output of the encoder block of shape B, S, E
    """    
    def __init__(self, embed_dim, n_attention_heads, forward_mul, dropout=0.0):
        super().__init__()
        self.norm1      = nn.LayerNorm(embed_dim)
        self.attention  = SelfAttention(embed_dim, n_attention_heads)
        self.dropout1   = nn.Dropout(dropout)
        
        self.norm2      = nn.LayerNorm(embed_dim)
        self.fc1        = nn.Linear(embed_dim, embed_dim * forward_mul)
        self.activation = nn.GELU()
        self.fc2        = nn.Linear(embed_dim * forward_mul, embed_dim)
        self.dropout2   = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout1(self.attention(self.norm1(x)))                                # Skip connections
        x = x + self.dropout2(self.fc2(self.activation(self.fc1(self.norm2(x)))))           # Skip connections
        return x


class Classifier(nn.Module):
    """
    Classification module of the Vision Transformer. Uses the embedding of the classification token to generate logits.

    Parameters:
        embed_dim (int) : Embedding dimension
        n_classes (int) : Number of classes
    
    Input:
        x (tensor): Tensor of shape B, S, E

    Returns:
        Tensor: Logits of shape B, CL
    """    

    def __init__(self, embed_dim, n_classes):
        super().__init__()
        # New architectures skip fc1 and activations and directly apply fc2.
        self.fc1        = nn.Linear(embed_dim, embed_dim)
        self.activation = nn.Tanh()
        self.fc2        = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        x = x[:, 0, :]              # B, S, E -> B, E          Get CLS token
        x = self.fc1(x)             # B, E    -> B, E
        x = self.activation(x)      # B, E    -> B, E    
        x = self.fc2(x)             # B, E    -> B, CL
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer Class.

    Parameters:
        n_channels (int)        : Number of channels of the input image
        embed_dim  (int)        : Embedding dimension
        n_layers   (int)        : Number of encoder blocks to use
        n_attention_heads (int) : Number of attention heads to use for performing MultiHeadAttention
        forward_mul (float)     : Used to calculate dimension of the hidden fc layer = embed_dim * forward_mul
        image_size (int)        : Image size
        patch_size (int)        : Patch size
        n_classes (int)         : Number of classes
        dropout  (float)        : dropout value
    
    Input:
        x (tensor): Image Tensor of shape B, C, IW, IH

    Returns:
        Tensor: Logits of shape B, CL
    """    
    def __init__(self, n_channels, embed_dim, n_layers, n_attention_heads, forward_mul, image_size, patch_size, n_classes, dropout=0.1):
        super().__init__()
        self.embedding  = EmbedLayer(n_channels, embed_dim, image_size, patch_size, dropout=dropout)
        self.encoder    = nn.ModuleList([Encoder(embed_dim, n_attention_heads, forward_mul, dropout=dropout) for _ in range(n_layers)])
        self.norm       = nn.LayerNorm(embed_dim)                                       # Final normalization layer after the last block
        self.classifier = Classifier(embed_dim, n_classes)

        self.apply(vit_init_weights)                                                    # Weight initalization

    def forward(self, x):
        x = self.embedding(x)
        for block in self.encoder:
            x = block(x)
        x = self.norm(x)
        x = self.classifier(x)
        return x


def vit_init_weights(m): 
    """
    function for initializing the weights of the Vision Transformer.
    """    

    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    elif isinstance(m, EmbedLayer):
        nn.init.trunc_normal_(m.cls_token, mean=0.0, std=0.02)
        nn.init.trunc_normal_(m.pos_embedding, mean=0.0, std=0.02)
