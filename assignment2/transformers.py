"""
Implements a Transformer in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""

import torch
from torch import Tensor, nn, optim
from torch.nn import functional as F


def hello_transformers():
    print("Hello from transformers.py!")


def generate_token_dict(vocab):
    """
    The function creates a hash map from the elements in the vocabulary to
    to a unique positive integer value.

    args:
        vocab: This is a 1D list of strings containing all the items in the vocab

    Returns:
        token_dict: a python dictionary with key as the string item in the vocab
            and value as a unique integer value
    """
    # initialize a empty dictionary
    token_dict = {}
    ##############################################################################
    # TODO: Use this function to assign a unique whole number element to each    #
    # element present in the vocab list. To do this, map the first element in the#
    # vocab to 0 and the last element in the vocab to len(vocab), and the        #
    # elements in between as consequetive number.                                #
    ##############################################################################
    # Replace "pass" statement with your code
    pass
    ##############################################################################
    #               END OF YOUR CODE                                             #
    ##############################################################################
    return token_dict


def prepocess_input_sequence(
    input_str: str, token_dict: dict, spc_tokens: list
) -> list:
    """
    The goal of this fucntion is to convert an input string into a list of positive
    integers that will enable us to process the string using neural nets further. We
    will use the dictionary made in the previous function to map the elements in the
    string to a unique value. Keep in mind that we assign a value for each integer
    present in the input sequence. For example, for a number present in the input
    sequence "33", you should break it down to a list of digits,
    ['0', '3'] and assign it to a corresponding value in the token_dict.

    args:
        input_str: A single string in the input data
                 e.g.: "BOS POSITIVE 0333 add POSITIVE 0696 EOS"

        token_dict: The token dictionary having key as elements in the string and
            value as a unique positive integer. This is generated  using
            generate_token_dict fucntion

        spc_tokens: The special tokens apart from digits.
    Returns:
        out_tokens: a list of integers corresponding to the input string


    """
    out = []
    ##############################################################################
    # TODO: for each number present in the input sequence, break it down into a
    # list of digits and use this list of digits to assign an appropriate value
    # from token_dict. For special tokens present in the input string, assign an
    # appropriate value for the complete token.
    ##############################################################################
    # Replace "pass" statement with your code
    pass
    ##############################################################################
    #               END OF YOUR CODE                                             #
    ##############################################################################
    return out


def scaled_dot_product_two_loop_single(
    query: Tensor, key: Tensor, value: Tensor
) -> Tensor:
    """
    The function performs a fundamental block for attention mechanism, the scaled
    dot product. We map the input query, key, and value to the output. Follow the
    description in TODO for implementation.

    args:
        query: a Tensor of shape (K, M) where K is the sequence length and M is
            the sequence embeding dimension

        key: a Tensor of shape (K, M) where K is the sequence length and M is the
            sequence embeding dimension

        value: a Tensor of shape (K, M) where K is the sequence length and M is
            the sequence embeding dimension


    Returns
        out: a tensor of shape (K, M) which is the output of self-attention from
        the function
    """
    K, M = query.shape
    
    # 初始化输出张量
    out = torch.zeros(K, M, device=query.device, dtype=query.dtype)
    
    # 计算缩放因子
    scale = 1.0 / torch.sqrt(torch.tensor(M, dtype=query.dtype, device=query.device))
    
    # 对每个查询向量（外部循环）
    for i in range(K):
        # 存储当前查询与所有键的点积结果
        scores = torch.zeros(K, device=query.device, dtype=query.dtype)
        
        # 对每个键向量（内部循环）
        for j in range(K):
            # 计算查询i与键j的点积
            scores[j] = torch.dot(query[i], key[j])
        
        # 应用缩放因子
        scores = scores * scale
        
        # 应用softmax获取注意力权重
        attn_weights = torch.softmax(scores, dim=0)
        
        # 计算加权和的输出向量
        output_vector = torch.zeros(M, device=query.device, dtype=query.dtype)
        for j in range(K):
            output_vector += attn_weights[j] * value[j]
        
        # 保存输出向量
        out[i] = output_vector
    
    return out
    return out


def scaled_dot_product_two_loop_batch(
    query: Tensor, key: Tensor, value: Tensor
) -> Tensor:

    """
    The function performs a fundamental block for attention mechanism, the scaled
    dot product. We map the input query, key, and value to the output. Follow the
    description in TODO for implementation.

    args:
        query: a Tensor of shape (N,K, M) where N is the batch size, K is the
            sequence length and  M is the sequence embeding dimension

        key: a Tensor of shape (N, K, M) where N is the batch size, K is the
            sequence length and M is the sequence embeding dimension


        value: a Tensor of shape (N, K, M) where N is the batch size, K is the
            sequence length and M is the sequence embeding dimension


    Returns:
        out: a tensor of shape (N, K, M) that contains the weighted sum of values


    """
    N, K, M = query.shape
    
    # 初始化输出张量
    out = torch.zeros(N, K, M, device=query.device, dtype=query.dtype)
    
    # 计算缩放因子
    scale = 1.0 / torch.sqrt(torch.tensor(M, dtype=query.dtype, device=query.device))
    
    # 对每个批次使用两个循环
    for n in range(N):
        # 转置键矩阵，准备计算点积
        key_transposed = key[n].transpose(0, 1)  # 形状变为 (M, K)
        
        # 计算查询与键的点积，然后应用缩放因子
        scores = query[n] @ key_transposed  # 形状为 (K, K)
        scores = scores * scale
        
        # 应用softmax获取注意力权重
        attn_weights = torch.softmax(scores, dim=1)  # 形状为 (K, K)
        
        # 使用注意力权重对值进行加权求和
        out[n] = attn_weights @ value[n]
    
    return out
    #               END OF YOUR CODE                                             #
    ##############################################################################
    return out


def scaled_dot_product_no_loop_batch(
    query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None
) -> Tensor:
    """

    The function performs a fundamental block for attention mechanism, the scaled
    dot product. We map the input query, key, and value to the output. It uses
    Matrix-matrix multiplication to find the scaled weights and then matrix-matrix
    multiplication to find the final output.

    args:
        query: a Tensor of shape (N,K, M) where N is the batch size, K is the
            sequence length and M is the sequence embeding dimension

        key:  a Tensor of shape (N, K, M) where N is the batch size, K is the
            sequence length and M is the sequence embeding dimension


        value: a Tensor of shape (N, K, M) where N is the batch size, K is the
            sequence length and M is the sequence embeding dimension


        mask: a Bool Tensor of shape (N, K, K) that is used to mask the weights
            used for computing weighted sum of values


    return:
        y: a tensor of shape (N, K, M) that contains the weighted sum of values

        weights_softmax: a tensor of shape (N, K, K) that contains the softmaxed
            weight matrix.

    """

    N, K, M = query.shape
    
    # 计算缩放因子
    scale = 1.0 / torch.sqrt(torch.tensor(M, dtype=query.dtype, device=query.device))
    
    # 转置键矩阵，准备计算点积
    # 我们需要将key的形状从 (N, K, M) 转换为 (N, M, K)
    key_transposed = key.transpose(1, 2)
    
    # 使用批矩阵乘法计算查询与键的点积
    # query: (N, K, M)  @ key_transposed: (N, M, K) -> scores: (N, K, K)
    scores = torch.bmm(query, key_transposed)
    
    # 应用缩放因子
    scores = scores * scale
    
    # 应用掩码（如果提供）
    if mask is not None:
        # 将掩码位置的分数设为极小值，这样在softmax后这些位置的权重将接近0
        scores = scores.masked_fill(mask, -1e9)
    
    # 应用softmax获取注意力权重
    weights_softmax = torch.softmax(scores, dim=2)
    
    # 使用注意力权重对值进行加权求和
    # weights_softmax: (N, K, K)  @ value: (N, K, M) -> y: (N, K, M)
    y = torch.bmm(weights_softmax, value)
    
    return y, weights_softmax


class SelfAttention(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_v: int):
        super().__init__()

        """
        This class encapsulates the implementation of self-attention layer. We map 
        the input query, key, and value using MLP layers and then use 
        scaled_dot_product_no_loop_batch to the final output.
        
        args:
            dim_in: an int value for input sequence embedding dimension
            dim_q: an int value for output dimension of query and ley vector
            dim_v: an int value for output dimension for value vectors

        """
        # 初始化三个Linear层用于将输入转换为query、key和value向量
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_q)
        self.v = nn.Linear(dim_in, dim_v)
        self.weights_softmax = None
        
        # 为每个Linear层应用特定的权重初始化方法
        for layer in [self.q, self.k, self.v]:
            D_in = layer.in_features
            D_out = layer.out_features
            c = torch.sqrt(torch.tensor(6.0 / (D_in + D_out)))
            # 使用均匀分布初始化权重：[-c, c]
            nn.init.uniform_(layer.weight, -c, c)
            # 初始化偏置为0
            nn.init.zeros_(layer.bias)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None
    ) -> Tensor:

        """
        An implementation of the forward pass of the self-attention layer.

        args:
            query: Tensor of shape (N, K, M)
            key: Tensor of shape (N, K, M)
            value: Tensor of shape (N, K, M)
            mask: Tensor of shape (N, K, K)
        return:
            y: Tensor of shape (N, K, dim_v)
        """
        # 使用初始化的Linear层转换输入
        query_transformed = self.q(query)
        key_transformed = self.k(key)
        value_transformed = self.v(value)
        
        # 调用scaled_dot_product_one_loop_batch函数计算注意力
        y, self.weights_softmax = scaled_dot_product_one_loop_batch(
            query_transformed, key_transformed, value_transformed, mask
        )
        
        return y
        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return y


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_out: int):
        super().__init__()

        """
        
        A naive implementation of the MultiheadAttention layer for Transformer model.
        We use multiple SelfAttention layers parallely on the same input and then concat
        them to into a single tensor. This Tensor is then passed through an MLP to 
        generate the final output. The input shape will look like (N, K, M) where  
        N is the batch size, K is the batch size and M is the sequence embedding  
        dimension.
        args:
            num_heads: int value specifying the number of heads
            dim_in: int value specifying the input dimension of the query, key
                and value. This will be the input dimension to each of the
                SingleHeadAttention blocks
            dim_out: int value specifying the output dimension of the complete 
                MultiHeadAttention block



        NOTE: Here, when we say dimension, we mean the dimesnion of the embeddings.
              In Transformers the input is a tensor of shape (N, K, M), here N is
              the batch size , K is the sequence length and M is the size of the
              input embeddings. As the sequence length(K) and number of batches(N)
              don't change usually, we mostly transform
              the dimension(M) dimension.


        """

        # 初始化一个SingleHeadAttention层列表
        self.heads = nn.ModuleList([
            SelfAttention(dim_in, dim_out, dim_out) for _ in range(num_heads)
        ])
        
        # 初始化一个Linear层将输出映射回dim_in
        self.out_proj = nn.Linear(dim_out * num_heads, dim_in)
        
        # 初始化out_proj的权重
        D_in = self.out_proj.in_features
        D_out = self.out_proj.out_features
        c = torch.sqrt(torch.tensor(6.0 / (D_in + D_out)))
        nn.init.uniform_(self.out_proj.weight, -c, c)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None
    ) -> Tensor:

        """
        An implementation of the forward pass of the MultiHeadAttention layer.

        args:
            query: Tensor of shape (N, K, M) where N is the number of sequences in
                the batch, K is the sequence length and M is the input embedding
                dimension. M should be equal to dim_in in the init function

            key: Tensor of shape (N, K, M) where N is the number of sequences in
                the batch, K is the sequence length and M is the input embedding
                dimension. M should be equal to dim_in in the init function

            value: Tensor of shape (N, K, M) where N is the number of sequences in
                the batch, K is the sequence length and M is the input embedding
                dimension. M should be equal to dim_in in the init function

            mask: Tensor of shape (N, K, K) where N is the number of sequences in
                the batch, K is the sequence length and M is the input embedding
                dimension. M should be equal to dim_in in the init function

        returns:
            y: Tensor of shape (N, K, M)
        """
        # 将输入传递给每个SingleHeadAttention并收集结果
        head_outputs = []
        for head in self.heads:
            head_output = head(query, key, value, mask)
            head_outputs.append(head_output)
        
        # 在最后一个维度上连接所有头的输出
        # 输入形状：(N, K, dim_out) * num_heads
        # 连接后形状：(N, K, dim_out * num_heads)
        concatenated = torch.cat(head_outputs, dim=-1)
        
        # 通过Linear层映射回原始维度
        y = self.out_proj(concatenated)
        
        return y


class LayerNormalization(nn.Module):
    def __init__(self, emb_dim: int, epsilon: float = 1e-10):
        super().__init__()
        """
        The class implements the Layer Normalization for Linear layers in 
        Transformers.  Unlike BathcNorm ,it estimates the normalization statistics 
        for each element present in the batch and hence does not depend on the  
        complete batch.
        The input shape will look something like (N, K, M) where N is the batch 
        size, K is the sequence length and M is the sequence length embedding. We 
        compute the  mean with shape (N, K) and standard deviation with shape (N, K) 
        and use them to normalize each sequence.
        
        args:
            emb_dim: int representing embedding dimension
            epsilon: float value

        """

        self.epsilon = epsilon

        # 初始化scale参数(gamma)为全1，shape为(input_dim,)
        self.gamma = nn.Parameter(torch.ones(input_dim))
        # 初始化shift参数(beta)为全0，shape为(input_dim,)
        self.beta = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x: Tensor):
        """
        An implementation of the forward pass of the Layer Normalization.

        args:
            x: a Tensor of shape (N, K, M) or (N, K) where N is the batch size, K
                is the sequence length and M is the embedding dimension

        returns:
            y: a Tensor of shape (N, K, M) or (N, K) after applying layer
                normalization

        """
        # 根据输入维度确定在哪些维度上计算均值和方差
        # 对于形状为(N, K, M)或(N, K)的输入，我们需要在最后一个维度上进行标准化
        # 计算输入在最后一个维度上的均值
        mean = x.mean(dim=-1, keepdim=True)
        
        # 计算输入在最后一个维度上的方差（不使用torch.std）
        variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        
        # 计算标准差
        std = torch.sqrt(variance + self.epsilon)
        
        # 标准化输入
        x_normalized = (x - mean) / std
        
        # 应用scale和shift参数
        # 注意需要将gamma和beta扩展到与输入匹配的维度
        y = self.gamma.unsqueeze(0).unsqueeze(0) * x_normalized + self.beta.unsqueeze(0).unsqueeze(0)
        
        # 如果输入是二维的(N, K)，则相应地调整
        if x.ndim == 2:
            y = self.gamma.unsqueeze(0) * x_normalized + self.beta.unsqueeze(0)
        
        return y


class FeedForwardBlock(nn.Module):
    def __init__(self, inp_dim: int, hidden_dim_feedforward: int):
        super().__init__()

        """
        An implementation of the FeedForward block in the Transformers. We pass  
        the input through stacked 2 MLPs and 1 ReLU layer. The forward pass has  
        following architecture:
        
        linear - relu -linear
        
        The input will have a shape of (N, K, M) where N is the batch size, K is 
        the sequence length and M is the embedding dimension. 
        
        args:
            inp_dim: int representing embedding dimension of the input tensor
                     
            hidden_dim_feedforward: int representing the hidden dimension for
                the feedforward block
        """

        # 初始化第一个Linear层：inp_dim -> hidden_dim_feedforward
        self.linear1 = nn.Linear(inp_dim, hidden_dim_feedforward)
        # 初始化第二个Linear层：hidden_dim_feedforward -> inp_dim（保持输入输出形状一致）
        self.linear2 = nn.Linear(hidden_dim_feedforward, inp_dim)
        
        # 应用与SelfAttention相同的权重初始化策略
        for layer in [self.linear1, self.linear2]:
            D_in = layer.in_features
            D_out = layer.out_features
            c = torch.sqrt(torch.tensor(6.0 / (D_in + D_out)))
            # 使用均匀分布初始化权重：[-c, c]
            nn.init.uniform_(layer.weight, -c, c)
            # 初始化偏置为0
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        """
        An implementation of the forward pass of the FeedForward block.

        args:
            x: a Tensor of shape (N, K, M) which is the output of
               MultiHeadAttention
        returns:
            y: a Tensor of shape (N, K, M)
        """
        # 执行前向传播：linear1 -> relu -> linear2
        x = self.linear1(x)
        x = torch.relu(x)
        y = self.linear2(x)
        
        return y


class EncoderBlock(nn.Module):
    def __init__(
        self, num_heads: int, emb_dim: int, feedforward_dim: int, dropout: float
    ):
        super().__init__()
        """
        This class implements the encoder block for the Transformer model, the 
        original paper used 6 of these blocks sequentially to train the final model. 
        Here, we will first initialize the required layers using the building  
        blocks we have already  implemented, and then finally write the forward     
        pass using these initialized layers, residual connections and dropouts.        
        
        As shown in the Figure 1 of the paper attention is all you need
        https://arxiv.org/pdf/1706.03762.pdf, the encoder consists of four components:
        
        1. MultiHead Attention
        2. FeedForward layer
        3. Residual connections after MultiHead Attention and feedforward layer
        4. LayerNorm
        
        The architecture is as follows:
        
       inp - multi_head_attention - out1 - layer_norm(out1 + inp) - dropout - out2 \ 
        - feedforward - out3 - layer_norm(out3 + out2) - dropout - out
        
        Here, inp is input of the MultiHead Attention of shape (N, K, M), out1, 
        out2 and out3 are the outputs of the corresponding layers and we add these 
        outputs to their respective inputs for implementing residual connections.

        args:
            num_heads: int value specifying the number of heads in the
                MultiHeadAttention block of the encoder

            emb_dim: int value specifying the embedding dimension of the input
                sequence

            feedforward_dim: int value specifying the number of hidden units in the 
                FeedForward layer of Transformer

            dropout: float value specifying the dropout value


        """

        if emb_dim % num_heads != 0:
            raise ValueError(
                f"""The value emb_dim = {emb_dim} is not divisible
                             by num_heads = {num_heads}. Please select an
                             appropriate value."""
            )

        # 计算每个head的输出维度
        # 根据提示，我们希望连接后的输出与输入维度相同，所以每个head的输出维度应该是emb_dim / num_heads
        dim_per_head = emb_dim // num_heads
        
        # 1. 初始化MultiHeadAttention块
        self.multihead_attention = MultiHeadAttention(
            dim_in=emb_dim,
            num_heads=num_heads,
            dim_out=dim_per_head
        )
        
        # 2. 初始化两个LayerNorm层
        self.layernorm1 = LayerNorm(emb_dim)
        self.layernorm2 = LayerNorm(emb_dim)
        
        # 3. 初始化FeedForward块
        self.feedforward = FeedForwardBlock(emb_dim, feedforward_dim)
        
        # 4. 初始化Dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        """

        An implementation of the forward pass of the EncoderBlock of the
        Transformer model.
        args:
            x: a Tensor of shape (N, K, M) as input sequence
        returns:
            y: a Tensor of shape (N, K, M) as the output of the forward pass
        """
        # 第一步：多头注意力
        # 对于自注意力，query、key和value都是相同的输入x
        attn_output = self.multihead_attention(x, x, x)
        
        # 残差连接 + LayerNorm
        # out1 + inp -> layer_norm
        x = x + attn_output  # 残差连接
        x = self.layernorm1(x)  # LayerNorm
        
        # Dropout
        x = self.dropout(x)
        
        # 第二步：前馈网络
        ff_output = self.feedforward(x)
        
        # 残差连接 + LayerNorm
        # out3 + out2 -> layer_norm
        x = x + ff_output  # 残差连接
        x = self.layernorm2(x)  # LayerNorm
        
        # 最后一个Dropout
        y = self.dropout(x)
        
        return y


def get_subsequent_mask(seq):
    """
    An implementation of the decoder self attention mask. This will be used to
    mask the target sequence while training the model. The input shape here is
    (N, K) where N is the batch size and K is the sequence length.

    args:
        seq: a tensor of shape (N, K) where N is the batch sieze and K is the
             length of the sequence
    return:
        mask: a tensor of shape (N, K, K) where N is the batch sieze and K is the
              length of the sequence

    Given a sequence of length K, we want to mask the weights inside the function
    `self_attention_no_loop_batch` so that it prohibits the decoder to look ahead
    in the future
    """
    N, K = seq.shape
    
    # 创建一个上三角矩阵，对角线以上的元素为True（需要掩码）
    # 使用triu创建上三角矩阵，然后调整使其符合我们的需求
    # 创建一个(N, K, K)的掩码，其中mask[i, j]为True表示位置i不能看到位置j
    # 对于解码器自注意力，每个位置只能看到它自己和之前的位置
    # 所以对于位置j来说，所有大于j的位置i都应该被掩码
    mask = torch.triu(torch.ones(K, K, device=seq.device, dtype=torch.bool), diagonal=1)
    
    # 将掩码扩展到批次维度
    mask = mask.unsqueeze(0).repeat(N, 1, 1)  # 形状变为 (N, K, K)
    
    return mask


class DecoderBlock(nn.Module):
    def __init__(
        self, num_heads: int, emb_dim: int, feedforward_dim: int, dropout: float
    ):
        super().__init__()
        if emb_dim % num_heads != 0:
            raise ValueError(
                f"""The value emb_dim = {emb_dim} is not divisible
                             by num_heads = {num_heads}. Please select an
                             appropriate value."""
            )

        """
        The function implements the DecoderBlock for the Transformer model. In the 
        class we learned about encoder only model that can be used for tasks like 
        sequence classification but for more complicated tasks like sequence to 
        sequence we need a decoder network that can transformt the output of the 
        encoder to a target sequence. This kind of architecture is important in 
        tasks like language translation where we have a sequence as input and a 
        sequence as output. 
        
        As shown in the Figure 1 of the paper attention is all you need
        https://arxiv.org/pdf/1706.03762.pdf, the encoder consists of 5 components:   
        
        1. Masked MultiHead Attention
        2. MultiHead Attention
        3. FeedForward layer
        4. Residual connections after MultiHead Attention and feedforward layer
        5. LayerNorm        
        
        The Masked MultiHead Attention takes the target, masks it as per the 
        function get_subsequent_mask and then gives the output as per the MultiHead  
        Attention layer. Further, another Multihead Attention block here takes the  
        encoder output and the output from Masked Multihead Attention layer giving  
        the output that helps the model create interaction between input and 
        targets. As this block helps in interation of the input and target, it  
        is also sometimes called the cross attention.

        The architecture is as follows:
        
        inp - masked_multi_head_attention - out1 - layer_norm(inp + out1) - \
        dropout - (out2 and enc_out) -  multi_head_attention - out3 - \
        layer_norm(out3 + out2) - dropout - out4 - feed_forward - out5 - \
        layer_norm(out5 + out4) - dropout - out
        
        Here, out1, out2, out3, out4, out5 are the corresponding outputs for the 
        layers, enc_out is the encoder output and we add these outputs to their  
        respective inputs for implementing residual connections.
        
        args:
            num_heads: int value representing number of heads

            emb_dim: int value representing embedding dimension

            feedforward_dim: int representing hidden layers in the feed forward 
                model

            dropout: float representing the dropout value
        """
        # 计算每个头的维度
        dim_per_head = emb_dim // num_heads
        
        # 初始化自注意力层（掩码多头注意力）
        self.attention_self = MultiHeadAttention(
            dim_in=emb_dim, num_heads=num_heads, dim_out=dim_per_head
        )
        
        # 初始化交叉注意力层（多头注意力，用于连接编码器和解码器）
        self.attention_cross = MultiHeadAttention(
            dim_in=emb_dim, num_heads=num_heads, dim_out=dim_per_head
        )
        
        # 初始化前馈网络层
        self.feed_forward = FeedForwardBlock(
            inp_dim=emb_dim, hidden_dim_feedforward=feedforward_dim
        )
        
        # 初始化三个LayerNorm层，分别用于每个主要操作后的归一化
        self.norm1 = LayerNorm(emb_dim)
        self.norm2 = LayerNorm(emb_dim)
        self.norm3 = LayerNorm(emb_dim)
        
        # 初始化Dropout层
        self.dropout = nn.Dropout(dropout)
        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def forward(
        self, dec_inp: Tensor, enc_inp: Tensor, mask: Tensor = None
    ) -> Tensor:

        """
        args:
            dec_inp: a Tensor of shape (N, K, M)
            enc_inp: a Tensor of shape (N, K, M)
            mask: a Tensor of shape (N, K, K)

        This function will handle the forward pass of the Decoder block. It takes
        in input as enc_inp which is the encoder output and a tensor dec_inp which
        is the target sequence shifted by one in case of training and an initial
        token "BOS" during inference
        """
        # 第一步：掩码自注意力 + 残差连接 + LayerNorm + Dropout
        # 自注意力使用三个相同的输入和提供的掩码
        attention_output = self.attention_self(dec_inp, dec_inp, dec_inp, mask)
        
        # 残差连接：将输入添加到注意力输出
        attention_with_residual = dec_inp + attention_output
        
        # LayerNorm
        norm1_output = self.norm1(attention_with_residual)
        
        # Dropout
        dropout1_output = self.dropout(norm1_output)
        
        # 第二步：交叉注意力 + 残差连接 + LayerNorm + Dropout
        # 交叉注意力使用dropout1_output作为查询，enc_inp作为键和值
        cross_attention_output = self.attention_cross(dropout1_output, enc_inp, enc_inp)
        
        # 残差连接
        cross_attention_with_residual = dropout1_output + cross_attention_output
        
        # LayerNorm
        norm2_output = self.norm2(cross_attention_with_residual)
        
        # Dropout
        dropout2_output = self.dropout(norm2_output)
        
        # 第三步：前馈网络 + 残差连接 + LayerNorm + Dropout
        ff_output = self.feed_forward(dropout2_output)
        
        # 残差连接
        ff_with_residual = dropout2_output + ff_output
        
        # LayerNorm
        norm3_output = self.norm3(ff_with_residual)
        
        # Dropout
        y = self.dropout(norm3_output)
        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################
        return y


class Encoder(nn.Module):
    def __init__(
        self,
        num_heads: int,
        emb_dim: int,
        feedforward_dim: int,
        num_layers: int,
        dropout: float,
    ):
        """
        The class encapsulates the implementation of the final Encoder that use
        multiple EncoderBlock layers.

        args:
            num_heads: int representing number of heads to be used in the
                EncoderBlock
            emb_dim: int repreesenting embedding dimension for the Transformer
                model
            feedforward_dim: int representing hidden layer dimension for the
                feed forward block

        """

        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderBlock(num_heads, emb_dim, feedforward_dim, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, src_seq: Tensor):
        for _layer in self.layers:
            src_seq = _layer(src_seq)

        return src_seq


class Decoder(nn.Module):
    def __init__(
        self,
        num_heads: int,
        emb_dim: int,
        feedforward_dim: int,
        num_layers: int,
        dropout: float,
        vocab_len: int,
    ):
        super().__init__()
        """
        The Decoder takes the input from the encoder and the target
        sequence to generate the final sequence for the output. We
        first pass the input through stacked DecoderBlocks and then
        project the output to vocab_len which is required to get the
        actual sequence.
        
        args:
            num_heads: Int representing number of heads in the MultiheadAttention
            for Transformer
            emb_dim: int representing the embedding dimension
            of the sequence
            feedforward_dim: hidden layers in the feed forward block
            num_layers: int representing the number of DecoderBlock in Decoder
            dropout: float representing the dropout in each DecoderBlock
            vocab_len: length of the vocabulary


        """

        self.layers = nn.ModuleList(
            [
                DecoderBlock(num_heads, emb_dim, feedforward_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.proj_to_vocab = nn.Linear(emb_dim, vocab_len)
        a = (6 / (emb_dim + vocab_len)) ** 0.5
        nn.init.uniform_(self.proj_to_vocab.weight, -a, a)

    def forward(self, target_seq: Tensor, enc_out: Tensor, mask: Tensor):

        out = target_seq.clone()
        for _layer in self.layers:
            out = _layer(out, enc_out, mask)
        out = self.proj_to_vocab(out)
        return out


def position_encoding_simple(K: int, M: int) -> Tensor:
    """
    An implementation of the simple positional encoding using uniform intervals
    for a sequence.

    args:
        K: int representing sequence length
        M: int representing embedding dimension for the sequence

    return:
        y: a Tensor of shape (1, K, M)
    """
    # 创建一个从0到K-1的序列，然后除以K得到0/K, 1/K, ..., (K-1)/K
    positions = torch.arange(K).float() / K
    
    # 扩展维度并复制M次
    # 首先添加一个维度，使形状变为(K, 1)
    # 然后使用repeat扩展到(K, M)
    positions_encoded = positions.unsqueeze(1).repeat(1, M)
    
    # 最后添加批次维度，得到(1, K, M)的形状
    y = positions_encoded.unsqueeze(0)
    ##############################################################################
    #               END OF YOUR CODE                                             #
    ##############################################################################
    return y


def position_encoding_sinusoid(K: int, M: int) -> Tensor:

    """
    An implementation of the sinousoidal positional encodings.

    args:
        K: int representing sequence length
        M: int representing embedding dimension for the sequence

    return:
        y: a Tensor of shape (1, K, M)

    """
    # 创建位置编码矩阵，形状为(1, K, M)
    # 初始化一个全零矩阵
    pe = torch.zeros(1, K, M)
    
    # 创建位置索引矩阵，形状为(K, 1)
    position = torch.arange(0, K, dtype=torch.float).unsqueeze(1)
    
    # 创建分母项，按照论文公式：10000^(2i/M)
    # 计算每个维度的位置编码频率
    div_term = torch.exp(torch.arange(0, M, 2).float() * (-math.log(10000.0) / M))
    
    # 偶数索引维度使用sin，奇数索引维度使用cos
    pe[:, :, 0::2] = torch.sin(position * div_term)
    pe[:, :, 1::2] = torch.cos(position * div_term)
    
    y = pe
    return y


class Transformer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        emb_dim: int,
        feedforward_dim: int,
        dropout: float,
        num_enc_layers: int,
        num_dec_layers: int,
        vocab_len: int,
    ):
        super().__init__()

        """
        The class implements Transformer model with encoder and decoder. The input
        to the model is a tensor of shape (N, K) and the output is a tensor of shape
        (N*O, V). Here, N is the batch size, K is the input sequence length, O is  
        the output sequence length and V is the Vocabulary size. The input is passed  
        through shared nn.Embedding layer and then added to input positonal 
        encodings. Similarily, the target is passed through the same nn.Embedding
        layer and added to the target positional encodings. The only difference
        is that we take last but one  value in the target. The summed 
        inputs(look at the code for detials) are then sent through the encoder and  
        decoder blocks  to get the  final output.
        args:
            num_heads: int representing number of heads to be used in Encoder
                       and decoder
            emb_dim: int representing embedding dimension of the Transformer
            dim_feedforward: int representing number of hidden layers in the
                             Encoder and decoder
            dropout: a float representing probability for dropout layer
            num_enc_layers: int representing number of encoder blocks
            num_dec_layers: int representing number of decoder blocks

        """
        # 初始化嵌入层，将词汇表索引映射到嵌入维度
        self.emb_layer = nn.Embedding(vocab_len, emb_dim)
        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################
        self.encoder = Encoder(
            num_heads, emb_dim, feedforward_dim, num_enc_layers, dropout
        )
        self.decoder = Decoder(
            num_heads,
            emb_dim,
            feedforward_dim,
            num_dec_layers,
            dropout,
            vocab_len,
        )

    def forward(
        self, ques_b: Tensor, ques_pos: Tensor, ans_b: Tensor, ans_pos: Tensor
    ) -> Tensor:

        """

        An implementation of the forward pass of the Transformer.

        args:
            ques_b: Tensor of shape (N, K) that consists of input sequence of
                the arithmetic expression
            ques_pos: Tensor of shape (N, K, M) that consists of positional
                encodings of the input sequence
            ans_b: Tensor of shape (N, K) that consists of target sequence
                of arithmetic expression
            ans_pos: Tensor of shape (N, K, M) that consists of positonal
                encodings of the target sequence

        returns:
            dec_out: Tensor of shape (N*O, M) where O is the size of
                the target sequence.
        """
        q_emb = self.emb_layer(ques_b)
        a_emb = self.emb_layer(ans_b)
        q_emb_inp = q_emb + ques_pos
        a_emb_inp = a_emb[:, :-1] + ans_pos[:, :-1]
        
        # 第一步：将输入通过编码器
        enc_out = self.encoder(q_emb_inp)
        
        # 第二步：为解码器创建掩码，掩码形状取决于ans_b的形状
        # 注意我们需要使用ans_b的前n-1个元素来创建掩码，因为a_emb_inp也是ans_b的前n-1个元素
        mask = get_subsequent_mask(ans_b[:, :-1])
        
        # 第三步：将解码器输入、编码器输出和掩码传递给解码器
        dec_out = self.decoder(a_emb_inp, enc_out, mask)
        
        # 将输出重塑为(N*O, V)，其中O是目标序列长度，V是词汇表大小
        # 首先获取批次大小和序列长度
        N, O, V = dec_out.shape
        dec_out = dec_out.view(N * O, V)
        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return dec_out


class AddSubDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        input_seqs,
        target_seqs,
        convert_str_to_tokens,
        special_tokens,
        emb_dim,
        pos_encode,
    ):

        """
        The class implements the dataloader that will be used for the toy dataset.

        args:
            input_seqs: A list of input strings
            target_seqs: A list of output strings
            convert_str_to_tokens: Dictionary to convert input string to tokens
            special_tokens: A list of strings
            emb_dim: embedding dimension of the transformer
            pos_encode: A function to compute positional encoding for the data
        """

        self.input_seqs = input_seqs
        self.target_seqs = target_seqs
        self.convert_str_to_tokens = convert_str_to_tokens
        self.emb_dim = emb_dim
        self.special_tokens = special_tokens
        self.pos_encode = pos_encode

    def preprocess(self, inp):
        return prepocess_input_sequence(
            inp, self.convert_str_to_tokens, self.special_tokens
        )

    def __getitem__(self, idx):
        """
        The core fucntion to get element with index idx in the data.
        args:
            idx: index of the element that we need to extract from the data
        returns:
            preprocess_inp: A 1D tensor of length K, where K is the input sequence
                length
            inp_pos_enc: A tensor of shape (K, M), where K is the sequence length
                and M is the embedding dimension
            preprocess_out: A 1D tensor of length O, where O is the output
                sequence length
            out_pos_enc: A tensor of shape (O, M), where O is the sequence length
                and M is the embedding dimension
        """

        inp = self.input_seqs[idx]
        out = self.target_seqs[idx]
        preprocess_inp = torch.tensor(self.preprocess(inp))
        preprocess_out = torch.tensor(self.preprocess(out))
        inp_pos = len(preprocess_inp)
        inp_pos_enc = self.pos_encode(inp_pos, self.emb_dim)
        out_pos = len(preprocess_out)
        out_pos_enc = self.pos_encode(out_pos, self.emb_dim)

        return preprocess_inp, inp_pos_enc[0], preprocess_out, out_pos_enc[0]

    def __len__(self):
        return len(self.input_seqs)


def LabelSmoothingLoss(pred, ground):
    """
    args:
        pred: predicted tensor of shape (N*O, V) where N is the batch size, O
            is the target sequence length and V is the size of the vocab
        ground: ground truth tensor of shape (N, O) where N is the batch size, O
            is the target sequence
    """
    ground = ground.contiguous().view(-1)
    eps = 0.1
    n_class = pred.size(1)
    one_hot = torch.nn.functional.one_hot(ground).to(pred.dtype)
    one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
    log_prb = F.log_softmax(pred, dim=1)
    loss = -(one_hot * log_prb).sum(dim=1)
    loss = loss.sum()
    return loss


def CrossEntropyLoss(pred, ground):
    """
    args:
        pred: predicted tensor of shape (N*O, V) where N is the batch size, O
            is the target sequence length and V is the size of the vocab
        ground: ground truth tensor of shape (N, O) where N is the batch size, O
            is the target sequence
    """
    loss = F.cross_entropy(pred, ground, reduction="sum")
    return loss
