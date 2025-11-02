import math
from typing import Optional, Tuple

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision.models import feature_extraction


def hello_rnn_lstm_captioning():
    print("Hello from rnn_lstm_captioning.py!")


class ImageEncoder(nn.Module):
    """
    Convolutional network that accepts images as input and outputs their spatial
    grid features. This module servesx as the image encoder in image captioning
    model. We will use a tiny RegNet-X 400MF model that is initialized with
    ImageNet-pretrained weights from Torchvision library.

    NOTE: We could use any convolutional network architecture, but we opt for a
    tiny RegNet model so it can train decently with a single K80 Colab GPU.
    """

    def __init__(self, pretrained: bool = True, verbose: bool = True):
        """
        Args:
            pretrained: Whether to initialize this model with pretrained weights
                from Torchvision library.
            verbose: Whether to log expected output shapes during instantiation.
        """
        super().__init__()
        self.cnn = torchvision.models.regnet_x_400mf(pretrained=pretrained)

        # Torchvision models return global average pooled features by default.
        # Our attention-based models may require spatial grid features. So we
        # wrap the ConvNet with torchvision's feature extractor. We will get
        # the spatial features right before the final classification layer.
        self.backbone = feature_extraction.create_feature_extractor(
            self.cnn, return_nodes={"trunk_output.block4": "c5"}
        )
        # We call these features "c5", a name that may sound familiar from the
        # object detection assignment. :-)

        # Pass a dummy batch of input images to infer output shape.
        dummy_out = self.backbone(torch.randn(2, 3, 224, 224))["c5"]
        self._out_channels = dummy_out.shape[1]

        if verbose:
            print("For input images in NCHW format, shape (2, 3, 224, 224)")
            print(f"Shape of output c5 features: {dummy_out.shape}")

        # Input image batches are expected to be float tensors in range [0, 1].
        # However, the backbone here expects these tensors to be normalized by
        # ImageNet color mean/std (as it was trained that way).
        # We define a function to transform the input images before extraction:
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    @property
    def out_channels(self):
        """
        Number of output channels in extracted image features. You may access
        this value freely to define more modules to go with this encoder.
        """
        return self._out_channels

    def forward(self, images: torch.Tensor):
        # Input images may be uint8 tensors in [0-255], change them to float
        # tensors in [0-1]. Get float type from backbone (could be float32/64).
        if images.dtype == torch.uint8:
            images = images.to(dtype=self.cnn.stem[0].weight.dtype)
            images /= 255.0

        # Normalize images by ImageNet color mean/std.
        images = self.normalize(images)

        # Extract c5 features from encoder (backbone) and return.
        # shape: (B, out_channels, H / 32, W / 32)
        features = self.backbone(images)["c5"]
        return features


##############################################################################
# Recurrent Neural Network                                                   #
##############################################################################
def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Args:
        x: Input data for this timestep, of shape (N, D).
        prev_h: Hidden state from previous timestep, of shape (N, H)
        Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
        Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
        b: Biases, of shape (H,)

    Returns a tuple of:
        next_h: Next hidden state, of shape (N, H)
        cache: Tuple of values needed for the backward pass.
    """
    # 计算线性组合: x @ Wx + prev_h @ Wh + b
    affine = x @ Wx + prev_h @ Wh + b
    # 应用tanh激活函数
    next_h = torch.tanh(affine)
    # 保存反向传播需要的值
    cache = (x, prev_h, Wx, Wh, affine)
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Args:
        dnext_h: Gradient of loss with respect to next hidden state, of shape (N, H)
        cache: Cache object from the forward pass

    Returns a tuple of:
        dx: Gradients of input data, of shape (N, D)
        dprev_h: Gradients of previous hidden state, of shape (N, H)
        dWx: Gradients of input-to-hidden weights, of shape (D, H)
        dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
        db: Gradients of bias vector, of shape (H,)
    """
    x, prev_h, Wx, Wh, affine = cache
    
    # 计算tanh的导数: 1 - tanh(x)^2 = 1 - next_h^2
    # 但我们可以直接用affine计算：d(tanh(affine))/daffine = 1 - tanh(affine)^2
    dtanh = 1 - torch.tanh(affine)**2
    
    # 梯度流动：dnext_h -> dtanh -> daffine
    daffine = dnext_h * dtanh
    
    # 计算各参数的梯度
    dx = daffine @ Wx.T
    dprev_h = daffine @ Wh.T
    dWx = x.T @ daffine
    dWh = prev_h.T @ daffine
    db = daffine.sum(dim=0)
    
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Args:
        x: Input data for the entire timeseries, of shape (N, T, D).
        h0: Initial hidden state, of shape (N, H)
        Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
        Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
        b: Biases, of shape (H,)

    Returns a tuple of:
        h: Hidden states for the entire timeseries, of shape (N, T, H).
        cache: Values needed in the backward pass
    """
    N, T, D = x.shape
    H = h0.shape[1]
    
    # 初始化隐藏状态矩阵和缓存列表
    h = torch.zeros(N, T, H, device=x.device, dtype=x.dtype)
    cache = []
    
    # 当前隐藏状态初始化为h0
    current_h = h0
    
    # 对每个时间步进行前向计算
    for t in range(T):
        # 取出当前时间步的输入
        x_t = x[:, t, :]
        # 计算当前时间步的隐藏状态
        current_h, step_cache = rnn_step_forward(x_t, current_h, Wx, Wh, b)
        # 保存当前隐藏状态
        h[:, t, :] = current_h
        # 保存当前时间步的缓存
        cache.append(step_cache)
    
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Args:
        dh: Upstream gradients of all hidden states, of shape (N, T, H).

    NOTE: 'dh' contains the upstream gradients produced by the
    individual loss functions at each timestep, *not* the gradients
    being passed between timesteps (which you'll have to compute yourself
    by calling rnn_step_backward in a loop).

    Returns a tuple of:
        dx: Gradient of inputs, of shape (N, T, D)
        dh0: Gradient of initial hidden state, of shape (N, H)
        dWx: Gradient of input-to-hidden weights, of shape (D, H)
        dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
        db: Gradient of biases, of shape (H,)
    """
    N, T, H = dh.shape
    # 从缓存中获取输入维度D
    x_t, _, Wx, _, _ = cache[0]
    D = x_t.shape[1]
    
    # 初始化各梯度
    dx = torch.zeros(N, T, D, device=dh.device, dtype=dh.dtype)
    dh0 = torch.zeros(N, H, device=dh.device, dtype=dh.dtype)
    dWx = torch.zeros(D, H, device=dh.device, dtype=dh.dtype)
    dWh = torch.zeros(H, H, device=dh.device, dtype=dh.dtype)
    db = torch.zeros(H, device=dh.device, dtype=dh.dtype)
    
    # 从最后一个时间步开始反向传播
    # 初始时，没有后续时间步的梯度，所以只使用dh[:, T-1, :]
    dnext_h = dh[:, T-1, :]
    
    for t in reversed(range(T)):
        # 计算当前时间步的梯度
        step_dx, dprev_h, step_dWx, step_dWh, step_db = rnn_step_backward(dnext_h, cache[t])
        
        # 保存当前时间步的输入梯度
        dx[:, t, :] = step_dx
        
        # 累加到参数梯度上
        dWx += step_dWx
        dWh += step_dWh
        db += step_db
        
        # 更新dnext_h为前一个时间步的梯度 + dh[:, t-1, :]
        if t > 0:
            dnext_h = dprev_h + dh[:, t-1, :]
        else:
            # 对于第一个时间步，dprev_h就是dh0
            dh0 = dprev_h
    
    return dx, dh0, dWx, dWh, db


class RNN(nn.Module):
    """
    Single-layer vanilla RNN module.

    You don't have to implement anything here but it is highly recommended to
    read through the code as you will implement subsequent modules.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Initialize an RNN. Model parameters to initialize:
            Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
            Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
            b: Biases, of shape (H,)

        Args:
            input_dim: Input size, denoted as D before
            hidden_dim: Hidden size, denoted as H before
        """
        super().__init__()

        # Register parameters
        self.Wx = nn.Parameter(
            torch.randn(input_dim, hidden_dim).div(math.sqrt(input_dim))
        )
        self.Wh = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim).div(math.sqrt(hidden_dim))
        )
        self.b = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x, h0):
        """
        Args:
            x: Input data for the entire timeseries, of shape (N, T, D)
            h0: Initial hidden state, of shape (N, H)

        Returns:
            hn: The hidden state output
        """
        hn, _ = rnn_forward(x, h0, self.Wx, self.Wh, self.b)
        return hn

    def step_forward(self, x, prev_h):
        """
        Args:
            x: Input data for one time step, of shape (N, D)
            prev_h: The previous hidden state, of shape (N, H)

        Returns:
            next_h: The next hidden state, of shape (N, H)
        """
        next_h, _ = rnn_step_forward(x, prev_h, self.Wx, self.Wh, self.b)
        return next_h


class WordEmbedding(nn.Module):
    """
    Simplified version of torch.nn.Embedding.

    We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    word to a vector of dimension D.

    Args:
        x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.

    Returns a tuple of:
        out: Array of shape (N, T, D) giving word vectors for all input words.
    """

    def __init__(self, vocab_size: int, embed_size: int):
        super().__init__()

        # Register parameters
        self.W_embed = nn.Parameter(
            torch.randn(vocab_size, embed_size).div(math.sqrt(vocab_size))
        )

    def forward(self, x):
        # 使用x中的索引从词嵌入矩阵中查找对应的词向量
        out = self.W_embed[x]
        return out


def temporal_softmax_loss(x, y, ignore_index=None):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, *summing* the loss over all timesteps and *averaging* across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional ignore_index argument
    tells us which elements in the caption should not contribute to the loss.

    Args:
        x: Input scores, of shape (N, T, V)
        y: Ground-truth indices, of shape (N, T) where each element is in the
            range 0 <= y[i, t] < V

    Returns a tuple of:
        loss: Scalar giving loss
    """
    # 使用PyTorch的交叉熵函数实现，按照要求使用单行代码
    loss = F.cross_entropy(x.reshape(-1, x.shape[-1]), y.reshape(-1), ignore_index=ignore_index, reduction='mean')
    return loss


class CaptioningRNN(nn.Module):
    """
    A CaptioningRNN produces captions from images using a recurrent
    neural network.

    The RNN receives input vectors of size D, has a vocab size of V, works on
    sequences of length T, has an RNN hidden dimension of H, uses word vectors
    of dimension W, and operates on minibatches of size N.

    Note that we don't use any regularization for the CaptioningRNN.

    You will implement the `__init__` method for model initialization and
    the `forward` method first, then come back for the `sample` method later.
    """

    def __init__(
        self,
        word_to_idx,
        input_dim: int = 512,
        wordvec_dim: int = 128,
        hidden_dim: int = 128,
        cell_type: str = "rnn",
        image_encoder_pretrained: bool = True,
        ignore_index: Optional[int] = None,
    ):
        """
        Construct a new CaptioningRNN instance.

        Args:
            word_to_idx: A dictionary giving the vocabulary. It contains V
                entries, and maps each string to a unique integer in the
                range [0, V).
            input_dim: Dimension D of input image feature vectors.
            wordvec_dim: Dimension W of word vectors.
            hidden_dim: Dimension H for the hidden state of the RNN.
            cell_type: What type of RNN to use; either 'rnn' or 'lstm'.
        """
        super().__init__()
        if cell_type not in {"rnn", "lstm", "attn"}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}

        vocab_size = len(word_to_idx)

        self._null = word_to_idx["<NULL>"]
        self._start = word_to_idx.get("<START>", None)
        self._end = word_to_idx.get("<END>", None)
        self.ignore_index = ignore_index

        ######################################################################
        # TODO: Initialize the image captioning module. Refer to the TODO
        # in the captioning_forward function on layers you need to create
        #
        # You may want to check the following pre-defined classes:
        # ImageEncoder WordEmbedding, RNN, LSTM, AttentionLSTM, nn.Linear
        #
        # (1) output projection (from RNN hidden state to vocab probability)
        # (2) feature projection (from CNN pooled feature to h0)
        ######################################################################
        # Replace "pass" statement with your code
        pass
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

    def forward(self, images, captions):
        """
        Compute training-time loss for the RNN. We input images and the GT
        captions for those images, and use an RNN (or LSTM) to compute loss. The
        backward part will be done by torch.autograd.

        Args:
            images: Input images, of shape (N, 3, 112, 112)
            captions: Ground-truth captions; an integer array of shape (N, T + 1)
                where each element is in the range 0 <= y[i, t] < V

        Returns:
            loss: A scalar loss
        """
        # Cut captions into two pieces: captions_in has everything but the last
        # word and will be input to the RNN; captions_out has everything but the
        # first word and this is what we will expect the RNN to generate. These
        # are offset by one relative to each other because the RNN should produce
        # word (t+1) after receiving word t. The first element of captions_in
        # will be the START token, and the first element of captions_out will
        # be the first word.
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        loss = 0.0
        ######################################################################
        # TODO: Implement the forward pass for the CaptioningRNN.
        # In the forward pass you will need to do the following:
        # (1) Use an affine transformation to project the image feature to
        #     the initial hidden state $h0$ (for RNN/LSTM, of shape (N, H)) or
        #     the projected CNN activation input $A$ (for Attention LSTM,
        #     of shape (N, H, 4, 4).
        # (2) Use a word embedding layer to transform the words in captions_in
        #     from indices to vectors, giving an array of shape (N, T, W).
        # (3) Use either a vanilla RNN or LSTM (depending on self.cell_type) to
        #     process the sequence of input word vectors and produce hidden state
        #     vectors for all timesteps, producing an array of shape (N, T, H).
        # (4) Use a (temporal) affine transformation to compute scores over the
        #     vocabulary at every timestep using the hidden states, giving an
        #     array of shape (N, T, V).
        # (5) Use (temporal) softmax to compute loss using captions_out, ignoring
        #     the points where the output word is <NULL>.
        #
        # Do not worry about regularizing the weights or their gradients!
        ######################################################################
        # 提取图像特征
        image_features = self.cnn(images)
        
        # 使用仿射变换将图像特征投影到初始隐藏状态
        h0 = self.W_proj(image_features)
        
        # 使用词嵌入层将captions_in中的词从索引转换为向量
        word_embeddings = self.word_embedding(captions_in)
        
        # 使用RNN或LSTM处理输入词向量序列
        if self.cell_type == 'rnn':
            h = self.rnn(word_embeddings, h0)
        elif self.cell_type == 'lstm':
            h = self.lstm(word_embeddings, h0)
        elif self.cell_type == 'attn':
            # 对于注意力机制，需要投影CNN特征
            A = self.W_proj_cnn(image_features)
            h = self.attn_lstm(word_embeddings, A)
        
        # 使用仿射变换计算每个时间步在词汇表上的分数
        scores = self.W_output(h)
        
        # 使用temporal_softmax计算损失，忽略输出词为<NULL>的点
        loss = temporal_softmax_loss(scores, captions_out, self.ignore_index)
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        return loss

    def sample(self, images, max_length=15):
        """
        Run a test-time forward pass for the model, sampling captions for input
        feature vectors.

        At each timestep, we embed the current word, pass it and the previous hidden
        state to the RNN to get the next hidden state, use the hidden state to get
        scores for all vocab words, and choose the word with the highest score as
        the next word. The initial hidden state is computed by applying an affine
        transform to the image features, and the initial word is the <START>
        token.

        For LSTMs you will also have to keep track of the cell state; in that case
        the initial cell state should be zero.

        Args:
            images: Input images, of shape (N, 3, 112, 112)
            max_length: Maximum length T of generated captions

        Returns:
            captions: Array of shape (N, max_length) giving sampled captions,
                where each element is an integer in the range [0, V). The first
                element of captions should be the first sampled word, not the
                <START> token.
        """
        N = images.shape[0]
        captions = self._null * images.new(N, max_length).fill_(1).long()

        if self.cell_type == "attn":
            attn_weights_all = images.new(N, max_length, 4, 4).fill_(0).float()

        # 提取图像特征
        image_features = self.cnn(images)
        
        # 根据不同的cell_type初始化不同的状态
        if self.cell_type == 'rnn' or self.cell_type == 'lstm':
            # 使用仿射变换将图像特征投影到初始隐藏状态
            h_prev = self.W_proj(image_features)
            
            # LSTM需要初始化cell state
            if self.cell_type == 'lstm':
                c_prev = torch.zeros_like(h_prev)
        elif self.cell_type == 'attn':
            # 对于注意力机制，投影CNN特征
            A = self.W_proj_cnn(image_features)
            h_prev = A.mean(dim=(2, 3))
            c_prev = h_prev
        
        # 初始化第一个输入为START token
        current_word = torch.full((N,), self._start, dtype=torch.long, device=images.device)
        
        # 逐时间步生成caption
        for t in range(max_length):
            # 将当前词转换为词嵌入
            word_emb = self.word_embedding(current_word.unsqueeze(1))  # 形状为(N, 1, W)
            
            # 通过RNN/LSTM处理当前词
            if self.cell_type == 'rnn':
                h = self.rnn(word_emb, h_prev)
                h_prev = h[:, -1]  # 取出最后一个时间步的隐藏状态
            elif self.cell_type == 'lstm':
                h = self.lstm(word_emb, h_prev)
                h_prev = h[:, -1]  # 对于简单实现，我们假设返回的是隐藏状态序列
            elif self.cell_type == 'attn':
                # 计算注意力权重
                attn, attn_weights = dot_product_attention(h_prev, A)
                attn_weights_all[:, t] = attn_weights
                # 通过注意力LSTM处理
                h_prev, c_prev = self.attn_lstm.step_forward(word_emb.squeeze(1), h_prev, c_prev, attn)
                h_prev = h_prev.unsqueeze(1)
            
            # 计算词汇表上的分数
            scores = self.W_output(h_prev.squeeze(1))  # 形状为(N, V)
            
            # 使用贪婪解码，选择概率最大的词
            current_word = torch.argmax(scores, dim=1)
            
            # 保存生成的词
            captions[:, t] = current_word
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################
        if self.cell_type == "attn":
            return captions, attn_weights_all.cpu()
        else:
            return captions


class LSTM(nn.Module):
    """Single-layer, uni-directional LSTM module."""

    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Initialize a LSTM. Model parameters to initialize:
            Wx: Weights for input-to-hidden connections, of shape (D, 4H)
            Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
            b: Biases, of shape (4H,)

        Args:
            input_dim: Input size, denoted as D before
            hidden_dim: Hidden size, denoted as H before
        """
        super().__init__()

        # Register parameters
        self.Wx = nn.Parameter(
            torch.randn(input_dim, hidden_dim * 4).div(math.sqrt(input_dim))
        )
        self.Wh = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim * 4).div(math.sqrt(hidden_dim))
        )
        self.b = nn.Parameter(torch.zeros(hidden_dim * 4))

    def step_forward(
        self, x: torch.Tensor, prev_h: torch.Tensor, prev_c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single timestep of an LSTM.
        The input data has dimension D, the hidden state has dimension H, and
        we use a minibatch size of N.

        Args:
            x: Input data for one time step, of shape (N, D)
            prev_h: The previous hidden state, of shape (N, H)
            prev_c: The previous cell state, of shape (N, H)
            Wx: Input-to-hidden weights, of shape (D, 4H)
            Wh: Hidden-to-hidden weights, of shape (H, 4H)
            b: Biases, of shape (4H,)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]
                next_h: Next hidden state, of shape (N, H)
                next_c: Next cell state, of shape (N, H)
        """
        ######################################################################
        # Implement the forward pass for a single timestep of an LSTM.
        ######################################################################
        # Compute affine transformations for all gates
        affine_x = x @ self.Wx
        affine_h = prev_h @ self.Wh
        affine = affine_x + affine_h + self.b
        
        # Split into four gates
        N, H4 = affine.shape
        H = H4 // 4
        i, f, o, g = torch.split(affine, H, dim=1)
        
        # Apply activation functions
        i = torch.sigmoid(i)  # input gate
        f = torch.sigmoid(f)  # forget gate
        o = torch.sigmoid(o)  # output gate
        g = torch.tanh(g)     # candidate cell state
        
        # Update cell state
        next_c = f * prev_c + i * g
        
        # Update hidden state
        next_h = o * torch.tanh(next_c)
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################
        return next_h, next_c

    def forward(self, x: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for an LSTM over an entire sequence of data. We assume an
        input sequence composed of T vectors, each of dimension D. The LSTM
        uses a hidden size of H, and we work over a minibatch containing N
        sequences. After running the LSTM forward, we return the hidden states
        for all timesteps.

        Note that the initial cell state is passed as input, but the initial
        cell state is set to zero. Also note that the cell state is not returned;
        it is an internal variable to the LSTM and is not accessed from outside.

        Args:
            x: Input data for the entire timeseries, of shape (N, T, D)
            h0: Initial hidden state, of shape (N, H)

        Returns:
            hn: The hidden state output.
        """

        c0 = torch.zeros_like(
            h0
        )  # we provide the intial cell state c0 here for you!
        ######################################################################
        # TODO: Implement the forward pass for an LSTM over entire timeseries
        ######################################################################
        N, T, D = x.shape
        H = h0.shape[1]
        
        # 初始化隐藏状态矩阵
        hn = torch.zeros(N, T, H, device=x.device, dtype=x.dtype)
        
        # 当前隐藏状态初始化为h0，细胞状态初始化为c0
        current_h = h0
        current_c = c0
        
        # 对每个时间步进行前向计算
        for t in range(T):
            # 取出当前时间步的输入
            x_t = x[:, t, :]
            # 计算当前时间步的隐藏状态和细胞状态
            current_h, current_c = self.step_forward(x_t, current_h, current_c)
            # 保存当前隐藏状态
            hn[:, t, :] = current_h
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        return hn


def dot_product_attention(prev_h, A):
    """
    A simple scaled dot-product attention layer.

    Args:
        prev_h: The LSTM hidden state from previous time step, of shape (N, H)
        A: **Projected** CNN feature activation, of shape (N, H, 4, 4),
         where H is the LSTM hidden state size

    Returns:
        attn: Attention embedding output, of shape (N, H)
        attn_weights: Attention weights, of shape (N, 4, 4)

    """
    N, H, _, _ = A.shape
    
    # 将A重塑为(N, H, 16)，将空间维度展平
    A_flat = A.view(N, H, 16)  # 形状为(N, H, 16)
    
    # 计算注意力分数：h_prev与每个CNN特征的点积
    # h_prev形状为(N, H)，需要扩展为(N, H, 1)
    h_prev_reshaped = prev_h.unsqueeze(2)  # 形状为(N, H, 1)
    
    # 计算点积：(N, H, 16) @ (N, H, 1) 沿着H维度求和
    scores = (A_flat * h_prev_reshaped).sum(dim=1)  # 形状为(N, 16)
    
    # 应用softmax获取注意力权重
    attn_weights = torch.softmax(scores, dim=1)  # 形状为(N, 16)
    
    # 重塑注意力权重为(N, 1, 16)
    attn_weights_reshaped = attn_weights.unsqueeze(1)  # 形状为(N, 1, 16)
    
    # 使用权重对CNN特征进行加权求和
    attn = (A_flat * attn_weights_reshaped).sum(dim=2)  # 形状为(N, H)
    
    # 将注意力权重重塑回空间形状
    attn_weights = attn_weights.view(N, 4, 4)  # 形状为(N, 4, 4)
    
    return attn, attn_weights


class AttentionLSTM(nn.Module):
    """
    This is our single-layer, uni-directional Attention module.

    Args:
        input_dim: Input size, denoted as D before
        hidden_dim: Hidden size, denoted as H before
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Initialize a LSTM. Model parameters to initialize:
            Wx: Weights for input-to-hidden connections, of shape (D, 4H)
            Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
            Wattn: Weights for attention-to-hidden connections, of shape (H, 4H)
            b: Biases, of shape (4H,)
        """
        super().__init__()

        # Register parameters
        self.Wx = nn.Parameter(
            torch.randn(input_dim, hidden_dim * 4).div(math.sqrt(input_dim))
        )
        self.Wh = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim * 4).div(math.sqrt(hidden_dim))
        )
        self.Wattn = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim * 4).div(math.sqrt(hidden_dim))
        )
        self.b = nn.Parameter(torch.zeros(hidden_dim * 4))

    def step_forward(
        self,
        x: torch.Tensor,
        prev_h: torch.Tensor,
        prev_c: torch.Tensor,
        attn: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input data for one time step, of shape (N, D)
            prev_h: The previous hidden state, of shape (N, H)
            prev_c: The previous cell state, of shape (N, H)
            attn: The attention embedding, of shape (N, H)

        Returns:
            next_h: The next hidden state, of shape (N, H)
            next_c: The next cell state, of shape (N, H)
        """
        # Compute affine transformations for all gates
        affine_x = x @ self.Wx
        affine_h = prev_h @ self.Wh
        affine_attn = attn @ self.Wattn
        affine = affine_x + affine_h + affine_attn + self.b
        
        # Split into four gates
        N, H4 = affine.shape
        H = H4 // 4
        i, f, o, g = torch.split(affine, H, dim=1)
        
        # Apply activation functions
        i = torch.sigmoid(i)  # input gate
        f = torch.sigmoid(f)  # forget gate
        o = torch.sigmoid(o)  # output gate
        g = torch.tanh(g)     # candidate cell state
        
        # Update cell state
        next_c = f * prev_c + i * g
        
        # Update hidden state
        next_h = o * torch.tanh(next_c)
        
        return next_h, next_c

    def forward(self, x: torch.Tensor, A: torch.Tensor):
        """
        Forward pass for an LSTM over an entire sequence of data. We assume an
        input sequence composed of T vectors, each of dimension D. The LSTM uses
        a hidden size of H, and we work over a minibatch containing N sequences.
        After running the LSTM forward, we return hidden states for all timesteps.

        Note that the initial cell state is passed as input, but the initial cell
        state is set to zero. Also note that the cell state is not returned; it
        is an internal variable to the LSTM and is not accessed from outside.

        h0 and c0 are same initialized as the global image feature (meanpooled A)
        For simplicity, we implement scaled dot-product attention, which means in
        Eq. 4 of the paper (https://arxiv.org/pdf/1502.03044.pdf),
        f_{att}(a_i, h_{t-1}) equals to the scaled dot product of a_i and h_{t-1}.

        Args:
            x: Input data for the entire timeseries, of shape (N, T, D)
            A: The projected CNN feature activation, of shape (N, H, 4, 4)

        Returns:
            hn: The hidden state output
        """

        # The initial hidden state h0 and cell state c0 are initialized
        # differently in AttentionLSTM from the original LSTM and hence
        # we provided them for you.
        h0 = A.mean(dim=(2, 3))  # Initial hidden state, of shape (N, H)
        c0 = h0  # Initial cell state, of shape (N, H)

        # 获取输入数据的形状
        N, T, D = x.shape
        H = h0.shape[1]
        
        # 初始化隐藏状态矩阵
        hn = torch.zeros(N, T, H, device=x.device, dtype=x.dtype)
        
        # 当前隐藏状态初始化为h0，细胞状态初始化为c0
        current_h = h0
        current_c = c0
        
        # 对每个时间步进行前向计算
        for t in range(T):
            # 取出当前时间步的输入
            x_t = x[:, t, :]
            
            # 计算注意力权重和加权特征
            attn, _ = dot_product_attention(current_h, A)
            
            # 通过Attention LSTM处理当前词和注意力特征
            current_h, current_c = self.step_forward(x_t, current_h, current_c, attn)
            
            # 保存当前隐藏状态
            hn[:, t, :] = current_h
            
        return hn
