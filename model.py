"""
Attention Is All You Need: Build the Transformer From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_token_to_id_vocab
def build_token_to_id_vocab(sentences, specials=('<pad>', '<bos>', '<eos>', '<unk>')):
    """
    Build a token-to-id dict with specials first, then corpus tokens in first-seen order.
    
    Args:
        sentences: List of whitespace-tokenized string sentences.
        specials: Iterable of special tokens to initialize at the start of the vocab.
        
    Returns:
        A dictionary mapping string tokens to unique integer IDs.
    """
    vocab = {}
    
    # 1. Map special tokens to sequential IDs starting from 0
    for special in specials:
        if special not in vocab:
            vocab[special] = len(vocab)
            
    # 2. Iterate through sentences and words to map remaining unique tokens
    for sentence in sentences:
        # Split on whitespace to extract tokens
        for token in sentence.split():
            if token not in vocab:
                vocab[token] = len(vocab)
                
    return vocab

# Step 2 - build_id_to_token_vocab
def build_id_to_token_vocab(token_to_id):
    """
    Build the inverse id-to-token dictionary from token_to_id.
    
    Args:
        token_to_id: A dictionary mapping string tokens to integer IDs.
        
    Returns:
        A dictionary mapping integer IDs back to string tokens.
    """
    # Use a dictionary comprehension to swap keys (tokens) and values (ids)
    return {token_id: token for token, token_id in token_to_id.items()}

# Step 3 - encode_sentence_to_ids
def encode_sentence_to_ids(sentence, token_to_id, unk_token='<unk>'):
    """
    Convert whitespace tokens of `sentence` to ids via `token_to_id`, 
    using `unk_token`'s id for OOV (Out-Of-Vocabulary) tokens.
    
    Args:
        sentence: A string sentence to be split by whitespace.
        token_to_id: A dictionary mapping string tokens to integer IDs.
        unk_token: The string representation of the unknown token fallback.
        
    Returns:
        A list of integer token IDs.
    """
    # 1. Fetch the default fallback ID for unknown tokens.
    # If '<unk>' itself isn't in the vocab, we default to None or raise an error.
    unk_id = token_to_id.get(unk_token)
    
    # 2. Convert each token to its ID, falling back to unk_id if missing
    token_ids = [token_to_id.get(token, unk_id) for token in sentence.split()]
    
    return token_ids

# Step 4 - decode_ids_to_tokens
def decode_ids_to_tokens(ids, id_to_token):
    """
    Map each id in ids to its token string via id_to_token and return the list.
    
    Args:
        ids: A list of integer token IDs to decode.
        id_to_token: A dictionary mapping integer IDs to string tokens.
        
    Returns:
        A list of string tokens matching the length and order of the input IDs.
    """
    # Look up each ID directly in the inverse vocabulary dictionary
    return [id_to_token[token_id] for token_id in ids]

# Step 5 - pad_id_sequence
def pad_id_sequence(ids, max_len, pad_id):
    # TODO: return a list of length exactly max_len, padding with pad_id or truncating.
    truncated=ids[:max_len]
    padding_needed=max_len-len(truncated)
    return truncated+[pad_id]*padding_needed

# Step 6 - stack_padded_sequences_to_batch
import torch

def stack_padded_sequences_to_batch(padded_sequences):
    """Stack a list of equal-length padded id sequences into a 2D LongTensor batch."""
    # TODO: stack padded id sequences into a (B, L) torch.long tensor
    return torch.tensor(padded_sequences,dtype=torch.long)

# Step 7 - scale_embeddings_by_sqrt_d_model
import math
import torch

def scale_embeddings_by_sqrt_d_model(embeddings, d_model):
    """Scale a token embedding tensor by sqrt(d_model)."""
    # TODO: rescale embeddings by sqrt(d_model) as in the original Transformer paper
    return embeddings*math.sqrt(d_model)

# Step 8 - compute_positional_div_term
import torch

def compute_positional_div_term(d_model):
    # TODO: return a 1D FloatTensor of length d_model // 2 holding the sinusoidal frequency divisors
    
    even_indices=torch.arange(0, d_model, 2,dtype=torch.float32)
    div_term=torch.exp(even_indices*-(torch.log(torch.tensor(10000.0))/d_model))
    return div_term

# Step 9 - build_position_index_column
import torch

def build_position_index_column(max_len):
    """Return a (max_len, 1) float tensor of [0, 1, ..., max_len-1]."""
    # TODO: build a column vector of position indices from 0 to max_len-1
    positions=torch.arange(max_len, dtype=torch.float32)
    position_column=positions.unsqueeze(1)
    return position_column

# Step 10 - fill_even_indices_with_sin
import torch

def fill_even_indices_with_sin(pe, position, div_term):
    """Fill even feature indices of pe with sin(position * div_term)."""
    # TODO: write sin(position * div_term) into the even-indexed columns of pe and return it
    sin_values=torch.sin(position*div_term)
    pe[:,0::2]=sin_values
    return pe

# Step 11 - fill_odd_indices_with_cos
import torch

def fill_odd_indices_with_cos(pe, position, div_term):
    # TODO: fill the odd-indexed columns of pe with cos(position * div_term)
    cos_values=torch.cos(position*div_term)
    pe[:,1::2]=cos_values
    return pe

# Step 12 - build_sinusoidal_positional_encoding
import torch

def build_sinusoidal_positional_encoding(max_len, d_model):
    """Assemble the (max_len, d_model) sinusoidal positional encoding matrix."""
    # TODO: build the (max_len, d_model) sinusoidal positional encoding matrix
    pe=torch.zeros(max_len,d_model)
    position=torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)
    div_term=torch.exp(torch.arange(0, d_model, 2,dtype=torch.float)*-(math.log(10000.0)/d_model))
    pe[:,0::2]=torch.sin(position*div_term)
    pe[:,1::2]=torch.cos(position*div_term)
    return pe

# Step 13 - add_positional_encoding_to_embeddings
import torch

def add_positional_encoding_to_embeddings(embedded_batch, positional_encoding):
    # TODO: add the first L rows of positional_encoding to embedded_batch and return the sum.
    return embedded_batch+positional_encoding[:embedded_batch.shape[1],:]

# Step 14 - build_padding_mask
import torch

def build_padding_mask(token_ids, pad_id):
    """Return a (B, 1, 1, L) bool mask: True where token_ids != pad_id."""
    # TODO: build a boolean mask marking non-pad positions, shaped for broadcasting against attention scores
    mask=token_ids!=pad_id
    mask=mask.unsqueeze(1).unsqueeze(2)
    return mask

# Step 15 - build_causal_mask
import torch

def build_causal_mask(seq_len):
    """Return a (1, 1, seq_len, seq_len) bool mask, True on and below diagonal."""
    # TODO: build a lower-triangular boolean causal mask of shape (1, 1, seq_len, seq_len)
    return torch.tril(torch.ones(seq_len,seq_len,dtype=torch.bool)).unsqueeze(0).unsqueeze(0)

# Step 16 - combine_padding_and_causal_masks
import torch

def combine_padding_and_causal_masks(padding_mask, causal_mask):
    # TODO: combine a (B,1,1,L) padding mask with a (1,1,L,L) causal mask into (B,1,L,L).
    combined_mask=padding_mask&causal_mask
    return combined_mask

# Step 17 - compute_raw_attention_scores
import torch

def compute_raw_attention_scores(query, key):
    """Compute raw attention scores Q @ K^T over the last two dimensions."""
    # TODO: matmul query with the transpose of key over the last two axes
    scores=torch.matmul(query, key.transpose(-2,-1))
    return scores

# Step 18 - scale_attention_scores
import torch
import math

def scale_attention_scores(scores, d_k):
    # TODO: divide raw attention scores by sqrt(d_k) to stabilize softmax inputs
    return scores/math.sqrt(d_k)

# Step 19 - mask_attention_scores_with_neg_inf
import torch

def mask_attention_scores_with_neg_inf(scores, mask):
    """Set entries of scores where mask is False to -inf."""
    # TODO: replace blocked positions of scores with negative infinity
    return scores.masked_fill(~mask, float('-inf'))

# Step 20 - softmax_attention_weights
import torch

def softmax_attention_weights(masked_scores):
    # TODO: softmax over the last axis, zeroing rows that are entirely -inf
    attention_weights=torch.softmax(masked_scores,dim=-1)

    row_is_nan=torch.isnan(attention_weights).all(dim=-1,keepdim=True)
    attention_weights=torch.where(row_is_nan,torch.zeros_like(attention_weights),attention_weights)
    return attention_weights

# Step 21 - apply_attention_weights_to_values
import torch

def apply_attention_weights_to_values(attention_weights, value):
    """Multiply attention weights by the value matrix to produce context vectors."""
    # TODO: combine attention weights (..., Lq, Lk) with value (..., Lk, d_v)
    return attention_weights @ value

# Step 22 - scaled_dot_product_attention
import torch
import math

def scaled_dot_product_attention(query, key, value, mask=None):
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    attention_weights = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
    context = torch.matmul(attention_weights, value)
    return context, attention_weights

# Step 23 - split_last_dim_into_heads
import torch

def split_last_dim_into_heads(tensor, num_heads):
    return tensor.view(*tensor.shape[:-1], num_heads, tensor.shape[-1] // num_heads)

# Step 24 - transpose_heads_before_sequence
import torch

def transpose_heads_before_sequence(split_tensor):
    # TODO: rearrange (B, L, num_heads, d_k) into (B, num_heads, L, d_k).
    return split_tensor.transpose(1,2)

# Step 25 - merge_heads_back_to_model_dim
import torch

def merge_heads_back_to_model_dim(multi_head_tensor):
    # TODO: merge the head axis back into the feature axis to reconstruct d_model
    return multi_head_tensor.transpose(1,2).reshape(
        multi_head_tensor.size(0),
        multi_head_tensor.size(2),
        -1
    ).contiguous()

# Step 26 - apply_linear_projection
import torch
import torch.nn.functional as F
def apply_linear_projection(x, weight, bias):
    # TODO: return x @ weight^T + bias (bias may be None) with shape (..., out_features)
    return F.linear(x, weight, bias)

# Step 27 - project_to_query_key_value
def project_to_query_key_value(x, w_q, b_q, w_k, b_k, w_v, b_v):
    # TODO: project x into separate query, key, and value tensors via three linear layers
    q=apply_linear_projection(x,w_q,b_q)
    k=apply_linear_projection(x,w_k,b_k)
    v=apply_linear_projection(x,w_v,b_v)

    return q,k,v

# Step 28 - split_qkv_into_heads
import torch

def split_qkv_into_heads(q, k, v, num_heads):
    return (transpose_heads_before_sequence(split_last_dim_into_heads(q, num_heads)),
            transpose_heads_before_sequence(split_last_dim_into_heads(k, num_heads)),
            transpose_heads_before_sequence(split_last_dim_into_heads(v, num_heads)))

# Step 29 - multi_head_scaled_dot_product_attention
import torch
import math

def multi_head_scaled_dot_product_attention(q_h, k_h, v_h, mask=None):
    """
    Run scaled dot-product attention over multi-head tensors using broadcasting.
    
    Args:
        q_h: Tensor of shape (B, num_heads, Lq, d_k)
        k_h: Tensor of shape (B, num_heads, Lk, d_k)
        v_h: Tensor of shape (B, num_heads, Lk, d_v)
        mask: Optional mask broadcastable to (B, num_heads, Lq, Lk)
    
    Returns:
        Tuple (context, attention_weights) with shapes:
        - context: (B, num_heads, Lq, d_v)
        - attention_weights: (B, num_heads, Lq, Lk)
    """
    # Step 1: Compute raw attention scores
    # (B, num_heads, Lq, d_k) @ (B, num_heads, d_k, Lk) -> (B, num_heads, Lq, Lk)
    raw_scores = torch.matmul(q_h, k_h.transpose(-2, -1))
    
    # Step 2: Scale scores by 1/sqrt(d_k)
    d_k = q_h.shape[-1]
    scaled_scores = raw_scores / math.sqrt(d_k)
    
    # Step 3: Apply mask if provided
    if mask is not None:
        scaled_scores = scaled_scores.masked_fill(~mask, float('-inf'))
    
    # Step 4: Apply softmax over key dimension (last axis)
    attention_weights = torch.softmax(scaled_scores, dim=-1)
    
    # Handle fully masked rows (all -inf) -> replace NaNs with 0
    attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
    
    # Step 5: Apply attention weights to values
    # (B, num_heads, Lq, Lk) @ (B, num_heads, Lk, d_v) -> (B, num_heads, Lq, d_v)
    context = torch.matmul(attention_weights, v_h)
    
    return context, attention_weights

# Step 30 - merge_heads_and_project_output
import torch

def merge_heads_and_project_output(context, w_o, b_o):
    # TODO: merge the head axis back into d_model and apply the output linear projection.
    return apply_linear_projection(merge_heads_back_to_model_dim(context),w_o,b_o)

# Step 31 - assemble_multi_head_attention_forward
import math
import torch

def assemble_multi_head_attention_forward(query, key, value, w_q, w_k, w_v, w_o, num_heads, mask=None):
    """
    Execute a multi-head attention forward pass from scratch.
    
    Args:
        query: Tensor of shape (batch_size, seq_len_q, d_model)
        key: Tensor of shape (batch_size, seq_len_k, d_model)
        value: Tensor of shape (batch_size, seq_len_k, d_model)
        w_q: Weight tensor for query projection of shape (d_model, d_model)
        w_k: Weight tensor for key projection of shape (d_model, d_model)
        w_v: Weight tensor for value projection of shape (d_model, d_model)
        w_o: Weight tensor for output projection of shape (d_model, d_model)
        num_heads: The number of attention heads (integer)
        mask: Optional binary/boolean mask tensor broadcastable to 
              (batch_size, num_heads, seq_len_q, seq_len_k)
              
    Returns:
        Tensor of shape (batch_size, seq_len_q, d_model) after output projection.
    """
    batch_size, seq_len_q, d_model = query.shape
    _, seq_len_k, _ = key.shape
    
    # Calculate the hidden dimension size of each head (d_k)
    d_k = d_model // num_heads
    
    # 1. Project input matrices using their respective weights
    # Shape: (batch_size, seq_len, d_model)
    Q = query @ w_q
    K = key @ w_k
    V = value @ w_v
    
    # 2. Reshape and Permute to split into heads
    # Original: (batch_size, seq_len, num_heads, d_k)
    # Permuted: (batch_size, num_heads, seq_len, d_k)
    Q = Q.view(batch_size, seq_len_q, num_heads, d_k).transpose(1, 2)
    K = K.view(batch_size, seq_len_k, num_heads, d_k).transpose(1, 2)
    V = V.view(batch_size, seq_len_k, num_heads, d_k).transpose(1, 2)
    
    # 3. Scaled Dot-Product Attention
    # Scores shape: (batch_size, num_heads, seq_len_q, seq_len_k)
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Apply attention mask if provided
    if mask is not None:
        # Replace masked positions with a large negative number so they become 0 after softmax
        scores = scores.masked_fill(mask == 0, -1e9)
        
    # Apply Softmax to get attention weights
    attn_weights = torch.softmax(scores, dim=-1)
    
    # Compute contextual representation for each head
    # Context shape: (batch_size, num_heads, seq_len_q, d_k)
    context = attn_weights @ V
    
    # 4. Merge heads back together
    # Permute back to: (batch_size, seq_len_q, num_heads, d_k)
    # Contiguous is required before view because transpose creates non-contiguous memory layouts
    context = context.transpose(1, 2).contiguous()
    context = context.view(batch_size, seq_len_q, d_model)
    
    # 5. Apply final linear output projection
    output = context @ w_o
    
    return output

# Step 32 - apply_ffn_first_linear_and_relu
def apply_ffn_first_linear_and_relu(x, w1, b1):
    # TODO: project x by w1, add b1, then apply a ReLU activation.
    linear_output=x @ w1+b1
    activated_output=torch.relu(linear_output)
    return activated_output

# Step 33 - apply_ffn_second_linear
import torch

def apply_ffn_second_linear(hidden, w2, b2):
    # TODO: project hidden (..., d_ff) back to (..., d_model) via w2 and b2.
    output=hidden@ w2 + b2
    return output

# Step 34 - position_wise_feed_forward_network
import torch

def apply_ffn_first_linear_and_relu(x, w1, b1):
    return torch.relu(x @ w1 + b1)

def apply_ffn_second_linear(hidden, w2, b2):
    return hidden @ w2 + b2

def position_wise_feed_forward_network(x, w1, b1, w2, b2):
    # TODO: compose the two FFN linears with a ReLU in between, returning shape (B, T, d_model).
    inner_activation=apply_ffn_first_linear_and_relu(x,w1,b1)
    output=apply_ffn_second_linear(inner_activation,w2,b2)
    return output

# Step 35 - compute_layer_norm_mean_and_variance
import torch

def compute_layer_norm_mean_and_variance(x):
    # TODO: return (mean, variance) reduced over the last dim with shape (..., 1)
    mean=torch.mean(x,dim=-1,keepdim=True)
    variance=torch.var(x,dim=-1,keepdim=True,correction=0)
    return mean,variance

# Step 36 - normalize_and_scale_with_gamma_beta
import torch
def compute_layer_norm_mean_and_variance(x):
    # Re-using your population-level mean and variance helper
    mean = torch.mean(x, dim=-1, keepdim=True)
    variance = torch.var(x, dim=-1, keepdim=True, correction=0)
    return mean, variance

def normalize_and_scale_with_gamma_beta(x, gamma, beta, eps=1e-5):
    # TODO: standardize x along the last axis then apply gamma and beta affine transform
    mean,variance=compute_layer_norm_mean_and_variance(x)
    x_normalized=(x-mean)/torch.sqrt(variance+eps)

    output=x_normalized*gamma+beta
    return output

# Step 37 - apply_residual_add_and_norm
import torch
import torch

def compute_layer_norm_mean_and_variance(x):
    mean = torch.mean(x, dim=-1, keepdim=True)
    variance = torch.var(x, dim=-1, keepdim=True, correction=0)
    return mean, variance

def normalize_and_scale_with_gamma_beta(x, gamma, beta, eps=1e-5):
    mean, variance = compute_layer_norm_mean_and_variance(x)
    x_normalized = (x - mean) / torch.sqrt(variance + eps)
    return x_normalized * gamma + beta
def apply_residual_add_and_norm(residual_input, sublayer_output, gamma, beta, eps=1e-5):
    # TODO: combine the residual with the sublayer output and layer-normalize the result.
    combined_signal=residual_input+sublayer_output

    output=normalize_and_scale_with_gamma_beta(combined_signal,gamma,beta,eps=eps)
    return output

# Step 38 - apply_dropout_with_keep_mask
def apply_dropout_with_keep_mask(x, keep_mask, keep_prob):
    # TODO: multiply x by the boolean keep_mask and rescale by 1/keep_prob.
    dropout_multiplier=keep_mask.to(x.dtype)/keep_prob

    return x*dropout_multiplier

# Step 39 - encoder_layer_self_attention_sublayer
import math
import torch

def assemble_multi_head_attention_forward(query, key, value, w_q, w_k, w_v, w_o, num_heads, mask=None):
    # Upstream MHA implementation
    batch_size, seq_len_q, d_model = query.shape
    _, seq_len_k, _ = key.shape
    d_k = d_model // num_heads
    
    Q = query @ w_q
    K = key @ w_k
    V = value @ w_v
    
    Q = Q.view(batch_size, seq_len_q, num_heads, d_k).transpose(1, 2)
    K = K.view(batch_size, seq_len_k, num_heads, d_k).transpose(1, 2)
    V = V.view(batch_size, seq_len_k, num_heads, d_k).transpose(1, 2)
    
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        
    attn_weights = torch.softmax(scores, dim=-1)
    context = attn_weights @ V
    
    context = context.transpose(1, 2).contiguous()
    context = context.view(batch_size, seq_len_q, d_model)
    return context @ w_o

def compute_layer_norm_mean_and_variance(x):
    mean = torch.mean(x, dim=-1, keepdim=True)
    variance = torch.var(x, dim=-1, keepdim=True, correction=0)
    return mean, variance

def normalize_and_scale_with_gamma_beta(x, gamma, beta, eps=1e-5):
    mean, variance = compute_layer_norm_mean_and_variance(x)
    x_normalized = (x - mean) / torch.sqrt(variance + eps)
    return x_normalized * gamma + beta

def apply_residual_add_and_norm(residual_input, sublayer_output, gamma, beta, eps=1e-5):
    combined_signal = residual_input + sublayer_output
    return normalize_and_scale_with_gamma_beta(combined_signal, gamma, beta, eps=eps)


def encoder_layer_self_attention_sublayer(x, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    """
    Run multi-head self-attention on x and wrap with residual add-and-norm.
    
    Args:
        x: Input tensor of shape (batch_size, seq_len, d_model)
        w_q, w_k, w_v: Projection weight tensors of shape (d_model, d_model)
        w_o: Output projection weight tensor of shape (d_model, d_model)
        gamma: Layer normalization scale weights of shape (d_model,)
        beta: Layer normalization bias weights of shape (d_model,)
        num_heads: Number of attention heads (integer)
        src_mask: Mask tensor broadcastable to (batch_size, num_heads, seq_len, seq_len)
        
    Returns:
        A torch.FloatTensor of shape (batch_size, seq_len, d_model)
    """
    # 1. Self-Attention: pass 'x' as query, key, and value simultaneously
    attention_output = assemble_multi_head_attention_forward(
        query=x, 
        key=x, 
        value=x, 
        w_q=w_q, 
        w_k=w_k, 
        w_v=w_v, 
        w_o=w_o, 
        num_heads=num_heads, 
        mask=src_mask
    )
    
    # 2. Add & Norm: Composed block using the original input 'x' as the residual shortcut
    output = apply_residual_add_and_norm(
        residual_input=x, 
        sublayer_output=attention_output, 
        gamma=gamma, 
        beta=beta
    )
    
    return output

# Step 40 - encoder_layer_feed_forward_sublayer
import torch

def apply_ffn_first_linear_and_relu(x, w1, b1):
    return torch.relu(x @ w1 + b1)

def apply_ffn_second_linear(hidden, w2, b2):
    return hidden @ w2 + b2

def position_wise_feed_forward_network(x, w1, b1, w2, b2):
    inner_activations = apply_ffn_first_linear_and_relu(x, w1, b1)
    return apply_ffn_second_linear(inner_activations, w2, b2)

def compute_layer_norm_mean_and_variance(x):
    mean = torch.mean(x, dim=-1, keepdim=True)
    variance = torch.var(x, dim=-1, keepdim=True, correction=0)
    return mean, variance

def normalize_and_scale_with_gamma_beta(x, gamma, beta, eps=1e-5):
    mean, variance = compute_layer_norm_mean_and_variance(x)
    x_normalized = (x - mean) / torch.sqrt(variance + eps)
    return x_normalized * gamma + beta

def apply_residual_add_and_norm(residual_input, sublayer_output, gamma, beta, eps=1e-5):
    combined_signal = residual_input + sublayer_output
    return normalize_and_scale_with_gamma_beta(combined_signal, gamma, beta, eps=eps)


def encoder_layer_feed_forward_sublayer(x, w1, b1, w2, b2, gamma, beta):
    """
    Run the position-wise FFN on x and wrap it with a residual add-and-norm layer.
    
    Args:
        x: Input tensor from the self-attention sublayer, shape (B, T, d_model)
        w1: First FFN projection weight matrix of shape (d_model, d_ff)
        b1: First FFN projection bias vector of shape (d_ff,)
        w2: Second FFN projection weight matrix of shape (d_ff, d_model)
        b2: Second FFN projection bias vector of shape (d_model,)
        gamma: Layer normalization scale weights of shape (d_model,)
        beta: Layer normalization bias weights of shape (d_model,)
        
    Returns:
        A torch.FloatTensor of shape (B, T, d_model)
    """
    # 1. Feed-Forward Network: process representations through the two-layer linear block
    ffn_output = position_wise_feed_forward_network(x, w1, b1, w2, b2)
    
    # 2. Add & Norm: Combine the FFN input 'x' as the shortcut with the processed ffn_output
    output = apply_residual_add_and_norm(
        residual_input=x, 
        sublayer_output=ffn_output, 
        gamma=gamma, 
        beta=beta
    )
    
    return output

# Step 41 - assemble_encoder_layer
import math
import torch

# --- Upstream Primitives (Pre-defined Helpers) ---

def assemble_multi_head_attention_forward(query, key, value, w_q, w_k, w_v, w_o, num_heads, mask=None):
    batch_size, seq_len_q, d_model = query.shape
    _, seq_len_k, _ = key.shape
    d_k = d_model // num_heads
    Q = query @ w_q
    K = key @ w_k
    V = value @ w_v
    Q = Q.view(batch_size, seq_len_q, num_heads, d_k).transpose(1, 2)
    K = K.view(batch_size, seq_len_k, num_heads, d_k).transpose(1, 2)
    V = V.view(batch_size, seq_len_k, num_heads, d_k).transpose(1, 2)
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn_weights = torch.softmax(scores, dim=-1)
    context = attn_weights @ V
    context = context.transpose(1, 2).contiguous()
    return context.view(batch_size, seq_len_q, d_model) @ w_o

def compute_layer_norm_mean_and_variance(x):
    return torch.mean(x, dim=-1, keepdim=True), torch.var(x, dim=-1, keepdim=True, correction=0)

def normalize_and_scale_with_gamma_beta(x, gamma, beta, eps=1e-5):
    mean, variance = compute_layer_norm_mean_and_variance(x)
    return ((x - mean) / torch.sqrt(variance + eps)) * gamma + beta

def apply_residual_add_and_norm(residual_input, sublayer_output, gamma, beta, eps=1e-5):
    return normalize_and_scale_with_gamma_beta(residual_input + sublayer_output, gamma, beta, eps=eps)

def encoder_layer_self_attention_sublayer(x, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    attention_output = assemble_multi_head_attention_forward(x, x, x, w_q, w_k, w_v, w_o, num_heads, mask=src_mask)
    return apply_residual_add_and_norm(x, attention_output, gamma, beta)

def encoder_layer_feed_forward_sublayer(x, w1, b1, w2, b2, gamma, beta):
    ffn_output = torch.relu(x @ w1 + b1) @ w2 + b2
    return apply_residual_add_and_norm(x, ffn_output, gamma, beta)


# --- Core Layer Assembly ---

def assemble_encoder_layer(x, layer_params, num_heads, src_mask):
    """
    Assemble and execute a full Transformer Encoder Layer block.
    
    Args:
        x: Input tensor of shape (batch_size, seq_len, d_model)
        layer_params: A dictionary containing all the weight and bias tensors:
                      w_q, w_k, w_v, w_o, attn_gamma, attn_beta,
                      w1, b1, w2, b2, ffn_gamma, ffn_beta
        num_heads: Number of attention heads (integer)
        src_mask: Source padding/attention mask tensor
        
    Returns:
        A torch.FloatTensor of shape (batch_size, seq_len, d_model)
    """
    # 1. Route through Sublayer 1: Multi-Head Self-Attention + Add & Norm
    attn_sublayer_output = encoder_layer_self_attention_sublayer(
        x=x,
        w_q=layer_params['w_q'],
        w_k=layer_params['w_k'],
        w_v=layer_params['w_v'],
        w_o=layer_params['w_o'],
        gamma=layer_params['attn_gamma'],
        beta=layer_params['attn_beta'],
        num_heads=num_heads,
        src_mask=src_mask
    )
    
    # 2. Route through Sublayer 2: Position-Wise Feed-Forward Network + Add & Norm
    final_output = encoder_layer_feed_forward_sublayer(
        x=attn_sublayer_output,
        w1=layer_params['w1'],
        b1=layer_params['b1'],
        w2=layer_params['w2'],
        b2=layer_params['b2'],
        gamma=layer_params['ffn_gamma'],
        beta=layer_params['ffn_beta']
    )
    
    return final_output

# Step 42 - stack_encoder_layers
import torch

def stack_encoder_layers(x, encoder_layer_params_list, num_heads, src_mask):
    """
    Stack encoder layers sequentially.
    """
    hidden_state = x
    
    for layer_params in encoder_layer_params_list:
        # The layer_params is a dictionary with all the parameters
        hidden_state = assemble_encoder_layer(hidden_state, layer_params, num_heads, src_mask)
    
    return hidden_state

# Step 43 - decoder_layer_masked_self_attention_sublayer
import torch

def decoder_layer_masked_self_attention_sublayer(y, w_q, w_k, w_v, w_o, gamma, beta, num_heads, tgt_mask):
    """
    Masked self-attention sublayer for Transformer decoder.
    """
    # Self-attention with mask
    attn_output = assemble_multi_head_attention_forward(
        y, y, y, w_q, w_k, w_v, w_o, num_heads, tgt_mask
    )
    
    # Residual connection + layer norm
    return layer_norm(y + attn_output, gamma, beta)


def apply_linear_projection(x, weight, bias):
    """Linear projection: x @ weight^T + bias"""
    out = torch.matmul(x, weight.T)
    if bias is not None:
        out = out + bias
    return out


def layer_norm(x, gamma, beta, eps=1e-5):
    """Layer normalization across last dimension"""
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    return x_norm * gamma + beta


def assemble_multi_head_attention_forward(query, key, value, w_q, w_k, w_v, w_o, num_heads, mask=None):
    """Multi-head attention forward pass."""
    d_model = query.shape[-1]
    d_k = d_model // num_heads
    
    # Project Q, K, V
    q = apply_linear_projection(query, w_q, None)
    k = apply_linear_projection(key, w_k, None)
    v = apply_linear_projection(value, w_v, None)
    
    # Split into heads
    B, Lq = q.shape[0], q.shape[1]
    Lk = k.shape[1]
    
    q = q.view(B, Lq, num_heads, d_k).transpose(1, 2)
    k = k.view(B, Lk, num_heads, d_k).transpose(1, 2)
    v = v.view(B, Lk, num_heads, d_k).transpose(1, 2)
    
    # Scaled dot-product attention
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
    
    if mask is not None:
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)  # Add head dimension
        scores = scores.masked_fill(~mask, float('-inf'))
    
    weights = torch.softmax(scores, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)
    
    context = torch.matmul(weights, v)
    
    # Merge heads
    context = context.transpose(1, 2).contiguous().view(B, Lq, d_model)
    
    # Output projection
    output = apply_linear_projection(context, w_o, None)
    
    return output

# Step 44 - decoder_layer_cross_attention_sublayer
import torch

def decoder_layer_cross_attention_sublayer(y, encoder_output, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    """
    Cross-attention sublayer for Transformer decoder.
    """
    # Multi-head cross-attention
    cross_attn_output = assemble_multi_head_attention_forward(
        query=y,
        key=encoder_output,
        value=encoder_output,
        w_q=w_q,
        w_k=w_k,
        w_v=w_v,
        w_o=w_o,
        num_heads=num_heads,
        mask=src_mask
    )
    
    # Residual connection (add) + layer norm
    y = y + cross_attn_output
    y = layer_norm(y, gamma, beta)
    
    return y


def layer_norm(x, gamma, beta, eps=1e-5):
    """Layer normalization across last dimension"""
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    return x_norm * gamma + beta


def apply_linear_projection(x, weight, bias):
    """Linear projection: x @ weight^T + bias"""
    out = torch.matmul(x, weight.T)
    if bias is not None:
        out = out + bias
    return out


def assemble_multi_head_attention_forward(query, key, value, w_q, w_k, w_v, w_o, num_heads, mask=None):
    """Multi-head attention forward pass."""
    d_model = query.shape[-1]
    d_k = d_model // num_heads
    
    # Project Q, K, V
    q = apply_linear_projection(query, w_q, None)
    k = apply_linear_projection(key, w_k, None)
    v = apply_linear_projection(value, w_v, None)
    
    # Split into heads
    B, Lq = q.shape[0], q.shape[1]
    Lk = k.shape[1]
    
    q = q.view(B, Lq, num_heads, d_k).transpose(1, 2)
    k = k.view(B, Lk, num_heads, d_k).transpose(1, 2)
    v = v.view(B, Lk, num_heads, d_k).transpose(1, 2)
    
    # Scaled dot-product attention
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
    
    if mask is not None:
        # Handle mask shape - should be broadcastable to (B, num_heads, Lq, Lk)
        if mask.dim() == 2:  # (B, Lk) - padding mask
            mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, Lk)
        elif mask.dim() == 3:  # (B, Lq, Lk) or (1, Lq, Lk)
            if mask.shape[0] == 1 and mask.shape[1] == Lq and mask.shape[2] == Lk:
                mask = mask.unsqueeze(1)  # (1, 1, Lq, Lk)
            else:
                mask = mask.unsqueeze(1)  # (B, 1, Lq, Lk)
        scores = scores.masked_fill(~mask, float('-inf'))
    
    weights = torch.softmax(scores, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)
    
    context = torch.matmul(weights, v)
    
    # Merge heads
    context = context.transpose(1, 2).contiguous().view(B, Lq, d_model)
    
    # Output projection
    output = apply_linear_projection(context, w_o, None)
    
    return output

# Step 45 - decoder_layer_feed_forward_sublayer
import torch

def decoder_layer_feed_forward_sublayer(y, w1, b1, w2, b2, gamma, beta):
    """
    Run the position-wise feed-forward network on y and wrap it with residual add-and-norm.
    
    Args:
        y: Decoder hidden state of shape (B, tgt_seq, d_model)
        w1: First linear layer weight of shape (d_ff, d_model)
        b1: First linear layer bias of shape (d_ff,)
        w2: Second linear layer weight of shape (d_model, d_ff)
        b2: Second linear layer bias of shape (d_model,)
        gamma: Layer normalization scale parameter of shape (d_model,)
        beta: Layer normalization shift parameter of shape (d_model,)
    
    Returns:
        Output of shape (B, tgt_seq, d_model) after FFN + residual + layer norm
    """
    # Step 1: Apply position-wise feed-forward network
    ff_output = position_wise_feed_forward_network(y, w1, b1, w2, b2)
    
    # Step 2: Residual connection (add) + layer normalization
    output = apply_residual_add_and_norm(y, ff_output, gamma, beta)
    
    return output

# Step 46 - assemble_decoder_layer
# ── Step 045  assemble_decoder_layer ──
def assemble_decoder_layer(y, encoder_output, layer_params, num_heads, src_mask, tgt_mask):
    """
    Assemble and execute a full Transformer Decoder Layer block.
    
    Args:
        y: Target hidden state tensor of shape (batch_size, tgt_seq, d_model)
        encoder_output: Encoder output tensor of shape (batch_size, src_seq, d_model)
        layer_params: A dictionary containing all the weight, bias, and normalization tensors
        num_heads: Number of attention heads (integer)
        src_mask: Source attention/padding mask tensor for cross-attention
        tgt_mask: Target causal/padding mask tensor for self-attention
        
    Returns:
        A torch.FloatTensor of shape (batch_size, tgt_seq, d_model)
    """
    # Helper to safely look up keys without triggering boolean evaluation on tensors
    def get_param(keys, default_key=None):
        for key in keys:
            if key in layer_params:
                return layer_params[key]
        return layer_params.get(default_key) if default_key else None

    # ── 1. Sublayer 1: Masked Multi-Head Self-Attention ──
    w_q_self = get_param(['self_w_q', 'w_q_self'], 'w_q')
    w_k_self = get_param(['self_w_k', 'w_k_self'], 'w_k')
    w_v_self = get_param(['self_w_v', 'w_v_self'], 'w_v')
    w_o_self = get_param(['self_w_o', 'w_o_self'], 'w_o')
    
    gamma_self = get_param(['self_attn_gamma', 'self_gamma', 'attn_gamma'], 'gamma')
    beta_self = get_param(['self_attn_beta', 'self_beta', 'attn_beta'], 'beta')

    y = decoder_layer_masked_self_attention_sublayer(
        y=y,
        w_q=w_q_self,
        w_k=w_k_self,
        w_v=w_v_self,
        w_o=w_o_self,
        gamma=gamma_self,
        beta=beta_self,
        num_heads=num_heads,
        tgt_mask=tgt_mask
    )
    
    # ── 2. Sublayer 2: Multi-Head Cross-Attention ──
    w_q_cross = get_param(['cross_w_q', 'w_q_cross'], 'w_q')
    w_k_cross = get_param(['cross_w_k', 'w_k_cross'], 'w_k')
    w_v_cross = get_param(['cross_w_v', 'w_v_cross'], 'w_v')
    w_o_cross = get_param(['cross_w_o', 'w_o_cross'], 'w_o')
    
    gamma_cross = get_param(['cross_attn_gamma', 'cross_gamma', 'attn_gamma'], 'gamma')
    beta_cross = get_param(['cross_attn_beta', 'cross_beta', 'attn_beta'], 'beta')

    y = decoder_layer_cross_attention_sublayer(
        y=y,
        encoder_output=encoder_output,
        w_q=w_q_cross,
        w_k=w_k_cross,
        w_v=w_v_cross,
        w_o=w_o_cross,
        gamma=gamma_cross,
        beta=beta_cross,
        num_heads=num_heads,
        src_mask=src_mask
    )
    
    # ── 3. Sublayer 3: Position-Wise Feed-Forward Network ──
    w1 = layer_params.get('w1')
    b1 = layer_params.get('b1')
    w2 = layer_params.get('w2')
    b2 = layer_params.get('b2')
    gamma_ffn = get_param(['ffn_gamma'], 'gamma')
    beta_ffn = get_param(['ffn_beta'], 'beta')

    y = encoder_layer_feed_forward_sublayer(
        x=y,
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        gamma=gamma_ffn,
        beta=beta_ffn
    )
    
    return y

# Step 47 - stack_decoder_layers
# ── Step 046  stack_decoder_layers ──
import torch

def stack_decoder_layers(y, encoder_output, decoder_layer_params_list, num_heads, src_mask=None, tgt_mask=None):
    """
    Build the full Transformer decoder by sequentially applying each decoder layer 
    to the running target-side hidden state using positional arguments.
    
    Args:
        y: Target-side input/hidden state tensor of shape (B, T_tgt, d_model)
        encoder_output: Encoder output tensor of shape (B, T_src, d_model)
        decoder_layer_params_list: List of parameter dictionaries, one per decoder layer.
        num_heads: Number of attention heads (integer)
        src_mask: Source padding/attention mask tensor
        tgt_mask: Target causal/padding mask tensor
        
    Returns:
        A torch.FloatTensor of shape (B, T_tgt, d_model) after processing through all layers.
    """
    hidden_state = y
    
    for layer_params in decoder_layer_params_list:
        # Pass parameters positionally to avoid parameter naming conflicts
        hidden_state = assemble_decoder_layer(
            hidden_state, 
            encoder_output, 
            layer_params, 
            num_heads, 
            src_mask, 
            tgt_mask
        )
        
    return hidden_state

# Step 48 - apply_final_output_projection
def apply_final_output_projection(decoder_output, output_projection_weight, output_projection_bias=None):
    # TODO: project decoder hidden states (B, T, D) to vocabulary logits (B, T, V).
    return apply_linear_projection(decoder_output,output_projection_weight,output_projection_bias)

# Step 49 - tie_output_projection_to_token_embeddings
import torch

def tie_output_projection_to_token_embeddings(token_embedding_weight):
    """Return an output projection weight that shares storage with token_embedding_weight.

    Input shape: (vocab_size, d_model). Output shape: (d_model, vocab_size).
    """
    # TODO: return an output projection weight tied to the token embedding matrix
    return token_embedding_weight.T

# Step 50 - apply_log_softmax_over_vocab
def apply_log_softmax_over_vocab(logits):
    # TODO: Convert decoder logits (B, T, V) into log probabilities over the vocabulary axis.
    return F.log_softmax(logits,dim=-1)

# Step 51 - run_transformer_forward
def run_transformer_forward(src_ids, tgt_ids, model_params, num_heads, pad_id):
    """
    Run the full encoder-decoder Transformer forward pass.
    
    Args:
        src_ids: Source token IDs, shape (batch, src_seq)
        tgt_ids: Target token IDs, shape (batch, tgt_seq)
        model_params: Dict with keys 'token_embedding', 'encoder_layers', 'decoder_layers', 'output_projection'
        num_heads: Number of attention heads
        pad_id: Padding token ID
    
    Returns:
        Log probabilities over vocabulary, shape (batch, tgt_seq, vocab_size)
    """
    # Extract model parameters
    token_embedding = model_params['token_embedding']
    encoder_layers = model_params['encoder_layers']
    decoder_layers = model_params['decoder_layers']
    output_projection = model_params['output_projection']
    d_model = token_embedding.shape[1]
    
    # ---- 1. Embed source and target ----
    # Token embeddings: (batch, seq_len, d_model)
    src_emb = scale_embeddings_by_sqrt_d_model(token_embedding[src_ids], d_model)
    tgt_emb = scale_embeddings_by_sqrt_d_model(token_embedding[tgt_ids], d_model)
    
    # ---- 2. Add sinusoidal positional encoding ----
    src_len = src_ids.shape[1]
    tgt_len = tgt_ids.shape[1]
    pos_enc = build_sinusoidal_positional_encoding(max(src_len, tgt_len), d_model)
    
    src_emb = add_positional_encoding_to_embeddings(src_emb, pos_enc)
    tgt_emb = add_positional_encoding_to_embeddings(tgt_emb, pos_enc)
    
    # ---- 3. Build masks ----
    # Source padding mask: (batch, 1, 1, src_len)
    src_padding_mask = build_padding_mask(src_ids, pad_id)
    
    # Target padding mask: (batch, 1, 1, tgt_len)
    tgt_padding_mask = build_padding_mask(tgt_ids, pad_id)
    
    # Target causal mask: (1, 1, tgt_len, tgt_len)
    causal_mask = build_causal_mask(tgt_len)
    
    # Combined target mask: (batch, 1, tgt_len, tgt_len)
    tgt_mask = combine_padding_and_causal_masks(tgt_padding_mask, causal_mask)
    
    # ---- 4. Run encoder stack ----
    # Encoder output: (batch, src_len, d_model)
    encoder_output = stack_encoder_layers(src_emb, encoder_layers, num_heads, src_padding_mask)
    
    # ---- 5. Run decoder stack ----
    # Decoder output: (batch, tgt_len, d_model)
    decoder_output = stack_decoder_layers(
        tgt_emb, encoder_output, decoder_layers, num_heads, src_padding_mask, tgt_mask
    )
    
    # ---- 6. Project to vocabulary logits ----
    # Use output_projection as weight (vocab_size, d_model) -> need to transpose
    logits = apply_final_output_projection(decoder_output, output_projection)
    
    # ---- 7. Apply log-softmax ----
    log_probs = apply_log_softmax_over_vocab(logits)
    
    return log_probs

# Step 52 - init_encoder_layer_parameters
import torch
import math

def init_encoder_layer_parameters(d_model, num_heads, d_ff):
    """Return a dict of leaf tensors with requires_grad=True for one encoder layer."""
    # Attention projection weights (no biases) - use randn directly with scaling
    w_q = torch.randn(d_model, d_model, dtype=torch.float32, requires_grad=True) * 0.02
    # This creates a non-leaf tensor. Instead, we should use torch.nn.init or create with uniform distribution.
    
    # Better approach: use torch.empty() and then fill with Xavier/Glorot initialization
    # Or use torch.nn.init directly
    
    # Let's use the approach that creates leaf tensors:
    w_q = torch.empty(d_model, d_model, dtype=torch.float32)
    torch.nn.init.xavier_uniform_(w_q)
    w_q.requires_grad_(True)
    
    w_k = torch.empty(d_model, d_model, dtype=torch.float32)
    torch.nn.init.xavier_uniform_(w_k)
    w_k.requires_grad_(True)
    
    w_v = torch.empty(d_model, d_model, dtype=torch.float32)
    torch.nn.init.xavier_uniform_(w_v)
    w_v.requires_grad_(True)
    
    w_o = torch.empty(d_model, d_model, dtype=torch.float32)
    torch.nn.init.xavier_uniform_(w_o)
    w_o.requires_grad_(True)
    
    # Feed-forward weights and biases
    w1 = torch.empty(d_model, d_ff, dtype=torch.float32)
    torch.nn.init.xavier_uniform_(w1)
    w1.requires_grad_(True)
    
    b1 = torch.zeros(d_ff, dtype=torch.float32, requires_grad=True)
    
    w2 = torch.empty(d_ff, d_model, dtype=torch.float32)
    torch.nn.init.xavier_uniform_(w2)
    w2.requires_grad_(True)
    
    b2 = torch.zeros(d_model, dtype=torch.float32, requires_grad=True)
    
    # Layer normalization parameters
    attn_gamma = torch.ones(d_model, dtype=torch.float32, requires_grad=True)
    attn_beta = torch.zeros(d_model, dtype=torch.float32, requires_grad=True)
    ffn_gamma = torch.ones(d_model, dtype=torch.float32, requires_grad=True)
    ffn_beta = torch.zeros(d_model, dtype=torch.float32, requires_grad=True)
    
    return {
        'w_q': w_q,
        'w_k': w_k,
        'w_v': w_v,
        'w_o': w_o,
        'w1': w1,
        'b1': b1,
        'w2': w2,
        'b2': b2,
        'attn_gamma': attn_gamma,
        'attn_beta': attn_beta,
        'ffn_gamma': ffn_gamma,
        'ffn_beta': ffn_beta
    }

# Step 53 - init_decoder_layer_parameters
import torch

def init_decoder_layer_parameters(d_model, num_heads, d_ff):
    """Return a dict of leaf tensors with requires_grad=True for one decoder layer."""
    # ---- Masked Self-Attention projections (no biases) ----
    w_q_self = torch.empty(d_model, d_model, dtype=torch.float32)
    torch.nn.init.xavier_uniform_(w_q_self)
    w_q_self.requires_grad_(True)
    
    w_k_self = torch.empty(d_model, d_model, dtype=torch.float32)
    torch.nn.init.xavier_uniform_(w_k_self)
    w_k_self.requires_grad_(True)
    
    w_v_self = torch.empty(d_model, d_model, dtype=torch.float32)
    torch.nn.init.xavier_uniform_(w_v_self)
    w_v_self.requires_grad_(True)
    
    w_o_self = torch.empty(d_model, d_model, dtype=torch.float32)
    torch.nn.init.xavier_uniform_(w_o_self)
    w_o_self.requires_grad_(True)
    
    # ---- Cross-Attention projections (no biases) ----
    w_q_cross = torch.empty(d_model, d_model, dtype=torch.float32)
    torch.nn.init.xavier_uniform_(w_q_cross)
    w_q_cross.requires_grad_(True)
    
    w_k_cross = torch.empty(d_model, d_model, dtype=torch.float32)
    torch.nn.init.xavier_uniform_(w_k_cross)
    w_k_cross.requires_grad_(True)
    
    w_v_cross = torch.empty(d_model, d_model, dtype=torch.float32)
    torch.nn.init.xavier_uniform_(w_v_cross)
    w_v_cross.requires_grad_(True)
    
    w_o_cross = torch.empty(d_model, d_model, dtype=torch.float32)
    torch.nn.init.xavier_uniform_(w_o_cross)
    w_o_cross.requires_grad_(True)
    
    # ---- Feed-Forward Network weights and biases ----
    w1 = torch.empty(d_model, d_ff, dtype=torch.float32)
    torch.nn.init.xavier_uniform_(w1)
    w1.requires_grad_(True)
    
    b1 = torch.zeros(d_ff, dtype=torch.float32, requires_grad=True)
    
    w2 = torch.empty(d_ff, d_model, dtype=torch.float32)
    torch.nn.init.xavier_uniform_(w2)
    w2.requires_grad_(True)
    
    b2 = torch.zeros(d_model, dtype=torch.float32, requires_grad=True)
    
    # ---- Layer Normalization parameters ----
    self_gamma = torch.ones(d_model, dtype=torch.float32, requires_grad=True)
    self_beta = torch.zeros(d_model, dtype=torch.float32, requires_grad=True)
    
    cross_gamma = torch.ones(d_model, dtype=torch.float32, requires_grad=True)
    cross_beta = torch.zeros(d_model, dtype=torch.float32, requires_grad=True)
    
    ffn_gamma = torch.ones(d_model, dtype=torch.float32, requires_grad=True)
    ffn_beta = torch.zeros(d_model, dtype=torch.float32, requires_grad=True)
    
    return {
        'w_q_self': w_q_self,
        'w_k_self': w_k_self,
        'w_v_self': w_v_self,
        'w_o_self': w_o_self,
        'w_q_cross': w_q_cross,
        'w_k_cross': w_k_cross,
        'w_v_cross': w_v_cross,
        'w_o_cross': w_o_cross,
        'w1': w1,
        'b1': b1,
        'w2': w2,
        'b2': b2,
        'self_gamma': self_gamma,
        'self_beta': self_beta,
        'cross_gamma': cross_gamma,
        'cross_beta': cross_beta,
        'ffn_gamma': ffn_gamma,
        'ffn_beta': ffn_beta
    }

# Step 54 - init_embedding_and_projection_parameters
import torch

def init_embedding_and_projection_parameters(vocab_size, d_model, tie_weights=True):
    """Allocate src/tgt embeddings and output projection (optionally tied)."""
    # Source embedding
    src_embedding = torch.empty(vocab_size, d_model, dtype=torch.float32)
    torch.nn.init.xavier_uniform_(src_embedding)
    src_embedding.requires_grad_(True)
    
    # Target embedding
    tgt_embedding = torch.empty(vocab_size, d_model, dtype=torch.float32)
    torch.nn.init.xavier_uniform_(tgt_embedding)
    tgt_embedding.requires_grad_(True)
    
    # Output projection
    if tie_weights:
        # When tie_weights=True, output_projection is the SAME tensor as tgt_embedding
        output_projection = tgt_embedding
    else:
        # When tie_weights=False, allocate an independent tensor
        output_projection = torch.empty(vocab_size, d_model, dtype=torch.float32)
        torch.nn.init.xavier_uniform_(output_projection)
        output_projection.requires_grad_(True)
    
    return {
        'src_embedding': src_embedding,
        'tgt_embedding': tgt_embedding,
        'output_projection': output_projection
    }

# Step 55 - collect_model_parameters_into_list
import torch

def collect_model_parameters_into_list(encoder_layer_params, decoder_layer_params, embedding_params):
    """Collect all unique trainable parameters into a flat list, preserving order."""
    param_list = []
    seen = set()
    
    # 1. Collect encoder layer parameters in order
    for layer_dict in encoder_layer_params:
        for tensor in layer_dict.values():
            if tensor not in seen:
                seen.add(tensor)
                param_list.append(tensor)
    
    # 2. Collect decoder layer parameters in order
    for layer_dict in decoder_layer_params:
        for tensor in layer_dict.values():
            if tensor not in seen:
                seen.add(tensor)
                param_list.append(tensor)
    
    # 3. Collect embedding/projection parameters
    for key in ['src_embedding', 'tgt_embedding', 'output_projection']:
        tensor = embedding_params[key]
        if tensor not in seen:
            seen.add(tensor)
            param_list.append(tensor)
    
    return param_list

# Step 56 - shift_targets_right_with_start_token
import torch

def shift_targets_right_with_start_token(target_ids, start_token_id):
    """
    Prepare teacher-forcing decoder input by shifting target ids right with start token.
    
    Args:
        target_ids: Tensor of shape (batch, tgt_seq) with gold target token IDs
        start_token_id: Integer ID of the start token
    
    Returns:
        Tensor of same shape as target_ids with first column = start_token_id
        and remaining columns = target_ids shifted right by one position.
    """
    batch_size, seq_len = target_ids.shape
    
    # Create tensor filled with start_token_id
    shifted = torch.full_like(target_ids, start_token_id)
    
    # Copy target_ids[:, :-1] to shifted[:, 1:]
    shifted[:, 1:] = target_ids[:, :-1]
    
    return shifted

# Step 57 - compute_noam_learning_rate
import math

def compute_noam_learning_rate(step, d_model, warmup_steps):
    """
    Compute the Noam learning rate from the original Transformer paper.
    
    Formula: lr = (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)
    
    Args:
        step: Current training step (integer, 1-indexed)
        d_model: Model dimension
        warmup_steps: Number of warmup steps
    
    Returns:
        float: Learning rate for the given step
    """
    # Convert step to float for calculations
    step = float(step)
    
    # Compute the two components
    # 1. d_model ** -0.5
    d_model_factor = d_model ** -0.5
    
    # 2. min(step ** -0.5, step * warmup_steps ** -1.5)
    step_inv_sqrt = step ** -0.5
    step_linear = step * (warmup_steps ** -1.5)
    min_factor = min(step_inv_sqrt, step_linear)
    
    # Combined learning rate
    lr = d_model_factor * min_factor
    
    return float(lr)

# Step 58 - build_uniform_smoothing_distribution
import torch

def build_uniform_smoothing_distribution(shape, vocab_size, epsilon):
    """
    Create the uniform smoothing tensor for label-smoothed cross entropy.
    
    Args:
        shape: Tuple like (batch, tgt_seq, vocab_size)
        vocab_size: Total vocabulary size
        epsilon: Smoothing parameter
    
    Returns:
        torch.Tensor of shape `shape` filled with epsilon / (vocab_size - 2)
    """
    value = epsilon / (vocab_size - 2)
    return torch.full(shape, value, dtype=torch.float32)

# Step 59 - set_confidence_on_gold_tokens
import torch

def set_confidence_on_gold_tokens(smoothed_distribution, gold_token_ids, confidence):
    """Place confidence mass at gold-token positions of a smoothed target distribution."""
    # Create a copy of the smoothed distribution to avoid modifying the input
    result = smoothed_distribution.clone()
    
    # Get the indices for the gold token positions
    batch_size, seq_len, vocab_size = result.shape
    
    # Create batch and sequence indices
    batch_indices = torch.arange(batch_size, device=result.device).unsqueeze(1)
    seq_indices = torch.arange(seq_len, device=result.device).unsqueeze(0)
    
    # Expand to match gold_token_ids shape
    batch_indices = batch_indices.expand(batch_size, seq_len)
    seq_indices = seq_indices.expand(batch_size, seq_len)
    
    # Use advanced indexing to set the confidence values
    result[batch_indices, seq_indices, gold_token_ids] = confidence
    
    return result

# Step 60 - zero_pad_column_and_pad_token_rows
import torch

def zero_pad_column_and_pad_token_rows(smoothed_distribution, gold_token_ids, pad_id):
    """Zero the pad column and rows where gold token equals pad_id."""
    # Create a copy to avoid modifying the input
    result = smoothed_distribution.clone()
    
    # 1. Zero out the pad column (the entire vocabulary index for pad_id)
    result[..., pad_id] = 0.0
    
    # 2. Zero out rows where the gold token equals pad_id
    # Create a mask of shape (batch, tgt_seq) where True means gold token is pad_id
    pad_mask = (gold_token_ids == pad_id)
    
    # Expand the mask to match the distribution shape (batch, tgt_seq, vocab_size)
    # Use unsqueeze(-1) to add the vocabulary dimension
    pad_mask_expanded = pad_mask.unsqueeze(-1)
    
    # Zero out those rows
    result = result.masked_fill(pad_mask_expanded, 0.0)
    
    return result

# Step 61 - compute_label_smoothed_kl_loss
import torch

def compute_label_smoothed_kl_loss(log_probabilities, smoothed_distribution):
    """Return the summed KL loss over all (batch, time, vocab) entries."""
    # Cross-entropy loss: -sum(P * log(Q))
    loss = -torch.sum(smoothed_distribution * log_probabilities)
    
    # Handle potential -0.0 by clamping very small negative values to 0
    # If loss is extremely close to 0 (within floating point precision), set to 0
    if torch.abs(loss) < 1e-10:
        loss = torch.tensor(0.0, device=loss.device)
    
    return loss

# Step 62 - average_loss_over_non_pad_tokens
import torch

def average_loss_over_non_pad_tokens(total_loss, gold_token_ids, pad_id):
    """Divide total_loss by the number of non-pad tokens."""
    # Count non-pad tokens
    non_pad_mask = (gold_token_ids != pad_id)
    num_non_pad = non_pad_mask.sum().float()
    
    # If every position is padding, return the total_loss (no division)
    if num_non_pad == 0:
        return total_loss
    
    # Average the loss over non-pad tokens
    avg_loss = total_loss / num_non_pad
    
    return avg_loss

# Step 63 - compute_token_accuracy_ignoring_pad
import torch

def compute_token_accuracy_ignoring_pad(log_probabilities, gold_token_ids, pad_id):
    """Compute accuracy over non-pad positions only."""
    # Get the predicted token IDs (argmax over vocabulary axis)
    pred_token_ids = torch.argmax(log_probabilities, dim=-1)  # shape: (batch, tgt_seq)
    
    # Create mask for non-pad positions
    non_pad_mask = (gold_token_ids != pad_id)
    
    # Check where predictions match gold (ignoring pad positions)
    correct = (pred_token_ids == gold_token_ids) & non_pad_mask
    
    # Count correct and total non-pad positions
    num_correct = correct.sum().float()
    num_non_pad = non_pad_mask.sum().float()
    
    # If every gold token is pad, return 0.0
    if num_non_pad == 0:
        return torch.tensor(0.0, device=log_probabilities.device)
    
    # Compute accuracy
    accuracy = num_correct / num_non_pad
    
    return accuracy

# Step 64 - initialize_adam_optimizer_state
import torch

def initialize_adam_optimizer_state(parameter_list):
    """Allocate Adam m, v zero buffers and a step counter t=0."""
    m = []
    v = []
    
    for param in parameter_list:
        # Create zero buffers with the same shape and dtype as the parameter
        # Use torch.zeros_like to match shape and dtype
        # Set requires_grad=False so buffers don't track gradients
        m.append(torch.zeros_like(param, requires_grad=False))
        v.append(torch.zeros_like(param, requires_grad=False))
    
    # Step counter starts at 0
    t = 0
    
    return {
        'm': m,
        'v': v,
        't': t
    }

# Step 65 - update_adam_first_moment
import torch

def update_adam_first_moment(m_prev, grad, beta1):
    """Return m_t = beta1 * m_prev + (1 - beta1) * grad."""
    # Apply the Adam first-moment EMA update
    m_t = beta1 * m_prev + (1 - beta1) * grad
    
    # Detach to ensure the returned tensor doesn't track gradients
    return m_t.detach()

# Step 66 - update_adam_second_moment
import torch

def update_adam_second_moment(v_prev, grad, beta2):
    """Return v_t = beta2 * v_prev + (1 - beta2) * grad ** 2."""
    # Apply Adam's EMA update for the second moment
    v_t = beta2 * v_prev + (1 - beta2) * (grad ** 2)
    
    # Detach to ensure the returned tensor doesn't track gradients
    return v_t.detach()

# Step 67 - apply_adam_bias_correction
import torch

def apply_adam_bias_correction(m_t, v_t, beta1, beta2, step):
    """Return bias-corrected (m_hat, v_hat) for Adam at the given step."""
    # Bias correction for first moment: divide by (1 - beta1**step)
    m_hat = m_t / (1 - beta1 ** step)
    
    # Bias correction for second moment: divide by (1 - beta2**step)
    v_hat = v_t / (1 - beta2 ** step)
    
    return m_hat, v_hat

# Step 69 - apply_adam_step_to_all_parameters
import torch

def apply_adam_step_to_all_parameters(parameter_list, optimizer_state, learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-9):
    """Run one full Adam update over every parameter in the list."""
    # Increment the step counter
    optimizer_state['t'] += 1
    step = optimizer_state['t']
    
    # Get the moment buffers
    m_list = optimizer_state['m']
    v_list = optimizer_state['v']
    
    # Iterate over parameters and their corresponding buffers
    for idx, param in enumerate(parameter_list):
        # Skip if gradient is None
        if param.grad is None:
            continue
        
        grad = param.grad
        
        # 1. Update first moment
        m_t = beta1 * m_list[idx] + (1 - beta1) * grad
        
        # 2. Update second moment
        v_t = beta2 * v_list[idx] + (1 - beta2) * (grad ** 2)
        
        # 3. Store updated buffers back
        m_list[idx] = m_t
        v_list[idx] = v_t
        
        # 4. Apply bias correction
        m_hat = m_t / (1 - beta1 ** step)
        v_hat = v_t / (1 - beta2 ** step)
        
        # 5. Compute parameter update and apply in place
        delta = learning_rate * m_hat / (torch.sqrt(v_hat) + epsilon)
        param.data -= delta
    
    return optimizer_state

# Step 70 - zero_all_parameter_gradients
import torch

def zero_all_parameter_gradients(parameter_list):
    """Clear the .grad of every parameter tensor before the next backward pass."""
    for param in parameter_list:
        if param.grad is not None:
            # Set grad to None to clear it (preferred method in PyTorch)
            param.grad = None

# Step 71 - compute_batch_training_loss
import torch

def compute_batch_training_loss(src_batch, tgt_batch, model_params, config):
    """Perform one teacher-forced forward pass and return the averaged KL loss."""
    # Extract config values
    pad_id = config['pad_id']
    start_id = config['start_id']
    vocab_size = config['vocab_size']
    smoothing = config['smoothing']
    num_heads = config['num_heads']
    d_model = config['d_model']
    
    # Ensure model_params has token_embedding key for run_transformer_forward
    # The token embedding is src_embedding from the embedding params
    if 'token_embedding' not in model_params:
        model_params['token_embedding'] = model_params['src_embedding']
    
    # 1. Shift target sequence right with start token for teacher forcing
    decoder_input = shift_targets_right_with_start_token(tgt_batch, start_id)
    
    # 2. Run the forward pass
    # log_probabilities has shape (batch, tgt_seq, vocab_size)
    log_probabilities = run_transformer_forward(src_batch, decoder_input, model_params, num_heads, pad_id)
    
    # 3. Build smoothed target distribution
    # Get the target sequence length
    batch_size, tgt_seq = tgt_batch.shape
    
    # Build uniform smoothing distribution
    shape = (batch_size, tgt_seq, vocab_size)
    smoothed_dist = build_uniform_smoothing_distribution(shape, vocab_size, smoothing)
    
    # Place confidence on gold tokens
    confidence = 1.0 - smoothing
    smoothed_dist = set_confidence_on_gold_tokens(smoothed_dist, tgt_batch, confidence)
    
    # Zero out pad column and pad token rows
    smoothed_dist = zero_pad_column_and_pad_token_rows(smoothed_dist, tgt_batch, pad_id)
    
    # 4. Compute KL loss (summed over all entries)
    total_loss = compute_label_smoothed_kl_loss(log_probabilities, smoothed_dist)
    
    # 5. Average loss over non-pad tokens
    avg_loss = average_loss_over_non_pad_tokens(total_loss, tgt_batch, pad_id)
    
    return avg_loss

# Step 72 - run_training_step_with_backprop
import torch

def run_training_step_with_backprop(src_batch, tgt_batch, parameter_list, model_params, optimizer_state, step_number, config):
    """Run one training iteration: zero grads, forward, backward, Noam LR, Adam step.

    Returns the scalar loss value for the step as a Python float.
    """
    # 1. Zero stale gradients before the forward pass
    zero_all_parameter_gradients(parameter_list)
    
    # 2. Forward pass: compute the loss (scalar tensor with autograd)
    loss_tensor = compute_batch_training_loss(src_batch, tgt_batch, model_params, config)
    
    # 3. Backward pass: compute gradients
    loss_tensor.backward()
    
    # 4. Compute Noam learning rate for this step
    d_model = config.get('d_model')
    warmup_steps = config.get('warmup_steps')
    learning_rate = compute_noam_learning_rate(step_number, d_model, warmup_steps)
    
    # 5. Get Adam hyperparameters from config with defaults
    beta1 = config.get('beta1', 0.9)
    beta2 = config.get('beta2', 0.98)
    epsilon = config.get('epsilon', 1e-9)
    
    # 6. Apply Adam update
    optimizer_state = apply_adam_step_to_all_parameters(
        parameter_list, 
        optimizer_state, 
        learning_rate, 
        beta1=beta1, 
        beta2=beta2, 
        epsilon=epsilon
    )
    
    # 7. Return the loss as a Python float
    return loss_tensor.item()

# Step 73 - run_training_loop_for_steps
def run_training_loop_for_steps(batches, parameter_list, model_params, optimizer_state, num_steps, config):
    """Run num_steps training iterations, cycling through batches, and return per-step losses."""
    losses = []
    num_batches = len(batches)
    
    for step in range(1, num_steps + 1):
        # Cycle through batches: use (step - 1) % num_batches to wrap around
        batch_idx = (step - 1) % num_batches
        src_batch, tgt_batch = batches[batch_idx]
        
        # Run one training step and get the loss
        loss = run_training_step_with_backprop(
            src_batch, 
            tgt_batch, 
            parameter_list, 
            model_params, 
            optimizer_state, 
            step,  # step number starts at 1
            config
        )
        
        losses.append(loss)
    
    return losses

# Step 74 - pick_next_token_by_argmax (not yet solved)
# TODO: implement

# Step 75 - compute_length_penalty (not yet solved)
# TODO: implement

# Step 76 - compute_candidate_scores (not yet solved)
# TODO: implement

# Step 77 - select_top_k_candidates (not yet solved)
# TODO: implement

# Step 78 - append_tokens_to_beam_sequences (not yet solved)
# TODO: implement

# Step 79 - mark_finished_beams (not yet solved)
# TODO: implement

# Step 80 - select_best_finished_beam (not yet solved)
# TODO: implement

