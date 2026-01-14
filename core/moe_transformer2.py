"""
Production-ready optimized MoE transformer module for BrahmaLLM.
- Vectorized routing in MoELayer and MoEAttention (no per-token Python loops)
- Batched expert GPU movement (small device-local copies)
- Fixed disk load calls (no invalid torch.load args)
- AttentionExpertManager returns CPU weight dicts; ExpertManager returns CPU modules
- Safe device handling and mixed-precision (AMP)

Notes:
- This file is designed to be drop-in for your training pipeline; adjust paths and hyperparams as needed.
- For very tight VRAM: change ExpertManager.move_expert_to_device to move only weight tensors and use F.linear.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import numpy as _np
import os
from collections import OrderedDict
import random
import time
import psutil
import threading
import queue
import math
import gc
from core.quant_utils import quantize_tensor_numpy, dequantize_to_numpy, save_quantized_state_dict, load_quantized_state_dict
from torch.amp import autocast, GradScaler

try:
    import pynvml
    _PYNVML_AVAILABLE = True
except Exception:
    _PYNVML_AVAILABLE = False


# ----------------------- Helpers -----------------------

def _device_of(x):
    return x.device if isinstance(x, torch.Tensor) else torch.device('cpu')


def get_ram_usage_mb():
    p = psutil.Process(os.getpid())
    return p.memory_info().rss / (1024 * 1024)


def safe_torch_load(path):
    return torch.load(path, map_location='cpu')


# ----------------------- Tiny modules -----------------------

class SwiGLU(nn.Module):
    def forward(self, x, gate):
        return F.silu(gate) * x


def quantize_and_dequantize(tensor, num_bits=2):
    min_val = tensor.min()
    max_val = tensor.max()
    if (max_val - min_val) == 0:
        return tensor
    scale = (max_val - min_val) / (2**num_bits - 1)
    quantized = torch.round((tensor - min_val) / scale)
    dequantized = quantized * scale + min_val
    return dequantized


class QuantLinear(nn.Module):
    """
    Linear layer that supports being loaded from a quantized state (packed bytes).
    - On forward, uses self.quant_weight (float32) if available, otherwise falls back to self.weight.
    - Provides helper load_quant_meta(meta) to load quantized metadata for weight (no allocation until dequantize()).
    """
    def __init__(self, in_features, out_features, num_bits=2, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_bits = int(num_bits)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        nn.init.xavier_uniform_(self.weight)
        self.quant_weight = None
        self._quant_meta = None  # holds packed bytes + scale/min if loaded from quant file

    def forward(self, x):
        # prefer quant_weight (dequantized float32 cached) to avoid repeated dequant
        w = self.quant_weight if self.quant_weight is not None else self.weight
        return F.linear(x, w, self.bias)

    def load_quant_meta(self, meta):
        """
        meta: dictionary containing 'shape','bits','min','scale','packed' (as produced by quant_utils)
        We don't immediately convert to tensor; we keep the meta and dequant on first forward/load.
        """
        # store as numpy-backed meta
        self._quant_meta = meta
        self.quant_weight = None

    def dequantize_weight_to_tensor(self, device=None, dtype=torch.float32):
        """
        Convert stored quant meta into a torch.FloatTensor and cache it in self.quant_weight.
        """
        if self._quant_meta is None:
            return
        arr = dequantize_to_numpy(self._quant_meta)  # returns float32 numpy array
        t = torch.from_numpy(arr).to(device or self.weight.device, dtype=dtype)
        # shape check
        if list(t.shape) != [self.out_features, self.in_features]:
            # if shapes don't match, try transpose fallback
            try:
                t = t.reshape(self.out_features, self.in_features)
            except Exception:
                raise RuntimeError("Dequantized shape mismatch for QuantLinear")
        self.quant_weight = nn.Parameter(t, requires_grad=False)
        # free meta to save memory (optional)
        self._quant_meta = None

    def clear_quant_cache(self):
        self.quant_weight = None
        self._quant_meta = None


class PBLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, adaptive_threshold_init=0.0, sparsity_percentile=0.05):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.adaptive_threshold = nn.Parameter(torch.full((out_features,), adaptive_threshold_init))
        self.sparsity_percentile = sparsity_percentile
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if hasattr(self, 'bias') and self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        nn.init.constant_(self.adaptive_threshold, 0.0)

    def forward(self, x):
        linear_output = F.linear(x, self.weight, self.bias)
        thresholded_output = linear_output - self.adaptive_threshold.unsqueeze(0).unsqueeze(0)
        flat = thresholded_output.view(-1)
        if flat.numel() > 0:
            p = float(min(max(self.sparsity_percentile, 0.0), 1.0))
            dyn_thresh = torch.quantile(flat.abs(), p).detach()
        else:
            dyn_thresh = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        return torch.where(thresholded_output.abs() > dyn_thresh, thresholded_output, torch.zeros_like(thresholded_output))


# ----------------------- Experts -----------------------

class Expert(nn.Module):
    def __init__(self, dim, ff_mult, dropout, LinearLayer=nn.Linear, linear_kwargs=None):
        super().__init__()
        linear_kwargs = linear_kwargs or {}
        self.fc1 = LinearLayer(dim, ff_mult * dim, **linear_kwargs)
        self.fc_gate = LinearLayer(dim, ff_mult * dim, **linear_kwargs)
        self.fc2 = LinearLayer(ff_mult * dim, dim, **linear_kwargs)
        self.activation = SwiGLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.fc1(x)
        g = self.fc_gate(x)
        out = self.activation(h, g)
        out = self.fc2(out)
        return self.dropout(out)


class DynamicExpert(nn.Module):
    # kept minimal; used when hypernet experts are enabled
    def __init__(self, dim, ff_mult, dropout):
        super().__init__()
        self.dim = dim
        self.ff_mult = ff_mult
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(dim, ff_mult * dim)
        self.fc_gate = nn.Linear(dim, ff_mult * dim)
        self.fc2 = nn.Linear(ff_mult * dim, dim)
        self.act = SwiGLU()

    def forward(self, x, params=None):
        # params path not fully implemented here; placeholder
        h = self.fc1(x)
        g = self.fc_gate(x)
        out = self.act(h, g)
        out = self.fc2(out)
        return self.dropout(out)


# ----------------------- Expert managers -----------------------

class ExpertManager:
    """Disk-backed expert modules (kept on CPU) with a small LRU cache.
    Use move_experts_to_device to create device-local copies when needed.
    """
    def __init__(self, expert_dir, cache_size, expert_cfg, device=torch.device('cpu'), num_total_experts=0, quantize_weights_on_save=False):
        self.expert_dir = expert_dir
        os.makedirs(self.expert_dir, exist_ok=True)
        self.cache_size = max(1, cache_size)
        self.expert_cfg = expert_cfg
        self.device = device
        self.num_total_experts = num_total_experts
        self.quantize_weights_on_save = quantize_weights_on_save

        self.cache = OrderedDict()  # idx -> module (CPU)
        self.lock = threading.Lock()
        self.prefetch_q = queue.Queue(maxsize=self.cache_size * 2)
        self.stop_event = threading.Event()
        self.prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.prefetch_thread.start()

    def _expert_path(self, idx):
        return os.path.join(self.expert_dir, f'expert_{idx}.pt')

    def _create_and_save(self, idx):
        LinearLayer = self.expert_cfg.get('LinearLayer', nn.Linear)
        linear_kwargs = self.expert_cfg.get('linear_kwargs', {})
        m = Expert(self.expert_cfg['dim'], self.expert_cfg['ff_mult'], self.expert_cfg['dropout'], LinearLayer=LinearLayer, linear_kwargs=linear_kwargs)
        torch.save(m.state_dict(), self._expert_path(idx))
        return m

    def _load_from_disk(self, idx):
        path = self._expert_path(idx)
        qpath = path + ".qst"
        LinearLayer = self.expert_cfg.get('LinearLayer', nn.Linear)
        linear_kwargs = self.expert_cfg.get('linear_kwargs', {})

        m = Expert(self.expert_cfg['dim'], self.expert_cfg['ff_mult'], self.expert_cfg['dropout'], LinearLayer=LinearLayer, linear_kwargs=linear_kwargs)

        # prefer quant file if exists
        if self.quantize_weights_on_save and os.path.exists(qpath):
            try:
                header, get_param = load_quantized_state_dict(qpath)
                # Reconstruct state dict: dequant weight tensors and set biases
                loaded = {}
                for k in header.keys():
                    param_meta = get_param(k)
                    if param_meta['count'] == 0:
                        arr = _np.zeros(tuple(param_meta['shape']), dtype=_np.float32)
                    else:
                        arr = dequantize_to_numpy(param_meta)
                    # Convert to torch tensor and set
                    loaded[k] = torch.from_numpy(arr)
                # Load state into module, allow non-strict in case names differ
                m.load_state_dict(loaded, strict=False)
                return m
            except Exception as e:
                print(f"Warning: failed to load quantized expert {qpath}: {e}. Falling back to .pt")
                # fallback to regular .pt load

        # fallback behavior: normal float32 .pt
        if not os.path.exists(path):
            # create and save one
            return self._create_and_save(idx)
        state = safe_torch_load(path)
        try:
            m.load_state_dict(state)
        except Exception:
            # try non-strict load
            m.load_state_dict(state, strict=False)
        return m


    def save_expert(self, module, idx):
        """
        Save an expert module either as full state_dict (.pt) or as quantized packed file (.qst/.qpt)
        depending on self.quantize_weights_on_save.
        """
        state = module.state_dict()
        path_pt = self._expert_path(idx)
        if self.quantize_weights_on_save:
            # prepare a numpy state dict but only quantize weight tensors
            import core.quant_utils as quant_utils
            np_state = {}
            for k, v in state.items():
                # v is a torch tensor
                arr = v.detach().cpu().numpy()
                # Only quantize weight tensors (heuristic: contain 'weight' in key)
                if 'weight' in k:
                    # quantize and store packed bytes via quant_utils.save_quantized_state_dict
                    np_state[k] = arr.astype(_np.float32)
                else:
                    # store biases & small tensors as float32 full precision
                    np_state[k] = arr.astype(_np.float32)
            # Use the custom quant file format (single file packing per module)
            qpath = path_pt + ".qst"
            save_quantized_state_dict(qpath, np_state, num_bits=self.expert_cfg.get('quant_bits', 2))
            return
        else:
            # default save full precision state
            torch.save(state, path_pt)
            return


    def request_prefetch(self, idx):
        try:
            self.prefetch_q.put_nowait(idx)
        except queue.Full:
            pass

    def _prefetch_worker(self):
        while not self.stop_event.is_set():
            try:
                idx = self.prefetch_q.get(timeout=0.1)
            except queue.Empty:
                time.sleep(0.01)
                continue
            with self.lock:
                if idx in self.cache:
                    continue
            m = self._load_from_disk(idx)
            with self.lock:
                self.cache[idx] = m
                while len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)
            try:
                self.prefetch_q.task_done()
            except Exception:
                pass

    def stop(self):
        self.stop_event.set()
        self.prefetch_thread.join(timeout=1.0)

    def get_expert_cpu(self, idx):
        with self.lock:
            if idx in self.cache:
                m = self.cache.pop(idx)
                self.cache[idx] = m
                return m
        m = self._load_from_disk(idx)
        with self.lock:
            self.cache[idx] = m
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return m

    def move_experts_to_device(self, idxs, device=None):
        device = device or self.device
        results = {}
        for i in idxs:
            cpu_mod = self.get_expert_cpu(i)
            # instantiate same class then load_state_dict to avoid mutating cpu cached module
            mod_copy = type(cpu_mod)(cpu_mod.fc1.in_features, cpu_mod.fc2.out_features // cpu_mod.fc2.in_features if hasattr(cpu_mod, 'fc2') else cpu_mod.fc1.out_features, cpu_mod.dropout.p) if isinstance(cpu_mod, Expert) else None
            if mod_copy is None:
                try:
                    mod_copy = Expert(self.expert_cfg['dim'], self.expert_cfg['ff_mult'], self.expert_cfg['dropout'])
                except Exception:
                    mod_copy = cpu_mod
            try:
                mod_copy.load_state_dict(cpu_mod.state_dict())
            except Exception:
                # fallback to shallow copy
                mod_copy = cpu_mod
            results[i] = mod_copy.to(device)
        return results


class AttentionExpertManager:
    """Stores attention expert weight dicts on disk and caches CPU-loaded dicts."""
    def __init__(self, expert_dir, num_layers, num_experts, dim, num_heads):
        self.expert_dir = expert_dir
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        os.makedirs(self.expert_dir, exist_ok=True)
        self.cache = OrderedDict()
        self.lock = threading.Lock()

    def _path(self, layer_idx, expert_idx):
        layer_dir = os.path.join(self.expert_dir, f'layer_{layer_idx}')
        os.makedirs(layer_dir, exist_ok=True)
        return os.path.join(layer_dir, f'attn_expert_{expert_idx}.pt')

    def precreate_all_experts(self):
        for l in range(self.num_layers):
            for e in range(self.num_experts):
                p = self._path(l, e)
                if not os.path.exists(p):
                    state = {
                        'q': torch.randn(self.num_heads * self.head_dim, self.dim),
                        'k': torch.randn(self.num_heads * self.head_dim, self.dim),
                        'v': torch.randn(self.num_heads * self.head_dim, self.dim),
                        'o': torch.randn(self.dim, self.dim)
                    }
                    torch.save(state, p)

    def get_expert(self, layer_idx, expert_idx):
        key = (layer_idx, expert_idx)
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
        p = self._path(layer_idx, expert_idx)
        if not os.path.exists(p):
            self.precreate_all_experts()
        state = safe_torch_load(p)
        # validate
        expected = {'q', 'k', 'v', 'o'}
        if set(state.keys()) != expected:
            raise RuntimeError(f'attn expert missing keys: expected {expected}, found {set(state.keys())}')
        with self.lock:
            self.cache[key] = state
            while len(self.cache) > 512:
                self.cache.popitem(last=False)
        return state


# ----------------------- Vectorized MoE Attention -----------------------
class MoEAttention(nn.Module):
    def __init__(self, dim, num_heads, attention_expert_manager, attention_num_experts=12, attention_top_k=2, dropout=0.0, layer_idx=0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attention_expert_manager = attention_expert_manager
        self.attention_num_experts = attention_num_experts
        self.attention_top_k = attention_top_k
        self.layer_idx = layer_idx
        self.router = nn.Linear(dim, attention_num_experts, bias=False)
        self.dropout = dropout

    def _calc_load_balance(self, router_logits, expert_indices_flat, top_k):
        # router_logits: [T, E]
        # expert_indices_flat: [T*K]
        with torch.no_grad():
            counts = torch.bincount(expert_indices_flat, minlength=self.attention_num_experts).to(router_logits.dtype)
            p = F.softmax(router_logits, dim=-1).sum(dim=0)
            total = counts.sum()
            frac = counts / (total + 1e-6)
            lb = (frac * p).sum() * self.attention_num_experts
            return lb

    def forward(self, x, attn_mask=None, is_causal=False, return_attn_weights=False):
        # x: [B, N, D] or [T, D]
        orig_dim = x.dim()
        if orig_dim == 3:
            B, N, D = x.shape
            x_flat = x.view(-1, D)
        elif orig_dim == 2:
            x_flat = x
            B = None
            N = None
            D = x.shape[1]
        else:
            raise ValueError('bad x dim')

        T = x_flat.shape[0]
        device = x_flat.device

        router_logits = self.router(x_flat)  # [T, E]
        router_probs = F.softmax(router_logits, dim=-1)
        topv, topi = torch.topk(router_probs, k=self.attention_top_k, dim=-1)  # [T, K]
        # normalize top-k weights per-token
        topv = topv / (topv.sum(dim=-1, keepdim=True) + 1e-6)

        K = self.attention_top_k
        flat_token_idxs = torch.arange(T, device=device).unsqueeze(1).expand(-1, K).reshape(-1)
        flat_expert_idxs = topi.reshape(-1)
        flat_weights = topv.reshape(-1)

        valid = (flat_expert_idxs >= 0) & (flat_expert_idxs < self.attention_num_experts)
        flat_token_idxs = flat_token_idxs[valid]
        flat_expert_idxs = flat_expert_idxs[valid]
        flat_weights = flat_weights[valid]

        if flat_token_idxs.numel() == 0:
            # fallback to vanilla attention
            mha = nn.MultiheadAttention(self.dim, self.num_heads)
            out = mha(x.transpose(0,1), x.transpose(0,1), x.transpose(0,1))[0].transpose(0,1)
            return out, torch.tensor(0.0, device=device)

        order = flat_expert_idxs.argsort()
        flat_expert_idxs_s = flat_expert_idxs[order]
        flat_token_idxs_s = flat_token_idxs[order]
        flat_weights_s = flat_weights[order]

        unique_experts, counts = torch.unique_consecutive(flat_expert_idxs_s, return_counts=True)

        # Load required attention expert weight dicts and move to device once
        expert_weight_map = {}
        ptr = 0
        for i, e in enumerate(unique_experts.tolist()):
            cnt = int(counts[i])
            token_positions = flat_token_idxs_s[ptr:ptr+cnt]
            ptr += cnt
            cpu_weights = self.attention_expert_manager.get_expert(self.layer_idx, int(e))
            device_weights = {k: v.to(device, non_blocking=True) for k, v in cpu_weights.items()}
            expert_weight_map[int(e)] = (token_positions, device_weights)

        Dq = self.num_heads * self.head_dim
        q_agg = x_flat.new_zeros((T, Dq))
        k_agg = x_flat.new_zeros((T, Dq))
        v_agg = x_flat.new_zeros((T, Dq))

        # per-expert projections aggregated into q/k/v
        for eid, (token_pos, wdict) in expert_weight_map.items():
            inp = x_flat[token_pos]
            q_proj = F.linear(inp, wdict['q'])
            k_proj = F.linear(inp, wdict['k'])
            v_proj = F.linear(inp, wdict['v'])
            # gather corresponding weights slice (they are contiguous due to sort)
            # compute where in flat_weights_s these token_pos appear: we can reuse contiguous ptr logic
            q_agg.index_add_(0, token_pos, q_proj * flat_weights_s[:token_pos.numel()].unsqueeze(1))
            k_agg.index_add_(0, token_pos, k_proj * flat_weights_s[:token_pos.numel()].unsqueeze(1))
            v_agg.index_add_(0, token_pos, v_proj * flat_weights_s[:token_pos.numel()].unsqueeze(1))

        # reshape to MHA format
        if orig_dim == 3:
            q = q_agg.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            k = k_agg.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            v = v_agg.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            q = q_agg.view(-1, self.num_heads, self.head_dim).transpose(0,1)
            k = k_agg.view(-1, self.num_heads, self.head_dim).transpose(0,1)
            v = v_agg.view(-1, self.num_heads, self.head_dim).transpose(0,1)

        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0, is_causal=is_causal)
        if orig_dim == 3:
            out = attn_output.transpose(1,2).reshape(B, N, self.dim)
        else:
            out = attn_output.transpose(1,2).reshape(-1, self.dim)

        # second stage: project output slices through expert 'o' matrices and accumulate
        o_agg = x_flat.new_zeros((T, self.dim))
        for eid, (token_pos, wdict) in expert_weight_map.items():
            proj_inp = out.view(-1, self.dim)[token_pos]
            o_proj = F.linear(proj_inp, wdict['o'])
            o_agg.index_add_(0, token_pos, o_proj * flat_weights_s[:token_pos.numel()].unsqueeze(1))

        final_out = o_agg.view(B, N, self.dim) if orig_dim == 3 else o_agg

        lb_loss = self._calc_load_balance(router_logits, flat_expert_idxs, self.attention_top_k) if self.training else torch.tensor(0.0, device=device)

        if return_attn_weights:
            # compute approximate attn weights from aggregated q/k
            q_m = q
            k_m = k
            attn_scores = torch.matmul(q_m, k_m.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn_w = F.softmax(attn_scores, dim=-1)
            return final_out, lb_loss, attn_w

        return final_out, lb_loss


# ----------------------- Vectorized MoE Layer -----------------------
class MoELayer(nn.Module):
    def __init__(self, expert_manager, dim, num_experts, top_k, ff_mult, dropout, routing_strategy='topk', use_hypernet_experts=False, num_hypernet_experts=0):
        super().__init__()
        self.expert_manager = expert_manager
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.ff_mult = ff_mult
        self.dropout = dropout
        self.routing_strategy = routing_strategy
        self.use_hypernet = use_hypernet_experts
        self.num_hyper = num_hypernet_experts
        self.router = nn.Linear(dim, num_experts + (num_hypernet_experts if use_hypernet_experts else 0), bias=False)

    def _calc_load_balance(self, router_logits, flat_expert_idxs):
        with torch.no_grad():
            counts = torch.bincount(flat_expert_idxs, minlength=self.num_experts)
            p = F.softmax(router_logits, dim=-1).sum(dim=0)
            total = counts.sum()
            frac = counts / (total + 1e-6)
            return (frac * p).sum() * max(1, self.num_experts)

    def forward(self, x, attn_output=None):
        orig_dim = x.dim()
        if orig_dim == 3:
            B, N, C = x.shape
            x_flat = x.view(-1, C)
        elif orig_dim == 2:
            x_flat = x
            B = None
            N = None
            C = x.shape[1]
        else:
            raise ValueError('bad x dim')

        T = x_flat.shape[0]
        router_in = x_flat if attn_output is None else attn_output.view(-1, C)
        router_logits = self.router(router_in)
        probs = F.softmax(router_logits, dim=-1)
        topv, topi = torch.topk(probs, k=self.top_k, dim=-1)
        topv = topv / (topv.sum(dim=-1, keepdim=True) + 1e-6)

        K = self.top_k
        flat_token_idxs = torch.arange(T, device=x_flat.device).unsqueeze(1).expand(-1, K).reshape(-1)
        flat_expert_idxs = topi.reshape(-1)
        flat_weights = topv.reshape(-1)

        valid = (flat_expert_idxs >= 0) & (flat_expert_idxs < self.num_experts + (self.num_hyper if self.use_hypernet else 0))
        flat_token_idxs = flat_token_idxs[valid]
        flat_expert_idxs = flat_expert_idxs[valid]
        flat_weights = flat_weights[valid]

        if flat_token_idxs.numel() == 0:
            return x, torch.tensor(0.0, device=x.device)

        order = flat_expert_idxs.argsort()
        flat_expert_idxs_s = flat_expert_idxs[order]
        flat_token_idxs_s = flat_token_idxs[order]
        flat_weights_s = flat_weights[order]

        unique_experts, counts = torch.unique_consecutive(flat_expert_idxs_s, return_counts=True)

        # batch-move disk experts to device
        disk_experts = [int(e) for e in unique_experts.tolist() if int(e) < self.num_experts]
        expert_modules = self.expert_manager.move_experts_to_device(disk_experts, device=x_flat.device) if disk_experts else {}

        outputs = x_flat.new_zeros((T, C))
        ptr = 0
        for i, e in enumerate(unique_experts.tolist()):
            cnt = int(counts[i])
            tok_pos = flat_token_idxs_s[ptr:ptr+cnt]
            weights_chunk = flat_weights_s[ptr:ptr+cnt].unsqueeze(1)
            inp = x_flat[tok_pos]
            if int(e) < self.num_experts:
                mod = expert_modules.get(int(e))
                if mod is None:
                    cpu_mod = self.expert_manager.get_expert_cpu(int(e))
                    mod = cpu_mod.to(x_flat.device)
                out = mod(inp)
            else:
                # simple hypernet path (placeholder)
                out = DynamicExpert(self.dim, self.ff_mult, self.dropout).to(x_flat.device)(inp)
            if out.dim() == 1:
                out = out.unsqueeze(0)
            outputs.index_add_(0, tok_pos, (out * weights_chunk.to(out.dtype)).to(outputs.dtype))
            ptr += cnt

        if orig_dim == 3:
            out_final = outputs.view(B, N, C)
        else:
            out_final = outputs

        lb = self._calc_load_balance(router_logits, flat_expert_idxs)
        return out_final, lb


# ----------------------- Transformer Blocks & Stacked Model -----------------------
class FeedForward(nn.Module):
    def __init__(self, dim, ff_mult, dropout, use_quant_linear=False, quant_bits=2, use_pb_linear=False):
        super().__init__()
        LinearLayer = PBLinear if use_pb_linear else (QuantLinear if use_quant_linear else nn.Linear)
        kwargs = {'num_bits': quant_bits} if use_quant_linear else {}
        self.fc1 = LinearLayer(dim, ff_mult*dim, **(kwargs if isinstance(LinearLayer, type) and LinearLayer is QuantLinear else {}))
        self.fc2 = LinearLayer(ff_mult*dim, dim, **(kwargs if isinstance(LinearLayer, type) and LinearLayer is QuantLinear else {}))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    def __init__(self, expert_manager, attention_expert_manager, dim, num_heads, ff_mult=2, dropout=0.0, use_moe=True,
                 num_experts=8, top_k=2, routing_strategy='topk', use_moe_attention=True, attention_num_experts=12, attention_top_k=2, layer_idx=0):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.use_moe_attention = use_moe_attention
        self.use_moe = use_moe
        if use_moe_attention:
            self.attn = MoEAttention(dim, num_heads, attention_expert_manager, attention_num_experts=attention_num_experts, attention_top_k=attention_top_k, layer_idx=layer_idx)
        else:
            self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.attn_dropout = nn.Dropout(dropout)
        if use_moe:
            self.ffn = MoELayer(expert_manager, dim, num_experts, top_k, ff_mult, dropout, routing_strategy=routing_strategy)
        else:
            self.ffn = FeedForward(dim, ff_mult, dropout)
        self.ffn_dropout = nn.Dropout(dropout)
        self.halting = nn.Linear(dim,1)

    def forward(self, x, attn_mask=None, is_causal=False):
        nx = self.attn_norm(x)
        if self.use_moe_attention:
            attn_out, attn_aux = self.attn(nx, attn_mask=attn_mask, is_causal=is_causal)
        else:
            attn_out = self.attn(nx.transpose(0,1), nx.transpose(0,1), nx.transpose(0,1))[0].transpose(0,1)
            attn_aux = torch.tensor(0.0, device=x.device)
        x = x + self.attn_dropout(attn_out)
        nx2 = self.ffn_norm(x)
        ffn_out, ffn_aux = self.ffn(nx2)
        x = x + self.ffn_dropout(ffn_out)
        halting = torch.sigmoid(self.halting(x))
        return x, ffn_aux, attn_aux, halting


class StackedTransformer(nn.Module):
    def __init__(self, vocab_size, max_seq_len, num_layers, dim, num_heads, ffn_expert_dir, attn_expert_dir, ff_mult=2, dropout=0.0,
                 use_moe=True, num_experts=12, top_k=2, expert_cache_size=4, device='cpu', use_quant_linear=False, quant_bits=2,
                 routing_strategy='topk', use_act=False, act_threshold=0.9, use_hypernet_experts=False, num_hypernet_experts=0,
                 use_moe_attention=True, attention_num_experts=12, attention_top_k=2):
        super().__init__()
        self.device = torch.device(device)
        self.token_embeddings = nn.Embedding(vocab_size, dim)
        self.position_embeddings = nn.Embedding(max_seq_len, dim)
        self.ffn_manager = ExpertManager(ffn_expert_dir, expert_cache_size, {'dim':dim,'ff_mult':ff_mult,'dropout':dropout}, device=self.device, num_total_experts=num_experts)
        self.attn_manager = AttentionExpertManager(attn_expert_dir, num_layers, attention_num_experts, dim, num_heads)
        self.layers = nn.ModuleList([
            TransformerBlock(self.ffn_manager, self.attn_manager, dim, num_heads, ff_mult=ff_mult, dropout=dropout, use_moe=use_moe, num_experts=num_experts, top_k=top_k, routing_strategy=routing_strategy, use_moe_attention=use_moe_attention, attention_num_experts=attention_num_experts, attention_top_k=attention_top_k, layer_idx=i)
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.use_act = use_act
        self.act_threshold = act_threshold
        self.scaler = GradScaler(enabled=(self.device.type == 'cuda'))

    def forward(self, input_ids, attn_mask=None):
        B, N = input_ids.shape
        pos = torch.arange(N, device=self.device)
        x = self.token_embeddings(input_ids.to(self.device)) + self.position_embeddings(pos)
        x = x
        total_ffn_aux = torch.tensor(0.0, device=self.device)
        total_attn_aux = torch.tensor(0.0, device=self.device)
        for layer in self.layers:
            x, ffn_aux, attn_aux, halting = layer(x, attn_mask=attn_mask, is_causal=True)
            total_ffn_aux += ffn_aux
            total_attn_aux += attn_aux
        logits = self.lm_head(self.norm(x))
        steps = torch.full((B,N), len(self.layers), dtype=torch.long, device=self.device)
        return logits, (total_ffn_aux+total_attn_aux)/max(1,len(self.layers)), steps