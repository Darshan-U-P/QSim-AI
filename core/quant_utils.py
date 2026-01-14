# quant_utils.py
# Simple per-tensor low-bit quantization utilities (1-4 bits).
# Stores metadata + bit-packed payload to reduce on-disk size.
# NOTE: This uses per-tensor symmetric quantization with zero point = 0 offset.
# It's simple and portable. For best accuracy or large models, consider per-channel
# quantization or GPTQ/AWQ style techniques.

import numpy as np
import struct
import json
import os

def _pack_bits(arr_uint, bits):
    """
    Pack an array of unsigned integers (each smaller than 2**bits) into bytes.
    arr_uint: 1D numpy uint8/uint32 ints containing values in range [0, 2**bits - 1].
    bits: 1..8
    Returns: bytes
    """
    assert 1 <= bits <= 8
    if bits == 8:
        return arr_uint.astype(np.uint8).tobytes()
    bit_cursor = 0
    out_bytes = bytearray()
    cur = 0
    bits_in_cur = 0
    mask = (1 << bits) - 1
    for v in arr_uint:
        cur |= (int(v) & mask) << bits_in_cur
        bits_in_cur += bits
        while bits_in_cur >= 8:
            out_bytes.append(cur & 0xFF)
            cur >>= 8
            bits_in_cur -= 8
    if bits_in_cur > 0:
        out_bytes.append(cur & 0xFF)
    return bytes(out_bytes)

def _unpack_bits(packed_bytes, count, bits):
    """
    Unpack `count` values of `bits` width from packed_bytes.
    Returns a numpy uint8 array (values <= 255), but values may be up to 2**bits-1.
    """
    assert 1 <= bits <= 8
    if bits == 8:
        arr = np.frombuffer(packed_bytes, dtype=np.uint8, count=count)
        return arr
    out = np.empty(count, dtype=np.uint8)
    bit_cursor = 0
    byte_idx = 0
    cur = 0
    bits_in_cur = 0
    mask = (1 << bits) - 1
    i = 0
    total_bytes = len(packed_bytes)
    while i < count:
        while bits_in_cur < bits and byte_idx < total_bytes:
            cur |= (packed_bytes[byte_idx] & 0xFF) << bits_in_cur
            bits_in_cur += 8
            byte_idx += 1
        if bits_in_cur >= bits:
            out[i] = cur & mask
            cur >>= bits
            bits_in_cur -= bits
            i += 1
        else:
            # not enough bits; treat remaining as zeros
            out[i] = cur & mask
            i += 1
    return out

def quantize_tensor_numpy(tensor: np.ndarray, num_bits: int = 2):
    """
    Quantize a floating numpy array to num_bits using min/max scale and produce packed bytes.
    Returns metadata dict:
      {
        'shape': list(tensor.shape),
        'dtype': 'float32',
        'bits': num_bits,
        'min': float(min_val),
        'scale': float(scale),
        'count': int(number_of_elements),
        'packed': bytes
      }
    """
    assert tensor.dtype == np.float32 or tensor.dtype == np.float64
    flat = tensor.ravel().astype(np.float32)
    if flat.size == 0:
        return {
            'shape': list(tensor.shape),
            'dtype': 'float32',
            'bits': num_bits,
            'min': 0.0,
            'scale': 1.0,
            'count': 0,
            'packed': b''
        }
    min_val = float(flat.min())
    max_val = float(flat.max())
    qlevels = (1 << num_bits) - 1
    # Handle constant tensor edge-case
    if max_val - min_val <= 1e-12:
        scale = 1.0
        q = np.zeros(flat.shape, dtype=np.uint8)
    else:
        scale = (max_val - min_val) / float(qlevels)
        inv_scale = 1.0 / scale
        q = np.round((flat - min_val) * inv_scale).astype(np.int64)
        q = np.clip(q, 0, qlevels).astype(np.uint8)
    packed = _pack_bits(q, num_bits)
    return {
        'shape': list(tensor.shape),
        'dtype': 'float32',
        'bits': int(num_bits),
        'min': float(min_val),
        'scale': float(scale),
        'count': int(flat.size),
        'packed': packed
    }

def dequantize_to_numpy(meta):
    """
    Reconstruct float32 numpy array from metadata produced by quantize_tensor_numpy.
    """
    count = int(meta['count'])
    bits = int(meta['bits'])
    if count == 0:
        return np.zeros(tuple(meta['shape']), dtype=np.float32)
    packed = meta['packed']
    q = _unpack_bits(packed, count, bits).astype(np.float32)
    min_val = float(meta['min'])
    scale = float(meta['scale'])
    flat = q * scale + min_val
    return flat.reshape(tuple(meta['shape'])).astype(np.float32)

def save_quantized_state_dict(path, state_dict, num_bits=2):
    """
    Save a state_dict (dict of numpy arrays or torch tensors) to a .qst file as quantized entries.
    We save a JSON header mapping param name -> metadata (excluding the large packed bytes stored separately).
    Format:
      <4-byte header len><JSON header bytes><concatenated packed bytes>
    Where JSON header contains for each param: shape, dtype, bits, min, scale, count, offset, length
    """
    header = {}
    body_parts = []
    offset = 0
    for k, v in state_dict.items():
        arr = np.array(v) if not isinstance(v, np.ndarray) else v
        if arr.size == 0:
            meta = {
                'shape': list(arr.shape),
                'dtype': 'float32',
                'bits': int(num_bits),
                'min': 0.0,
                'scale': 1.0,
                'count': int(arr.size),
                'offset': offset,
                'length': 0
            }
            header[k] = meta
            continue
        # ensure float32
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        qmeta = quantize_tensor_numpy(arr, num_bits=num_bits)
        b = qmeta['packed']
        length = len(b)
        meta = {
            'shape': qmeta['shape'],
            'dtype': qmeta['dtype'],
            'bits': qmeta['bits'],
            'min': qmeta['min'],
            'scale': qmeta['scale'],
            'count': qmeta['count'],
            'offset': offset,
            'length': length
        }
        header[k] = meta
        body_parts.append(b)
        offset += length
    header_json = json.dumps(header).encode('utf-8')
    with open(path, 'wb') as f:
        f.write(struct.pack('<I', len(header_json)))
        f.write(header_json)
        for b in body_parts:
            f.write(b)
    return True

def load_quantized_state_dict(path):
    """
    Load quantized state dict previously saved with save_quantized_state_dict.
    Returns a dict mapping param name -> metadata dict (including 'packed' bytes slice)
    but to conserve memory we return metadata plus a function to dequantize param on demand.
    """
    with open(path, 'rb') as f:
        hdr_len = struct.unpack('<I', f.read(4))[0]
        hdr_bytes = f.read(hdr_len)
        header = json.loads(hdr_bytes.decode('utf-8'))
        # Read body into memory once (it's smaller than float32 full)
        body = f.read()
    # attach helper
    def get_param(name):
        meta = header[name]
        off = meta['offset']
        length = meta['length']
        packed = b'' if length == 0 else body[off:off+length]
        m = {
            'shape': meta['shape'],
            'dtype': meta['dtype'],
            'bits': meta['bits'],
            'min': meta['min'],
            'scale': meta['scale'],
            'count': meta['count'],
            'packed': packed
        }
        return m
    return header, get_param
