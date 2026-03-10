import cuda_attn_backend as _backend

def naive_attn(q,k,v,scaling):
    return _backend.naive_attention_fwd(q,k,v,scaling)