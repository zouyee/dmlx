#!/usr/bin/env python3
"""Extract DSpark shared expert weights from safetensors into per-layer binary files."""
import json, struct, glob, sys, os
from pathlib import Path
import numpy as np

def read_raw_tensor(filepath, header, key):
    info = header[key]
    start, end = info['data_offsets']
    with open(filepath, 'rb') as f:
        hs = struct.unpack('<Q', f.read(8))[0]
        f.seek(8 + hs + start)
        return f.read(end - start)

def main():
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.path.expanduser('~/models/DeepSeek-V4-Flash-DSpark'))
    output_dir = model_dir / 'dspark_weights'
    
    files = sorted(glob.glob(str(model_dir / '*.safetensors')))
    # Read all headers
    shard_headers = {}
    for f in files:
        with open(f, 'rb') as fp:
            hs = struct.unpack('<Q', fp.read(8))[0]
            header = json.loads(fp.read(hs))
        shard_headers[f] = {k: v for k, v in header.items() if k != '__metadata__'}
    
    for layer_idx in range(3):
        layer_dir = output_dir / f'layer_{layer_idx:02d}'
        layer_dir.mkdir(parents=True, exist_ok=True)
        
        prefix = f'mtp.{layer_idx}.ffn.shared_experts.'
        components = [
            ('w1', 'weight'), ('w1', 'scale'),
            ('w3', 'weight'), ('w3', 'scale'),
            ('w2', 'weight'), ('w2', 'scale'),
        ]
        
        for proj, attr in components:
            key = f'{prefix}{proj}.{attr}'
            for fname, header in shard_headers.items():
                if key in header:
                    data = read_raw_tensor(fname, header, key)
                    out_name = f'shared_{proj}_{attr}.bin'
                    out_path = layer_dir / out_name
                    with open(out_path, 'wb') as f:
                        f.write(data)
                    info = header[key]
                    print(f'  {out_path.name}: {info["shape"]} {info["dtype"]} ({len(data)} bytes)')
                    break
            else:
                print(f'  WARNING: {key} not found!')
    
    print('Done.')

if __name__ == '__main__':
    main()
