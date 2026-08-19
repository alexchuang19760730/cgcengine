import sys

from cgc_engine.model_parsers import GGUFParser, parsed_model_to_pytorch
from cgc_engine.model_parsers.parsed_model_adapter import _gguf_to_pytorch_name, _adjust_tensor_shape
import torch
import traceback

try:
    parser = GGUFParser('/Users/alexchuang/Documents/flashkv0430/qwen2.5-7b-q4_k_m.gguf')
    parsed_model = parser.parse_model()
    weights = parser.load_weights()
    
    print(f'Parsed: {parsed_model.hidden_dim} hidden, {parsed_model.num_layers} layers')
    
    # Test weight name mapping
    print('\nTesting weight mapping:')
    for w in weights[:10]:
        pt_name = _gguf_to_pytorch_name(w.name)
        print(f'  GGUF: {w.name:<35} → PyTorch: {pt_name}')
    
    # Manual conversion to debug
    print('\n=== Manual Conversion Debug ===')
    
    # Create model first
    from cgc_engine.model_parsers.parsed_model_adapter import AdapterLLM
    print('Creating AdapterLLM...')
    model = AdapterLLM(parsed_model)
    print(f'Model created with {len(list(model.parameters()))} parameters')
    
    # Now load weights manually
    print('\nLoading weights manually...')
    state_dict = {}
    loaded_count = 0
    skipped_count = 0
    shape_mismatch_count = 0
    
    for i, w in enumerate(weights):
        pt_name = _gguf_to_pytorch_name(w.name)
        if pt_name:
            # Check if this name exists in model
            if pt_name in model.state_dict():
                target_shape = model.state_dict()[pt_name].shape
                source_shape = w.tensor.shape
                
                if source_shape == target_shape:
                    state_dict[pt_name] = w.tensor
                    loaded_count += 1
                elif w.tensor.T.shape == target_shape:
                    state_dict[pt_name] = w.tensor.T
                    loaded_count += 1
                else:
                    print(f'  Shape mismatch: {pt_name}')
                    print(f'    Source: {source_shape}, Target: {target_shape}')
                    shape_mismatch_count += 1
            else:
                skipped_count += 1
        else:
            skipped_count += 1
    
    print(f'Loaded: {loaded_count}, Skipped: {skipped_count}, Shape mismatch: {shape_mismatch_count}')
    
    # Try to load state dict
    print('\nLoading state dict...')
    model.load_state_dict(state_dict, strict=False)
    print('State dict loaded!')
    
    # Test inference
    print('\nTesting inference...')
    model.eval()
    with torch.no_grad():
        test_input = torch.randint(0, parsed_model.vocab_size, (1, 32))
        output = model(test_input)
    print(f'Inference successful! Output shape: {output.shape}')
    
except Exception as e:
    traceback.print_exc()
    print(f'Error type: {type(e).__name__}')
    print(f'Error: {e}')
