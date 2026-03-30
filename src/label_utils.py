import torch
from . import global_variables

def encode_to_labels(txt):
    encoded = [global_variables.CHAR_LIST.index(char) + 1 for char in txt]  # 0 is reserved for blank
    padded = encoded + [0] * (global_variables.MAX_LABEL_LEN - len(encoded)) # Pads sequence to 46 characters
    return torch.tensor(padded, dtype=torch.long), len(encoded)

def decode_to_text(label_ids):
    chars = []
    for idx in label_ids:
        idx = int(idx)
        if idx == 0:
            continue
        chars.append(global_variables.CHAR_LIST[idx - 1])
    return "".join(chars)
    

def decode_prediction(log_probs):
    # log_probs: (T, N, C)
    greedy = log_probs.argmax(dim=2).permute(1, 0)  # (N, T)
    decoded = []
    for seq in greedy:
        out = []
        prev = 0
        for idx in seq.tolist():
            if idx != 0 and idx != prev:
                out.append(idx)
            prev = idx
        decoded.append(out)
    return decoded

def target_to_list(target, target_len):
    return target[:int(target_len)].tolist()