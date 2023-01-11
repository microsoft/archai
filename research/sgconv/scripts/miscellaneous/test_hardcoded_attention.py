import torch


def make_asso_map(input_ids, mask):
    T = input_ids.shape[1]
    tokens = input_ids.unsqueeze(1).float()
    v = torch.ones(1, 1, T, device=input_ids.device).float()
    asso_map = (
        v.transpose(-1, -2) @ tokens**2 + (tokens**2).transpose(-1, -2) @ v - 2 * tokens.transpose(-1, -2) @ tokens
    )
    asso_map = (asso_map.long() == 0).float()
    idx = torch.arange(T, device=input_ids.device)
    asso_map[:, idx, idx] = 0
    asso_map *= mask.unsqueeze(-1) * mask.unsqueeze(1)
    asso_map /= asso_map.sum(-1, keepdim=True) + 1e-6

    return asso_map.unsqueeze(1)


def make_broadcast_map(input_ids, mask, eos_id=103):
    T = input_ids.shape[1]
    eos_map = (input_ids == eos_id).float()
    eos_map = eos_map.unsqueeze(1).expand(-1, T, -1)
    eos_mapp = eos_map * (mask.unsqueeze(-1) * mask.unsqueeze(1))
    eos_map = eos_mapp / (eos_map.sum(dim=-1, keepdim=True) + 1e-6)

    return eos_map.unsqueeze(1)


def main():
    input_ids = torch.tensor([[27, 3, 27, 103, 19, 1, 27]])
    mask = torch.ones_like(input_ids)
    _ = make_asso_map(input_ids, mask)
    _ = make_broadcast_map(input_ids, mask, eos_id=103)

    print("done")


if __name__ == "__main__":
    main()
