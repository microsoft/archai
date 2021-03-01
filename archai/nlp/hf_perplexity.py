import torch
from datasets import load_dataset

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import tqdm

from archai.common import utils

cache_dir = utils.full_path('~/dataroot/huggingface/transformers', create=True)

device = 'cuda'
model_id = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_id, cache_dir=cache_dir).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id, cache_dir=cache_dir)

data_dir = utils.full_path('~/dataroot/huggingface/datasets', create=True)

test = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test', data_dir=data_dir)
# below will be just one giant tensor of all tokens
encodings = tokenizer('\n\n'.join(test['text']), return_tensors='pt')

max_length = model.config.n_positions
stride = 512

lls = []
with torch.no_grad():
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i    # may be different from stride on last loop
        input_ids = encodings.input_ids[:,begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()

        # -100 is special number to indicate ignore in loss calculation
        # token ids are otherwise >=0
        target_ids[:,:-trg_len] = -100

        outputs = model(input_ids, labels=target_ids)
        log_likelihood = outputs[0] * trg_len # total loss = avg_loss per prediction * num_predictions

        #print(begin_loc, end_loc, trg_len)

        lls.append(log_likelihood)

    ppl = torch.exp(torch.stack(lls).sum() / end_loc).item()

print(ppl)