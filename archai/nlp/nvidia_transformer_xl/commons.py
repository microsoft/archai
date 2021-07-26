
# general utils

import torch

def encode(list_of_strings, pad_token_id=0):
    # max_length = max([len(string) for string in list_of_strings])
    max_length = -1
    for string in list_of_strings:
        if not isinstance(string, bytes):
            string = str.encode(string)
        max_length = max(max_length, len(string))

    # create emtpy tensors
    attention_masks = torch.zeros((len(list_of_strings), max_length), dtype=torch.long)
    input_ids = torch.full((len(list_of_strings), max_length), pad_token_id, dtype=torch.long)

    for idx, string in enumerate(list_of_strings):
        # make sure string is in byte format
        if not isinstance(string, bytes):
            string = str.encode(string)
        input_ids[idx, :len(string)] = torch.tensor([x + 2 for x in string])
        attention_masks[idx, :len(string)] = 1

    return input_ids, attention_masks

def decode(outputs_ids):
    decoded_outputs = []
    for output_ids in outputs_ids.tolist():
        # transform id back to char IDs < 2 are simply transformed to ""
        decoded_outputs.append("".join([chr(x - 2) if x > 1 else "" for x in output_ids]))
    return decoded_outputs

def reformer_check():
    from transformers import ReformerModelWithLMHead
    model = ReformerModelWithLMHead.from_pretrained("google/reformer-enwik8")
    encoded, attention_masks = encode(["In 1965, Brooks left IBM to found the Department of", "this is cool"])
    print(decode(model.generate(encoded, do_sample=True, max_length=150)))

def read_data(prompt_context_percent):
    encoded = []
    with open("/home/t-gjawahar/dataroot/wiki.test.tokens", 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if len(line) == 0 or len(line.split()) <= 1:
                continue
            tokens = line.split()
            num_ptokens = int(prompt_context_percent * len(tokens))
            prompt_tokens = tokens[0:num_ptokens]
            target_tokens = tokens[num_ptokens:]
            target_tokens = target_tokens[0:3]
            if len(prompt_tokens) == 0 or len(target_tokens) == 0:
                continue
            encoded.append((prompt_tokens, target_tokens))
            if len(encoded) == 500:
                break
    return encoded

def exact_match_reformer():
    from transformers import ReformerModelWithLMHead
    model = ReformerModelWithLMHead.from_pretrained("google/reformer-enwik8")
    from collections import Counter
    for p in [0.25, 0.5, 0.75]: # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        encoded = read_data(p)
        exact_matches, total, partial_matches = Counter(), Counter(), Counter()
        idx = 0
        for prompt_tokens, target_tokens in encoded:
            encoded, attention_masks = encode([" ".join(prompt_tokens)])
            generated_text = decode(model.generate(encoded, do_sample=True, max_length=encoded.size(1)+100))[0]
            generated_tokens = generated_text.split()[len(prompt_tokens):]
            # compute match metrics
            #generated_tokens = "".join(generated_text).split()
            for suggestion_len in range(0, min(3, len(target_tokens))):
                exact_match = True
                for token_i in range(0, suggestion_len+1):
                    if len(generated_tokens) < token_i + 1 or generated_tokens[token_i] != target_tokens[token_i]:
                        exact_match = False
                        break
                if exact_match:
                    exact_matches[suggestion_len+1] += 1
                    #if suggestion_len > 2:
                    #    sys.exit(0)
                total[suggestion_len+1] += 1
                partial_matches[suggestion_len+1] += get_prefix_overlap_len(" ".join(generated_tokens[0:suggestion_len+1]), " ".join(target_tokens[0:suggestion_len+1]))
            print("Index: %d\nPrompt: %s\nGenerated Text: %s\nTarget Text: %s\n\n"%(idx, " ".join(prompt_tokens), " ".join(generated_tokens), " ".join(target_tokens)), flush=True)
            idx += 1
        res = ""
        for suggestion_len in range(1, len(total)+1):
            res += "%d: %.2f (%d/%d),"%(suggestion_len, float(exact_matches[suggestion_len])/total[suggestion_len] if total[suggestion_len]!= 0 else 0, exact_matches[suggestion_len], total[suggestion_len])
        print("context=%.2f %s"%(p, res), flush=True)
        res = ""
        for suggestion_len in range(1, len(total)+1):
            res += "%d: %.2f (%d/%d),"%(suggestion_len, float(partial_matches[suggestion_len])/total[suggestion_len] if total[suggestion_len]!= 0 else 0, partial_matches[suggestion_len], total[suggestion_len])
        print("context=%.2f %s"%(p, res), flush=True)
        # break

def get_prefix_overlap_len(string_a, string_b):
    num_match = 0
    for ai, bi in zip(string_a, string_b):
        if ai == bi:
            num_match += 1
        else:
            return float(num_match/len(string_b))
    return float(num_match/len(string_b))
    
#reformer_check()    
#exact_match_reformer()

def exact_match_pipeline(model):
    from transformers import pipeline
    generator = pipeline('text-generation', model='gpt2')
    from collections import Counter
    for p in [0.25, 0.5, 0.75]: # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        encoded = read_data(p)
        exact_matches, total, partial_matches = Counter(), Counter(), Counter()
        idx = 0
        for prompt_tokens, target_tokens in encoded:
            generated_text = generator(" ".join(prompt_tokens), max_length=int((1.5 * len(prompt_tokens))+20), do_sample=True, top_p=0.95, top_k=0, num_return_sequences=1)[0]
            generated_text = generated_text["generated_text"]
            generated_tokens = generated_text.split()[len(prompt_tokens):]
            # compute match metrics
            #generated_tokens = "".join(generated_text).split()
            for suggestion_len in range(0, min(3, len(target_tokens))):
                exact_match = True
                for token_i in range(0, suggestion_len+1):
                    if len(generated_tokens) < token_i + 1 or generated_tokens[token_i] != target_tokens[token_i]:
                        exact_match = False
                        break
                if exact_match:
                    exact_matches[suggestion_len+1] += 1
                    #if suggestion_len > 2:
                    #    sys.exit(0)
                total[suggestion_len+1] += 1
                partial_matches[suggestion_len+1] += get_prefix_overlap_len(" ".join(generated_tokens[0:suggestion_len+1]), " ".join(target_tokens[0:suggestion_len+1]))
            print("Index: %d\nPrompt: %s\nGenerated Text: %s\nTarget Text: %s\n\n"%(idx, " ".join(prompt_tokens), " ".join(generated_tokens), " ".join(target_tokens)), flush=True)
            idx += 1
        res = ""
        for suggestion_len in range(1, len(total)+1):
            res += "%d: %.2f (%d/%d),"%(suggestion_len, float(exact_matches[suggestion_len])/total[suggestion_len] if total[suggestion_len]!= 0 else 0, exact_matches[suggestion_len], total[suggestion_len])
        print("context=%.2f %s"%(p, res), flush=True)
        res = ""
        for suggestion_len in range(1, len(total)+1):
            res += "%d: %.2f (%d/%d),"%(suggestion_len, float(partial_matches[suggestion_len])/total[suggestion_len] if total[suggestion_len]!= 0 else 0, partial_matches[suggestion_len], total[suggestion_len])
        print("context=%.2f %s"%(p, res), flush=True)
        # break
exact_match_pipeline("gpt2")


def plot_exact_match_score():
    lines = []
    for line in open("/home/t-gjawahar/archai/out_reformer"):
        line = line.strip()
        if line.startswith("context="):
            items = line.split()[0:14]
            context = items[0].split("=")[-1]
            items = " ".join(items[1:]).split(",")
            lines.append("%s %s %s %s %s %s %s %s"%(context, items[0][3:].replace(" ", "-"), items[1][3:].replace(" ", "-"), items[2][3:].replace(" ", "-"), items[3][3:].replace(" ", "-"), items[4][3:].replace(" ", "-"), items[5][3:].replace(" ", "-"), items[6][3:].replace(" ", "-")))
    for i in range(2):
        print(",".join(lines[i].split()))
        #print(",".join(lines[7+i].split()))

#plot_exact_match_score()

def plot_learning_curve():
    import glob
    f2scores = {}
    for f in glob.glob("/home/t-gjawahar/object_dir/logs_vcorrected/vcorr*"):
        f2scores[f.split("/")[-1].split(".")[0]] = []
        for line in open(f):
            line = line.strip()
            if "valid ppl" in line:
                items = line.split()
                f2scores[f.split("/")[-1].split(".")[0]].append(float(items[-5]))
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10,5))
    plt.grid(color='gray', linestyle='dashed')
    xaxis = [10000*(i+1) for i in range(40)]
    colors = ['red', 'cyan', 'green', 'orange']
    plt.plot(xaxis, f2scores["vcorrbase"][0:40], color=colors[0], marker='o', label="base")
    plt.plot(xaxis, f2scores["vcorr1"][0:40], color=colors[1], marker='o', label="half layer")
    plt.plot(xaxis, f2scores["vcorr2"][0:40], color=colors[2], marker='o', label="half d_inner")
    plt.plot(xaxis, f2scores["vcorr3"][0:40], color=colors[3], marker='o', label="half d_model")
    #plt.plot(xaxis, f2scores["vcorr4"][0:40], color=colors[3], marker='o', label="half d_model, d_inner, layer")
    plt.xlabel("Steps")
    plt.ylabel("Valid Loss")
    plt.legend(loc="upper left")
    plt.show()
    fig.savefig("/home/t-gjawahar/object_dir/logs_vcorrected/%s_lcurve.pdf"%('full'), bbox_inches='tight')
#plot_learning_curve()