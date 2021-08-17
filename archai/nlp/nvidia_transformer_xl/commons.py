
# general utils

import torch
import glob, sys
import matplotlib.pyplot as plt

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
    print(sum(p.numel() for p in model.parameters()))
#reformer_check()

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
#exact_match_pipeline("gpt2")


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


# script generator

'''
#latency-vs-params
latency_vs_params_command = "CUDA_VISIBLE_DEVICES="" python archai/nlp/nvidia_transformer_xl/exact_match.py --work_dir /home/t-gjawahar/archai/amlt/transxl_char_exp2_randsearch/??? --data /home/t-gjawahar/dataroot --dataset wt2 --tgt_len 512 --mem_len 2000 --same_length --model /home/t-gjawahar/archai/amlt/transxl_char_exp2_randsearch/???/checkpoint_best.pt --experiment_name transxl_char_exp3_wikifull_select_v1corrected --cache_dir /home/t-gjawahar/dataroot --batch_size 1 --prompt_context_percent 0.5 --num_prompts 200 --num_chars_generate 100"
models = ["transxl_char_params_5M", "transxl_char_params_50M", "v1corrected", "transxl_char_base_lr_0p001", "transxl_char_params_80M", "transxl_char_params_100M", "transxl_char_params_200M", "transxl_char_large_lr_0p001", "transxl_char_large_lr_0p001_layer28", "transxl_char_large_lr_0p001_layer32"]
for num_chars in [100]: #25, 50, 100]:
    for model in models:
        print(latency_vs_params_command.replace("???", model).replace("--num_chars_generate 100", "--num_chars_generate %d"%num_chars))
sys.exit(0)
'''

'''
def plot_EB():
    import glob
    xaxis = []
    prefixlen2ebs = {"data_prefix": [], "model_prefix": [], "EB-M": []}
    lines = []
    for line in open("/home/t-gjawahar/archai/out_transxl_exposure_bias"):
        if "BLEU = " in line.strip():
            lines.append(line.strip())

    for i in range(5):
        data_prefix = float(lines[2*i].split()[2])
        model_prefix = float(lines[(2*i)+1].split()[2])
        prefix_len = 10*(i+1)
        xaxis.append(prefix_len)
        prefixlen2ebs["data_prefix"].append(data_prefix)
        prefixlen2ebs["model_prefix"].append(model_prefix)
        prefixlen2ebs["EB-M"].append(data_prefix/model_prefix)
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10,5))
    plt.grid(color='gray', linestyle='dashed')
    xaxis = [xaxis[i] for i in range(len(xaxis))]
    colors = ['red', 'cyan', 'green']
    plt.plot(xaxis, prefixlen2ebs["data_prefix"], color=colors[0], marker='o', label="data_prefix_bleu")
    plt.plot(xaxis, prefixlen2ebs["model_prefix"], color=colors[1], marker='o', label="model_prefix_bleu")
    plt.plot(xaxis, prefixlen2ebs["EB-M"], color=colors[2], marker='o', label="EB-M")
    plt.xlabel("Prefix Length")
    plt.ylabel("Score")
    plt.legend(loc="upper left")
    plt.show()
    fig.savefig("/home/t-gjawahar/object_dir/logs_vcorrected/%s_eb_8_3.pdf"%('full'), bbox_inches='tight')
plot_EB()
'''

def plot_latency_params():    
    # experiments
    exp2scores = {}
    lines = []
    for line in open("/home/t-gjawahar/archai/outputs/83_plots.out"):
        lines.append(line.strip())

    for i in range(7):
        exp_name = lines[4*i].strip()
        time = float(lines[(4*i)+1].split()[2])/float(lines[(4*i)+1].split()[6])
        perpl = float(lines[(4*i)+2])
        word_1 = float(lines[(4*i)+3].split()[0])
        word_2 = float(lines[(4*i)+3].split()[2])
        word_3 = float(lines[(4*i)+3].split()[4])
        exp2scores[exp_name] = [time, perpl, word_1, word_2, word_3]
    
    exp2goodname = {}
    exp2goodname["v1corrected"] = "char13M"
    exp2goodname["transxl_char_base_lr_0p001"] = "char41M"
    exp2goodname["transxl_char_params_80M"] = "char80M"
    exp2goodname["transxl_char_large_lr_0p001"] = "char277M"
    exp2goodname["transxl_char_large_lr_0p001_layer28"] = "char323M"
    exp2goodname["transxl_char_large_lr_0p001_layer32"] = "char369M"
    exp2goodname["word"] = "word80M"
    exp2goodname["transxl_char_params_5M"] = "char5M"
    exp2goodname["transxl_char_params_50M"] = "char50M"
    exp2goodname["transxl_char_params_100M"] = "char100M"

    '''
    # plot perpl vs num_word (old)
    colors = ['b', 'c', 'y', 'm', 'r', 'g', 'k']
    for num_word in [1, 2, 3]:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10,5))
        ei = 0
        points = []
        goodnames = []
        for exp in sorted(exp2goodname):
            goodname = exp2goodname[exp]
            pt = plt.scatter(exp2scores[exp][1], exp2scores[exp][1+num_word], marker='x', c=colors[ei])
            ei += 1
            points.append(pt)
            goodnames.append(goodname)
        plt.xlabel("Perplexity")
        plt.ylabel("Match@"+str(num_word))
        plt.legend(points, goodnames, scatterpoints=1, loc="upper right", fontsize=10)
        plt.show()
        fig.savefig("/home/t-gjawahar/object_dir/logs_vcorrected/%d_perpl_num_word.pdf"%(num_word), bbox_inches='tight')
        #break
    '''
    
    '''
    # plot perpl vs num_word (old)
    colors = ['b', 'c', 'y', 'm', 'r', 'g', 'k']
    for num_word in [1, 2, 3]:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10,5))
        ei = 0
        points = []
        goodnames = []
        for exp in sorted(exp2goodname):
            goodname = exp2goodname[exp]
            pt = plt.scatter(exp2scores[exp][0], exp2scores[exp][1+num_word], marker='x', c=colors[ei])
            ei += 1
            points.append(pt)
            goodnames.append(goodname)
        plt.xlabel("Latency")
        plt.ylabel("Match@"+str(num_word))
        plt.legend(points, goodnames, scatterpoints=1, loc="upper right")
        plt.show()
        fig.savefig("/home/t-gjawahar/object_dir/logs_vcorrected/%d_latency_num_word.pdf"%(num_word), bbox_inches='tight')
    '''

    # new experiments (greedy analysis)
    newexp2scores = {}
    import glob, sys
    for f in glob.glob("/home/t-gjawahar/object_dir/greedy_analysis/*"):
        exp_name = "char" + f.split("/")[-1].split(".")[0]
        word_1, word_2, word_3 = None, None, None
        time, ppl = None, None
        for line in open(f):
            line = line.strip()
            if "context=0.50" in line: # and not word_1:
                items = line.split()
                word_1 = float(items[3])
                word_2 = float(items[5])
                word_3 = float(items[7])
            elif "num_chars_generate = 100" in line:
                items = line.split()
                time = float(items[3])/200
            elif "Word-PPL" in line:
                items = line.split()
                ppl = float(items[-7])
        newexp2scores[exp_name] = [time, ppl, word_1, word_2, word_3]
    
    # plot perpl vs num_word
    colors = ['b', 'c', 'y', 'm', 'r', 'g', 'k', "indigo", "violet", "springgreen", "olive"]
    # cat outputs/8_3_word_level_diff_sampling.out
    exp2scores["word"][-3], exp2scores["word"][-2], exp2scores["word"][-1] = 0.28, 0.05, 0.02 # full match
    exp2scores["word"][-3], exp2scores["word"][-2], exp2scores["word"][-1] = 0.29, 0.16, 0.11 # partial match
    for num_word in [1, 2, 3]:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10,5))
        ei = 0
        points = []
        goodnames = []
        #fig, ax = plt.subplots()
        for goodname in sorted(newexp2scores):
            pt = plt.scatter(newexp2scores[goodname][1], newexp2scores[goodname][1+num_word], marker='x', c=colors[ei])
            #pt = plt.scatter(newexp2scores[goodname][1], newexp2scores[goodname][1+num_word])
            #ax.annotate(goodname, (newexp2scores[goodname][1], newexp2scores[goodname][1+num_word]))
            ei += 1
            points.append(pt)
            goodnames.append(goodname)
        pt = plt.scatter(exp2scores["word"][1], exp2scores["word"][1+num_word], marker='x', c=colors[ei])
        #pt = plt.scatter(exp2scores["word"][1], exp2scores["word"][1+num_word])
        #ax.annotate(goodname, (newexp2scores[goodname][1], newexp2scores[goodname][1+num_word]))
        points.append(pt)
        goodnames.append("word80M")
        
        plt.xlabel("Perplexity")
        plt.ylabel("Match@"+str(num_word))
        plt.legend(points, goodnames, scatterpoints=1, loc="upper right", fontsize=10)
        plt.show()
        fig.savefig("/home/t-gjawahar/object_dir/logs_vcorrected/%d_perpl_num_word_partial_greedy.pdf"%(num_word), bbox_inches='tight')
        #break
    sys.exit(0)
    
    # plot latency vs num_word (greedy)
    colors = ['b', 'c', 'y', 'm', 'r', 'g', 'k', "indigo", "violet", "springgreen", "olive"]
    models = ["transxl_char_params_5M", "transxl_char_params_50M", "v1corrected", "transxl_char_base_lr_0p001", "transxl_char_params_80M", "transxl_char_params_100M", "word"]
    scores = []
    for line in open("outputs/8-4-latency-size-greedy.out"):
        line = line.strip()
        if "Time = " in line:
            scores.append(float(line.split()[2])/200.0)
    scores.append(1299.0/200.0)
    newexp2scores["word80M"] = [None, None, 0.28, 0.05, 0.02]
    for num_word in [1, 2, 3]:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10,5))
        ei = 0
        points = []
        goodnames = []
        for color, model, score in zip(colors, models, scores):
            goodname = exp2goodname[model]
            pt = plt.scatter(score, newexp2scores[goodname][1+num_word], marker='x', c=colors[ei])
            ei += 1
            points.append(pt)
            goodnames.append(goodname)
        
        plt.xlabel("Latency")
        plt.ylabel("Match@"+str(num_word))
        plt.legend(points, goodnames, scatterpoints=1, loc="upper right", fontsize=10)
        plt.show()
        fig.savefig("/home/t-gjawahar/object_dir/logs_vcorrected/%d_latency_num_word_greedy.pdf"%(num_word), bbox_inches='tight')

layers_info = {""}

#plot_latency_params()



def plot_learning_curve2():
    import glob
    f2scores = {}
    for f in glob.glob("/home/t-gjawahar/object_dir/char_archi_modifications/80M*"):
        f2scores[f.split("/")[-1].split(".")[0]] = []
        for line in open(f):
            line = line.strip()
            if "valid ppl" in line:
                items = line.split()
                f2scores[f.split("/")[-1].split(".")[0]].append(float(items[-1]))
    max_val = -1
    for key in f2scores:
        print(key, len(f2scores[key]))
        max_val = max(max_val,  len(f2scores[key]))
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10,5))
    plt.grid(color='gray', linestyle='dashed')
    xaxis = [10000*(i+1) for i in range(40)]
    colors =['b', 'c', 'y', 'm', 'r', 'g', 'k', "indigo", "violet", "springgreen", "olive"]
    ei = 0
    for key in sorted(f2scores):
        if not ("mean" in key or "max" in key or "sum" in key or key=="80M"):
            continue
        if len(f2scores[key]) == 40:
            plt.plot(xaxis, f2scores[key][0:40], color=colors[ei], marker='o', label=key)
        else:
            scores = f2scores[key] + [None] * (40 - len( f2scores[key]))
            plt.plot(xaxis, scores, color=colors[ei], marker='o', label=key)
        ei += 1
    #plt.plot(xaxis, f2scores["80M"][0:40], color=colors[0], marker='o', label="char80M")
    #plt.plot(xaxis, f2scores["bertstyle_lr0p0001"][0:6], color=colors[1], marker='o', label="bertstyle_lr0.0001")
    #plt.plot(xaxis, f2scores["bertstyle_lr0p001"][0:6], color=colors[2], marker='o', label="bertstyle_lr0.001")
    #plt.plot(xaxis, f2scores["bertstyle_lr0p01"][0:6], color=colors[3], marker='o', label="bertstyle_lr0.01")
    #plt.plot(xaxis, f2scores["vcorr4"][0:40], color=colors[3], marker='o', label="half d_model, d_inner, layer")
    plt.xlabel("Steps")
    plt.ylabel("Valid BPC")
    plt.legend(loc="upper right")
    plt.show()
    #fig.savefig("/home/t-gjawahar/object_dir/char_archi_modifications/%s_bertstyle_lcurve.pdf"%('full'), bbox_inches='tight')
    fig.savefig("/home/t-gjawahar/object_dir/char_archi_modifications/%s_mean_max_sum_lcurve.pdf"%('full'), bbox_inches='tight')

#plot_learning_curve2()

def run_param_imp_char():
    res = ""
    for line in open("/home/t-gjawahar/archai/archai/nlp/nvidia_transformer_xl/run_param_imp_char.yaml"):
        if "name:" in line.strip() and "ms-shared" not in line.strip():
            content = line.split("name: ")[-1].strip()
            res += ":" + content + " "
    res = res.strip()
    print("amlt run archai/nlp/nvidia_transformer_xl/run_param_imp_char.yaml %s param_imp_char --upload-data"%res)
#run_param_imp_char()


def plot_word_level_models():
    newexp2scores = {}
    # add match scores
    for f in glob.glob("/home/t-gjawahar/object_dir/inference_word_match/word*"):
        exp_name = f.split("/")[-1].split(".")[0]
        word_1, word_2, word_3 = None, None, None
        partial_word_1, partial_word_2, partial_word_3 = None, None, None
        for line in open(f):
            line = line.strip()
            if "context=0.50" in line:
                items = line.split()
                if not word_1:
                    word_1 = float(items[3])
                    word_2 = float(items[5])
                    word_3 = float(items[7])
                elif not partial_word_1:
                    partial_word_1 = float(items[3])
                    partial_word_2 = float(items[5])
                    partial_word_3 = float(items[7])
        newexp2scores[exp_name] = [word_1, word_2, word_3, partial_word_1, partial_word_2, partial_word_3]
    # add ppl
    for f in glob.glob("/home/t-gjawahar/object_dir/perplexity_word_match/word*"):
        exp_name = f.split("/")[-1].split(".")[0]
        ppl = None
        for line in open(f):
            line = line.strip()
            if "valid ppl" in line:
                if line.strip().split()[-1] != "nan":
                    ppl = float(line.strip().split()[-1])
        newexp2scores[exp_name].append(ppl)
    # add char
    for f in glob.glob("/home/t-gjawahar/object_dir/greedy_analysis/*"):
        exp_name = "char" + f.split("/")[-1].split(".")[0]
        word_1, word_2, word_3 = None, None, None
        partial_word_1, partial_word_2, partial_word_3 = None, None, None
        time, ppl = None, None
        for line in open(f):
            line = line.strip()
            if "context=0.50" in line:
                items = line.split()
                if not word_1:
                    word_1 = float(items[3])
                    word_2 = float(items[5])
                    word_3 = float(items[7])
                elif not partial_word_1:
                    partial_word_1 = float(items[3])
                    partial_word_2 = float(items[5])
                    partial_word_3 = float(items[7])
            elif "num_chars_generate = 100" in line:
                items = line.split()
                time = float(items[3])/200
            elif "Word-PPL" in line:
                items = line.split()
                ppl = float(items[-7])
        newexp2scores[exp_name] = [word_1, word_2, word_3, partial_word_1, partial_word_2, partial_word_3, ppl]
    colors = ['b', 'c', 'y', 'm', 'r', 'g', 'k', "indigo", "violet", "springgreen", "olive", "firebrick", "gold"]
    for num_word in [1, 2, 3]:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10,5))
        plt.grid(color='gray', linestyle='dashed')
        xaxis = [5, 40, 50, 80, 100, 200]
        plt.plot(xaxis, [newexp2scores["char"+model][3+num_word-1] for model in ["5M", "41M", "50M", "80M", "100M", "200M"]], color=colors[0], marker='o', label="char")
        plt.plot(xaxis, [newexp2scores["word"+model][3+num_word-1] for model in ["5M", "40M", "50M", "80M", "100M", "200M"]], color=colors[1], marker='x', label="word")
        plt.xlabel("# parameters")
        plt.ylabel("Match@"+str(num_word))
        plt.legend(loc="upper left")
        plt.show()
        fig.savefig("/home/t-gjawahar/object_dir/inference_word_match/%d_params_num_word_partial_greedy.pdf"%(num_word), bbox_inches='tight')

    def check_within_paramset(modelname):
        for param in ["5M", "41M", "50M", "80M", "100M", "200M", "40M"]:
            if modelname.endswith(param):
                return True
        return False
    
    for num_word in [1, 2, 3]:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10,5))
        ei = 0
        points = []
        goodnames = []
        #fig, ax = plt.subplots()
        for goodname in sorted(newexp2scores):
            if not check_within_paramset(goodname):
                continue
            pt = plt.scatter(newexp2scores[goodname][-1], newexp2scores[goodname][num_word-1], marker='x' if "char" in goodname else 'o', c=colors[ei])
            ei += 1
            points.append(pt)
            goodnames.append(goodname)
        plt.xlabel("Perplexity")
        plt.ylabel("Match@"+str(num_word))
        plt.legend(points, goodnames, scatterpoints=1, loc="upper right", fontsize=10)
        plt.show()
        fig.savefig("/home/t-gjawahar/object_dir/inference_word_match/%d_perpl_num_word_full_greedy_wvsch.pdf"%(num_word), bbox_inches='tight')

#plot_word_level_models()

def read_all_param_imp_char():
    import os
    output_master_folder = "/".join(os.environ["AMLT_OUTPUT_DIR"].split("/")[0:-1])
    results_f = os.environ["AMLT_OUTPUT_DIR"] + "/results.csv"
    results_file_w = open(results_f, "w")
    print(glob.glob(output_master_folder + "/*"))
    for folder in glob.glob(output_master_folder + "/*"):
        if os.path.exists(folder + "/train_log.log"):
            valid_f = None
            lines = []
            for line in open(folder + "/train_log.log"):
                if "param_imp_char" in line.strip():
                    valid_f = True
                    results_file_w.write(line.strip()+"\n")
                if "valid ppl" in line.strip():
                    #results_file_w.write(line.strip()+"\n")
                    lines.append(line.strip()+"\n")
            if valid_f:
                results_file_w.write(folder+"\n")
                for line in lines:
                    results_file_w.write(line+"\n")
    results_file_w.close()

#read_all_param_imp_char()

def char_expoure_bias():
    newexp2scores = {}
    params = []
    for f in glob.glob("/home/t-gjawahar/object_dir/char_exposure_bias/*M.*"):
        exp_name = "char" + f.split("/")[-1].split(".")[0]
        eb = []
        for line in open(f):
            line = line.strip()
            if "BLEU = " in line:
                eb.append(float(line.split("BLEU = ")[-1].split()[0]))
        assert(len(eb) == 2)
        newexp2scores[exp_name] = [eb[0], eb[1], eb[0]/eb[1]]
        params.append(int(f.split("/")[-1].split(".")[0][0:-1]))
    colors = ['b', 'c', 'y', 'm', 'r', 'g', 'k', "indigo", "violet", "springgreen", "olive"]
    for idx, name in [(-1, "EB-M"), (0, "DataPref"), (1, "ModelPref")]:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10,5))
        plt.grid(color='gray', linestyle='dashed')
        params.sort()
        xaxis = params
        yaxis = []
        for param in xaxis:
            yaxis.append(newexp2scores["char"+str(param)+"M"][idx])
        plt.plot(xaxis, yaxis, color=colors[0], marker='o', label=name)
        plt.xlabel("# parameters")
        plt.ylabel(name)
        plt.legend(loc="upper left")
        plt.show()
        fig.savefig("/home/t-gjawahar/object_dir/char_exposure_bias/params_vs_%s.pdf"%name, bbox_inches='tight')

#char_expoure_bias()

# generate commands for word, char models w. test accuracy, latency, peak mem. utilization
def gen_commands_for_metrics_sandbox():
    word_accuracy = "python archai/nlp/nvidia_transformer_xl/exact_match.py --work_dir <WORK-DIR> --dataset wt2 --tgt_len 192 --mem_len 192 --same_length --model <WORK-DIR>/checkpoint_best.pt --experiment_name word_accuracy --batch_size 128 --prompt_context_percent 0.5 --cuda"
    char_accuracy = "python archai/nlp/nvidia_transformer_xl/exact_match.py --work_dir <WORK-DIR> --dataset wt2 --tgt_len 512 --mem_len 2000 --same_length --model <WORK-DIR>/checkpoint_best.pt --experiment_name char_accuracy --batch_size 128 --prompt_context_percent 0.5 --cuda"
    word_latency = "CUDA_VISIBLE_DEVICES= python archai/nlp/nvidia_transformer_xl/exact_match.py --work_dir <WORK-DIR> --dataset wt2 --tgt_len 192 --mem_len 192 --same_length --model <WORK-DIR>/checkpoint_best.pt --experiment_name word_latency --batch_size 1  --prompt_context_percent 0.5 --num_prompts 100 --num_chars_generate 100"
    char_latency = "CUDA_VISIBLE_DEVICES= python archai/nlp/nvidia_transformer_xl/exact_match.py --work_dir <WORK-DIR> --dataset wt2 --tgt_len 512 --mem_len 2000 --same_length --model <WORK-DIR>/checkpoint_best.pt --experiment_name char_latency --batch_size 1 --prompt_context_percent 0.5 --num_prompts 100 --num_chars_generate 100"
    word_memutil = "CUDA_VISIBLE_DEVICES= python archai/nlp/nvidia_transformer_xl/exact_match.py --work_dir <WORK-DIR> --dataset wt2 --tgt_len 192 --mem_len 192 --same_length --model <WORK-DIR>/checkpoint_best.pt --experiment_name word_memutil --batch_size 1  --prompt_context_percent 0.5 --num_prompts 5 --num_chars_generate 100 --memstat"
    char_pemutil = "CUDA_VISIBLE_DEVICES= python archai/nlp/nvidia_transformer_xl/exact_match.py --work_dir <WORK-DIR> --dataset wt2 --tgt_len 512 --mem_len 2000 --same_length --model <WORK-DIR>/checkpoint_best.pt --experiment_name char_pemutil --batch_size 1 --prompt_context_percent 0.5 --num_prompts 5 --num_chars_generate 100 --memstat"

    # 8-12
    WORD_MODEL_DIRS = ["/home/t-gjawahar/archai/amlt/word_train/word40M_nofp", "/home/t-gjawahar/archai/amlt/word_train/word50M_nofp", "/home/t-gjawahar/archai/amlt/word_train/word80M_nofp"]
    CHAR_MODEL_DIRS = ["/home/t-gjawahar/archai/amlt/transxl_char_exp2_randsearch/transxl_char_base", "/home/t-gjawahar/archai/amlt/transxl_char_exp2_randsearch/transxl_char_params_50M", "/home/t-gjawahar/archai/amlt/transxl_char_exp2_randsearch/transxl_char_params_80M"]

    # 8-12-enron
    WORD_MODEL_DIRS = ["/home/t-gjawahar/archai/amlt/word_train/word80M_nofp_enron"]
    CHAR_MODEL_DIRS = ["/home/t-gjawahar/archai/amlt/word_train/char80M_nofp_enron"]

    exp_dir = "/home/t-gjawahar/archai/scripts/8-12-enron"
    for name, cmd, mdirs in [('word_accuracy', word_accuracy, WORD_MODEL_DIRS), ('char_accuracy', char_accuracy, CHAR_MODEL_DIRS)]: #, ('word_latency', word_latency, WORD_MODEL_DIRS), ('char_latency', char_latency, CHAR_MODEL_DIRS), ('word_pemutil', word_pemutil, WORD_MODEL_DIRS), ('char_pemutil', char_pemutil, CHAR_MODEL_DIRS)]:
        print("bash %s/%s.sh > %s/%s.out"%(exp_dir, name, exp_dir, name))
        w = open("%s/%s.sh"%(exp_dir, name), "w")
        for mdir in mdirs:
            w.write(cmd.replace("<WORK-DIR>", mdir))
            w.write("\n")
        w.close()
#gen_commands_for_metrics_sandbox()

def plot_acc_lat_mem():
    exp_dir = "/home/t-gjawahar/archai/scripts/8-12"
    models = ["40M", "50M", "80M"]

    # read accuracy (fullMatch-1)
    model2scores = {}
    for typ in ["char", "word"]:
        scores = []
        for line in open(exp_dir + "/" + typ + "_accuracy.out"):
            line = line.strip()
            if "context=" in line:
                scores.append(float(line.split()[2]))
        j = 0
        for i in range(0, len(scores), 2): # get only fullMatch
            model2scores[typ+"_"+models[j]] = [scores[i]]
            j += 1
    
    # read latency
    for typ in ["char", "word"]:
        scores = []
        for line in open(exp_dir + "/" + typ + "_latency.out"):
            line = line.strip()
            if "Time" in line:
                scores.append(float(line.split()[2]))
        for si, score in enumerate(scores):
            model2scores[typ+"_"+models[si]].append(scores[si])
    
    # read peak memory utilization
    for typ in ["char", "word"]:
        scores = []
        cur_max = None
        for line in open(exp_dir + "/" + typ + "_pemutil.out"):
            line = line.strip()
            if "program_start" in line:
                if cur_max:
                    scores.append(cur_max)
                cur_max = float(line.split()[-1])
            elif "memstat output of" in line:
                cur_max = max(cur_max, float(line.split()[-1]))     
        if cur_max:
            scores.append(cur_max)
        for si, score in enumerate(scores):
            model2scores[typ+"_"+models[si]].append(scores[si])
    
    # plot bar plots
    # https://www.datasciencemadesimple.com/bar-plot-bar-chart-in-python-legend-using-matplotlib/
    import numpy as np
    import matplotlib.pyplot as plt
    colors = ['forestgreen', 'indianred']
    for metric_idx, metric_name in enumerate(["accuracy", "latency", "peakmem"]):
        data = [[], []]
        for ti, typ in enumerate(["char", "word"]):
            for mi, model in enumerate(models):
                data[ti].append(model2scores[typ+"_"+model][metric_idx])
        X = np.arange(len(models))
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        offset = 0.00
        for ti in range(len(data)):
            ax.bar(X + offset, data[ti], color = colors[ti], width = 0.25, edgecolor='black')
            offset += 0.25
        plt.xticks(X+0.125, models)
        plt.xlabel("#params", fontsize=16)
        plt.ylabel(metric_name, fontsize=16)
        #plt.title("#params vs. %s"%metric_name)    
        plt.legend(["char", "word"], loc="upper center")
        plt.show()
        fig.savefig(exp_dir + "/bar_%s.pdf"%metric_name, bbox_inches='tight')
        #if metric_idx == 1:
        #    break

#plot_acc_lat_mem()


def plot_enron_stats():
    # match results

    char_results = []
    for line in open("/home/t-gjawahar/archai/scripts/8-12-enron/char_accuracy.out"):
        content = line.strip()
        if "context=0.50" in content:
            items = content.split()
            res = []
            for i in range(10):
                res.append(float(items[2* (i+1)]))
            char_results.append(res)
    
    word_results = []
    for line in open("/home/t-gjawahar/archai/scripts/8-12-enron/word_accuracy.out"):
        content = line.strip()
        if "context=0.50" in content:
            items = content.split()
            res = []
            for i in range(10):
                res.append(float(items[2* (i+1)]))
            word_results.append(res)
    
    import matplotlib.pyplot as plt

    for i in range(2):
        fig = plt.figure(figsize=(10,5))
        plt.grid(color='gray', linestyle='dashed')
        xaxis = [i for i in range(10)]
        colors = ['red', 'cyan', 'green', 'orange']
        plt.plot(xaxis, char_results[i], color=colors[0], marker='o', label="char80M")
        plt.plot(xaxis, word_results[i], color=colors[1], marker='x', label="word80M")
        plt.xlabel("#words")
        plt.ylabel("%s Match"%("Full" if i == 0 else "Partial"))
        plt.legend(loc="upper left")
        plt.show()
        fig.savefig("/home/t-gjawahar/archai/scripts/8-12-enron/%s_accuracy.pdf"%("Full" if i == 0 else "Partial"), bbox_inches='tight')

#plot_enron_stats()

def count_num_tokens():
    num_tokens = 0
    for line in open("/home/t-gjawahar/object_dir//wikitext-2-raw-v1-char/wiki.train.tokens"):
        content = line.strip()
        num_tokens += len(content.split())
    print(num_tokens)

#count_num_tokens()

def process_reddit():
    src_dir = "/home/t-gjawahar/object_dir/data_v2_splitted_twoturn_ner_non_offensive"
    out_dir = "/home/t-gjawahar/object_dir/reddit_non_offensive"
    import nltk, json
    for src_f, dest_f in [("train.ljson.thread", "wiki.train.tokens"), ("dev.ljson.thread", "wiki.valid.tokens"), ("test.ljson.thread", "wiki.test.tokens")]:
        w = open(out_dir + "/" + dest_f, "w")
        for line in open(src_dir + "/" + src_f):
            line = json.loads(line.strip())["current"]
            tokens = nltk.word_tokenize(line)
            w.write(" %s \n"%(" ".join(tokens)))
        w.close()
#process_reddit()

def check_reddit():
    out_dir = "/home/t-gjawahar/object_dir/reddit_non_offensive"
    for f in glob.glob(out_dir + "/*train.tokens.full"): # 800M words
        chars = {}
        num_words = 0
        for line in open(f):
            for ch in line.strip():
                chars[ch] = True
            num_words += len(line.strip().split())
        print(len(chars))
        print(chars)
        print(num_words)
#check_reddit()

def check_param_imp_char():
    char_results = {}
    for f in glob.glob("/home/t-gjawahar/object_dir/param_imp_char/80M*"):
        for line in open(f):
            content = line.strip()
            if "context=0.50" in content:
                items = content.split()
                res = []
                for i in range(3):
                    res.append(float(items[1 + 2* (i+1)]))
                char_results[f.split("/")[-1].split(".")[0]] = res
                break
    keys = sorted(list(char_results.keys()))
    print(keys)
    
    '''
    # plot bar plots
    # https://www.datasciencemadesimple.com/bar-plot-bar-chart-in-python-legend-using-matplotlib/
    import numpy as np
    import matplotlib.pyplot as plt
    colors = ['forestgreen']
    for metric_idx, metric_name in enumerate(["accuracy"]):
        data = [[]]
        for ti, typ in enumerate(["char"]):
            for key in keys:
                data[ti].append(char_results[key][0])
        X = np.arange(len(keys))
        fig = plt.figure(figsize=(12,5))
        ax = fig.add_axes([0, 0, 1, 1])
        offset = 0.00
        for ti in range(len(data)):
            ax.bar(X + offset, data[ti], color = colors[ti], width = 0.25, edgecolor='black')
            offset += 0.25
        plt.xticks(X+0.05, [ "def" if key == "80M" else "-".join(key.split("_")[1:]) for key in keys])
        plt.xlabel("hyperparameter changes", fontsize=16)
        plt.ylabel(metric_name, fontsize=16)
        #plt.title("#params vs. %s"%metric_name)    
        plt.legend(["char"], loc="upper center")
        plt.show()
        fig.savefig("/home/t-gjawahar/object_dir/param_imp_char/bar_%s.pdf"%metric_name, bbox_inches='tight')
        #if metric_idx == 1:
        #    break
    '''

check_param_imp_char()
sys.exit(0)

def plot_reddit_stats():
    # match results

    char_results = []
    char_lines = ["context=0.50 1: 0.13 (63/500),2: 0.02 (11/500),3: 0.00 (1/481),", "context=0.50 1: 0.14 (68/500),2: 0.07 (36/500),3: 0.04 (21/481),"]
    for line in char_lines:
        content = line.strip()
        if "context=0.50" in content:
            items = content.split()
            res = []
            for i in range(3):
                res.append(float(items[2* (i+1)]))
            char_results.append(res)
    
    word_results = []
    word_lines = ["context=0.50 1: 0.14 (71/500),2: 0.03 (14/500),3: 0.00 (1/481),", "context=0.50 1: 0.16 (77/500),2: 0.08 (41/500),3: 0.05 (25/481),"]
    for line in word_lines:
        content = line.strip()
        if "context=0.50" in content:
            items = content.split()
            res = []
            for i in range(3):
                res.append(float(items[2* (i+1)]))
            word_results.append(res)
    
    import matplotlib.pyplot as plt

    for i in range(2):
        fig = plt.figure(figsize=(10,5))
        plt.grid(color='gray', linestyle='dashed')
        xaxis = [i+1 for i in range(3)]
        colors = ['red', 'cyan', 'green', 'orange']
        plt.plot(xaxis, char_results[i], color=colors[0], marker='o', label="char80M")
        plt.plot(xaxis, word_results[i], color=colors[1], marker='x', label="word80M")
        plt.xlabel("#words")
        plt.ylabel("%s Match"%("Full" if i == 0 else "Partial"))
        plt.legend(loc="upper right")
        plt.show()
        fig.savefig("/home/t-gjawahar/archai/scripts/%s_reddit_accuracy.pdf"%("Full" if i == 0 else "Partial"), bbox_inches='tight')

#plot_reddit_stats()


def tokenize_office_dataset():
    import nltk
    for split in ["train", "valid"]: #, "test"]:
        w = open("/home/t-gjawahar/object_dir/WordData20210110/gan/tokenized_%s.txt"%split, "w")
        for line in open("/home/t-gjawahar/object_dir/WordData20210110/WordData20210110/%s.txt"%split):
            line = line.strip()
            if len(line) != 0:
                tokens = nltk.word_tokenize(line)
                w.write(" %s \n"%(" ".join(tokens)))
        w.close()

#tokenize_office_dataset()

def plot_office_acc_lat_mem():
    '''
    exp_dir = "/home/t-gjawahar/archai/scripts/8-16"
    models = ["40M", "50M", "80M"]

    # read accuracy (fullMatch-1)
    model2scores = {}
    model2partial_scores = {}
    for typ in ["char", "word"]:
        scores = []
        for line in open(exp_dir + "/" + typ + "_accuracy.out"):
            line = line.strip()
            if "context=" in line:
                #scores.append(float(line.split()[2]))
                items = line.split()
                res = []
                for i in range(3):
                    res.append(float(items[2* (i+1)]))
                scores.append(res)
        j = 0
        for i in range(0, len(scores), 2): # get only fullMatch
            model2scores[typ+"_"+models[j]] = scores[i]
            j += 1
        j = 0
        for i in range(1, len(scores), 2): # get only partialMatch
            model2partial_scores[typ+"_"+models[j]] = scores[i]
            j += 1
    '''

    exp_f = "/home/t-gjawahar/archai/scripts/8-16/reddit_models_on_office.out"
    models = ["80M"]
    model2scores = {}
    model2partial_scores = {}
    scores = []
    for line in open(exp_f):
        line = line.strip()
        if "context=" in line:
            items = line.split()
            res = []
            for i in range(3):
                res.append(float(items[2* (i+1)]))
            scores.append(res)
    print(scores)
    j = 0
    for i in range(0, len(scores), 2): # get only fullMatch
        typ = "word_80M" if i == 0 else "char_80M"
        model2scores[typ] = scores[i]
        j += 1
    j = 0
    for i in range(1, len(scores), 2): # get only partialMatch
        typ = "word_80M" if i == 1 else "char_80M"
        model2partial_scores[typ] = scores[i]
        j += 1

    for i in range(2):
        fig = plt.figure(figsize=(10,5))
        plt.grid(color='gray', linestyle='dashed')
        xaxis = [i+1 for i in range(3)]
        colors = ['red', 'cyan', 'green', 'orange', "indigo", "violet", "springgreen", "olive"]
        ci = 0
        for model in models:
            if i == 0:
                plt.plot(xaxis, [model2scores["char_"+model][j] for j in range(3)], color=colors[ci], marker='o', label="char_"+model)
                plt.plot(xaxis, [model2scores["word_"+model][j] for j in range(3)], color=colors[ci+1], marker='x', label="word_"+model)
            else:
                plt.plot(xaxis, [model2partial_scores["char_"+model][j] for j in range(3)], color=colors[ci], marker='o', label="char_"+model)
                plt.plot(xaxis, [model2partial_scores["word_"+model][j] for j in range(3)], color=colors[ci+1], marker='x', label="word_"+model)
            ci += 2
        plt.xlabel("#words")
        plt.ylabel("%s Match"%("Full" if i == 0 else "Partial"))
        plt.legend(loc="upper right")
        plt.show()
        #fig.savefig("/home/t-gjawahar/archai/scripts/8-16/%s_wikitrain_office_accuracy.pdf"%("Full" if i == 0 else "Partial"), bbox_inches='tight')
        fig.savefig("/home/t-gjawahar/archai/scripts/8-16/%s_reddit_office_accuracy.pdf"%("Full" if i == 0 else "Partial"), bbox_inches='tight')


#plot_office_acc_lat_mem()


