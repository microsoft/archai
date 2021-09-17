
# general utils

from math import radians
from matplotlib.colors import cnames
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

def check_num_emb_params():
    from transformers import AutoTokenizer, AutoModelWithLMHead  
    tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    model = AutoModelWithLMHead.from_pretrained("gpt2-large")
    print(sum(p.numel() for p in model.parameters()))

#check_num_emb_params()

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


# amlt log transxl_char_exp2_randsearch :transxl_char_params_80M_clean_vocab :transxl_char_params_80M :transxl_char_params_80M_bertstyle_lr0p001 :transxl_char_params_80M_bertstyle_lr0p01_restart_10K :transxl_char_params_80M_char_emb_from_word_max_lr0p001_g4 :transxl_char_params_80M_char_emb_from_word_mean_lr0p001_g4 :transxl_char_params_80M_char_emb_from_word_sum_lr0p001_g4         
def plot_learning_curve2():
    import glob
    f2scores = {}
    '''
    for f in glob.glob("/home/t-gjawahar/object_dir/char_archi_modifications/80M*"):
        f2scores[f.split("/")[-1].split(".")[0]] = []
        for line in open(f):
            line = line.strip()
            if "valid ppl" in line:
                items = line.split()
                f2scores[f.split("/")[-1].split(".")[0]].append(float(items[-1]))
    '''
    for f in glob.glob("/home/t-gjawahar/archai/amlt/transxl_char_exp2_randsearch/transxl_char_params_80M*/stdout.txt"):
        f2scores[f.split("/")[-2].split("transxl_char_params_")[-1]] = []
        for line in open(f):
            line = line.strip()
            if "valid ppl" in line:
                items = line.split()
                f2scores[f.split("/")[-2].split("transxl_char_params_")[-1]].append(float(items[-1]))
        print(f.split("/")[-2], len(f2scores[f.split("/")[-2].split("transxl_char_params_")[-1]]))
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
        #if not ("mean" in key or "max" in key or "sum" in key or key=="80M"):
        #    continue
        if ("mean" in key or "max" in key or "sum" in key or "restart" in key or "clean" in key):
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
    plt.ylabel("Valid Loss")
    plt.legend(loc="upper right")
    plt.show()
    fig.savefig("/home/t-gjawahar/object_dir/char_archi_modifications/%s_bertstyle_lcurve.pdf"%('full'), bbox_inches='tight')
    #fig.savefig("/home/t-gjawahar/object_dir/char_archi_modifications/%s_mean_max_sum_lcurve.pdf"%('full'), bbox_inches='tight')

#plot_learning_curve2()
#sys.exit(0)

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

# amlt logs param_imp_char :transxl_char_params_80M :transxl_char_params_80M_dembed1000 :transxl_char_params_80M_dembed500 :transxl_char_params_80M_dhead32 :transxl_char_params_80M_dinner512 :transxl_char_params_80M_layer14 :transxl_char_params_80M_layer18 :transxl_char_params_80M_layer20 :transxl_char_params_80M_m1024_t256 :transxl_char_params_80M_m768_t256 :transxl_char_params_80M_nhead12 :transxl_char_params_80M_nhead2 :transxl_char_params_80M_nhead4
def check_param_imp_char():
    # read accuracy
    char_results = {}
    '''
    for f in glob.glob("/home/t-gjawahar/object_dir/param_imp_char/80M*"):
        for line in open(f):
            content = line.strip()
            if "context=0.50" in content:
                items = content.split()
                res = []
                for i in range(3):
                    res.append(float(items[1 + 2* (i+1)]))
                char_results[f.split("/")[-1].split(".")[0]] = [res]
                break
    '''

    for f in glob.glob("/home/t-gjawahar/archai/amlt/inference_transxl_char/inference_char-transxl_char_params_80M*/stdout.txt"):
        for line in open(f):
            content = line.strip()
            if "context=0.50" in content:
                items = content.split()
                res = []
                for i in range(3):
                    res.append(float(items[2 + 2* (i+1)]))
                char_results[f.split("/")[-2].split("inference_char-transxl_char_params_")[-1]] = [res]
                #break

    keys = sorted(list(char_results.keys()))
    print(keys)
    
    # read embed params, non embed params, valid ppl
    for f in glob.glob("/home/t-gjawahar/archai/amlt/param_imp_char/transxl_char_params_80M*/stdout.txt"):
        key = f.split("/")[-2].split("transxl_char_params_")[-1]
        valid_loss, n_all_param, n_nonemb_param, eval_step = None, None, None, None
        for line in open(f):
            line = line.strip()
            if "valid loss" in line:
                if int(line.split()[4]) > 20:
                    break
                eval_step = int(line.split()[4])
                valid_loss = float(line.split()[-5])
            elif "n_all_param" in line:
                n_all_param = int(line.split()[-1])
            elif "n_nonemb_param" in line:
                n_nonemb_param = int(line.split()[-1])
        if eval_step == 20:
            char_results[key] += [valid_loss, n_all_param, n_nonemb_param]
        if eval_step == 20:
            print(key, [char_results[key][0][0], valid_loss, n_all_param, n_nonemb_param], eval_step)
    
    # plot scatter plot
    colors = ['b', 'c', 'y', 'm', 'r', 'g', 'k', "indigo", "violet", "springgreen", "olive", "firebrick", "gold"]
    for i, term in enumerate(["Valid Loss.", "#all_params", "#nonemb_params"]):
        fig = plt.figure(figsize=(10,5))
        ei = 0
        points, names = [], []
        for key in keys:
            if key in char_results and len(char_results[key]) == 4:
                pt = plt.scatter(char_results[key][0][0], char_results[key][1+i], marker='x', c=colors[ei])
                ei += 1
                points.append(pt)
                names.append(key)
        plt.xlabel("PartialMatch@1")
        plt.ylabel(term)
        plt.legend(points, names, scatterpoints=1, loc="upper right", fontsize=10, bbox_to_anchor=(1.25, 0.55))
        plt.show()
        fig.savefig("/home/t-gjawahar/object_dir/param_imp_char/PartialMatch1_vs_%s_accuracy.pdf"%(term), bbox_inches='tight')
    
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

#check_param_imp_char()
#sys.exit(0)

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
        #plt.grid(color='gray', linestyle='dashed')
        xaxis = [i+1 for i in range(3)]
        colors = ['red', 'cyan', 'green', 'orange']
        #plt.plot(xaxis, char_results[i], color=colors[0], marker='o', label="char80M")
        #plt.plot(xaxis, word_results[i], color=colors[1], marker='x', label="word80M")
        plt.plot(xaxis, [526.0/1827.0, 167.0/1815.0, 51.0/1789.0], color=colors[0], marker='o', label="char80M")
        plt.plot(xaxis, [498.0/1827.0, 136.0/1815.0, 40.0/1789.0], color=colors[1], marker='x', label="word80M")
        plt.xlabel("N")
        plt.ylabel("%sMatch@N"%("Exact" if i == 0 else "Partial"))
        #plt.title("ExactMatch@N vs. N - Reddit")
        plt.title("ExactMatch@N vs. N - Wikitext")
        plt.legend(loc="upper right")
        plt.show()
        fig.savefig("/home/t-gjawahar/archai/scripts/%s_wikitext_accuracy.pdf"%("Full" if i == 0 else "Partial"), bbox_inches='tight')
        break

#plot_reddit_stats()
#sys.exit(0)

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


def plot_small_models_analysis():
    # read accuracy
    char_results = {}
    percent = "0.20"
    fullmatch_n = 2
    for f in glob.glob("/home/t-gjawahar/archai/amlt/inference_transxl/inference_char_valid-small_*/stdout.txt"):
        for line in open(f):
            content = line.strip()
            if "context="+percent in content:
                items = content.split()
                #res = []
                #for i in range(3):
                #    res.append(float(items[2 + 2* (i+1)]))
                score_1 = float(items[5].split(",")[0][1:-1].split("/")[0]) /  float(items[5].split(",")[0][1:-1].split("/")[1])
                score_2 = float(items[7].split(",")[0][1:-1].split("/")[0]) /  float(items[7].split(",")[0][1:-1].split("/")[1])
                score_3 = float(items[-1].split(",")[0][1:-1].split("/")[0]) /  float(items[-1].split(",")[0][1:-1].split("/")[1])
                score_1, score_2, score_3 = float("%.2f"%(100.0*score_1)), float("%.2f"%(100.0*score_2)), float("%.2f"%(100.0*score_3))
                char_results[f.split("/")[-2].split("inference_char_valid-small_")[-1]] =  [[score_1, score_2, score_3]]
                break
    print(char_results)

    keys = sorted(list(char_results.keys()))
    print(keys)
    
    # read embed params, non embed params, valid ppl
    for f in glob.glob("/home/t-gjawahar/archai/amlt/transxl_char_exp2_randsearch/small_*/stdout.txt"):
        key = f.split("/")[-2].split("small_")[-1]
        valid_loss, n_all_param, n_nonemb_param, eval_step = None, None, None, None
        for line in open(f):
            line = line.strip()
            if "valid loss" in line:
                #if int(line.split()[4]) > 20:
                #    break
                eval_step = int(line.split()[4])
                valid_loss = float(line.split()[-5])
            elif "n_all_param" in line:
                n_all_param = int(line.split()[-1])
            elif "n_nonemb_param" in line:
                n_nonemb_param = int(line.split()[-1])
        
        #if eval_step == 20:
        char_results[key] += [valid_loss, n_all_param, n_nonemb_param]
        #if eval_step == 20:
        print(key, [char_results[key][0][0], valid_loss, n_all_param, n_nonemb_param], eval_step)
    print(char_results)

    # word results
    word_results = {}
    for f in glob.glob("/home/t-gjawahar/archai/amlt/inference_word/inference_word_model_metrics-word*_nofp/stdout.txt"):
        for line in open(f):
            content = line.strip()
            if "context=0.50" in content:
                items = content.split()
                res = []
                for i in range(3):
                    res.append(float(items[2 + 2* (i+1)]))
                print(f.split("/")[-2])
                score_1 = float(items[5].split(",")[0][1:-1].split("/")[0]) /  float(items[5].split(",")[0][1:-1].split("/")[1])
                score_2 = float(items[7].split(",")[0][1:-1].split("/")[0]) /  float(items[7].split(",")[0][1:-1].split("/")[1])
                score_3 = float(items[-1].split(",")[0][1:-1].split("/")[0]) /  float(items[-1].split(",")[0][1:-1].split("/")[1])
                score_1, score_2, score_3 = float("%.2f"%(100.0*score_1)), float("%.2f"%(100.0*score_2)), float("%.2f"%(100.0*score_3))
                word_results[f.split("/")[-2].split("inference_word_model_metrics-word")[-1].split("_")[0]] = [[score_1, score_2, score_3]]
                break
    print(word_results)

    # read embed params, non embed params, valid ppl
    for f in glob.glob("/home/t-gjawahar/archai/amlt/word_train/word*_nofp/stdout.txt"):
        key = f.split("/")[-2].split("word")[-1].split("_")[0]
        valid_loss, n_all_param, n_nonemb_param, eval_step = None, None, None, None
        valid_ppl = None
        for line in open(f):
            line = line.strip()
            if "valid loss" in line:
                #if int(line.split()[4]) > 20:
                #    break
                eval_step = int(line.split()[4])
                valid_loss = float(line.split()[-5])
                valid_ppl = float(line.split()[-1])
            elif "n_all_param" in line:
                n_all_param = int(line.split()[-1])
            elif "n_nonemb_param" in line:
                n_nonemb_param = int(line.split()[-1])
        
        #if eval_step == 20:
        word_results[key] += [valid_loss, n_all_param, n_nonemb_param, valid_ppl]
        #if eval_step == 20:
        #print(key, [char_results[key][0][0], valid_loss, n_all_param, n_nonemb_param], eval_step)
    print(word_results)
    wordkeys = sorted(list(word_results.keys()))

    # plot scatter plot
    colors = ['b', 'c', 'y', 'm', 'r', 'g', 'k', "indigo", "violet", "springgreen", "olive", "firebrick", "gold", "darkorange", "teal", "slategrey", "crimson", "peru", "olive", "dimgray"]
    markers = {"5M": "x", "10M": "o", "20M": "*", "30M": "<", "40M": "+", "80M": "D", "50M": "s"}
    word_layers = {"5M": 3 , "10M": 4, "20M": 6, "30M": 8, "40M": 14, "50M": 16, "80M": 16}
    embed_size = {"5M": 18 , "10M": 36, "20M": 74, "30M": 100, "40M": 128, "50M": 160, "80M": 256}
    subword_results = {"0.20": [21.73, 6.34, 1.53], "0.50": [25.23, 6.67, 1.96], "0.80": [28.83, 8.79, 3.03]}
    for i, term in enumerate(["Valid Loss.", "#all_params", "#nonemb_params"]):
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(111)
        ei = 0
        points, names = [], []
        for key in keys:
            if key in char_results and len(char_results[key]) == 4:
                pt = plt.scatter(char_results[key][0][fullmatch_n-1], char_results[key][1+i], marker='o', c=colors[ei]) #markers[key.split("_")[0]]
                ei += 1
                points.append(pt)
                names.append("char_"+key)
        for key in wordkeys:
            if key in word_results and len(word_results[key]) == 5:
                pt = plt.scatter(word_results[key][0][fullmatch_n-1], word_results[key][1+i], marker='x', c=colors[ei])
                ei += 1
                #points.append(pt)
                #names.append("word_"+key)
                ax.annotate("word_"+key+"_"+str(word_layers[key])+"_"+str(embed_size[key]), xy=(word_results[key][0][fullmatch_n-1], word_results[key][1+i]), textcoords='data')
        if i == 1:
            # subword model
            pt = plt.scatter(subword_results[percent][fullmatch_n-1], 8551955, marker='+', c=colors[ei])
            ax.annotate("subword_4L", xy=(subword_results[percent][fullmatch_n-1], 8551955), textcoords='data')
        plt.xlabel("FullMatch@%d %s"%(fullmatch_n, percent))
        plt.ylabel(term)
        plt.legend(points, names, scatterpoints=1, loc="upper left", fontsize=8)#, bbox_to_anchor=(1.25, 0.55))
        #plt.legend(points, names, scatterpoints=1, loc="upper right", fontsize=10, bbox_to_anchor=(1.25, 0.55))
        plt.show()
        #fig.savefig("/home/t-gjawahar/object_dir/param_imp_char/PartialMatch1_vs_%s_accuracy.pdf"%(term), bbox_inches='tight')
        fig.savefig("/home/t-gjawahar/archai/scripts/8-23/Small_FullMatch%d%s_vs_%s_accuracy.pdf"%(fullmatch_n, percent,  term), bbox_inches='tight')
    
    # print gap
    # accuracy gain vs. param savings
    points = []
    for word_model in word_results:
        word_nparam = int(word_model[0:-1])
        word_acc = word_results[word_model][0][0]
        best_char_nparam, best_char_nacc = None, None
        #layer = "12L"
        params_def = "10M"
        layer_def = "12L"
        for param in [params_def]: # ["5M", "10M"]:#, "20M"]:
            for layer in [layer_def]: # ["1L", "2L", "8L", "12L"]:
                if char_results[param+"_"+layer][0][0] >= word_acc:
                    best_char_nparam = int(param[0:-1]) #char_results[param+"_"+layer][2]
                    best_char_nacc = char_results[param+"_"+layer][0][0]
                    break
            if best_char_nparam:
                break
        if not best_char_nparam:
            #param, layer = "20M", "12L"
            param, layer = params_def,layer_def
            best_char_nparam =  int(param[0:-1]) # char_results[param+"_"+layer][2]
            best_char_nacc = char_results[param+"_"+layer][0][0]
        #print(word_model[0:-1], best_char_nacc-word_acc, word_nparam-best_char_nparam)
        #points.append((word_model[0:-1], best_char_nacc-word_acc, (word_nparam-best_char_nparam)*1000000)) # raw
        points.append((word_model[0:-1], best_char_nacc-word_acc, ((float(word_nparam-best_char_nparam)/float(word_nparam))*100.0))) # percent
        print(word_model[0:-1], best_char_nacc-word_acc, ((float(word_nparam-best_char_nparam)/float(word_nparam))*100.0))
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    ei = 0
    mpts, names = [], []
    for point in points:
        word_key, word_acc_gain, word_param_gain = point
        pt = plt.scatter(word_acc_gain, word_param_gain, marker='x', c=colors[ei]) 
        ei += 1
        mpts.append(pt)
        names.append("word_"+word_key)
        #ax.annotate("word_"+word_key+"M", xy=(word_acc_gain-0.15, word_param_gain), textcoords='data')
        '''
        if "80" in word_key or "10" in word_key:
            plt.text(word_acc_gain-0.15, word_param_gain-3500000, "word_"+word_key+"M", fontsize=15)
        elif "30" in  word_key:
            plt.text(word_acc_gain-0.05, word_param_gain+2500000, "word_"+word_key+"M", fontsize=15)
        elif "20" in  word_key:
            plt.text(word_acc_gain-0.25, word_param_gain+2500000, "word_"+word_key+"M", fontsize=15)
        else:
            plt.text(word_acc_gain-0.15, word_param_gain+2500000, "word_"+word_key+"M", fontsize=15)
        '''
        plt.text(word_acc_gain, word_param_gain, "word_"+word_key+"M", fontsize=15)
    plt.xlabel("Accuracy Gain w. Char-%s-%s"%(params_def, layer_def), fontsize=15)
    plt.ylabel("#Params Savings", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    #plt.legend(mpts, names, scatterpoints=1, loc="upper right", fontsize=10) #, bbox_to_anchor=(1.25, 0.55))
    plt.show()
    #fig.savefig("/home/t-gjawahar/archai/scripts/8-23/accgain_paramsavings.pdf", bbox_inches='tight')
    #fig.savefig("/home/t-gjawahar/archai/scripts/8-23/accgain_paramsavings_onlychar12L.pdf", bbox_inches='tight')
    fig.savefig("/home/t-gjawahar/archai/scripts/8-23/accgain%s_paramsavings_percent_%s_%s.pdf"%(percent, params_def, layer_def), bbox_inches='tight')
    
#plot_small_models_analysis()

def plot_subword_analysis():
    # word results
    word_results = {}
    for f in glob.glob("/home/t-gjawahar/archai/amlt/inference_word/inference_word_model_metrics-subword_nofp_wt103_g4_*/stdout.txt"):
        for line in open(f):
            content = line.strip()
            if "context=0.50" in content:
                items = content.split()
                res = []
                for i in range(3):
                    res.append(float(items[2 + 2* (i+1)]))
                print(f.split("/")[-2])
                word_results[f.split("/")[-2].split("_")[-1]] = [res]
                break
    print(word_results)

    # read embed params, non embed params, valid ppl
    for f in glob.glob("/home/t-gjawahar/archai/amlt/word_train/subword_nofp_wt103_g4_*/stdout.txt"):
        key = f.split("/")[-2].split("word")[-1].split("_")[-1]
        valid_loss, n_all_param, n_nonemb_param, eval_step = None, None, None, None
        valid_ppl = None
        for line in open(f):
            line = line.strip()
            if "valid loss" in line:
                #if int(line.split()[4]) > 20:
                #    break
                eval_step = int(line.split()[4])
                valid_loss = float(line.split()[-5])
                valid_ppl = float(line.split()[-1])
            elif "n_all_param" in line:
                n_all_param = int(line.split()[-1])
            elif "n_nonemb_param" in line:
                n_nonemb_param = int(line.split()[-1])
        
        #if eval_step == 20:
        if key in word_results:
            word_results[key] += [valid_loss, n_all_param, n_nonemb_param, valid_ppl]
    print(word_results)
    
    colors = ['b', 'c', 'y', 'm', 'r', 'g', 'k']
    for num_word in [1, 2, 3]:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10,5))
        ei = 0
        points = []
        goodnames = []
        for key in sorted(word_results):
            goodname = key
            pt = plt.scatter(word_results[key][2], word_results[key][0][num_word-1], marker='x', c=colors[ei])
            ei += 1
            points.append(pt)
            goodnames.append(goodname)
        plt.xlabel("Params")
        plt.ylabel("FullMatch@"+str(num_word))
        plt.legend(points, goodnames, scatterpoints=1, loc="upper right", fontsize=10, bbox_to_anchor=(1.15, 0.45))
        plt.show()
        fig.savefig("/home/t-gjawahar/archai/scripts/8-23/subword_params_vs_fmatch@%d.pdf"%(num_word), bbox_inches='tight')

# plot_subword_analysis()


def plot_latency_params():
    points = []
    lines = []
    for line in open("/home/t-gjawahar/archai/scripts/latency"):
        lines.append(line.strip())
    char_latencies = {}
    for i in range(12):
        key = lines[3*i].split("small_")[-1]
        latency = float(lines[(3*i)+2][0:-1])*1000 if lines[(3*i)+2][-1] == 's' else float(lines[(3*i)+2])
        latency = (latency/1024) * (5)
        points.append((key, int(key.split("M")[0]), latency))
        char_latencies[key] = latency
    print(points)
    print(char_latencies)
    lines = []
    for line in open("/home/t-gjawahar/archai/scripts/latencyword"):
        lines.append(line.strip())
    word_latencies = {}
    for i in range(7):
        key = lines[3*i].strip()
        latency = float(lines[(3*i)+2][0:-1])*1000 if lines[(3*i)+2][-1] == 's' else float(lines[(3*i)+2])
        latency = (latency/192) 
        points.append((key, int(key.split("M")[0]), latency))
        word_latencies[key] = latency
    print(word_latencies)

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    ei = 0
    mpts, names = [], []
    colors = ['b', 'c', 'y', 'm', 'r', 'g', 'k', "indigo", "violet", "springgreen", "olive", "firebrick", "gold", "darkorange", "teal", "slategrey", "crimson", "peru", "olive"]
    for point in points:
        word_key, word_acc_gain, word_param_gain = point
        pt = plt.scatter(word_acc_gain, word_param_gain, marker='x', c=colors[ei]) 
        ei += 1
        mpts.append(pt)
        names.append(word_key)
    plt.xlabel("Total Params")
    plt.ylabel("Latency (ms)")
    plt.legend(mpts, names, scatterpoints=1, loc="upper right", fontsize=10, bbox_to_anchor=(1.25, 0.55))
    plt.show()
    fig.savefig("/home/t-gjawahar/archai/scripts/8-23/params_vs_latency.pdf", bbox_inches='tight')

    # read accuracy
    char_results = {}
    for f in glob.glob("/home/t-gjawahar/archai/amlt/inference_transxl/inference_char_valid-small_*/stdout.txt"):
        for line in open(f):
            content = line.strip()
            if "context=0.50" in content:
                items = content.split()
                #res = []
                #for i in range(3):
                #    res.append(float(items[2 + 2* (i+1)]))
                score_1 = float(items[5].split(",")[0][1:-1].split("/")[0]) /  float(items[5].split(",")[0][1:-1].split("/")[1])
                score_2 = float(items[7].split(",")[0][1:-1].split("/")[0]) /  float(items[7].split(",")[0][1:-1].split("/")[1])
                score_3 = float(items[-1].split(",")[0][1:-1].split("/")[0]) /  float(items[-1].split(",")[0][1:-1].split("/")[1])
                score_1, score_2, score_3 = float("%.2f"%(100.0*score_1)), float("%.2f"%(100.0*score_2)), float("%.2f"%(100.0*score_3))
                char_results[f.split("/")[-2].split("inference_char_valid-small_")[-1]] = [[score_1, score_2, score_3]]
                break
    print(char_results)

    keys = sorted(list(char_results.keys()))
    print(keys)
    
    # read embed params, non embed params, valid ppl
    for f in glob.glob("/home/t-gjawahar/archai/amlt/transxl_char_exp2_randsearch/small_*/stdout.txt"):
        key = f.split("/")[-2].split("small_")[-1]
        valid_loss, n_all_param, n_nonemb_param, eval_step = None, None, None, None
        for line in open(f):
            line = line.strip()
            if "valid loss" in line:
                #if int(line.split()[4]) > 20:
                #    break
                eval_step = int(line.split()[4])
                valid_loss = float(line.split()[-5])
            elif "n_all_param" in line:
                n_all_param = int(line.split()[-1])
            elif "n_nonemb_param" in line:
                n_nonemb_param = int(line.split()[-1])
        
        #if eval_step == 20:
        char_results[key] += [valid_loss, n_all_param, n_nonemb_param]
        #if eval_step == 20:
        print(key, [char_results[key][0][0], valid_loss, n_all_param, n_nonemb_param], eval_step)
    print(char_results)

    # word results
    word_results = {}
    for f in glob.glob("/home/t-gjawahar/archai/amlt/inference_word/inference_word_model_metrics-word*_nofp/stdout.txt"):
        for line in open(f):
            content = line.strip()
            if "context=0.50" in content:
                items = content.split()
                #res = []
                #for i in range(3):
                #    res.append(float(items[2 + 2* (i+1)]))
                #print(f.split("/")[-2])
                score_1 = float(items[5].split(",")[0][1:-1].split("/")[0]) /  float(items[5].split(",")[0][1:-1].split("/")[1])
                score_2 = float(items[7].split(",")[0][1:-1].split("/")[0]) /  float(items[7].split(",")[0][1:-1].split("/")[1])
                score_3 = float(items[-1].split(",")[0][1:-1].split("/")[0]) /  float(items[-1].split(",")[0][1:-1].split("/")[1])
                score_1, score_2, score_3 = float("%.2f"%(100.0*score_1)), float("%.2f"%(100.0*score_2)), float("%.2f"%(100.0*score_3))
                
                word_results[f.split("/")[-2].split("inference_word_model_metrics-word")[-1].split("_")[0]] = [[score_1, score_2, score_3]]
                break
    print(word_results)

    # read embed params, non embed params, valid ppl
    for f in glob.glob("/home/t-gjawahar/archai/amlt/word_train/word*_nofp/stdout.txt"):
        key = f.split("/")[-2].split("word")[-1].split("_")[0]
        valid_loss, n_all_param, n_nonemb_param, eval_step = None, None, None, None
        valid_ppl = None
        for line in open(f):
            line = line.strip()
            if "valid loss" in line:
                #if int(line.split()[4]) > 20:
                #    break
                eval_step = int(line.split()[4])
                valid_loss = float(line.split()[-5])
                valid_ppl = float(line.split()[-1])
            elif "n_all_param" in line:
                n_all_param = int(line.split()[-1])
            elif "n_nonemb_param" in line:
                n_nonemb_param = int(line.split()[-1])
        
        #if eval_step == 20:
        word_results[key] += [valid_loss, n_all_param, n_nonemb_param, valid_ppl]
        #if eval_step == 20:
        #print(key, [char_results[key][0][0], valid_loss, n_all_param, n_nonemb_param], eval_step)
    print(word_results)
    wordkeys = sorted(list(word_results.keys()))

    # plot scatter plot
    colors = ['b', 'c', 'y', 'm', 'r', 'g', 'k', "indigo", "violet", "springgreen", "olive", "firebrick", "gold", "darkorange", "teal", "slategrey", "crimson", "peru", "olive"]
    markers = {"5M": "x", "10M": "o", "20M": "*", "30M": "<", "40M": "+", "80M": "D", "50M": "s"}
    word_layers = {"5M": 3 , "10M": 4, "20M": 6, "30M": 8, "40M": 14, "50M": 16, "80M": 16}
    embed_size = {"5M": 18 , "10M": 36, "20M": 74, "30M": 100, "40M": 128, "50M": 160, "80M": 256}
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    ei = 0
    points, names = [], []
    for key in keys:
        if key in char_results and len(char_results[key]) == 4:
            pt = plt.scatter(char_results[key][0][0], char_latencies[key], marker='o', c=colors[ei]) #markers[key.split("_")[0]]
            ei += 1
            #points.append(pt)
            #names.append("char_"+key)
            ax.annotate("char"+key, xy=(char_results[key][0][0], char_latencies[key]), textcoords='data')
    for key in wordkeys:
        if key in word_results and len(word_results[key]) == 5:
            pt = plt.scatter(word_results[key][0][0], word_latencies[key], marker='x', c=colors[ei])
            ei += 1
            points.append(pt)
            names.append("word_"+key+"_"+str(word_layers[key])+"_"+str(embed_size[key]))
            #ax.annotate("word_"+key+"_"+str(word_layers[key])+"_"+str(embed_size[key]), xy=(word_results[key][0][0], word_latencies[key]), textcoords='data')
    plt.xlabel("FullMatch@1")
    plt.ylabel("Latency")
    plt.legend(points, names, scatterpoints=1, loc="upper left", fontsize=8)
    plt.show()
    fig.savefig("/home/t-gjawahar/archai/scripts/8-23/Small_FullMatch1_vs_latency.pdf", bbox_inches='tight')

    points = []
    for word_model in word_results:
        word_nparam = word_latencies[word_model]  # int(word_model[0:-1])
        word_acc = word_results[word_model][0][0]
        best_char_nparam, best_char_nacc = None, None
        #layer = "12L"
        params_def = "5M"
        layer_def = "12L"
        for param in [params_def]: #["5M", "10M", "20M"]:
            for layer in [layer_def]: #["1L", "2L", "8L", "12L"]:
                if char_results[param+"_"+layer][0][0] >= word_acc:
                    best_char_nparam = char_latencies[param+"_"+layer] # int(param[0:-1]) #char_results[param+"_"+layer][2]
                    best_char_nacc = char_results[param+"_"+layer][0][0]
                    break
            if best_char_nparam:
                break
        if not best_char_nparam:
            #param, layer = "20M", "12L"
            param, layer = params_def, layer_def
            best_char_nparam = char_latencies[param+"_"+layer] # int(param[0:-1]) # char_results[param+"_"+layer][2]
            best_char_nacc = char_results[param+"_"+layer][0][0]
        #points.append((word_model[0:-1], best_char_nacc-word_acc, (word_nparam-best_char_nparam))) # raw
        points.append((word_model[0:-1], best_char_nacc-word_acc, 100.0*((word_nparam-best_char_nparam)/float(word_nparam))))
        print((word_model[0:-1], best_char_nacc-word_acc, word_nparam, best_char_nparam,  100.0*((word_nparam-best_char_nparam)/float(word_nparam))))
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    ei = 0
    mpts, names = [], []
    for point in points:
        word_key, word_acc_gain, word_param_gain = point
        pt = plt.scatter(word_acc_gain, word_param_gain, marker='x', c=colors[ei]) 
        ei += 1
        mpts.append(pt)
        names.append("word_"+word_key)
        print(word_key)
        '''
        if word_key in ["5", "40"]:
            plt.text(word_acc_gain-0.15, word_param_gain-0.6, "word_"+word_key+"M",fontsize=15)
        elif "10" in word_key:
            plt.text(word_acc_gain-0.06, word_param_gain-0.6, "word_"+word_key+"M",fontsize=15)
        elif "20" in word_key:
            plt.text(word_acc_gain-0.26, word_param_gain-0.6, "word_"+word_key+"M",fontsize=15)
        else:
            plt.text(word_acc_gain-0.15, word_param_gain+0.4, "word_"+word_key+"M",fontsize=15)
        '''
        plt.text(word_acc_gain, word_param_gain, "word_"+word_key+"M",fontsize=15)
    plt.xlabel("Accuracy Gain w. Char-%s-%s"%(params_def, layer_def),fontsize=15)
    plt.ylabel("Latency Savings (ms)",fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    #plt.legend(mpts, names, scatterpoints=1, loc="upper right", fontsize=10) #, bbox_to_anchor=(1.25, 0.55))
    plt.show()
    fig.savefig("/home/t-gjawahar/archai/scripts/8-23/accgain_latencylosses_%s_%s.pdf"%(params_def, layer_def), bbox_inches='tight')

#plot_latency_params()

def plot_memory_params():
    colors = ['b', 'c', 'y', 'm', 'r', 'g', 'k', "indigo", "violet", "springgreen", "olive", "firebrick", "gold", "darkorange", "teal", "slategrey", "crimson", "peru", "olive"]
    points = []
    char_latencies = {}
    for f in glob.glob("/home/t-gjawahar/archai/amlt/inference_transxl/inference_char_valid_memstat-small*/stdout.txt"):
        max_peak = 0
        for line in open(f):
            if "rss output of" in line.strip():
                try:
                    max_peak = max(max_peak, float(line.split()[-1]))
                    if "program_end" in line.strip():
                        break
                except:
                    print('error')
                    continue
        char_latencies[f.split("/")[-2].split("small_")[-1]] = max_peak
    print(char_latencies)
    lines = []
    for line in open("/home/t-gjawahar/archai/scripts/latencyword"):
        lines.append(line.strip())
    word_latencies = {}
    for f in glob.glob("/home/t-gjawahar/archai/amlt/inference_word/inference_word_model_metrics_memstat-word*/stdout.txt"):
        max_peak = 0
        for line in open(f):
            if "rss output of" in line.strip():
                try:
                    max_peak = max(max_peak, float(line.split()[-1]))
                except:
                    print('error')
                    continue
        word_latencies[f.split("/")[-2].split("word")[-1].split("_")[0]] = max_peak
    print(word_latencies)

    # read accuracy
    char_results = {}
    for f in glob.glob("/home/t-gjawahar/archai/amlt/inference_transxl/inference_char_valid-small_*/stdout.txt"):
        for line in open(f):
            content = line.strip()
            if "context=0.50" in content:
                items = content.split()
                #res = []
                #for i in range(3):
                #    res.append(float(items[2 + 2* (i+1)]))
                score_1 = float(items[5].split(",")[0][1:-1].split("/")[0]) /  float(items[5].split(",")[0][1:-1].split("/")[1])
                score_2 = float(items[7].split(",")[0][1:-1].split("/")[0]) /  float(items[7].split(",")[0][1:-1].split("/")[1])
                score_3 = float(items[-1].split(",")[0][1:-1].split("/")[0]) /  float(items[-1].split(",")[0][1:-1].split("/")[1])
                score_1, score_2, score_3 = float("%.2f"%(100.0*score_1)), float("%.2f"%(100.0*score_2)), float("%.2f"%(100.0*score_3))
                char_results[f.split("/")[-2].split("inference_char_valid-small_")[-1]] = [[score_1, score_2, score_3]]
                break
    print(char_results)

    keys = sorted(list(char_results.keys()))
    print(keys)
    
    # read embed params, non embed params, valid ppl
    for f in glob.glob("/home/t-gjawahar/archai/amlt/transxl_char_exp2_randsearch/small_*/stdout.txt"):
        key = f.split("/")[-2].split("small_")[-1]
        valid_loss, n_all_param, n_nonemb_param, eval_step = None, None, None, None
        for line in open(f):
            line = line.strip()
            if "valid loss" in line:
                #if int(line.split()[4]) > 20:
                #    break
                eval_step = int(line.split()[4])
                valid_loss = float(line.split()[-5])
            elif "n_all_param" in line:
                n_all_param = int(line.split()[-1])
            elif "n_nonemb_param" in line:
                n_nonemb_param = int(line.split()[-1])
        
        #if eval_step == 20:
        char_results[key] += [valid_loss, n_all_param, n_nonemb_param]
        #if eval_step == 20:
        print(key, [char_results[key][0][0], valid_loss, n_all_param, n_nonemb_param], eval_step)
    print(char_results)

    # word results
    word_results = {}
    for f in glob.glob("/home/t-gjawahar/archai/amlt/inference_word/inference_word_model_metrics-word*_nofp/stdout.txt"):
        for line in open(f):
            content = line.strip()
            if "context=0.50" in content:
                items = content.split()
                #res = []
                #for i in range(3):
                #    res.append(float(items[2 + 2* (i+1)]))
                #print(f.split("/")[-2])
                score_1 = float(items[5].split(",")[0][1:-1].split("/")[0]) /  float(items[5].split(",")[0][1:-1].split("/")[1])
                score_2 = float(items[7].split(",")[0][1:-1].split("/")[0]) /  float(items[7].split(",")[0][1:-1].split("/")[1])
                score_3 = float(items[-1].split(",")[0][1:-1].split("/")[0]) /  float(items[-1].split(",")[0][1:-1].split("/")[1])
                score_1, score_2, score_3 = float("%.2f"%(100.0*score_1)), float("%.2f"%(100.0*score_2)), float("%.2f"%(100.0*score_3))
                
                word_results[f.split("/")[-2].split("inference_word_model_metrics-word")[-1].split("_")[0]] = [[score_1, score_2, score_3]]
                break
    print(word_results)

    # read embed params, non embed params, valid ppl
    for f in glob.glob("/home/t-gjawahar/archai/amlt/word_train/word*_nofp/stdout.txt"):
        key = f.split("/")[-2].split("word")[-1].split("_")[0]
        valid_loss, n_all_param, n_nonemb_param, eval_step = None, None, None, None
        valid_ppl = None
        for line in open(f):
            line = line.strip()
            if "valid loss" in line:
                #if int(line.split()[4]) > 20:
                #    break
                eval_step = int(line.split()[4])
                valid_loss = float(line.split()[-5])
                valid_ppl = float(line.split()[-1])
            elif "n_all_param" in line:
                n_all_param = int(line.split()[-1])
            elif "n_nonemb_param" in line:
                n_nonemb_param = int(line.split()[-1])
        
        #if eval_step == 20:
        word_results[key] += [valid_loss, n_all_param, n_nonemb_param, valid_ppl]
        #if eval_step == 20:
        #print(key, [char_results[key][0][0], valid_loss, n_all_param, n_nonemb_param], eval_step)
    print(word_results)
    wordkeys = sorted(list(word_results.keys()))

    # plot scatter plot
    colors = ['b', 'c', 'y', 'm', 'r', 'g', 'k', "indigo", "violet", "springgreen", "olive", "firebrick", "gold", "darkorange", "teal", "slategrey", "crimson", "peru", "olive"]
    markers = {"5M": "x", "10M": "o", "20M": "*", "30M": "<", "40M": "+", "80M": "D", "50M": "s"}
    word_layers = {"5M": 3 , "10M": 4, "20M": 6, "30M": 8, "40M": 14, "50M": 16, "80M": 16}
    embed_size = {"5M": 18 , "10M": 36, "20M": 74, "30M": 100, "40M": 128, "50M": 160, "80M": 256}
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    ei = 0
    points, names = [], []
    for key in keys:
        if key in char_results and len(char_results[key]) == 4:
            pt = plt.scatter(char_results[key][0][0], char_latencies[key], marker='o', c=colors[ei]) #markers[key.split("_")[0]]
            ei += 1
            #points.append(pt)
            #names.append("char_"+key)
            ax.annotate("char"+key, xy=(char_results[key][0][0], char_latencies[key]), textcoords='data')
    for key in wordkeys:
        if key in word_results and len(word_results[key]) == 5:
            pt = plt.scatter(word_results[key][0][0], word_latencies[key], marker='x', c=colors[ei])
            ei += 1
            points.append(pt)
            names.append("word_"+key+"_"+str(word_layers[key])+"_"+str(embed_size[key]))
            #ax.annotate("word_"+key+"_"+str(word_layers[key])+"_"+str(embed_size[key]), xy=(word_results[key][0][0], word_latencies[key]), textcoords='data')
    plt.xlabel("FullMatch@1")
    plt.ylabel("Peak Memory utilization")
    plt.legend(points, names, scatterpoints=1, loc="upper left", fontsize=8)
    plt.show()
    fig.savefig("/home/t-gjawahar/archai/scripts/8-23/Small_FullMatch1_vs_pmu.pdf", bbox_inches='tight')

    points = []
    for word_model in word_results:
        word_nparam = word_latencies[word_model]  # int(word_model[0:-1])
        word_acc = word_results[word_model][0][0]
        best_char_nparam, best_char_nacc = None, None
        #layer = "12L"
        params_def = "20M"
        layer_def = "12L"
        for param in [params_def]: #["5M", "10M", "20M"]:
            for layer in [layer_def]: # ["1L", "2L", "8L", "12L"]:
                if char_results[param+"_"+layer][0][0] >= word_acc:
                    best_char_nparam = char_latencies[param+"_"+layer] # int(param[0:-1]) #char_results[param+"_"+layer][2]
                    best_char_nacc = char_results[param+"_"+layer][0][0]
                    break
            if best_char_nparam:
                break
        if not best_char_nparam:
            #param, layer = "20M", "12L"
            param, layer = params_def, layer_def
            best_char_nparam = char_latencies[param+"_"+layer] # int(param[0:-1]) # char_results[param+"_"+layer][2]
            best_char_nacc = char_results[param+"_"+layer][0][0]
        #print(word_model[0:-1], best_char_nacc-word_acc, word_nparam-best_char_nparam)
        # points.append((word_model[0:-1], best_char_nacc-word_acc, (word_nparam-best_char_nparam))) # raw
        points.append((word_model[0:-1], best_char_nacc-word_acc, 100.0*((word_nparam-best_char_nparam)/float(word_nparam))))
        print(word_model[0:-1], best_char_nacc-word_acc, 100.0*((word_nparam-best_char_nparam)/float(word_nparam)))
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    ei = 0
    mpts, names = [], []
    for point in points:
        word_key, word_acc_gain, word_param_gain = point
        pt = plt.scatter(word_acc_gain, word_param_gain, marker='x', c=colors[ei]) 
        ei += 1
        mpts.append(pt)
        names.append("word_"+word_key)
        print(word_key)
        '''
        if word_key in ["10", "80"]: # down
            plt.text(word_acc_gain-0.15, word_param_gain-40, "word_"+word_key+"M",fontsize=15)    
        else: # up
            plt.text(word_acc_gain-0.15, word_param_gain+35, "word_"+word_key+"M",fontsize=15)
        '''
        plt.text(word_acc_gain, word_param_gain, "word_"+word_key+"M",fontsize=15)
    plt.xlabel("Accuracy Gain w. char-%s"%(params_def),fontsize=15)
    plt.ylabel("Peak Memory Savings (MB)",fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    #plt.legend(mpts, names, scatterpoints=1, loc="upper right", fontsize=10) #, bbox_to_anchor=(1.25, 0.55))
    plt.show()
    fig.savefig("/home/t-gjawahar/archai/scripts/8-23/accgain_pmugains_%s_%s.pdf"%(params_def, layer_def), bbox_inches='tight')

#plot_memory_params()
#sys.exit(0)


def generate_restart_commands():
    res = ""
    names = []
    for line in open("/home/t-gjawahar/object_dir/run_small.txt"):
        if "name: " in line.strip():
            res += line.replace("name: ", "name: char_restart_moresteps_")
            names.append((line.replace("name: ", "name: char_restart_moresteps_").split(": ")[-1], line.split(": ")[-1]))
        elif "sku: G4" in line.strip():
            res += line.replace("G4", "G2")
        elif "python -m" in line.strip():
            res += line.replace("--nproc_per_node=\"4\"", "--nproc_per_node=\"2\"").replace("--warmup_step 4000", "--warmup_step 0").replace("--max_step 400000", "--max_step 1600000").replace("--config dgx1_4gpu_fp16", "--config dgx1_2gpu_fp16")[0:-1]+ " --restart $$AMLT_MAP_INPUT_DIR/checkpoint_400000.pt\n"
        else:
            res += (line)
    #print(res)
    #for name, oldname in names:
    #    print("amlt map archai/nlp/nvidia_transformer_xl/run_char_philly_exp3_char_final_select.yaml :%s transxl_char_exp2_randsearch :%s misc_transxl_char --yes"%(name.strip(), oldname.strip()))
    cmd = "amlt log misc_transxl_char"
    for name, oldname in names:
        cmd += " :%s"%(name.strip()+"-"+oldname.strip())
    print(cmd)

#generate_restart_commands()

def generate_word_segment_small_models(f):
    command = "description: run small model scripts\n\ntarget:\n  service: amlk8s\n  name: ms-shared\n  vc: resrchvc\n\nenvironment:\n  image: pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel\n  registry: docker.io\n  setup:\n    - set -e -o xtrace\n    - sudo apt-get -y install git\n    - pip install --user tensorboard\n\ncode:\n  local_dir: /home/t-gjawahar/archai\n\ndata:\n  local_dir: /home/t-gjawahar/object_dir/wikitext-2-raw-v1-char\n  remote_dir: dataroot/textpred/wikitext-2-raw-v1-char\n\njobs:\n"
    job_cmd = "- name: small_5M_8L\n  sku: G2\n  command:\n  - set -e -o xtrace\n  - bash scripts/apex_install.sh\n  - pip install --user -e .\n  - python -m torch.distributed.launch --nproc_per_node=\"2\" archai/nlp/nvidia_transformer_xl/train.py --dataset wt2 --warmup_step 4000 --max_step 400000 --eval_interval 10000 --n_layer 8 --n_head 8 --d_head 32 --d_embed 350 --d_inner 256 --mem_len 512 --tgt_len 512 --d_model 350 --dropout 0.1 -dropatt 0.0 --config dgx1_2gpu_fp16 --experiment_name small_5M_8L --config_file char_base.yaml --eval_tgt_len 1024 --batch_size 512 --lr 0.001 --save_all"

    res = command
    names = []
    for segment_type in ["word", "subword"]:
        for sname, model_ext in [("bert", "bert_style_word_segment"), ("cpool", "char_emb_from_word")]:
            if sname == "bert":
                new_job_cmd = job_cmd.replace("small_5M_8L", "small_5M_8L_"+sname+"_"+segment_type) + " --model_ext bert_style_word_segment --segment_type "+segment_type
                res += new_job_cmd +"\n"
                names.append(":small_5M_8L_"+sname+"_"+segment_type)
            else:
                for pooling in ["mean", "sum", "max"]:
                    new_job_cmd = job_cmd.replace("small_5M_8L", "small_5M_8L_"+sname+"_"+segment_type+"_"+pooling) + " --model_ext char_emb_from_word --segment_type "+segment_type
                    res += new_job_cmd +"\n"
                    names.append(":small_5M_8L_"+sname+"_"+segment_type+"_"+pooling)
    #print(res)
    w = open(f, 'w')
    w.write(res)
    w.close()
    print("amlt run %s %s transxl_char_exp2_randsearch"%(f, " ".join(names)))
    # amlt status transxl_char_exp2_randsearch

#generate_word_segment_small_models("scripts/8-25/generate_word_segment_small_models.yaml")

def generate_integ_word_info():
    for f in glob.glob("/home/t-gjawahar/archai/amlt/inference_transxl/inference_char-transxl_char_params_80M*/*.txt"):
        for line in open(f):
            line = line.strip()
            if "context=" in line.strip():
                items = line.split(" ")
                score_1 = float(items[5].split(",")[0][1:-1].split("/")[0]) /  float(items[5].split(",")[0][1:-1].split("/")[1])
                score_2 = float(items[7].split(",")[0][1:-1].split("/")[0]) /  float(items[7].split(",")[0][1:-1].split("/")[1])
                score_3 = float(items[-1].split(",")[0][1:-1].split("/")[0]) /  float(items[-1].split(",")[0][1:-1].split("/")[1])
                print("%s %.2f %.2f %.2f"%(f.split("/")[-2].split("80M_")[-1], 100.0*score_1, 100.0*score_2, 100.0*score_3))
                break

#generate_integ_word_info()

def calculate_word_ppl():
    num_characters = 0
    num_tokens = 0
    for line in open("/home/t-gjawahar/object_dir/wikitext-2-raw-v1-char/wiki.valid.tokens"):
        line = line.strip()
        if len(line) != 0:
            num_characters += len(line)
            num_tokens += len(line.split())
    import math
    for bpc in [2.163, 2.174, 2.150, 2.151, 2.147]:
        avg_loss = math.log(bpc)
        bpc = avg_loss / math.log(2)
        word_ppl = math.pow(2, bpc * (num_characters / num_tokens))
        print("%.4f"%avg_loss, "%.2f"%word_ppl)

# calculate_word_ppl()

# amlt log inference_transxl :inference_char_valid-small_5M_1L :inference_char_valid-small_10M_2L :inference_char_valid-small_20M_12L :inference_char_valid-small_10M_8L :inference_char_valid-small_5M_2L :inference_char_valid-small_10M_12L :inference_char_valid-transxl_char_params_80M_char_emb_from_word_mean_lr0p001_g4 :inference_char_valid-transxl_char_params_80M :inference_char_valid-transxl_char_params_80M_bertstyle_lr0p001_8g :inference_char_valid-small_5M_8L :inference_char_valid-small_5M_12L :inference_char_valid-small_20M_8L :inference_char_valid-transxl_char_params_80M_char_emb_from_word_sum_lr0p001_g4 :inference_char_valid-small_20M_2L :inference_char_valid-small_10M_1L :inference_char_valid-small_20M_1L :inference_char_valid-transxl_char_params_80M_char_emb_from_word_max_lr0p001_g4
def generate_inference_logs_mlrg():
    res = "amlt log inference_word"
    for line in open("scripts/inference.sh"):
        line = line.strip()
        if "failed" in line:
            continue
        item = line.split()[0]
        res += " "+item
    print(res)
    
#generate_inference_logs_mlrg()


def generate_memstat_commands():
    commands = {}
    correct_cur_name = None
    for line in open("/home/t-gjawahar/archai/archai/nlp/nvidia_transformer_xl/run_char_philly_exp3_char_final_select.yaml"):
        line = line.strip()
        if "- name: " in line:
            if "- name: small_" in line:
                correct_cur_name = line.strip().split(": ")[-1]
            else:
                correct_cur_name = None
        if correct_cur_name:
            if "python -m" in line:
                commands[correct_cur_name] = line.strip()
    #print(commands)
    for expname in commands:
        cmd = commands[expname]
        print(cmd.replace("- python -m torch.distributed.launch --nproc_per_node=\"4\" archai/nlp/nvidia_transformer_xl/train.py --dataset wt2 --warmup_step 4000 --max_step 400000 --eval_interval 10000", "CUDA_VISIBLE_DEVICES=\"\" python archai/nlp/nvidia_transformer_xl/exact_match.py --data /home/t-gjawahar/object_dir/wikitext-2-raw-v1-char").replace("--config_file char_base.yaml --eval_tgt_len 1024 --batch_size 128 --lr 0.001 --save_all", "--batch_size 1 --prompt_context_percent 0.5 --num_prompts 1 --num_chars_generate 100 --memstat"))
        sys.exit(0)

#generate_memstat_commands()


def plot_char_modif_results():
    charscores = {}
    for f in glob.glob("/home/t-gjawahar/archai/amlt/inference_transxl/inference_char_valid-transxl_char_params_80M*/stdout.txt"):
        scores = []
        for line in open(f):
            if "context=" in line:
                items = line.split(" ")
                score_1 = float(items[5].split(",")[0][1:-1].split("/")[0]) /  float(items[5].split(",")[0][1:-1].split("/")[1])
                score_2 = float(items[7].split(",")[0][1:-1].split("/")[0]) /  float(items[7].split(",")[0][1:-1].split("/")[1])
                score_3 = float(items[-1].split(",")[0][1:-1].split("/")[0]) /  float(items[-1].split(",")[0][1:-1].split("/")[1])
                score_1, score_2, score_3 = float("%.2f"%(100.0*score_1)), float("%.2f"%(100.0*score_2)), float("%.2f"%(100.0*score_3))
                scores.append([score_1, score_2, score_3])
        goodname = "base"
        if "mean" in f.split("/")[-2]:
            goodname = "mean"
        elif "max" in f.split("/")[-2]:
            goodname = "max"
        elif "sum" in f.split("/")[-2]:
            goodname = "sum"
        elif "bertstyle" in f.split("/")[-2]:
            goodname = "bertstyle"
        charscores[goodname] = [scores[0], scores[2], scores[4]]
        #charscores[goodname] = [scores[1], scores[3], scores[5]]

    colors = ['b', 'c', 'y', 'm', 'r', 'g', 'k', "indigo", "violet", "springgreen", "olive", "firebrick", "gold", "darkorange", "teal", "slategrey", "crimson", "peru", "olive"]
    '''
    for j, context in enumerate([0.2, 0.5, 0.8]):
        fig = plt.figure(figsize=(10,5))
        plt.grid(color='gray', linestyle='dashed')
        xaxis = [i+1 for i in range(3)]
        ci = 0
        for model in ["base", "bertstyle", "mean", "max", "sum"]:
            plt.plot(xaxis, charscores[model][j], color=colors[ci], marker='o', label=model)
            ci += 1
        plt.xlabel("N (no. of words)")
        plt.ylabel("ExactMatch@N")
        plt.legend(loc="upper left")
        plt.show()
        fig.savefig("/home/t-gjawahar/archai/scripts/mlrg/wordonly_%.2f.pdf"%context, bbox_inches='tight')
    '''
    for j, N in enumerate([1, 2, 3]):
        fig = plt.figure(figsize=(10,5))
        #plt.grid(color='gray', linestyle='dashed')
        xaxis = [0.2, 0.5, 0.8]
        ci = 0
        for model in ["base", "bertstyle", "mean", "max", "sum"]:
            plt.plot(xaxis, [charscores[model][k][j] for k in range(len(xaxis))], color=colors[ci], marker='o', label=model)
            ci += 1
        plt.xlabel("Context Percent",fontsize=15)
        plt.ylabel("ExactMatch@%d"%N,fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend(loc="lower right",fontsize=12)
        plt.title("ExactMatch@%d vs. Context Percent"%N,fontsize=15)
        plt.show()
        fig.savefig("/home/t-gjawahar/archai/scripts/mlrg/wordonly_%d.pdf"%N, bbox_inches='tight')
    
#plot_char_modif_results()

def generate_char_archi_boxplot_commands():
    command = "python archai/nlp/nvidia_transformer_xl/train.py --dataset wt2 --warmup_step 4000 --max_step 400000 --eval_interval 10000 --n_layer <n_layer> --n_head <n_head> --d_head <d_head> --d_embed <d_embed> --d_inner <d_inner> --mem_len 512 --tgt_len 512 --d_model <d_model> --dropout 0.1 -dropatt 0.0 --config dgx1_1gpu_fp16 --experiment_name small_20M_2L --config_file char_base.yaml --eval_tgt_len 1024 --batch_size 128 --lr 0.001"

    LAYERS = [2, 4, 8, 12, 16, 24, 32]
    NHEAD = [2, 4, 8, 16, 32, 64]
    DHEAD = [8, 16, 32, 64, 128]
    DEMBED = [256, 512, 1024, 2048]
    DINNER = [256, 512, 1024, 2048]
    DMODEL = [256, 512, 1024, 2048]
    import random
    random.seed(123)
    for i in range(150):
        print(command.replace("<n_layer>", "%d"%random.choice(LAYERS)).replace("<n_head>", "%d"%random.choice(NHEAD)).replace("<d_head>", "%d"%random.choice(DHEAD)).replace("<d_embed>", "%d"%random.choice(DEMBED)).replace("<d_inner>", "%d"%random.choice(DINNER)).replace("<d_model>", "%d"%random.choice(DMODEL)))

#generate_char_archi_boxplot_commands()

def generate_plot_char_archi():
    import numpy as np
    params_adaptive_embedding_list, params_adaptive_softmax_list, params_attention_list, params_ff_list = [], [], [], []
    for line in open("scripts/mlrg/chararchi.out"):
        line = line.strip()
        if line.startswith("adaparams"):
            vals = [int(item.split("=")[-1]) for item in line.split()]
            vals = [float(val) for val in vals]
            total = vals[0]+vals[1]+vals[2]+vals[3]
            vals[0] = vals[0]/total
            vals[1] = vals[1]/total
            vals[2] = vals[2]/total
            vals[3] = vals[3]/total
            params_adaptive_embedding_list.append(int(100.0*vals[0]))
            params_adaptive_softmax_list.append(int(100.0*vals[1]))
            params_attention_list.append(int(100.0*vals[2]))
            params_ff_list.append(int(100.0*vals[3]))
            if len(params_ff_list) == 100:
                break

    #fig = plt.figure(figsize=(10,5))
    #ax = fig.add_subplot(111)
    fig, ax = plt.subplots()
    data = [params_adaptive_embedding_list, params_adaptive_softmax_list, params_attention_list, params_ff_list]
    bp = ax.boxplot(data, sym='k+', showmeans=True)
    m = [np.mean(d, axis=0) for d in data]
    for i, line in enumerate(bp['medians']):
        x, y = line.get_xydata()[1]
        text = ' μ={:.2f}'.format(m[i])
        print(text)
        #if i>0:
        #    ax.annotate(text, xy=(x-0.2, y+20))
        #elif i == 3:
        #    ax.annotate(text, xy=(x, y-30))
        #else:
        ax.annotate(text, xy=(x, y), fontsize=14)
    ax.grid(axis='y')
    plt.xticks(range(1, 5), ['AdaEmb', 'Sftmax', 'Attn', 'FFN'], fontsize=14)
    plt.savefig('/home/t-gjawahar/archai/scripts/mlrg/parameter_breakdown_char.png', bbox_inches="tight")
    fig.savefig("/home/t-gjawahar/archai/scripts/mlrg/parameter_breakdown_char.pdf", bbox_inches='tight')

#generate_plot_char_archi()


def integrate_word_subword_info():
    charscores = {}
    for f in glob.glob("/home/t-gjawahar/archai/amlt/inference_transxl/inference_char_valid-small_5M_8L_*/stdout.txt"):
        scores = []
        for line in open(f):
            if "context=" in line:
                items = line.split(" ")
                score_1 = float(items[5].split(",")[0][1:-1].split("/")[0]) /  float(items[5].split(",")[0][1:-1].split("/")[1])
                score_2 = float(items[7].split(",")[0][1:-1].split("/")[0]) /  float(items[7].split(",")[0][1:-1].split("/")[1])
                score_3 = float(items[-1].split(",")[0][1:-1].split("/")[0]) /  float(items[-1].split(",")[0][1:-1].split("/")[1])
                score_1, score_2, score_3 = float("%.2f"%(100.0*score_1)), float("%.2f"%(100.0*score_2)), float("%.2f"%(100.0*score_3))
                scores.append([score_1, score_2, score_3])
        goodname = f.split("/")[-2].split("inference_char_valid-small_5M_8L_")[-1]
        #if "pool" in goodname and "max" not in goodname:
            #continue
        charscores[goodname] = [scores[0], scores[2], scores[4]]
        #charscores[goodname] = [scores[1], scores[3], scores[5]]
    print(charscores)
    scores = []
    for line in open("/home/t-gjawahar/archai/amlt/inference_transxl/inference_char_valid_100K-small_5M_8L/stdout.txt"):
        if "context=" in line:
            items = line.split(" ")
            score_1 = float(items[5].split(",")[0][1:-1].split("/")[0]) /  float(items[5].split(",")[0][1:-1].split("/")[1])
            score_2 = float(items[7].split(",")[0][1:-1].split("/")[0]) /  float(items[7].split(",")[0][1:-1].split("/")[1])
            score_3 = float(items[-1].split(",")[0][1:-1].split("/")[0]) /  float(items[-1].split(",")[0][1:-1].split("/")[1])
            score_1, score_2, score_3 = float("%.2f"%(100.0*score_1)), float("%.2f"%(100.0*score_2)), float("%.2f"%(100.0*score_3))
            scores.append([score_1, score_2, score_3])
    charscores["5M_8L_base"] = [scores[0], scores[2], scores[4]]

    colors = ['b', 'c', 'y', 'm', 'r', 'g', 'k', "indigo", "violet", "springgreen", "olive", "firebrick", "gold", "darkorange", "teal", "slategrey", "crimson", "peru", "olive"]
    for j, N in enumerate([1, 2, 3]):
        fig = plt.figure(figsize=(10,5))
        #plt.grid(color='gray', linestyle='dashed')
        xaxis = [0.2, 0.5, 0.8]
        ci = 0
        for model in charscores.keys():
            plt.plot(xaxis, [charscores[model][k][j] for k in range(len(xaxis))], color=colors[ci], marker='o', label=model)
            ci += 1
        plt.xlabel("Context Percent")
        plt.ylabel("ExactMatch@%d"%N)
        plt.legend(loc="lower right")
        plt.title("ExactMatch@%d vs. Context Percent"%N)
        plt.show()
        fig.savefig("/home/t-gjawahar/archai/scripts/mlrg/word_subword_full_%d.pdf"%N, bbox_inches='tight')

#integrate_word_subword_info()

def generate_bpe_sweep_models(f):
    import random
    random.seed(123)

    command = "description: run bpe sweep model scripts\n\ntarget:\n  service: amlk8s\n  name: ms-shared\n  vc: resrchvc\n\nenvironment:\n  image: pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel\n  registry: docker.io\n  setup:\n    - set -e -o xtrace\n    - sudo apt-get -y install git\n    - pip install --user tensorboard\n\ncode:\n  local_dir: /home/t-gjawahar/archai\n\ndata:\n  local_dir: /home/t-gjawahar/object_dir/wikitext-103\n  remote_dir: dataroot/textpred/wikitext-103\n\njobs:\n"
    job_cmd = "- name: subword_space1_<n_layer>_<n_head>_<d_model>_<d_head>_<d_inner>_<vocab_size>_<lr>\n  sku: G2\n  command:\n  - set -e -o xtrace\n  - bash scripts/apex_install.sh\n  - pip install --user -e .\n  - python -m torch.distributed.launch --nproc_per_node=\"2\" archai/nlp/nvidia_transformer_xl/train.py --config dgx1_2gpu_fp16 --config_file wt103_base.yaml --n_layer <n_layer> --n_head <n_head> --d_model <d_model> --d_head <d_head> --d_inner <d_inner> --max_step 100000 --vocab bpe --vocab_size <vocab_size> --lr <lr>"
    
    # scripts/8-31-randsearch
    # search_space_1
    n_layers = [2, 4, 8]
    n_heads = [4, 8]
    d_models = [512]
    d_heads = [32, 64]
    d_inners = [512]
    vocab_sizes = [260, 1000, 5000, 10000, 25000, 50000, 100000] # >10K got cancelled
    lrs = [0.01, 0.001]

    res = command
    names = []
    job_cache = {}
    while len(job_cache) < 50:
        n_layer = random.choice(n_layers)
        n_head = random.choice(n_heads)
        d_model = random.choice(d_models)
        d_head = random.choice(d_heads)
        d_inner = random.choice(d_inners)
        vocab_size = random.choice(vocab_sizes)
        lr = random.choice(lrs)

        cand_name = ":subword_space1_<n_layer>_<n_head>_<d_model>_<d_head>_<d_inner>_<vocab_size>_<lr>".replace("<n_layer>", str(n_layer)).replace("<n_layer>", str(n_layer)).replace("<n_head>", str(n_head)).replace("<d_model>", str(d_model)).replace("<d_head>", str(d_head)).replace("<d_inner>", str(d_inner)).replace("<vocab_size>", str(vocab_size)).replace("<lr>", str(lr))
        if cand_name in job_cache:
            continue
        job_cache[cand_name] = True
        new_job_cmd = job_cmd.replace("<n_layer>", str(n_layer)).replace("<n_layer>", str(n_layer)).replace("<n_head>", str(n_head)).replace("<d_model>", str(d_model)).replace("<d_head>", str(d_head)).replace("<d_inner>", str(d_inner)).replace("<vocab_size>", str(vocab_size)).replace("<lr>", str(lr))
        res += new_job_cmd +"\n"
        names.append(cand_name)

    #print(res)
    res += "\n# amlt run %s %s word_train"%(f, " ".join(names))
    res += "\n# amlt log word_train %s"%(" ".join(names))
    res += "\n# amlt status word_train %s"%(" ".join(names))
    w = open(f, 'w')
    w.write(res)
    w.close()
    print("amlt run %s %s word_train"%(f, " ".join(names)))

#generate_bpe_sweep_models("scripts/8-31-randsearch/generate_bpe_sweep_models.yaml")

def results_generate_bpe_sweep_models():
    # num parameters
    hyp2params = {}
    for f in glob.glob("/home/t-gjawahar/archai/amlt/word_train/subword_space1_*/*"):
        num_params = None
        for line in open(f):
            line = line.strip()
            if "#params" in line:
                num_params = int(line.split()[-1])
        if num_params:
            hyp2params[f.split("/")[-2].split("subword_space1_")[-1]] = num_params
    print(hyp2params, len(hyp2params))

    # get exact match scores
    hyp2taskscores = {}
    for f in glob.glob("/home/t-gjawahar/archai/amlt/inference_word/inference_word_model_metrics_valid-subword_space1*/*"):
        hyp = f.split("/")[-2].split("inference_word_model_metrics_valid-subword_space1_")[-1]
        if hyp not in hyp2params:
            continue
        scores = []
        overall_numerator, overall_denominator = [0.0]*3, [0.0]*3 # 3 is # tokens (smart compose upto 15)
        for line in open(f):
            if "context=" in line:
                items = line.split(" ")
                score_1 = float(items[5].split(",")[0][1:-1].split("/")[0]) /  float(items[5].split(",")[0][1:-1].split("/")[1])
                score_2 = float(items[7].split(",")[0][1:-1].split("/")[0]) /  float(items[7].split(",")[0][1:-1].split("/")[1])
                score_3 = float(items[-1].split(",")[0][1:-1].split("/")[0]) /  float(items[-1].split(",")[0][1:-1].split("/")[1])
                score_1, score_2, score_3 = float("%.2f"%(100.0*score_1)), float("%.2f"%(100.0*score_2)), float("%.2f"%(100.0*score_3))
                scores.append([score_1, score_2, score_3])
                overall_numerator[0] += float(items[5].split(",")[0][1:-1].split("/")[0]) 
                overall_denominator[0] += float(items[5].split(",")[0][1:-1].split("/")[1])
                overall_numerator[1] +=  float(items[7].split(",")[0][1:-1].split("/")[0]) 
                overall_denominator[1] += float(items[7].split(",")[0][1:-1].split("/")[1])
                overall_numerator[2] += float(items[-1].split(",")[0][1:-1].split("/")[0])
                overall_denominator[2] += float(items[-1].split(",")[0][1:-1].split("/")[1])
        overall_score = 0.0
        for idx in range(len(overall_numerator)):
            overall_score += overall_numerator[idx]/overall_denominator[idx]
        overall_score /= 3.0
        hyp2taskscores[hyp] = [scores[0], scores[2], scores[4], overall_score]
    print(hyp2taskscores, len(hyp2taskscores))
    scores = []
    for hyp in hyp2taskscores:
        #scores.append((hyp2taskscores[hyp][0][0], [hyp2taskscores[hyp], hyp2params[hyp], hyp]))
        scores.append((hyp2taskscores[hyp][-1], [hyp2taskscores[hyp], hyp2params[hyp], hyp]))
    scores = sorted(scores)
    scores.reverse()
    for score in scores:
        print(score[1])

# results_generate_bpe_sweep_models()

def generate_bpe_sweep_models_space2(f):
    import random
    random.seed(123)

    command = "description: run bpe sweep model scripts\n\ntarget:\n  service: amlk8s\n  name: ms-shared\n  vc: resrchvc\n\nenvironment:\n  image: pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel\n  registry: docker.io\n  setup:\n    - set -e -o xtrace\n    - sudo apt-get -y install git\n    - pip install --user tensorboard\n\ncode:\n  local_dir: /home/t-gjawahar/archai\n\ndata:\n  local_dir: /home/t-gjawahar/object_dir/wikitext-103\n  remote_dir: dataroot/textpred/wikitext-103\n\njobs:\n"
    job_cmd = "- name: subword_space2_<n_layer>_<n_head>_<d_model>_<d_head>_<d_inner>_<vocab_size>_<lr>\n  sku: G4\n  command:\n  - set -e -o xtrace\n  - bash scripts/apex_install.sh\n  - pip install --user -e .\n  - python -m torch.distributed.launch --nproc_per_node=\"4\" archai/nlp/nvidia_transformer_xl/train.py --config dgx1_4gpu_fp16 --config_file wt103_base.yaml --n_layer <n_layer> --n_head <n_head> --d_model <d_model> --d_head <d_head> --d_inner <d_inner> --max_step 400000 --vocab bpe --vocab_size <vocab_size> --lr <lr> --save_all"
    
    # scripts/8-31-randsearch
    # search_space_1
    n_layers = [2, 4, 8]
    n_heads = [4, 8, 16]
    d_models = [256, 512, 768, 1024, 2048]
    d_heads = [32, 64, 128]
    d_inners = [256, 512, 768, 1024, 2048]
    vocab_sizes = [260, 1000, 5000, 10000, 25000, 50000, 100000]
    lrs = [0.01, 0.001]

    res = command
    names = []
    job_cache = {}
    while len(job_cache) < 50:
        n_layer = random.choice(n_layers)
        n_head = random.choice(n_heads)
        d_model = random.choice(d_models)
        d_head = random.choice(d_heads)
        d_inner = random.choice(d_inners)
        vocab_size = random.choice(vocab_sizes)
        lr = random.choice(lrs)

        cand_name = ":subword_space2_<n_layer>_<n_head>_<d_model>_<d_head>_<d_inner>_<vocab_size>_<lr>".replace("<n_layer>", str(n_layer)).replace("<n_layer>", str(n_layer)).replace("<n_head>", str(n_head)).replace("<d_model>", str(d_model)).replace("<d_head>", str(d_head)).replace("<d_inner>", str(d_inner)).replace("<vocab_size>", str(vocab_size)).replace("<lr>", str(lr))
        if cand_name in job_cache:
            continue
        job_cache[cand_name] = True
        new_job_cmd = job_cmd.replace("<n_layer>", str(n_layer)).replace("<n_layer>", str(n_layer)).replace("<n_head>", str(n_head)).replace("<d_model>", str(d_model)).replace("<d_head>", str(d_head)).replace("<d_inner>", str(d_inner)).replace("<vocab_size>", str(vocab_size)).replace("<lr>", str(lr))
        res += new_job_cmd +"\n"
        names.append(cand_name)

    #print(res)
    res += "\n# amlt run %s %s word_train"%(f, " ".join(names))
    res += "\n# amlt log word_train %s"%(" ".join(names))
    res += "\n# amlt status word_train %s"%(" ".join(names))
    w = open(f, 'w')
    w.write(res)
    w.close()
    print("amlt run %s %s word_train"%(f, " ".join(names)))

#generate_bpe_sweep_models_space2("scripts/9-13-randsearch/generate_bpe_sweep_models_space2.yaml")

def regenerate_bpe_sweep_models_space2(f):
    import random
    random.seed(123)

    command = "description: run bpe sweep model scripts\n\ntarget:\n  service: amlk8s\n  name: ms-shared\n  vc: resrchvc\n\nenvironment:\n  image: pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel\n  registry: docker.io\n  setup:\n    - set -e -o xtrace\n    - sudo apt-get -y install git\n    - pip install --user tensorboard\n\ncode:\n  local_dir: /home/t-gjawahar/archai\n\ndata:\n  local_dir: /home/t-gjawahar/object_dir/wikitext-103\n  remote_dir: dataroot/textpred/wikitext-103\n\njobs:\n"
    job_cmd = "- name: subword_space2_<n_layer>_<n_head>_<d_model>_<d_head>_<d_inner>_<vocab_size>_<lr>\n  sku: G4\n  command:\n  - set -e -o xtrace\n  - bash scripts/apex_install.sh\n  - pip install --user -e .\n  - python -m torch.distributed.launch --nproc_per_node=\"4\" archai/nlp/nvidia_transformer_xl/train.py --config dgx1_4gpu_fp16 --config_file wt103_base.yaml --n_layer <n_layer> --n_head <n_head> --d_model <d_model> --d_head <d_head> --d_inner <d_inner> --max_step 400000 --vocab bpe --vocab_size <vocab_size> --lr <lr> --save_all"
    
    # scripts/8-31-randsearch
    # search_space_1
    n_layers = [2, 4, 8]
    n_heads = [4, 8, 16]
    d_models = [256, 512, 768, 1024, 2048]
    d_heads = [32, 64, 128]
    d_inners = [256, 512, 768, 1024, 2048]
    vocab_sizes = [260, 1000, 5000, 10000, 25000, 50000, 100000]
    lrs = [0.01, 0.001]

    res = command
    names = []
    job_cache = {}
    models = ":subword_space2_2_8_256_64_768_260_0.01 :subword_space2_4_16_2048_64_768_100000_0.01 :subword_space2_2_4_768_128_768_50000_0.01 :subword_space2_2_4_1024_32_2048_10000_0.01 :subword_space2_2_8_1024_32_256_260_0.01 :subword_space2_2_4_768_64_2048_10000_0.001 :subword_space2_4_4_768_64_2048_100000_0.001 :subword_space2_2_16_2048_128_768_260_0.001 :subword_space2_8_16_1024_128_2048_50000_0.001 :subword_space2_8_16_1024_64_2048_260_0.01 :subword_space2_8_4_1024_128_768_1000_0.001 :subword_space2_4_16_1024_32_1024_10000_0.001 :subword_space2_2_4_2048_32_768_260_0.001 :subword_space2_2_8_2048_128_256_50000_0.001 :subword_space2_4_16_512_128_1024_1000_0.01 :subword_space2_8_8_1024_64_256_260_0.01 :subword_space2_4_4_256_64_768_50000_0.001 :subword_space2_4_16_1024_32_256_25000_0.01 :subword_space2_8_8_256_32_1024_260_0.001 :subword_space2_2_16_1024_32_768_5000_0.001 :subword_space2_2_16_768_64_768_260_0.01 :subword_space2_4_4_512_64_512_50000_0.01 :subword_space2_8_4_2048_64_2048_5000_0.01 :subword_space2_8_8_1024_64_512_25000_0.001 :subword_space2_8_4_512_128_1024_10000_0.01 :subword_space2_8_4_1024_128_256_5000_0.001 :subword_space2_8_16_1024_64_1024_100000_0.01 :subword_space2_2_16_768_128_2048_25000_0.01 :subword_space2_4_16_256_128_768_1000_0.001 :subword_space2_2_8_2048_32_512_5000_0.001 :subword_space2_4_4_2048_32_512_25000_0.01 :subword_space2_2_4_768_128_512_260_0.001 :subword_space2_2_8_2048_128_512_100000_0.01 :subword_space2_4_8_1024_32_768_50000_0.001 :subword_space2_4_8_512_128_512_25000_0.01 :subword_space2_4_8_768_128_1024_5000_0.001 :subword_space2_2_8_768_32_512_10000_0.01 :subword_space2_4_8_2048_64_512_100000_0.001 :subword_space2_4_4_768_32_768_25000_0.001 :subword_space2_2_8_256_128_1024_5000_0.01 :subword_space2_4_16_768_32_1024_5000_0.001 :subword_space2_8_16_768_32_1024_25000_0.001 :subword_space2_4_4_512_32_256_10000_0.001 :subword_space2_8_4_1024_128_256_10000_0.01 :subword_space2_4_8_512_128_256_10000_0.001 :subword_space2_8_16_256_64_1024_1000_0.01 :subword_space2_4_8_512_64_768_10000_0.01 :subword_space2_8_4_1024_128_512_1000_0.01 :subword_space2_4_16_512_128_2048_100000_0.01 :subword_space2_4_8_1024_32_768_25000_0.01"
    for model in models.split():
        n_layer, n_head, d_model, d_head, d_inner, vocab_size, lr = model.split(":subword_space2_")[-1].split('_')

        cand_name = ":subword_space2_<n_layer>_<n_head>_<d_model>_<d_head>_<d_inner>_<vocab_size>_<lr>".replace("<n_layer>", str(n_layer)).replace("<n_layer>", str(n_layer)).replace("<n_head>", str(n_head)).replace("<d_model>", str(d_model)).replace("<d_head>", str(d_head)).replace("<d_inner>", str(d_inner)).replace("<vocab_size>", str(vocab_size)).replace("<lr>", str(lr))
        if cand_name in job_cache:
            continue
        job_cache[cand_name] = True
        new_job_cmd = job_cmd.replace("<n_layer>", str(n_layer)).replace("<n_layer>", str(n_layer)).replace("<n_head>", str(n_head)).replace("<d_model>", str(d_model)).replace("<d_head>", str(d_head)).replace("<d_inner>", str(d_inner)).replace("<vocab_size>", str(vocab_size)).replace("<lr>", str(lr))
        res += new_job_cmd +"\n"
        names.append(cand_name)

    #print(res)
    res += "\n# amlt run %s %s word_train"%(f, " ".join(names))
    res += "\n# amlt log word_train %s"%(" ".join(names))
    res += "\n# amlt status word_train %s"%(" ".join(names))
    w = open(f, 'w')
    w.write(res)
    w.close()
    print("amlt run %s %s word_train"%(f, " ".join(names)))

#regenerate_bpe_sweep_models_space2("scripts/9-13-randsearch/generate_bpe_sweep_models_space2.yaml")

def results_generate_bpe_sweep_models_space2():
    # num parameters
    hyp2params = {}
    hyp2nonembparams = {}
    for f in glob.glob("/home/t-gjawahar/archai/amlt/word_train/subword_space2_*/*"):
        num_params = None
        num_ne_params = None
        for line in open(f):
            line = line.strip()
            if "#params" in line:
                num_params = int(line.split()[-1])
            elif "#non emb params" in line:
                num_ne_params = int(line.split()[-1])
        if num_params:
            hyp2params[f.split("/")[-2].split("subword_space2_")[-1]] = num_params
        if num_ne_params:
            hyp2nonembparams[f.split("/")[-2].split("subword_space2_")[-1]] = num_params
    print(hyp2params, len(hyp2params))

    # get exact match scores
    hyp2taskscores = {}
    for f in glob.glob("/home/t-gjawahar/archai/amlt/inference_misc_word/inference_word_valid_400K-subword_space2*/*"):
        hyp = f.split("/")[-2].split("inference_word_valid_400K-subword_space2_")[-1]
        print(f)
        print(hyp)
        if hyp not in hyp2params:
            continue
        scores = []
        overall_numerator, overall_denominator = [0.0]*3, [0.0]*3 # 3 is # tokens (smart compose upto 15)
        i = 0
        for line in open(f):
            if "context=" in line:
                items = line.split(" ")
                score_1 = float(items[5].split(",")[0][1:-1].split("/")[0]) /  float(items[5].split(",")[0][1:-1].split("/")[1])
                score_2 = float(items[7].split(",")[0][1:-1].split("/")[0]) /  float(items[7].split(",")[0][1:-1].split("/")[1])
                score_3 = float(items[-1].split(",")[0][1:-1].split("/")[0]) /  float(items[-1].split(",")[0][1:-1].split("/")[1])
                score_1, score_2, score_3 = float("%.2f"%(100.0*score_1)), float("%.2f"%(100.0*score_2)), float("%.2f"%(100.0*score_3))
                scores.append([score_1, score_2, score_3])
                if i % 2 == 0:
                    overall_numerator[0] += float(items[5].split(",")[0][1:-1].split("/")[0]) 
                    overall_denominator[0] += float(items[5].split(",")[0][1:-1].split("/")[1])
                    overall_numerator[1] +=  float(items[7].split(",")[0][1:-1].split("/")[0]) 
                    overall_denominator[1] += float(items[7].split(",")[0][1:-1].split("/")[1])
                    overall_numerator[2] += float(items[-1].split(",")[0][1:-1].split("/")[0])
                    overall_denominator[2] += float(items[-1].split(",")[0][1:-1].split("/")[1])
                i += 1
        if len(scores) == 0:
            continue
        overall_score = 0.0
        for idx in range(len(overall_numerator)):
            overall_score += overall_numerator[idx]/overall_denominator[idx]
        overall_score /= 3.0
        hyp2taskscores[hyp] = [scores[0], scores[2], scores[4], overall_score]
    print(hyp2taskscores, len(hyp2taskscores))
    scores = []
    for hyp in hyp2taskscores:
        #scores.append((hyp2taskscores[hyp][0][0], [hyp2taskscores[hyp], hyp2params[hyp], hyp]))
        scores.append((hyp2taskscores[hyp][-1], [hyp2taskscores[hyp], hyp2params[hyp], hyp2nonembparams[hyp], hyp]))
    scores = sorted(scores)
    scores.reverse()
    for score in scores:
        #print(score[1])
        print(score[1][-2], score[1][-1].split("_")[-2], score[1][0][-1])

#results_generate_bpe_sweep_models_space2()
#sys.exit(0)

def plot_learning_curve_more_steps():
    import glob
    initial_f2scores = {}
    epochs = {}
    for f in glob.glob("/home/t-gjawahar/archai/amlt/transxl_char_exp2_randsearch/small*/*"):
        initial_f2scores[f.split("/")[-2]] = []
        for line in open(f):
            line = line.strip()
            if "valid ppl" in line:
                items = line.split()
                initial_f2scores[f.split("/")[-2]].append(float(items[-1]))
    restart_f2scores = {}
    for f in glob.glob("/home/t-gjawahar/archai/amlt/misc_transxl_char/char_restart_moresteps_small_*/*"):
        restart_f2scores[f.split("/")[-2]] = []
        for line in open(f):
            line = line.strip()
            if "valid ppl" in line:
                items = line.split()
                restart_f2scores[f.split("/")[-2]].append(float(items[-1]))
    final_f2scores = {}
    for key in initial_f2scores:
        for key1 in restart_f2scores:
            if key in key1:
                #print(key, len(initial_f2scores[key]), len(restart_f2scores[key1]))
                final_f2scores[key] = initial_f2scores[key] + restart_f2scores[key1]
                break
    
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12,5))
    plt.grid(color='gray', linestyle='dashed')
    xaxis = [10000*(i+1) for i in range(160)]
    colors = ['b', 'c', 'y', 'm', 'r', 'g', 'k', "indigo", "violet", "springgreen", "olive", "firebrick", "gold"]
    ei = 0
    for key in sorted(final_f2scores):
        scores = final_f2scores[key] + [None] * (160 - len(final_f2scores[key]))
        plt.plot(xaxis, scores, color=colors[ei], marker='o', label=key)
        print(key, scores)
        ei += 1
    plt.xlabel("Steps")
    plt.ylabel("Valid BPC")
    plt.legend(loc="upper right")
    plt.show()
    fig.savefig("/home/t-gjawahar/archai/scripts/mlrg/char_train_longer.pdf", bbox_to_anchor=(2.25, 2.55)) # bbox_inches='tight')

# plot_learning_curve_more_steps()


def extract_more_steps_diff():
    more_steps_res = {}
    for f in glob.glob("/home/t-gjawahar/archai/amlt/inference_misc_transxl_char/inference_char_valid-char_restart_moresteps_small_*/*.txt"):
        scores = []
        for line in open(f):
            if "context=" in line:
                items = line.split(" ")
                score_1 = float(items[5].split(",")[0][1:-1].split("/")[0]) /  float(items[5].split(",")[0][1:-1].split("/")[1])
                score_2 = float(items[7].split(",")[0][1:-1].split("/")[0]) /  float(items[7].split(",")[0][1:-1].split("/")[1])
                score_3 = float(items[-1].split(",")[0][1:-1].split("/")[0]) /  float(items[-1].split(",")[0][1:-1].split("/")[1])
                score_1, score_2, score_3 = float("%.2f"%(100.0*score_1)), float("%.2f"%(100.0*score_2)), float("%.2f"%(100.0*score_3))
                scores.append([score_1, score_2, score_3])
        more_steps_res[f.split("/")[-2].split("inference_char_valid-char_restart_moresteps_small_")[-1].split("-")[-1][6:]] = [scores[0], scores[2], scores[4]]
    res_400K = {}
    for f in glob.glob("/home/t-gjawahar/archai/amlt/inference_transxl/inference_char_valid-small_*/stdout.txt"):
        if len(f.split("/")[-2].split("inference_char_valid-small_")[-1].split("_")) != 2:
            continue
        scores = []
        for line in open(f):
            if "context=" in line:
                items = line.split(" ")
                score_1 = float(items[5].split(",")[0][1:-1].split("/")[0]) /  float(items[5].split(",")[0][1:-1].split("/")[1])
                score_2 = float(items[7].split(",")[0][1:-1].split("/")[0]) /  float(items[7].split(",")[0][1:-1].split("/")[1])
                score_3 = float(items[-1].split(",")[0][1:-1].split("/")[0]) /  float(items[-1].split(",")[0][1:-1].split("/")[1])
                score_1, score_2, score_3 = float("%.2f"%(100.0*score_1)), float("%.2f"%(100.0*score_2)), float("%.2f"%(100.0*score_3))
                scores.append([score_1, score_2, score_3])
        res_400K[f.split("/")[-2].split("inference_char_valid-small_")[-1].split("-")[-1]] = [scores[0], scores[2], scores[4]]
    print(more_steps_res)
    print(res_400K)
    for key in sorted(res_400K.keys()):
        res = res_400K[key]
        more_steps = more_steps_res[key]
        fstr = key
        for res1, ms2 in zip(res, more_steps):
            for r1, m2 in zip(res1, ms2):
                fstr += ",%.2f"%(m2-r1)
        print(fstr)
        print(res, more_steps)
        break
#extract_more_steps_diff()

def generate_small_char10M_grid_search_models(f):
    import random
    random.seed(123)

    command = "description: run generate_small_char10M_grid_search_models scripts\n\ntarget:\n  service: amlk8s\n  name: ms-shared\n  vc: resrchvc\n\nenvironment:\n  image: pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel\n  registry: docker.io\n  setup:\n    - set -e -o xtrace\n    - sudo apt-get -y install git\n    - pip install --user tensorboard\n\ncode:\n  local_dir: /home/t-gjawahar/archai\n\ndata:\n  local_dir: /home/t-gjawahar/object_dir/wikitext-2-raw-v1-char\n  remote_dir: dataroot/textpred/wikitext-2-raw-v1-char\n\njobs:\n"
    job_cmd = "- name: char10M_grid_<n_layer>_<n_head>_<d_model>_<d_head>_<d_inner>\n  sku: G2\n  command:\n  - set -e -o xtrace\n  - bash scripts/apex_install.sh\n  - pip install --user -e .\n  - python -m torch.distributed.launch --nproc_per_node=\"2\" archai/nlp/nvidia_transformer_xl/train.py --dataset wt2 --warmup_step 4000 --max_step 2000000 --eval_interval 10000 --config dgx1_2gpu_fp16 --config_file char_base.yaml --n_layer <n_layer> --n_head <n_head> --d_model <d_model> --d_head <d_head> --d_inner <d_inner> --save_all --batch_size 128 --lr 0.001 --mem_len 512 --tgt_len 512 --eval_tgt_len 1024 --d_embed <d_model> --dropout 0.1 --dropatt 0.0 --experiment_name char10M_grid_<n_layer>_<n_head>_<d_model>_<d_head>_<d_inner>"
    
    # search_space_1
    n_layers = [8, 12, 16]
    n_heads = [8, 16]
    d_models = [256, 512, 768, 1024]
    d_heads = [32, 64]
    d_inners = [128, 165, 200, 300]

    res = command
    names = []
    job_cache = {}
    while len(job_cache) < 20:
        n_layer = random.choice(n_layers)
        n_head = random.choice(n_heads)
        d_model = random.choice(d_models)
        d_head = random.choice(d_heads)
        d_inner = random.choice(d_inners)

        cand_name = ":char10M_grid_<n_layer>_<n_head>_<d_model>_<d_head>_<d_inner>".replace("<n_layer>", str(n_layer)).replace("<n_layer>", str(n_layer)).replace("<n_head>", str(n_head)).replace("<d_model>", str(d_model)).replace("<d_head>", str(d_head)).replace("<d_inner>", str(d_inner))
        if cand_name in job_cache:
            continue
        job_cache[cand_name] = True
        new_job_cmd = job_cmd.replace("<n_layer>", str(n_layer)).replace("<n_layer>", str(n_layer)).replace("<n_head>", str(n_head)).replace("<d_model>", str(d_model)).replace("<d_head>", str(d_head)).replace("<d_inner>", str(d_inner))
        res += new_job_cmd +"\n"
        names.append(cand_name)

    exp_name = "transxl_char_exp2_randsearch"
    res += "\n# amlt run %s %s %s"%(f, " ".join(names), exp_name)
    res += "\n# amlt log %s %s"%(exp_name, " ".join(names))
    res += "\n# amlt status %s %s"%(exp_name, " ".join(names))
    w = open(f, 'w')
    w.write(res)
    w.close()
    print("amlt run %s %s %s"%(f, " ".join(names), exp_name))

#generate_small_char10M_grid_search_models("scripts/9-3-randsearch/generate_small_char10M_grid_search_models.yaml")

# amlt log inference_transxl :inference_char_valid-char10M_grid_12_8_512_32_300 :inference_char_valid-char10M_grid_8_16_256_32_200 :inference_char_valid-char10M_grid_12_8_256_64_200 :inference_char_valid-char10M_grid_8_16_256_64_200 :inference_char_valid-char10M_grid_16_16_256_32_128 :inference_char_valid-char10M_grid_8_8_512_64_200 :inference_char_valid-char10M_grid_16_8_512_32_200
# amlt log transxl_char_exp2_randsearch :char10M_grid_12_8_512_32_300 :char10M_grid_8_16_256_32_200 :char10M_grid_12_8_256_64_200 :char10M_grid_8_16_256_64_200 :char10M_grid_16_16_256_32_128 :char10M_grid_8_8_512_64_200 :char10M_grid_16_8_512_32_200
def results_generate_small_char10M_grid_search_models():
    # conversion
    num_characters = 0
    num_tokens = 0
    for line in open("/home/t-gjawahar/object_dir/wikitext-2-raw-v1-char/wiki.valid.tokens"):
        line = line.strip()
        if len(line) != 0:
            num_characters += len(line)
            num_tokens += len(line.split())
    import math
    def conversion(bpc):
        avg_loss = math.log(bpc)
        bpc = avg_loss / math.log(2)
        return math.pow(2, bpc * (num_characters / num_tokens))

    # calculate ppl
    hyp2ppl = {}
    hyp2params = {}
    for f in glob.glob("/home/t-gjawahar/archai/amlt/transxl_char_exp2_randsearch/char10M_grid*/*"):
        hyp2ppl[f.split("/")[-2]] = []
        for line in open(f):
            line = line.strip()
            if "valid ppl" in line:
                items = line.split()
                hyp2ppl[f.split("/")[-2]].append(float(items[-1]))
            elif "#params" in line:
                hyp2params[f.split("/")[-2]] = float(line.split()[-1])
    
    # calculate word ppl
    hyp2wordppl = {}
    for f in hyp2ppl:
        hyp2wordppl[f] = [hyp2ppl[f][20], conversion(hyp2ppl[f][20])]
    print(hyp2wordppl)
    
    # get exact match scores
    hyp2taskscores = {}
    for f in glob.glob("/home/t-gjawahar/archai/amlt/inference_transxl/inference_char_valid-char10M_grid*/stdout.txt"):
        scores = []
        for line in open(f):
            if "context=" in line:
                items = line.split(" ")
                score_1 = float(items[5].split(",")[0][1:-1].split("/")[0]) /  float(items[5].split(",")[0][1:-1].split("/")[1])
                score_2 = float(items[7].split(",")[0][1:-1].split("/")[0]) /  float(items[7].split(",")[0][1:-1].split("/")[1])
                score_3 = float(items[-1].split(",")[0][1:-1].split("/")[0]) /  float(items[-1].split(",")[0][1:-1].split("/")[1])
                score_1, score_2, score_3 = float("%.2f"%(100.0*score_1)), float("%.2f"%(100.0*score_2)), float("%.2f"%(100.0*score_3))
                scores.append([score_1, score_2, score_3])
        hyp2taskscores[f.split("/")[-2].split("inference_char_valid-")[-1]] = [scores[0], scores[2], scores[4]]
    print(hyp2taskscores)

    # get scores
    for hyp in hyp2taskscores:
        res = ",".join(hyp.split("char10M_grid_")[-1].split("_"))
        res += "," + str(hyp2taskscores[hyp][1][0]) + "," + str(hyp2taskscores[hyp][1][1]) + "," + str(hyp2taskscores[hyp][1][2])
        res += "," + str(hyp2wordppl[hyp][0]) + "," + "%.3f"%(hyp2wordppl[hyp][1])
        res += "," + str(hyp2params[hyp])
        print(res)

#results_generate_small_char10M_grid_search_models()

def generate_small_embed_sizes(f):
    import random
    random.seed(123)

    command = "description: run generate_small_embed_sizes scripts\n\ntarget:\n  service: amlk8s\n  name: ms-shared\n  vc: resrchvc\n\nenvironment:\n  image: pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel\n  registry: docker.io\n  setup:\n    - set -e -o xtrace\n    - sudo apt-get -y install git\n    - pip install --user tensorboard\n\ncode:\n  local_dir: /home/t-gjawahar/archai\n\ndata:\n  local_dir: /home/t-gjawahar/object_dir/wikitext-2-raw-v1-char\n  remote_dir: dataroot/textpred/wikitext-2-raw-v1-char\n\njobs:\n"
    job_cmd = "- name: small_embed_sizes_<n_layer>_<n_head>_<d_model>_<d_head>_<d_inner>\n  sku: G2\n  command:\n  - set -e -o xtrace\n  - bash scripts/apex_install.sh\n  - pip install --user -e .\n  - python -m torch.distributed.launch --nproc_per_node=\"2\" archai/nlp/nvidia_transformer_xl/train.py --dataset wt2 --warmup_step 4000 --max_step 2000000 --eval_interval 10000 --config dgx1_2gpu_fp16 --config_file char_base.yaml --n_layer <n_layer> --n_head <n_head> --d_model 750 --d_head <d_head> --d_inner <d_inner> --save_all --batch_size 128 --lr 0.001 --mem_len 512 --tgt_len 512 --eval_tgt_len 1024 --d_embed <d_model> --dropout 0.1 --dropatt 0.0 --experiment_name small_embed_sizes_<n_layer>_<n_head>_<d_model>_<d_head>_<d_inner>"

    # search_space_1
    n_layers = [16]
    n_heads = [8]
    d_models = [10, 50, 100, 150]
    d_heads = [64]
    d_inners = [2048]

    res = command
    names = []
    job_cache = {}
    while len(job_cache) < 4:
        n_layer = random.choice(n_layers)
        n_head = random.choice(n_heads)
        d_model = random.choice(d_models)
        d_head = random.choice(d_heads)
        d_inner = random.choice(d_inners)

        cand_name = ":small_embed_sizes_<n_layer>_<n_head>_<d_model>_<d_head>_<d_inner>".replace("<n_layer>", str(n_layer)).replace("<n_layer>", str(n_layer)).replace("<n_head>", str(n_head)).replace("<d_model>", str(d_model)).replace("<d_head>", str(d_head)).replace("<d_inner>", str(d_inner))
        if cand_name in job_cache:
            continue
        job_cache[cand_name] = True
        new_job_cmd = job_cmd.replace("<n_layer>", str(n_layer)).replace("<n_layer>", str(n_layer)).replace("<n_head>", str(n_head)).replace("<d_model>", str(d_model)).replace("<d_head>", str(d_head)).replace("<d_inner>", str(d_inner))
        res += new_job_cmd +"\n"
        names.append(cand_name)

    exp_name = "transxl_char_exp2_randsearch"
    res += "\n# amlt run %s %s %s"%(f, " ".join(names), exp_name)
    res += "\n# amlt log %s %s"%(exp_name, " ".join(names))
    res += "\n# amlt status %s %s"%(exp_name, " ".join(names))
    w = open(f, 'w')
    w.write(res)
    w.close()
    print("amlt run %s %s %s"%(f, " ".join(names), exp_name))

#generate_small_embed_sizes("scripts/9-5/generate_small_embed_sizes.yaml")

def generate_small_embed_sizes_g8(f):
    import random
    random.seed(123)

    command = "description: run generate_small_embed_sizes scripts\n\ntarget:\n  service: amlk8s\n  name: ms-shared\n  vc: resrchvc\n\nenvironment:\n  image: pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel\n  registry: docker.io\n  setup:\n    - set -e -o xtrace\n    - sudo apt-get -y install git\n    - pip install --user tensorboard\n\ncode:\n  local_dir: /home/t-gjawahar/archai\n\ndata:\n  local_dir: /home/t-gjawahar/object_dir/wikitext-2-raw-v1-char\n  remote_dir: dataroot/textpred/wikitext-2-raw-v1-char\n\njobs:\n"
    job_cmd = "- name: small_embed_sizes_g8_<n_layer>_<n_head>_<d_model>_<d_head>_<d_inner>\n  sku: G8\n  command:\n  - set -e -o xtrace\n  - bash scripts/apex_install.sh\n  - pip install --user -e .\n  - python -m torch.distributed.launch --nproc_per_node=\"8\" archai/nlp/nvidia_transformer_xl/train.py --dataset wt2 --warmup_step 4000 --max_step 2000000 --eval_interval 10000 --config dgx1_8gpu_fp16 --config_file char_base.yaml --n_layer <n_layer> --n_head <n_head> --d_model 750 --d_head <d_head> --d_inner <d_inner> --save_all --batch_size 128 --lr 0.001 --mem_len 512 --tgt_len 512 --eval_tgt_len 1024 --d_embed <d_model> --dropout 0.1 --dropatt 0.0 --experiment_name small_embed_sizes_g8_<n_layer>_<n_head>_<d_model>_<d_head>_<d_inner>"

    # search_space_1
    n_layers = [16]
    n_heads = [8]
    d_models = [10, 50, 100, 150]
    d_heads = [64]
    d_inners = [2048]

    res = command
    names = []
    job_cache = {}
    while len(job_cache) < 4:
        n_layer = random.choice(n_layers)
        n_head = random.choice(n_heads)
        d_model = random.choice(d_models)
        d_head = random.choice(d_heads)
        d_inner = random.choice(d_inners)

        cand_name = ":small_embed_sizes_g8_<n_layer>_<n_head>_<d_model>_<d_head>_<d_inner>".replace("<n_layer>", str(n_layer)).replace("<n_layer>", str(n_layer)).replace("<n_head>", str(n_head)).replace("<d_model>", str(d_model)).replace("<d_head>", str(d_head)).replace("<d_inner>", str(d_inner))
        if cand_name in job_cache:
            continue
        job_cache[cand_name] = True
        new_job_cmd = job_cmd.replace("<n_layer>", str(n_layer)).replace("<n_layer>", str(n_layer)).replace("<n_head>", str(n_head)).replace("<d_model>", str(d_model)).replace("<d_head>", str(d_head)).replace("<d_inner>", str(d_inner))
        res += new_job_cmd +"\n"
        names.append(cand_name)

    exp_name = "transxl_char_exp2_randsearch"
    res += "\n# amlt run %s %s %s"%(f, " ".join(names), exp_name)
    res += "\n# amlt log %s %s"%(exp_name, " ".join(names))
    res += "\n# amlt status %s %s"%(exp_name, " ".join(names))
    w = open(f, 'w')
    w.write(res)
    w.close()
    print("amlt run %s %s %s"%(f, " ".join(names), exp_name))

#generate_small_embed_sizes_g8("scripts/9-5/generate_small_embed_sizes_g8.yaml")


def generate_wordmodel_layer_copy():
    import random
    random.seed(123)
    
    # word80M g2
    job_cmd = "- name: wordmodel_80M_layer_copy_<lidx>\n  sku: G2\n  command:\n  - set -e -o xtrace\n  - bash scripts/apex_install.sh\n  - pip install --user -e .\n  - python -m torch.distributed.launch --nproc_per_node=\"2\" archai/nlp/nvidia_transformer_xl/train.py --dataset wt2 --warmup_step 4000 --max_step 2000000 --eval_interval 10000 --config dgx1_2gpu_fp16 --config_file char_base.yaml --n_layer 16 --n_head 8 --d_model 256 --d_head 32 --d_inner 768 --save_all --batch_size 128 --lr 0.001 --mem_len 512 --tgt_len 512 --eval_tgt_len 1024 --d_embed 1000 --dropout 0.1 --dropatt 0.0 --experiment_name wordmodel_80M_layer_copy_<lidx> --layer_init_from_ckpt $$AMLT_MAP_INPUT_DIR/checkpoint_best.pt --layer_idx_to_init <lidx>"

    # word40M g2
    job_cmd = "- name: wordmodel_40M_layer_copy_<lidx>\n  sku: G2\n  command:\n  - set -e -o xtrace\n  - bash scripts/apex_install.sh\n  - pip install --user -e .\n  - python -m torch.distributed.launch --nproc_per_node=\"2\" archai/nlp/nvidia_transformer_xl/train.py --dataset wt2 --warmup_step 4000 --max_step 2000000 --eval_interval 10000 --config dgx1_2gpu_fp16 --config_file char_base.yaml --n_layer 14 --n_head 8 --d_model 128 --d_head 32 --d_inner 900 --save_all --batch_size 128 --lr 0.001 --mem_len 512 --tgt_len 512 --eval_tgt_len 1024 --d_embed 1000 --dropout 0.1 --dropatt 0.0 --experiment_name wordmodel_40M_layer_copy_<lidx> --layer_init_from_ckpt $$AMLT_MAP_INPUT_DIR/checkpoint_best.pt --layer_idx_to_init <lidx>"

    # word50M g2
    job_cmd = "- name: wordmodel_50M_layer_copy_<lidx>\n  sku: G2\n  command:\n  - set -e -o xtrace\n  - bash scripts/apex_install.sh\n  - pip install --user -e .\n  - python -m torch.distributed.launch --nproc_per_node=\"2\" archai/nlp/nvidia_transformer_xl/train.py --dataset wt2 --warmup_step 4000 --max_step 2000000 --eval_interval 10000 --config dgx1_2gpu_fp16 --config_file char_base.yaml --n_layer 16 --n_head 8 --d_model 160 --d_head 32 --d_inner 800 --save_all --batch_size 128 --lr 0.001 --mem_len 512 --tgt_len 512 --eval_tgt_len 1024 --d_embed 1000 --dropout 0.1 --dropatt 0.0 --experiment_name wordmodel_50M_layer_copy_<lidx> --layer_init_from_ckpt $$AMLT_MAP_INPUT_DIR/checkpoint_best.pt --layer_idx_to_init <lidx>"
    
    lidx = ["0-100", "0-25", "0-50", "0-75", "75-100", "50-100", "25-100", "25-75", "0-0"]
    lidx = ["0-0", "0-10", "0-15", "0-20", "0-25"]
    #lidx = ["0-0", "0-10", "0-20", "0-25"]
    res = ""
    names = []
    for lid in lidx:
        new_job_cmd = job_cmd.replace("<lidx>", lid)
        res += new_job_cmd +"\n"
        names.append("wordmodel_50M_layer_copy_<lidx>".replace("<lidx>", lid))
    print(res)
    for name in names:
        print("amlt map archai/nlp/nvidia_transformer_xl/word_train.yaml :%s word_train :word50M misc_word --yes"%name)

#generate_wordmodel_layer_copy()

# amlt status misc_word
# amlt log misc_word :wordmodel_80M_layer_copy_0-100-word80M :wordmodel_80M_layer_copy_0-25-word80M :wordmodel_80M_layer_copy_0-50-word80M :wordmodel_80M_layer_copy_0-75-word80M :wordmodel_80M_layer_copy_75-100-word80M :wordmodel_80M_layer_copy_50-100-word80M :wordmodel_80M_layer_copy_25-100-word80M :wordmodel_80M_layer_copy_25-75-word80M :wordmodel_80M_layer_copy_0-0-word80M
def results_generate_wordmodel_layer_copy():
    # conversion
    num_characters = 0
    num_tokens = 0
    for line in open("/home/t-gjawahar/object_dir/wikitext-2-raw-v1-char/wiki.valid.tokens"):
        line = line.strip()
        if len(line) != 0:
            num_characters += len(line)
            num_tokens += len(line.split())
    import math
    def conversion(bpc):
        avg_loss = math.log(bpc)
        bpc = avg_loss / math.log(2)
        return math.pow(2, bpc * (num_characters / num_tokens))

    exp2scores = {}
    max_steps = 0
    for f in glob.glob("/home/t-gjawahar/archai/amlt/misc_word/wordmodel*/*"):
        scores = []
        for line in open(f):
            line = line.strip()
            if "valid ppl" in line:
                items = line.split()
                score = conversion(float(items[-1]))
                if score < 10000:
                    scores.append(score)
        exp2scores[f.split("/")[-2].split("_copy_")[-1].split("-word80M")[0]] = scores
        max_steps = max(max_steps, len(scores))
    print(exp2scores)

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12,5))
    plt.grid(color='gray', linestyle='dashed')
    xaxis = [10000*(i+1) for i in range(max_steps)]
    colors = ['b', 'c', 'y', 'm', 'r', 'g', 'k', "indigo", "violet", "springgreen", "olive", "firebrick", "gold"]
    ei = 0
    for key in sorted(exp2scores):
        #if key.startswith("0"):
        if key.startswith("0-0") or not key.startswith("0"):
            scores = exp2scores[key] + [None] * (max_steps - len(exp2scores[key]))
            plt.plot(xaxis, scores, color=colors[ei], marker='o', label=key)
            #print(key, scores)
            ei += 1
            print(key, len(exp2scores[key]))
    plt.xlabel("Steps")
    plt.ylabel("Valid PPL")
    plt.legend(loc="upper right")
    plt.show()
    fig.savefig("/home/t-gjawahar/archai/scripts/mlrg/results_generate_wordmodel_layer_copy.pdf", bbox_to_anchor=(2.25, 2.55)) # bbox_inches='tight')
    
    more_steps_res = {}
    for f in glob.glob("/home/t-gjawahar/archai/amlt/inference_misc_word/inference_char_valid_100K-*/*.txt"):
        scores = []
        for line in open(f):
            if "context=" in line:
                items = line.split(" ")
                score_1 = float(items[5].split(",")[0][1:-1].split("/")[0]) /  float(items[5].split(",")[0][1:-1].split("/")[1])
                score_2 = float(items[7].split(",")[0][1:-1].split("/")[0]) /  float(items[7].split(",")[0][1:-1].split("/")[1])
                score_3 = float(items[-1].split(",")[0][1:-1].split("/")[0]) /  float(items[-1].split(",")[0][1:-1].split("/")[1])
                score_1, score_2, score_3 = float("%.2f"%(100.0*score_1)), float("%.2f"%(100.0*score_2)), float("%.2f"%(100.0*score_3))
                scores.append([score_1, score_2, score_3])
        more_steps_res[f.split("/")[-2].split("_copy_")[-1].split("-word80M")[0]] = [scores[0], scores[2], scores[4]]
    print(more_steps_res)
    for res in sorted(more_steps_res):
        st = res
        for s in more_steps_res[res]:
            for s1 in s:
                st += ',' + str(s1)
        print(st)
    
#results_generate_wordmodel_layer_copy()

def generate_wordmodel_layer_copy_warmup_lr_tuning():
    import random
    random.seed(123)

    # word40M g2
    job_cmd = "- name: wordmodel_40M_layer_copy_<lidx>_<warmup>_<lr>\n  sku: G4\n  command:\n  - set -e -o xtrace\n  - bash scripts/apex_install.sh\n  - pip install --user -e .\n  - python -m torch.distributed.launch --nproc_per_node=\"4\" archai/nlp/nvidia_transformer_xl/train.py --dataset wt2 --warmup_step <warmup> --max_step 200000 --eval_interval 10000 --config dgx1_4gpu_fp16 --config_file char_base.yaml --n_layer 14 --n_head 8 --d_model 128 --d_head 32 --d_inner 900 --save_all --batch_size 128 --lr <lr> --mem_len 512 --tgt_len 512 --eval_tgt_len 1024 --d_embed 1000 --dropout 0.1 --dropatt 0.0 --experiment_name wordmodel_40M_layer_copy_<lidx>_<warmup>_<lr> --layer_init_from_ckpt $$AMLT_MAP_INPUT_DIR/checkpoint_best.pt --layer_idx_to_init <lidx>"
    
    #lidx = ["0-100", "0-25", "0-50", "0-75", "75-100", "50-100", "25-100", "25-75", "0-0"]
    #lidx = ["0-0", "0-10", "0-15", "0-20", "0-25"]
    #lidx = ["0-0", "0-10", "0-20", "0-25"]
    lid = "0-20"
    res = ""
    warmup_steps = [0, 4000]
    lrs = [0.01, 0.001, 0.0001]
    names = []
    for step in warmup_steps:
        for lr in lrs:
            new_job_cmd = job_cmd.replace("<lidx>", lid).replace("<warmup>", str(step)).replace("<lr>", str(lr))
            res += new_job_cmd +"\n"
            names.append("wordmodel_40M_layer_copy_<lidx>_<warmup>_<lr>".replace("<lidx>", lid).replace("<warmup>", str(step)).replace("<lr>", str(lr)))
    print(res)
    for name in names:
        print("amlt map archai/nlp/nvidia_transformer_xl/word_train.yaml :%s word_train :word40M misc_word --yes"%name)

#generate_wordmodel_layer_copy_warmup_lr_tuning()

def generate_subword2char_layer_copy():
    import random
    random.seed(123)

    # subword (top based on overall EM)  4_8_512_64_512_10000_0.01 (space1)
    # [[[21.62, 6.06, 1.64], [26.66, 7.71, 2.29], [29.37, 8.56, 3.38], 0.15195327250264076], 12484883, '4_8_512_64_512_10000_0.01']
    # scripts/8-31-randsearch/generate_bpe_sweep_models.yaml
    # python -m torch.distributed.launch --nproc_per_node="2" archai/nlp/nvidia_transformer_xl/train.py --config dgx1_2gpu_fp16 --config_file wt103_base.yaml --n_layer 4 --n_head 8 --d_model 512 --d_head 64 --d_inner 512 --max_step 100000 --vocab bpe --vocab_size 10000 --lr 0.01
    #job_cmd = "- name: subwordmodel_12M_layer_copy_<lidx>\n  sku: G4\n  command:\n  - set -e -o xtrace\n  - bash scripts/apex_install.sh\n  - pip install --user -e .\n  - python -m torch.distributed.launch --nproc_per_node=\"4\" archai/nlp/nvidia_transformer_xl/train.py --dataset wt2 --warmup_step 4000 --max_step 200000 --eval_interval 10000 --config dgx1_4gpu_fp16 --config_file char_base.yaml --n_layer 4 --n_head 8 --d_model 512 --d_head 64 --d_inner 512 --save_all --batch_size 128 --lr 0.001 --mem_len 512 --tgt_len 512 --eval_tgt_len 1024 --d_embed 512 --dropout 0.1 --dropatt 0.0 --experiment_name subwordmodel_12M_layer_copy_<lidx> --layer_init_from_ckpt $$AMLT_MAP_INPUT_DIR/checkpoint_best.pt --layer_idx_to_init <lidx>"

    # subword (top based on overall EM)  8_4_2048_64_2048_5000_0.01 (space1)
    # [[[24.9, 6.96, 2.38], [30.54, 9.42, 3.52], [32.48, 10.83, 5.01], 0.1401888142836454], 98430347, '8_4_2048_64_2048_5000_0.01']
    # scripts/9-13-randsearch/generate_bpe_sweep_models_space2.yaml
    # python -m torch.distributed.launch --nproc_per_node="4" archai/nlp/nvidia_transformer_xl/train.py --config dgx1_4gpu_fp16 --config_file wt103_base.yaml --n_layer 8 --n_head 4 --d_model 2048 --d_head 64 --d_inner 2048 --max_step 400000 --vocab bpe --vocab_size 5000 --lr 0.01 --save_all
    job_cmd = "- name: subwordmodel_98M_layer_copy_<lidx>\n  sku: G4\n  command:\n  - set -e -o xtrace\n  - bash scripts/apex_install.sh\n  - pip install --user -e .\n  - python -m torch.distributed.launch --nproc_per_node=\"4\" archai/nlp/nvidia_transformer_xl/train.py --dataset wt2 --warmup_step 4000 --max_step 200000 --eval_interval 10000 --config dgx1_4gpu_fp16 --config_file char_base.yaml --n_layer 8 --n_head 4 --d_model 2048 --d_head 64 --d_inner 2048 --save_all --batch_size 128 --lr 0.001 --mem_len 512 --tgt_len 512 --eval_tgt_len 1024 --d_embed 2048 --dropout 0.1 --dropatt 0.0 --experiment_name subwordmodel_98M_layer_copy_<lidx> --layer_init_from_ckpt $$AMLT_MAP_INPUT_DIR/checkpoint_best.pt --layer_idx_to_init <lidx>"
    
    lidx = ["0-0", "0-25", "0-50", "0-75", "0-100"]
    lidx = ["0-0", "0-20", "0-25", "0-40", "0-50", "0-70", "0-80", "0-90", "0-100"]
    res = ""
    names = []
    for lid in lidx:
        new_job_cmd = job_cmd.replace("<lidx>", lid)
        res += new_job_cmd +"\n"
        names.append("subwordmodel_98M_layer_copy_<lidx>".replace("<lidx>", lid))
    print(res)
    for name in names:
        print("amlt map scripts/9-13-randsearch/generate_bpe_sweep_models_space2.yaml :%s word_train :subword_space2_8_4_2048_64_2048_5000_0.01 misc_word --yes"%name)

#generate_subword2char_layer_copy()

def generate_word2subword_layer_copy():
    import random
    random.seed(123)

    # subword example
    # - python -m torch.distributed.launch --nproc_per_node="2" archai/nlp/nvidia_transformer_xl/train.py --config dgx1_2gpu_fp16 --config_file wt103_base.yaml --n_layer 2 --n_head 8 --d_model 512 --d_head 64 --d_inner 512 --max_step 100000 --vocab bpe --vocab_size 260 --lr 0.01

    # word80M g2
    #job_cmd = "- name: wordmodel_80M_layer_copy_<lidx>\n  sku: G2\n  command:\n  - set -e -o xtrace\n  - bash scripts/apex_install.sh\n  - pip install --user -e .\n  - python -m torch.distributed.launch --nproc_per_node=\"2\" archai/nlp/nvidia_transformer_xl/train.py --dataset wt2 --warmup_step 4000 --max_step 2000000 --eval_interval 10000 --config dgx1_2gpu_fp16 --config_file char_base.yaml --n_layer 16 --n_head 8 --d_model 256 --d_head 32 --d_inner 768 --save_all --batch_size 128 --lr 0.001 --mem_len 512 --tgt_len 512 --eval_tgt_len 1024 --d_embed 1000 --dropout 0.1 --dropatt 0.0 --experiment_name wordmodel_80M_layer_copy_<lidx> --layer_init_from_ckpt $$AMLT_MAP_INPUT_DIR/checkpoint_best.pt --layer_idx_to_init <lidx>"
    #job_cmd = "- name: word2subword_word80M_layer_copy_<lidx>\n  sku: G2\n  command:\n  - set -e -o xtrace\n  - bash scripts/apex_install.sh\n  - pip install --user -e .\n  - python -m torch.distributed.launch --nproc_per_node=\"2\" archai/nlp/nvidia_transformer_xl/train.py --max_step 100000 --config dgx1_2gpu_fp16 --config_file wt103_base_no_fp.yaml --n_layer 16 --n_head 8 --d_model 256 --d_head 32 --d_inner 768 --save_all --lr 0.01 --d_embed 512 --dropout 0.1 --dropatt 0.0 --experiment_name word2subword_word80M_layer_copy_<lidx> --layer_init_from_ckpt $$AMLT_MAP_INPUT_DIR/checkpoint_best.pt --layer_idx_to_init <lidx> --vocab bpe --vocab_size 10000"

    # word40M g2
    #job_cmd = "- name: wordmodel_40M_layer_copy_<lidx>\n  sku: G2\n  command:\n  - set -e -o xtrace\n  - bash scripts/apex_install.sh\n  - pip install --user -e .\n  - python -m torch.distributed.launch --nproc_per_node=\"2\" archai/nlp/nvidia_transformer_xl/train.py --dataset wt2 --warmup_step 4000 --max_step 2000000 --eval_interval 10000 --config dgx1_2gpu_fp16 --config_file char_base.yaml --n_layer 14 --n_head 8 --d_model 128 --d_head 32 --d_inner 900 --save_all --batch_size 128 --lr 0.001 --mem_len 512 --tgt_len 512 --eval_tgt_len 1024 --d_embed 1000 --dropout 0.1 --dropatt 0.0 --experiment_name wordmodel_40M_layer_copy_<lidx> --layer_init_from_ckpt $$AMLT_MAP_INPUT_DIR/checkpoint_best.pt --layer_idx_to_init <lidx>"
    job_cmd = "- name: word2subword_word40M_layer_copy_<lidx>\n  sku: G2\n  command:\n  - set -e -o xtrace\n  - bash scripts/apex_install.sh\n  - pip install --user -e .\n  - python -m torch.distributed.launch --nproc_per_node=\"2\" archai/nlp/nvidia_transformer_xl/train.py --max_step 100000 --config dgx1_2gpu_fp16 --config_file wt103_base_no_fp.yaml --n_layer 14 --n_head 8 --d_model 128 --d_head 32 --d_inner 900 --save_all --lr 0.01 --d_embed 512 --dropout 0.1 --dropatt 0.0 --experiment_name word2subword_word40M_layer_copy_<lidx> --layer_init_from_ckpt $$AMLT_MAP_INPUT_DIR/checkpoint_best.pt --layer_idx_to_init <lidx> --vocab bpe --vocab_size 10000"
    
    lidx = ["0-0", "0-10", "0-15", "0-20", "0-25", "0-50", "0-75", "0-100"]
    res = ""
    names = []
    for lid in lidx:
        new_job_cmd = job_cmd.replace("<lidx>", lid)
        res += new_job_cmd +"\n"
        names.append("word2subword_word40M_layer_copy_<lidx>".replace("<lidx>", lid))
    print(res)
    for name in names:
        print("amlt map archai/nlp/nvidia_transformer_xl/word_train.yaml :%s word_train :word40M misc_word --yes"%name)

#generate_word2subword_layer_copy()

def generate_word2subword_layer_copy_embed_copy():
    import random
    random.seed(123)

    # subword example
    # - python -m torch.distributed.launch --nproc_per_node="2" archai/nlp/nvidia_transformer_xl/train.py --config dgx1_2gpu_fp16 --config_file wt103_base.yaml --n_layer 2 --n_head 8 --d_model 512 --d_head 64 --d_inner 512 --max_step 100000 --vocab bpe --vocab_size 260 --lr 0.01
    # word40M g2
    #job_cmd = "- name: wordmodel_40M_layer_copy_<lidx>\n  sku: G2\n  command:\n  - set -e -o xtrace\n  - bash scripts/apex_install.sh\n  - pip install --user -e .\n  - python -m torch.distributed.launch --nproc_per_node=\"2\" archai/nlp/nvidia_transformer_xl/train.py --dataset wt2 --warmup_step 4000 --max_step 2000000 --eval_interval 10000 --config dgx1_2gpu_fp16 --config_file char_base.yaml --n_layer 14 --n_head 8 --d_model 128 --d_head 32 --d_inner 900 --save_all --batch_size 128 --lr 0.001 --mem_len 512 --tgt_len 512 --eval_tgt_len 1024 --d_embed 1000 --dropout 0.1 --dropatt 0.0 --experiment_name wordmodel_40M_layer_copy_<lidx> --layer_init_from_ckpt $$AMLT_MAP_INPUT_DIR/checkpoint_best.pt --layer_idx_to_init <lidx>"
    job_cmd = "- name: word2subword_word40M_layer_copy_<lidx>_<embed_layer_init>\n  sku: G2\n  command:\n  - set -e -o xtrace\n  - bash scripts/apex_install.sh\n  - pip install --user -e .\n  - python -m torch.distributed.launch --nproc_per_node=\"2\" archai/nlp/nvidia_transformer_xl/train.py --max_step 100000 --config dgx1_2gpu_fp16 --config_file wt103_base_no_fp.yaml --n_layer 14 --n_head 8 --d_model 128 --d_head 32 --d_inner 900 --save_all --lr 0.01 --d_embed 768 --dropout 0.1 --dropatt 0.0 --experiment_name word2subword_word40M_layer_copy_<lidx> --layer_init_from_ckpt $$AMLT_MAP_INPUT_DIR/checkpoint_best.pt --layer_idx_to_init <lidx> --vocab bpe --vocab_size 10000 --embed_layer_init <embed_layer_init>"
    
    lidx = ["0-0", "0-10", "0-15", "0-20"] #, "0-25", "0-50", "0-75", "0-100"]
    embed_layer_init = ["gpt2"]
    res = ""
    names = []
    for eli in embed_layer_init:
        for lid in lidx:
            new_job_cmd = job_cmd.replace("<lidx>", lid).replace("<embed_layer_init>", eli)
            res += new_job_cmd +"\n"
            names.append("word2subword_word40M_layer_copy_<lidx>_<embed_layer_init>".replace("<lidx>", lid).replace("<embed_layer_init>", eli))

    job_cmd = "- name: word2subword_word40M_layer_copy_<lidx>_None\n  sku: G2\n  command:\n  - set -e -o xtrace\n  - bash scripts/apex_install.sh\n  - pip install --user -e .\n  - python -m torch.distributed.launch --nproc_per_node=\"2\" archai/nlp/nvidia_transformer_xl/train.py --max_step 100000 --config dgx1_2gpu_fp16 --config_file wt103_base.yaml --n_layer 14 --n_head 8 --d_model 128 --d_head 32 --d_inner 900 --save_all --lr 0.01 --d_embed 768 --dropout 0.1 --dropatt 0.0 --experiment_name word2subword_word40M_layer_copy_<lidx> --layer_init_from_ckpt $$AMLT_MAP_INPUT_DIR/checkpoint_best.pt --layer_idx_to_init <lidx> --vocab bpe --vocab_size 10000"
    for lid in lidx:
        new_job_cmd = job_cmd.replace("<lidx>", lid)
        res += new_job_cmd +"\n"
        names.append("word2subword_word40M_layer_copy_<lidx>_None".replace("<lidx>", lid))
    print(res)
    for name in names:
        print("amlt map archai/nlp/nvidia_transformer_xl/word_train.yaml :%s word_train :word40M misc_word --yes"%name)

#generate_word2subword_layer_copy_embed_copy()

def generate_char2subword_layer_copy():
    import random
    random.seed(123)

    # subword example
    # - python -m torch.distributed.launch --nproc_per_node="2" archai/nlp/nvidia_transformer_xl/train.py --config dgx1_2gpu_fp16 --config_file wt103_base.yaml --n_layer 2 --n_head 8 --d_model 512 --d_head 64 --d_inner 512 --max_step 100000 --vocab bpe --vocab_size 260 --lr 0.01

    # transxl_char_params_80M g2
    # python -m torch.distributed.launch --nproc_per_node="8" archai/nlp/nvidia_transformer_xl/train.py --dataset wt2 --warmup_step 4000 --max_step 400000 --eval_interval 10000 --n_layer 16 --n_head 8 --d_head 64 --d_embed 750 --d_inner 2048 --mem_len 512 --tgt_len 512 --d_model 750 --dropout 0.1 -dropatt 0.0 --config dgx1_8gpu_fp16 --experiment_name transxl_char_base_enwiki --config_file char_no_fp.yaml --eval_tgt_len 1024 --batch_size 64 --lr 0.001
    job_cmd = "- name: char2subword_char80M_layer_copy_<lidx>\n  sku: G2\n  command:\n  - set -e -o xtrace\n  - bash scripts/apex_install.sh\n  - pip install --user -e .\n  - python -m torch.distributed.launch --nproc_per_node=\"2\" archai/nlp/nvidia_transformer_xl/train.py --max_step 100000 --config dgx1_2gpu_fp16 --config_file wt103_base.yaml --n_layer 16 --n_head 8 --d_model 750 --d_head 64 --d_inner 768 --save_all --lr 0.01 --d_embed 512 --dropout 0.1 --dropatt 0.0 --experiment_name char2subword_char80M_layer_copy_<lidx> --layer_init_from_ckpt $$AMLT_MAP_INPUT_DIR/checkpoint_best.pt --layer_idx_to_init <lidx> --vocab bpe --vocab_size 10000"

    lidx = ["0-0", "0-10", "0-15", "0-20", "0-25", "0-50", "0-75", "0-100"]
    res = ""
    names = []
    for lid in lidx:
        new_job_cmd = job_cmd.replace("<lidx>", lid)
        res += new_job_cmd +"\n"
        names.append("char2subword_char80M_layer_copy_<lidx>".replace("<lidx>", lid))
    print(res)
    for name in names:
        print("amlt map archai/nlp/nvidia_transformer_xl/run_char_philly_exp3_char_final_select.yaml :%s transxl_char_exp2_randsearch :transxl_char_params_80M misc_word --yes"%name)

#generate_char2subword_layer_copy()

def generate_small_char10M_grid_search_bertstyle_models(f):
    import random
    random.seed(123)

    command = "description: run generate_small_char10M_grid_search_bertstyle_models scripts\n\ntarget:\n  service: amlk8s\n  name: ms-shared\n  vc: resrchvc\n\nenvironment:\n  image: pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel\n  registry: docker.io\n  setup:\n    - set -e -o xtrace\n    - sudo apt-get -y install git\n    - pip install --user tensorboard\n\ncode:\n  local_dir: /home/t-gjawahar/archai\n\ndata:\n  local_dir: /home/t-gjawahar/object_dir/wikitext-2-raw-v1-char\n  remote_dir: dataroot/textpred/wikitext-2-raw-v1-char\n\njobs:\n"
    job_cmd = "- name: char10M_grid_bertstyle_<n_layer>_<n_head>_<d_model>_<d_head>_<d_inner>_<segment_type>\n  sku: G2\n  command:\n  - set -e -o xtrace\n  - bash scripts/apex_install.sh\n  - pip install --user -e .\n  - python -m torch.distributed.launch --nproc_per_node=\"2\" archai/nlp/nvidia_transformer_xl/train.py --dataset wt2 --warmup_step 4000 --max_step 2000000 --eval_interval 10000 --config dgx1_2gpu_fp16 --config_file char_base.yaml --n_layer <n_layer> --n_head <n_head> --d_model <d_model> --d_head <d_head> --d_inner <d_inner> --save_all --batch_size 128 --lr 0.001 --mem_len 512 --tgt_len 512 --eval_tgt_len 1024 --d_embed <d_model> --dropout 0.1 --dropatt 0.0 --experiment_name char10M_grid_bertstyle_<n_layer>_<n_head>_<d_model>_<d_head>_<d_inner> --model_ext bert_style_word_segment --segment_type <segment_type>"
    
    # search_space_1
    n_layers = [8, 12, 16]
    n_heads = [8, 16]
    d_models = [256, 512, 768, 1024]
    d_heads = [32, 64]
    d_inners = [128, 165, 200, 300]
    segment_types = ["word", "subword"]

    res = command
    names = []
    job_cache = {}
    while len(job_cache) < 50:
        n_layer = random.choice(n_layers)
        n_head = random.choice(n_heads)
        d_model = random.choice(d_models)
        d_head = random.choice(d_heads)
        d_inner = random.choice(d_inners)
        segment_type = random.choice(segment_types)

        cand_name = ":char10M_grid_bertstyle_<n_layer>_<n_head>_<d_model>_<d_head>_<d_inner>_<segment_type>".replace("<n_layer>", str(n_layer)).replace("<n_layer>", str(n_layer)).replace("<n_head>", str(n_head)).replace("<d_model>", str(d_model)).replace("<d_head>", str(d_head)).replace("<d_inner>", str(d_inner)).replace("<segment_type>", segment_type)
        if cand_name in job_cache:
            continue
        job_cache[cand_name] = True
        new_job_cmd = job_cmd.replace("<n_layer>", str(n_layer)).replace("<n_layer>", str(n_layer)).replace("<n_head>", str(n_head)).replace("<d_model>", str(d_model)).replace("<d_head>", str(d_head)).replace("<d_inner>", str(d_inner)).replace("<segment_type>", segment_type)
        res += new_job_cmd +"\n"
        names.append(cand_name)

    exp_name = "transxl_char_exp2_randsearch"
    res += "\n# amlt run %s %s %s"%(f, " ".join(names), exp_name)
    res += "\n# amlt log %s %s"%(exp_name, " ".join(names))
    res += "\n# amlt status %s %s"%(exp_name, " ".join(names))
    w = open(f, 'w')
    w.write(res)
    w.close()
    print("amlt run %s %s %s"%(f, " ".join(names), exp_name))

#generate_small_char10M_grid_search_bertstyle_models("scripts/9-3-randsearch/generate_small_char10M_grid_search_models_bertstyle.yaml")

def cancel_jobs():
    #cmd = "amlt abort transxl_char_exp2_randsearch"
    #for line in open("/home/t-gjawahar/archai/scripts/cancel.sh"):
    #    item = line.strip().split()[0]
    #    cmd += " %s"%item
    #print(cmd)
    
    cmd = "amlt map archai/nlp/nvidia_transformer_xl/run_char_philly_exp3_char_final_select.yaml :inference_char_valid_50K transxl_char_exp2_randsearch"
    for line in open("/home/t-gjawahar/archai/scripts/cancel.sh"):
        if "killed" in line:
            continue
        item = line.strip().split()[0]
        cmd += " %s"%item
    cmd += " inference_transxl"

    # cancel.sh
    cmd = "amlt run scripts/9-13-randsearch/generate_bpe_sweep_models_space2.yaml"
    for line in open("/home/t-gjawahar/archai/scripts/cancel.sh"):
        if "running" in line:
            continue
        item = line.strip().split()[0]
        cmd += " %s"%item
    cmd += " word_train"
    print(cmd)
#cancel_jobs()

def generate_inference_commands():
    #cmd = "amlt map archai/nlp/nvidia_transformer_xl/word_train.yaml :inference_char_valid_200K misc_word"
    #for line in open("/home/t-gjawahar/archai/scripts/inference.sh"):
    #    item = line.strip().split()[0]
    #    cmd += " %s"%item
    #cmd += " inference_misc_word"
    cmd = "amlt map scripts/9-13-randsearch/generate_bpe_sweep_models_space2.yaml :inference_word_valid_400K word_train"
    for line in open("/home/t-gjawahar/archai/scripts/inference.sh"):
        item = line.strip().split()[0]
        cmd += " %s"%item
    cmd += " inference_misc_word"
    print(cmd)
    
#generate_inference_commands()

# gather data for 3D plot
# scripts/9-13-3d-plot
import json
def write_jsonl_to_f(json_lines, f):
    w = open(f, 'w')
    for line in json_lines:
        w.write(json.dumps(line))
        w.write("\n")
    w.close()
def extract_hyp_values(f):
    params = ["vocab_size", "experiment_name", "vocab", "n_layer", "n_head", "d_head", "d_embed", "d_model", "d_inner", "adaptive", "div_val", "lr", "tgt_len", "mem_len", "eval_tgt_len", "n_all_param", "n_nonemb_param"]
    work_dir_starts = False
    values = {}
    for line in open(f):
        line = line.strip()
        if "work_dir" in line and "Namespace" not in line:
            work_dir_starts = True
        elif "Starting training..." in line:
            work_dir_starts = False
        if work_dir_starts:
            param = None
            for p in params:
                if p in line:
                    param = p
            if param and "lr_min" not in line:
                values[param] = line.strip().split()[-1]
    return values
def extract_scores(f):
    scores = []
    overall_numerator, overall_denominator = [0.0]*3, [0.0]*3 # 3 is # tokens (smart compose upto 15)
    partial_overall_numerator, partial_overall_denominator = [0.0]*3, [0.0]*3
    j = 0
    for line in open(f):
        if "context=" in line:
            items = line.split(" ")
            score_1 = float(items[5].split(",")[0][1:-1].split("/")[0]) /  float(items[5].split(",")[0][1:-1].split("/")[1])
            score_2 = float(items[7].split(",")[0][1:-1].split("/")[0]) /  float(items[7].split(",")[0][1:-1].split("/")[1])
            score_3 = float(items[-1].split(",")[0][1:-1].split("/")[0]) /  float(items[-1].split(",")[0][1:-1].split("/")[1])
            score_1, score_2, score_3 = float("%.2f"%(100.0*score_1)), float("%.2f"%(100.0*score_2)), float("%.2f"%(100.0*score_3))
            scores.append([score_1, score_2, score_3])
            if j % 2 == 0:
                overall_numerator[0] += float(items[5].split(",")[0][1:-1].split("/")[0]) 
                overall_denominator[0] += float(items[5].split(",")[0][1:-1].split("/")[1])
                overall_numerator[1] +=  float(items[7].split(",")[0][1:-1].split("/")[0]) 
                overall_denominator[1] += float(items[7].split(",")[0][1:-1].split("/")[1])
                overall_numerator[2] += float(items[-1].split(",")[0][1:-1].split("/")[0])
                overall_denominator[2] += float(items[-1].split(",")[0][1:-1].split("/")[1])
            else:
                partial_overall_numerator[0] += float(items[5].split(",")[0][1:-1].split("/")[0]) 
                partial_overall_denominator[0] += float(items[5].split(",")[0][1:-1].split("/")[1])
                partial_overall_numerator[1] +=  float(items[7].split(",")[0][1:-1].split("/")[0]) 
                partial_overall_denominator[1] += float(items[7].split(",")[0][1:-1].split("/")[1])
                partial_overall_numerator[2] += float(items[-1].split(",")[0][1:-1].split("/")[0])
                partial_overall_denominator[2] += float(items[-1].split(",")[0][1:-1].split("/")[1])
            j += 1
    final_scores = {}
    final_scores["em@0.2"] = scores[0]
    final_scores["em@0.5"] = scores[2]
    final_scores["em@0.8"] = scores[4]
    overall_score = 0.0
    for idx in range(len(overall_numerator)):
        overall_score += overall_numerator[idx]/overall_denominator[idx]
    overall_score /= 3.0
    final_scores["em@overall"] = float("%.2f"%(100.0*overall_score))
    overall_score = 0.0
    for idx in range(len(partial_overall_numerator)):
        overall_score += partial_overall_numerator[idx]/partial_overall_denominator[idx]
    overall_score /= 3.0
    final_scores["pm@overall"] = float("%.2f"%(100.0*overall_score))
    return final_scores

def gather_data_for_3d_plot():
    master_folder = "scripts/9-13-3d-plot"
    
    ##############
    # char model
    ##############
    # grid search around 10M
    charmod2params = {}
    for f in glob.glob("/home/t-gjawahar/archai/amlt/transxl_char_exp2_randsearch/char10M_grid*/*"):
        folder_name = f.split("/")[-2]
        if "bertstyle" in folder_name:
            continue
        values = extract_hyp_values(f)
        values["experiment_name"] = folder_name.split("_grid_")[-1]
        charmod2params[folder_name.split("_grid_")[-1]] = values
    for f in glob.glob("/home/t-gjawahar/archai/amlt/inference_transxl/inference_char_valid-char10M_grid*/stdout.txt"):
        final_scores = extract_scores(f)
        folder_name = f.split("/")[-2]
        for k in final_scores:
            charmod2params[folder_name.split("_grid_")[-1]][k] = final_scores[k]
    print(charmod2params, len(charmod2params))
    # small models
    for f in glob.glob("/home/t-gjawahar/archai/amlt/transxl_char_exp2_randsearch/small_*/stdout.txt"):
        folder_name = f.split("/")[-2]
        if "g8" in folder_name:
            continue
        values = extract_hyp_values(f)
        values["experiment_name"] = folder_name
        charmod2params[folder_name] = values
    for f in glob.glob("/home/t-gjawahar/archai/amlt/inference_transxl/inference_char_valid-small_*/stdout.txt"):
        folder_name = f.split("/")[-2]
        if "bert" in folder_name or "cpool" in folder_name:
            continue
        final_scores = extract_scores(f)
        for k in final_scores:
            charmod2params[folder_name.split("inference_char_valid-")[-1]][k] = final_scores[k]
    print(charmod2params, len(charmod2params))
    
    ##############
    # word model
    ##############
    # word model all sizes
    wordmod2params = {}
    for f in glob.glob("/home/t-gjawahar/archai/amlt/word_train/word*_nofp/stdout.txt"):
        folder_name = f.split("/")[-2]
        if "g4" not in folder_name and "nofp" in folder_name:
            values = extract_hyp_values(f)
            values["experiment_name"] = folder_name.split("_")[0]
            wordmod2params[folder_name.split("_")[0]] = values
    for f in glob.glob("/home/t-gjawahar/archai/amlt/inference_word/inference_word_model_metrics-word*_nofp/stdout.txt"):
        folder_name = f.split("/")[-2]
        final_scores = extract_scores(f)
        if folder_name.split("inference_word_model_metrics-")[-1].split("_nofp")[0] in wordmod2params:
            for k in final_scores:
                wordmod2params[folder_name.split("inference_word_model_metrics-")[-1].split("_nofp")[0]][k] = final_scores[k]
    #print(wordmod2params, wordmod2params.keys(), len(wordmod2params))
        
    ###############
    # subword model
    ###############
    subwordmod2params = {}
    for f in glob.glob("/home/t-gjawahar/archai/amlt/word_train/subword_space1_*/*"):
        folder_name = f.split("/")[-2]
        num_params = None
        for line in open(f):
            line = line.strip()
            if "#params" in line:
                num_params = int(line.split()[-1])
        if num_params:
            values = extract_hyp_values(f)
            values["experiment_name"] = folder_name.split("subword_")[-1]
            subwordmod2params[folder_name.split("subword_")[-1]] = values
    for f in glob.glob("/home/t-gjawahar/archai/amlt/inference_word/inference_word_model_metrics_valid-subword_space1*/*"):
        folder_name = f.split("/")[-2]
        if folder_name.split("inference_word_model_metrics_valid-subword_")[-1] in subwordmod2params:
            final_scores = extract_scores(f)
            for k in final_scores:
                subwordmod2params[folder_name.split("inference_word_model_metrics_valid-subword_")[-1]][k] = final_scores[k]
    
    write_jsonl_to_f([charmod2params[k] for k in sorted(charmod2params)], master_folder + "/char_acc.jsonl")
    write_jsonl_to_f([wordmod2params[k] for k in sorted(wordmod2params)], master_folder + "/word_acc.jsonl")
    write_jsonl_to_f([subwordmod2params[k] for k in sorted(subwordmod2params)], master_folder + "/subword_acc.jsonl")

#gather_data_for_3d_plot()

def generate_latency_commands_d3(fold):
    sample_word_cmd = "python archai/nlp/nvidia_transformer_xl/train.py --config_file wt103_base_no_fp.yaml --n_layer 16 --n_head 8 --d_model 256 --d_head 32 --d_inner 768 --batch_chunk 1 --eval_batch_size 1 --max_step 0 --config dgx1_1gpu_fp16 --eval_tgt_len 128 --mem_len 0 --tgt_len 128"
    template_word_cmd = "python archai/nlp/nvidia_transformer_xl/train.py --config_file wt103_base_no_fp.yaml --n_layer <n_layer> --n_head <n_head> --d_model <d_model> --d_head <d_head> --d_inner <d_inner> --batch_chunk 1 --eval_batch_size 1 --max_step 0 --config dgx1_1gpu_fp16 --eval_tgt_len 128 --mem_len 0 --tgt_len 128"
    sample_char_cmd = "python archai/nlp/nvidia_transformer_xl/train.py --config_file char_no_fp.yaml --n_layer 16 --n_head 8 --d_model 256 --d_head 32 --d_inner 768 --max_step 200000 --save_all --batch_chunk 1 --eval_batch_size 1 --max_step 0 --config dgx1_1gpu_fp16 --eval_tgt_len 600 --mem_len 0 --tgt_len 600 --vocab_size 128"
    template_char_cmd = "python archai/nlp/nvidia_transformer_xl/train.py --config_file char_no_fp.yaml --n_layer <n_layer> --n_head <n_head> --d_model <d_model> --d_head <d_head> --d_inner <d_inner> --max_step 200000 --save_all --batch_chunk 1 --eval_batch_size 1 --max_step 0 --config dgx1_1gpu_fp16 --eval_tgt_len 600 --mem_len 0 --tgt_len 600 --vocab_size 128"
    sample_subword_cmd = "python archai/nlp/nvidia_transformer_xl/train.py --config_file wt103_base_no_fp.yaml --n_layer 16 --n_head 8 --d_model 256 --d_head 32 --d_inner 768 --max_step 200000 --save_all --batch_chunk 1 --eval_batch_size 1 --max_step 0 --config dgx1_1gpu_fp16 --eval_tgt_len 600 --mem_len 0 --tgt_len 600 --vocab bpe --vocab_size 260"
    template_subword_cmd = "python archai/nlp/nvidia_transformer_xl/train.py --config_file wt103_base_no_fp.yaml --n_layer <n_layer> --n_head <n_head> --d_model <d_model> --d_head <d_head> --d_inner <d_inner> --max_step 200000 --save_all --batch_chunk 1 --eval_batch_size 1 --max_step 0 --config dgx1_1gpu_fp16 --eval_tgt_len 600 --mem_len 0 --tgt_len 600 --vocab bpe --vocab_size <vocab_size>"

    w = open("word_latency.sh", "w")
    for line in open(fold + "/word_acc.jsonl"):
        content = json.loads(line.strip())
        w.write("%s\n"%(template_word_cmd.replace("<n_layer>", content["n_layer"]).replace("<n_head>", content["n_head"]).replace("<d_model>", content["d_model"]).replace("<d_head>", content["d_head"]).replace("<d_inner>", content["d_inner"])))
    w.close()

    w = open("char_latency.sh", "w")
    for line in open(fold + "/char_acc.jsonl"):
        content = json.loads(line.strip())
        w.write("%s\n"%(template_word_cmd.replace("<n_layer>", content["n_layer"]).replace("<n_head>", content["n_head"]).replace("<d_model>", content["d_model"]).replace("<d_head>", content["d_head"]).replace("<d_inner>", content["d_inner"])))
    w.close()

    w = open("subword_latency.sh", "w")
    for line in open(fold + "/subword_acc.jsonl"):
        content = json.loads(line.strip())
        w.write("%s\n"%(template_word_cmd.replace("<n_layer>", content["n_layer"]).replace("<n_head>", content["n_head"]).replace("<d_model>", content["d_model"]).replace("<d_head>", content["d_head"]).replace("<d_inner>", content["d_inner"]).replace("<vocab_size>", content["vocab"])))
    w.close()

#generate_latency_commands_d3(fold)

def results_inference_valid_200K():
    for f in sorted(glob.glob("/home/t-gjawahar/archai/amlt/inference_misc_word/inference_char_valid_200K-subwordmodel_*/*")):
        folder = f.split("/")[-2]
        if folder.startswith("inference_char_valid_200K-subwordmodel_12M_layer_copy"):
            final_scores = extract_scores(f)
            res = folder.split("_copy_")[-1].split("-subword_space1_4_8_512_64_512_10000_0.01")[0]
            for p in ["0.2", "0.5", "0.8"]:
                for s in final_scores["em@"+p]:
                    res += "," + str(s)
            res += "," + str(final_scores["em@overall"])
            res += "," + str(final_scores["pm@overall"])
            print(res)

#results_inference_valid_200K()

def results_for_bpe_sweep_space1():
    scores = []
    for line in open("scripts/9-13-3d-plot/subword_acc.jsonl"):
        content = json.loads(line.strip())
        if "em@overall" not in content:
            continue
        scores.append((content["pm@overall"], content))
    scores = sorted(scores)
    scores.reverse()
    for score in scores:
        _, content = score
        res = '%s,%s,%s,%s,%s,%s,%s,%.2f,%.2f'%(content["n_layer"], content["n_head"], content["d_head"], content["lr"], content["vocab"], content["n_all_param"], content["n_nonemb_param"], content["em@overall"], content["pm@overall"])
        print(res)

#results_for_bpe_sweep_space1()

def compare_all_three_models(xaxis, yaxis):
    next_ans = False
    latency = []
    for line in open("scripts/9-13-3d-plot/char_latency_d3.out"):
        line = line.strip()
        if "model(data, target, mems)" in line:
            next_ans = True
        elif next_ans:
            latency.append((5.0*float(line.split()[0]))/512) if line.endswith("ms") else latency.append((5000.0*float(line.split()[0]))/512)
            next_ans = False
        else:
            next_ans = False
    pmu = []
    prev_memory = None
    for line in open("scripts/9-13-3d-plot/char_pmu_d3.out"):
        line = line.strip()
        if "TestMemoryUsed output of" in line:
            if "eval end:" in line:
                if prev_memory:
                    pmu.append(prev_memory)
                prev_memory = None
            else:
                if prev_memory:
                    prev_memory = max(prev_memory, float(line.split()[-1]))
                else:
                    prev_memory = float(line.split()[-1])
    charmod2scores = {}
    for li, line in enumerate(open("scripts/9-13-3d-plot/char_acc.jsonl")):
        content = json.loads(line.strip())
        content["latency"] = latency[li]
        content["pmu"] = pmu[li]
        charmod2scores[content["experiment_name"]] = content

    latency = []
    next_ans = False
    for line in open("scripts/9-13-3d-plot/word_latency_d3.out"):
        line = line.strip()
        if "model(data, target, mems)" in line:
            next_ans = True
        elif next_ans:
            latency.append((1.0*float(line.split()[0]))/192) if line.endswith("ms") else latency.append((1000.0*float(line.split()[0]))/192)
            next_ans = False
        else:
            next_ans = False
    pmu = []
    prev_memory = None
    for line in open("scripts/9-13-3d-plot/word_pmu_d3.out"):
        line = line.strip()
        if "TestMemoryUsed output of" in line:
            if "eval end:" in line:
                if prev_memory:
                    pmu.append(prev_memory)
                prev_memory = None
            else:
                if prev_memory:
                    prev_memory = max(prev_memory, float(line.split()[-1]))
                else:
                    prev_memory = float(line.split()[-1])
    wordmod2scores = {}
    for li, line in enumerate(open("scripts/9-13-3d-plot/word_acc.jsonl")):
        content = json.loads(line.strip())
        content["latency"] = latency[li]
        content["pmu"] = pmu[li]
        if "em@overall" not in content:
            continue
        wordmod2scores[content["experiment_name"]] = content

    latency = []
    next_ans = False
    for line in open("scripts/9-13-3d-plot/subword_latency_d3.out"):
        line = line.strip()
        if "model(data, target, mems)" in line:
            print(line)
            next_ans = True
        elif next_ans:
            print("---", line)
            if not line.endswith("ms") and not line.endswith("s"):
                latency.append(None)
            else:
                latency.append((1.0*float(line.split()[0]))/192) if line.endswith("ms") else latency.append((1000.0*float(line.split()[0]))/192)
            next_ans = False
        else:
            next_ans = False
    pmu = []
    prev_memory = None
    for line in open("scripts/9-13-3d-plot/subword_pmu_d3.out"):
        line = line.strip()
        if "TestMemoryUsed output of" in line:
            if "eval end:" in line:
                if prev_memory:
                    pmu.append(prev_memory)
                prev_memory = None
            else:
                if prev_memory:
                    prev_memory = max(prev_memory, float(line.split()[-1]))
                else:
                    prev_memory = float(line.split()[-1])
    subwordmod2scores = {}
    print('subword...')
    for li, line in enumerate(open("scripts/9-13-3d-plot/subword_acc.jsonl")):
        content = json.loads(line.strip())
        lat = None if len(latency) <= li else latency[li]
        pm = pmu[li]
        if "em@overall" not in content:
            continue
        if lat:
            #print(lat, content["vocab"])
            content["latency"] = lat
            content["pmu"] = pm
            subwordmod2scores[content["experiment_name"]] = content
            print(content["pmu"], content["n_all_param"], content["n_nonemb_param"])
    sys.exit(0)
    print(len(charmod2scores), len(wordmod2scores), len(subwordmod2scores))
    import matplotlib.patches as mpatches
    fig = plt.figure(figsize=(10,5))
    class_colours = ['cyan', 'red', 'green']
    classes = ['char', 'word', 'subword']
    #class_colours = ['cyan', 'green']
    #classes = ['char', 'subword']
    for mod in charmod2scores:
        pt = plt.scatter(float(charmod2scores[mod][xaxis]), float(charmod2scores[mod][yaxis]), marker='x', c=class_colours[0], label='char')
    for mod in wordmod2scores:
        pt = plt.scatter(float(wordmod2scores[mod][xaxis]), float(wordmod2scores[mod][yaxis]), marker='o', c=class_colours[1], label='word')
    for mod in subwordmod2scores:
        pt = plt.scatter(float(subwordmod2scores[mod][xaxis]), float(subwordmod2scores[mod][yaxis]), marker='+', c=class_colours[2], label='subword')
    plt.xlabel(xaxis,fontsize=16)
    plt.ylabel(yaxis,fontsize=16)
    recs = []
    for i in range(0,len(class_colours)):
        recs.append(mpatches.Rectangle((0,0),1,1,fc=class_colours[i]))
    plt.legend(recs,classes,loc=4,fontsize=16)
    plt.show()
    fig.savefig("/home/t-gjawahar/archai/scripts/9-14/%s_vs_%s.pdf"%(xaxis, yaxis), bbox_inches='tight')

compare_all_three_models("pmu", "em@overall")
sys.exit(0)

def spearman_nonemb_params_for_bpesweep():
    subwordmod2scores = {}
    for li, line in enumerate(open("scripts/9-13-3d-plot/subword_acc.jsonl")):
        content = json.loads(line.strip())
        subwordmod2scores[content["experiment_name"]] = content
    vocab2scores = {}
    for exp in subwordmod2scores:
        values = subwordmod2scores[exp]
        if 'em@overall' not in values:
            continue
        if values['vocab'] not in vocab2scores:
            vocab2scores[values['vocab']] = [[], []]
        vocab2scores[values['vocab']][0].append(values['em@overall'])
        vocab2scores[values['vocab']][1].append(values['n_nonemb_param'])
    for line in open("scripts/9-13-3d-plot/subword_space2.jsonl"):
        items = line.strip().split()
        params, vocab_size, score = int(items[0]), items[1], float(items[-1])
        if vocab_size not in vocab2scores:
            vocab2scores[vocab_size] = [[], []]
        vocab2scores[vocab_size][0].append(score)
        vocab2scores[vocab_size][1].append(params)
    print(vocab2scores)
    from scipy import stats
    for vocab in sorted(vocab2scores):
        rho, pval = stats.spearmanr(vocab2scores[vocab][0], vocab2scores[vocab][1])
        print("%s,%.2f,%.2f,%d"%(vocab, rho, pval, len(vocab2scores[vocab][0])))
#spearman_nonemb_params_for_bpesweep()

def check_word2subword_scores():
    f2scores = {}
    for f in glob.glob("/home/t-gjawahar/archai/amlt/misc_word/word2subword_word40M_layer_copy_*/*"):
        if "_" in f.split("/")[-2].split("word2subword_word40M_layer_copy_")[-1].split("-word40M")[0]:
            continue
        if "0-20" not in f.split("/")[-2].split("word2subword_word40M_layer_copy_")[-1].split("-word40M")[0]  and "0-0" not in f.split("/")[-2].split("word2subword_word40M_layer_copy_")[-1].split("-word40M")[0]:
            continue
        f2scores[f.split("/")[-2].split("word2subword_word40M_layer_copy_")[-1].split("-word40M")[0]] = []
        print(f)
        for line in open(f):
            line = line.strip()
            if "valid ppl" in line:
                items = line.split()
                f2scores[f.split("/")[-2].split("word2subword_word40M_layer_copy_")[-1].split("-word40M")[0]].append(float(items[-1]))
        print(f.split("/")[-2].split("word2subword_word40M_layer_copy_")[-1].split("-word40M")[0], len(f2scores[f.split("/")[-2].split("word2subword_word40M_layer_copy_")[-1].split("-word40M")[0]]))
    #del f2scores["0-75"]
    #del f2scores["0-100"]
    #del f2scores["0-50"]
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12,5))
    plt.grid(color='gray', linestyle='dashed')
    xaxis = [5000*(i+1) for i in range(20)]
    colors = ['b', 'c', 'y', 'm', 'r', 'g', 'k', "indigo", "violet", "springgreen", "olive", "firebrick", "gold"]
    ei = 0
    for key in sorted(f2scores):
        scores = f2scores[key] + [None] * (20 - len(f2scores[key]))
        plt.plot(xaxis, scores, color=colors[ei], marker='o', label=key)
        print(key, scores)
        ei += 1
    plt.xlabel("Steps")
    plt.ylabel("Valid PPL")
    plt.legend(loc="upper right")
    plt.show()
    fig.savefig("/home/t-gjawahar/archai/scripts/9-14/word2subword_40M_copy.pdf", bbox_to_anchor=(2.25, 2.55))

# check_word2subword_scores()

def plot_learning_curve_word40M2char6M_warmup_lr_study():
    import glob
    f2scores = {}
    for f in glob.glob("/home/t-gjawahar/archai/amlt/misc_word/wordmodel_40M_layer_copy_0-20_*/*"):
        f2scores[f.split("/")[-2].split("wordmodel_40M_layer_copy_0-20_")[-1].split("-word40M")[0]] = []
        for line in open(f):
            line = line.strip()
            if "valid ppl" in line:
                items = line.split()
                f2scores[f.split("/")[-2].split("wordmodel_40M_layer_copy_0-20_")[-1].split("-word40M")[0]].append(float(items[-1]))
        if len(f2scores[f.split("/")[-2].split("wordmodel_40M_layer_copy_0-20_")[-1].split("-word40M")[0]]) == 0:
            del f2scores[f.split("/")[-2].split("wordmodel_40M_layer_copy_0-20_")[-1].split("-word40M")[0]]
    print(f2scores)
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12,5))
    plt.grid(color='gray', linestyle='dashed')
    xaxis = [10000*(i+1) for i in range(7)]
    colors = ['b', 'c', 'y', 'm', 'r', 'g', 'k', "indigo", "violet", "springgreen", "olive", "firebrick", "gold"]
    ei = 0
    for key in sorted(f2scores):
        scores = f2scores[key] + [None] * (7 - len(f2scores[key]))
        plt.plot(xaxis, scores, color=colors[ei], marker='o', label=key)
        print(key, scores)
        ei += 1
    plt.xlabel("Steps")
    plt.ylabel("Valid BPC")
    plt.legend(loc="upper right")
    plt.show()
    fig.savefig("/home/t-gjawahar/archai/scripts/mlrg/word40M2char6M_warmup_lr.pdf", bbox_to_anchor=(2.25, 2.55))

#plot_learning_curve_word40M2char6M_warmup_lr_study()

def results_beam_search():
    exp2score = {}
    for f in glob.glob("/home/t-gjawahar/archai/amlt/inference_word/word_beam_search_metrics-subword_space1*/*") + glob.glob("/home/t-gjawahar/archai/amlt/inference_transxl/char_beam_search_metrics-*/*"):
        scores = []
        for line in open(f):
            line = line.strip()
            if "context=" in line:
                items = line.split(" ")
                score_1 = float(items[5].split(",")[0][1:-1].split("/")[0]) /  float(items[5].split(",")[0][1:-1].split("/")[1])
                score_2 = float(items[7].split(",")[0][1:-1].split("/")[0]) /  float(items[7].split(",")[0][1:-1].split("/")[1])
                score_3 = float(items[-1].split(",")[0][1:-1].split("/")[0]) /  float(items[-1].split(",")[0][1:-1].split("/")[1])
                score_1, score_2, score_3 = float("%.2f"%(100.0*score_1)), float("%.2f"%(100.0*score_2)), float("%.2f"%(100.0*score_3))
                scores.append([score_1, score_2, score_3])
        exp2score[f.split("/")[-2].split("beam_search_metrics-")[-1]] = scores
        print(f.split("/")[-2].split("beam_search_metrics-")[-1], len(scores))
    
    charexp2score = {}
    for f in glob.glob("/home/t-gjawahar/archai/amlt/inference_transxl/inference_char_valid-small_*/*"):
        if len(f.split("/")[-2].split("inference_char_valid-")[-1].split("_")) != 3:
            continue
        scores = []
        for line in open(f):
            line = line.strip()
            if "context=" in line:
                items = line.split(" ")
                score_1 = float(items[5].split(",")[0][1:-1].split("/")[0]) /  float(items[5].split(",")[0][1:-1].split("/")[1])
                score_2 = float(items[7].split(",")[0][1:-1].split("/")[0]) /  float(items[7].split(",")[0][1:-1].split("/")[1])
                score_3 = float(items[-1].split(",")[0][1:-1].split("/")[0]) /  float(items[-1].split(",")[0][1:-1].split("/")[1])
                score_1, score_2, score_3 = float("%.2f"%(100.0*score_1)), float("%.2f"%(100.0*score_2)), float("%.2f"%(100.0*score_3))
                scores.append([score_1, score_2, score_3])
        charexp2score[f.split("/")[-2].split("inference_char_valid-")[-1]] = scores
        print(f.split("/")[-2].split("inference_char_valid-")[-1], len(scores))
    
    # print char scores
    res = ""
    for model in ["5M", "10M", "20M"]:
        for layer in ["1L", "2L", "8L", "12L"]:
            base_scores = charexp2score["small_"+model+"_"+layer]
            beam_scores = exp2score["small_"+model+"_"+layer]
            res += "small_"+model+"_"+layer
            scores = base_scores + beam_scores
            print("small_"+model+"_"+layer, scores)
            for start_idx in range(0, len(scores), 6):
                s = scores[start_idx:start_idx+6]
                if len(s) != 6:
                    continue
                res += "," + str(s[0][0]) + "," + str(s[2][0]) + "," + str(s[4][0])
            res += "\n"
    
    swexp2score = {}
    for f in glob.glob("/home/t-gjawahar/archai/amlt/inference_word/inference_word_model_metrics_valid-subword_space1_*/*"):
        scores = []
        for line in open(f):
            line = line.strip()
            if "context=" in line:
                items = line.split(" ")
                score_1 = float(items[5].split(",")[0][1:-1].split("/")[0]) /  float(items[5].split(",")[0][1:-1].split("/")[1])
                score_2 = float(items[7].split(",")[0][1:-1].split("/")[0]) /  float(items[7].split(",")[0][1:-1].split("/")[1])
                score_3 = float(items[-1].split(",")[0][1:-1].split("/")[0]) /  float(items[-1].split(",")[0][1:-1].split("/")[1])
                score_1, score_2, score_3 = float("%.2f"%(100.0*score_1)), float("%.2f"%(100.0*score_2)), float("%.2f"%(100.0*score_3))
                scores.append([score_1, score_2, score_3])
        swexp2score[f.split("/")[-2].split("inference_word_model_metrics_valid-")[-1]] = scores
        print(f.split("/")[-2].split("inference_word_model_metrics_valid-")[-1], len(scores))
    print(swexp2score)

    # print subword scores
    res = ""
    for exp in exp2score:
        if "space1" in exp:
            base_scores = swexp2score[exp]
            beam_scores = exp2score[exp]
            scores = base_scores + beam_scores
            if len(scores) < 12:
                continue
            res += ",".join(exp.split("subword_space1_")[-1].split("_"))
            for start_idx in range(0, len(scores), 6):
                s = scores[start_idx:start_idx+6]
                if len(s) != 6:
                    continue
                res += "," + str(s[0][0]) + "," + str(s[2][0]) + "," + str(s[4][0])
            res += "\n"
    print(res)

#results_beam_search()

