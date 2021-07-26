# read logs programmtically without downloading to local
# prints metrics

import sys, os, glob, math

# read all lines to a list
def read_all_lines(f):
    lines = []
    for line in open(f):
        lines.append(line.strip().split(" - INFO - ")[1].strip())
    return lines

# checks if the output folder belongs to a given experiment
def check_experiment_name(f, exp_name):
    content = ""
    for line in open(f):
        content += line.strip() + "\n"
    if "experiment_name : %s"%exp_name in content:
        return True
    return False

# convert bpc to word-level perplexity
def bp2wordlevel_ramimethod(eval_bpc, eval_file):
    num_chars, num_words = 0, 0
    for line in open(eval_file):
        line = line.strip()
        num_chars += len(line)
        num_words += len(line.split())
    print(num_chars, num_words)
    return math.pow(2, (eval_bpc*float(num_chars))/num_words)

#print(bp2wordlevel_ramimethod(1.0, "/home/t-gjawahar/object_dir/wikitext-2-raw-v1-char/wiki.valid.tokens"))
#sys.exit(0)

# get values
def getval(log_lines):
    values = {}
    assert(log_lines[0].startswith("Namespace"))
    params = log_lines[0].split(",")
    for param in params:
        param = param.strip()
        if param.split("=")[0].strip() in headers:
            values[param.split("=")[0].strip()] = param.split("=")[1].strip()
    for line in log_lines:
        line = line.strip()
        if line.startswith("#params"):
            values["total_params"] = line.split("=")[-1].strip()
        elif line.startswith("#non emb params"):
            values["non_emb_params"] = line.split("=")[-1].strip()
            values["emb_params"] = str(int(values["total_params"]) - int(values["non_emb_params"]))
        elif line.startswith("Training time:"):
            values["total_time"] = line.split()[2]
        elif "valid loss" in line:
            items = line.split()
            values["valid_time"] = items[8][0:-1] # seconds
            values["valid_loss"] = items[12]
            values["valid_ppl"] = items[16]
    return values


experiment_name_prefix = "transxl_char_exp1_5_randsearch_wikifull_10Ksteps_"
headers="n_layer,n_head,d_head,d_embed,d_inner,mem_len,tgt_len,dropout,emb_params,non_emb_params,total_params,valid_ppl,valid_loss,total_time,valid_time,ckpt_size".split(",")
#train_log = getval(read_all_lines("/home/t-gjawahar/object_dir/train_log (2).log"))
#print(train_log)
#sys.exit(0)

results_f = os.environ["AMLT_OUTPUT_DIR"] + "/results.csv"
print('results file at = %s'%results_f)
results_file_w = open(results_f, "w")
output_master_folder = "/".join(os.environ["AMLT_OUTPUT_DIR"].split("/")[0:-1])
print('output master folder = %s'%output_master_folder)

results_file_w.write("%s\n"%",".join(headers))
for out_dir in glob.glob(output_master_folder + "/*"):
    if os.path.exists(out_dir + "/checkpoint_last.pt") and os.path.exists(out_dir + "/train_log.log") and check_experiment_name(out_dir + "/train_log.log", experiment_name_prefix):
        train_log = read_all_lines(out_dir + "/train_log.log")
        res = "" 
        try:
            all_values = getval(train_log)
            for header in headers:
                if header != "ckpt_size":
                    res += all_values[header] + ","
                else:
                    res += "%.2f,"%(os.path.getsize(out_dir + "/checkpoint_last.pt")/float(1024*1024))
            results_file_w.write(res+"\n")
        except:
            continue
results_file_w.close()

