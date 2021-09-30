 #%%
 
file = '/home/caiocesart/dataroot/textpred/olx/valid.txt'

with open(file, 'r', encoding='utf-8') as f:
    max_idx = 0
    for idx, line in enumerate(f):
        max_idx = idx
    
    print(max_idx)