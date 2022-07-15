from datasets import load_dataset

def main():

    dataset = load_dataset("the_pile")
    
    def calc_len(examples):
        return {"lengths": [len(t.split()) for t in examples['text']]}

    dataset.map(calc_len, batched=True, batch_size=10000, num_proc=1)

    chunk_size = int(1e6)
    num_train = len(dataset['train'])
    for i in range(0, num_train, chunk_size):
        start = i
        stop = min(i+chunk_size, num_train)
        chunk = dataset['train'][start:stop]
        print(len(chunk))

    print('done')

if __name__ == '__main__':
    main()