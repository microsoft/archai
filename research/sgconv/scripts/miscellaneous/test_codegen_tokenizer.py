import token

from transformers import AutoTokenizer, PreTrainedTokenizerFast


def main():
    # load codegen tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")

    codestr = "mylist = [this_data, 2, 3, 'abc']"

    _ = tokenizer.tokenize(codestr)

    print("done")


if __name__ == "__main__":
    main()
