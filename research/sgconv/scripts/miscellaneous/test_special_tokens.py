import token
from re import S

from transformers import AutoTokenizer, PreTrainedTokenizerFast


def main():
    # codestr = "[s_NAME]remove[e_NAME] [s_NAME]first[e_NAME] [s_NAME]and[e_NAME] [s_NAME]last[e_NAME] [s_NAME]lines[e_NAME] [s_NAME]of[e_NAME] [s_NAME]string[e_NAME] `[s_NAME]s[e_NAME]` [s_NEWLINE]\n[e_NEWLINE][s_INDENT] [e_INDENT][s_NAME]s[e_NAME][s_OP][[e_OP][s_NAME]s[e_NAME][s_OP].[e_OP][s_NAME]find[e_NAME][s_OP]([e_OP][s_STRING]'\\n'[e_STRING][s_OP])[e_OP] [s_OP]+[e_OP] [s_NUMBER]1[e_NUMBER][s_OP]:[e_OP][s_NAME]s[e_NAME][s_OP].[e_OP][s_NAME]rfind[e_NAME][s_OP]([e_OP][s_STRING]'\\n'[e_STRING][s_OP])[e_OP][s_OP]][e_OP][s_NEWLINE][e_NEWLINE][s_DEDENT][e_DEDENT][s_ENDMARKER][e_ENDMARKER] "
    codestr = "[s_COMMENT]# DExTer : Debugging Experience Tester[e_COMMENT][s_NL]\n[e_NL][s_COMMENT]"
    # load codegen tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")

    # create and add special tokens for python
    special_start_tokens = ["[" + "s_" + t + "]" for t in token.tok_name.values()]
    special_end_tokens = ["[" + "e_" + t + "]" for t in token.tok_name.values()]
    all_special_tokens = special_start_tokens + special_end_tokens

    tokenizer.add_tokens(all_special_tokens, special_tokens=True)

    _ = tokenizer.tokenize(codestr)

    # save tokenizer
    tokenizer.save_pretrained("/data1/codegen-350m-python-aware")

    # reload tokenizer
    loaded_tokenizer = PreTrainedTokenizerFast(tokenizer_file="/data1/codegen-350m-python-aware/tokenizer.json")
    _ = loaded_tokenizer.tokenize(codestr)

    # try with non special tokens
    orig_tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
    _ = orig_tokenizer.tokenize(codestr)

    print("done")


if __name__ == "__main__":
    main()
