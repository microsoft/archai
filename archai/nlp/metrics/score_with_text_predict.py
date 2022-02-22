# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Score models using Text Predict.
"""

from __future__ import annotations

__doc__ = """Command line program to run Text Prediction feature.
Example:
Run in console mode, where you type string and it predict completion with metrics
    tp_predict_text --verbose --type console \
        --model_type transformers --model distilgpt2 \
            --tokenizer_type transformers --tokenizer gpt2

Run in automatic mode, where you take input string from a SmartCompose .json file:
    tp_predict_text --type smartcompose \
        --model_type swiftkey --model ~/Swiftkey/model/WordData-v3-t20/lm+/model_opset12.onnx \
            --tokenizer_type swiftkey --tokenizer ~/Tok10k \
                --input Input.json --output Output.json
SmartCompose file format is like this:
    {"UniqueId":"0-1","From":"Phani Yadavalli <pyadavalli@avocadoit.com>","To":"Michael Liang <mliang@avocadoit.com>","Subject":"RE: Does anybody has a Neopoint phone I can borrow for etrade emergency?","Body":"","CursorPosition":"0","BodyContinued":"Hey Michael Phong Le has one Neopoint phone and he is not around. Thats the only neopoint phone we have. \\nThank you\\nPhani","Suggestions":[]}

Run in automatic mode, where you take input string from a text file and predict for each position:
    tp_predict_text --type text \
        --model_type micronet --model ~/neurips-micronet/results/TeamsData_8.3M_cache0 \
            --tokenizer_type micronet --tokenizer /mnt/data/TeamsDataMicronetV3/TokSB10k/ --bos_token_id 1
                --input Input.txt --output Output.json
In this file, each line has new example to predict, e.g.:
Setting
The novel is set in various locations in the Milky Way. The galaxy is divided into four concentric volumes.
"""

import os
import os.path as osp
import argparse
import logging
import re
import readline  # pylint: disable=unused-import
import time
import pathlib as pl

import pandas as pd
import numpy as np

from nlxpy.textprediction import (create_model, create_tokenizer, TextPredictor, TextPredictionSequence)

PROMPT = "> "
START_MSG = "Press CTRL-D or type 'exit' to exit."

def predict_console(predictor):
    """Console application showing predictions.
    """
    logging.info("Launching console")
    print(START_MSG)
    try:
        while True:
            line = input(PROMPT)
            if line.strip() == "exit":
                break

            if line[0:4] == "set ":
                try:
                    _, param, value = line.strip().split(maxsplit=3)
                except:
                    logging.warning("Could not split '%s' into keyword, param, value", line)
                    param = ""
                    value = 0

                predictor_param_names = [name for name in TextPredictor.__dict__ if isinstance(TextPredictor.__dict__[name], int)]
                if param in predictor_param_names:
                    predictor.__dict__[param] = int(value)

                for name in predictor_param_names:
                    value = predictor.__dict__[name] if name in predictor.__dict__ else TextPredictor.__dict__[name]
                    print(f"{name:30s}\t{value}")

                continue

            line = re.sub(r"\\n", r"\n", line)
            start = time.time()
            (best_prediction, predictions) = predictor.predict_full(line)
            msg = f"Prediction  : {best_prediction.text}\n"
            msg += f"P(Match)    : {best_prediction.p_match():.5f}\n"
            msg += f"CharAccepted: {best_prediction.char_accepted():.5f}\n"
            msg += f"Score       : {best_prediction.score():.5f}\n"
            msg += f"Time (ms)   : {1000*(time.time() - start):.3f}"
            print(msg)
            preds_dict = [p.to_odict() for p in predictions]
            df = pd.DataFrame(preds_dict)
            if 'Match' in df:
                df = df.drop(['Match'], axis=1)
            print(df)
    except (EOFError, KeyboardInterrupt):
        print("Exiting...")

def predict_website(predictor, args):
    """Web application showing predictions.
    """
    logging.info("Launching website...")

    import dash
    from dash.dependencies import Input, Output
    import dash_core_components as dcc
    import dash_html_components as html
    import dash_bootstrap_components as dbc
    import dash_table

    import flask

    title = f"Text Pediction: {args.model_type} / {args.model}"
    server = flask.Flask('app')
    app = dash.Dash('app', server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.title = title
    app.scripts.config.serve_locally = False

    def update_table(line):
        (best_prediction, predictions) = predictor.predict_full(line)
        preds_dict = [p.to_odict() for p in predictions]
        df = pd.DataFrame(preds_dict)
        if "Match" in df: 
            df = df.drop(["Match"], axis=1)
        if "Tokens" in df: 
            df = df.drop(["Tokens"], axis=1)
        print(f"Line: {line}")
        print(df)
        return df.to_dict("records")

    column_format = []
    for c, v in update_table("I am looking forw")[0].items():
        cf = {"name": c, "id": c}
        if isinstance(v, float):
            cf["type"] = 'numeric'
            cf["format"] = {'specifier': '.4f'}
        column_format.append(cf)

    app.layout = dbc.Container([
        html.H3(title),
        html.Hr(),
        dbc.Input("input_text", type="text", placeholder="Type something...", value=""),
        html.Hr(),
        html.P(id="output_text", children=""),
        html.Hr(),
        dash_table.DataTable(
            id="output_table",
            columns=column_format,
            data=update_table(""),
            style_cell={'minWidth': '100px'}
            )
    ])

    @app.callback(Output('output_table', 'data'),
                  [Input('input_text', 'value')])
    def update_table_callback(line):
        return update_table(line)

    @app.callback(Output('output_text', 'children'),
                  [Input('input_text', 'value')])
    def update_text_callback(line):
        prediction = predictor.predict(line)
        if prediction.score() > args.min_score:
            return f"Prediction: {prediction}"
        else:
            return "Prediction: "

    app.run_server(host='0.0.0.0', port=args.port)


TYPE_LIST = ["console", "website", "text", "smartcompose"]
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--type", type=str, choices=TYPE_LIST, default=TYPE_LIST[0], help="Input type for text prediction")
    parser.add_argument("--input", type=str, default='Input.json',
                        help="Input file. If none given, console mode is executed. For turing input type, it denotes prefix for files to be loaded")
    parser.add_argument("--output", type=str, default=None, help="Output file (default: input.pred)")
    parser.add_argument("--model_type", type=str, required=True, help="Define model type to run models from Transformers library")
    parser.add_argument("--tokenizer_type", type=str, required=True, help="Define tokenizer type to run models from Transformers library")
    parser.add_argument("--model", type=str, required=True, help="Model directory, file or name")
    parser.add_argument("--tokenizer", type=str, required=True, help="Tokenizer directory, file or name")
    parser.add_argument("--cache_dir", type=str, default=None, help="Huggingface transformers cache directory (default: None)")
    # Prediction settings
    parser.add_argument("--min_score", type=float, default=1.0, help="Minimum score to return the results")
    parser.add_argument("--save_step", type=int, default=100000, help="Save file every step predictions")
    parser.add_argument("--max_body_len", type=int, default=100000, help="Maximum length of a body to pass to text prediction")
    parser.add_argument("--max_seq_len", type=int, default=192, help="Maximum length to pass to the model")
    parser.add_argument("--current_paragraph_only",action="store_true", default=False,
                        help="Truncate the body to current paragraph only (remove anything before new line)")
    parser.add_argument("--bos_token_id", type=int, default=None, help="Token for beginning of the sentence/message")
    # Final scoring settings
    parser.add_argument("--score", action='store_true', default=False, help="Perform scoring at the end of the run?")
    parser.add_argument("--score_output_dir", type=str, default=None, help="Output summary file (default: input.dir)")
    # parser.add_argument("--min_score", type=float, default=2.0, help="Minimum score to check") (reused from above)
    parser.add_argument("--max_score", type=float, default=5.0, help="Maximum score to check")
    parser.add_argument("--score_step", type=float, default=0.1, help="Score step to check")
    parser.add_argument("--expected_match_rate", type=float, default=0.5, help="Match point to estimate parameters at")
    # Other
    parser.add_argument("--port", type=int, default=8890, help="Port for the website interface")
    parser.add_argument("--verbose", action='store_true', default=False, help="Verbose output")
    parser.add_argument("--amulet_data", action='store_true', default=False, help="Uses data from Amulet")
    parser.add_argument("--amulet_model", action='store_true', default=False, help="Uses model from Amulet")
    parser.add_argument("--amulet_output", action='store_true', default=False, help="Outputs to Amulet")

    if argv is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(argv)

    amlt_data = os.environ.get('AMLT_DATA_DIR', '')
    amlt_output = os.environ.get('AMLT_OUTPUT_DIR', '')

    if args.amulet_data:
        args.input = osp.join(amlt_data, args.input)
        
        if args.model_type == 'gpt2onnxprob':
            args.tokenizer = osp.join(amlt_data, args.tokenizer)
        else:
            args.tokenizer = osp.join(osp.dirname(amlt_output), args.model)
            args.tokenizer = osp.dirname(args.tokenizer)

    if args.amulet_model:
        args.model = osp.join(osp.dirname(amlt_output), args.model)

    if args.amulet_output:
        args.output = osp.join(amlt_output, args.output)

    print(f'input: {args.input}')
    print(f'tokenizer: {args.tokenizer}')
    print(f'model: {args.model}')

    args.output = f"{args.input}.pred" if args.output is None else args.output
    args.score_output_dir = f"{args.output}.dir" if args.score_output_dir is None else args.score_output_dir

    print(f'output: {args.output}')

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARN)

    predict(args)

def predict(args):
    model = create_model(args.model_type, args.model, cache_dir=args.cache_dir, max_seq_len=args.max_seq_len)
    tokenizer = create_tokenizer(args.tokenizer_type, args.tokenizer, cache_dir=args.cache_dir)

    predictor = TextPredictor(model, tokenizer)
    predictor.MAX_INPUT_TEXT_LEN = args.max_body_len
    predictor.BOS_TOKEN_ID = args.bos_token_id

    if args.type == "console":
        predict_console(predictor)
    elif args.type == "website":
        predict_website(predictor, args)
    elif args.type == "text" or args.type == "smartcompose":
        seq = TextPredictionSequence.from_file(args.input, args.type, predictor)
        # seq.MAX_BODY_LEN = args.max_body_len # Doesn't play well with BOS token
        seq.SAVE_STEP = args.save_step
        seq.MIN_SCORE = args.min_score
        seq.CURRENT_PARAGRAPH_ONLY = args.current_paragraph_only
        seq.predict(args.output)
        seq.save(args.output)
        if args.score:
            min_scores = np.arange(args.min_score, args.max_score, args.score_step).tolist()
            seq.score(min_scores, args.expected_match_rate)
            seq.save_all(args.score_output_dir, predict_file=None)
    else:
        raise ValueError(f"Unkown input type '{type}'")

if __name__ == '__main__':
    main()
