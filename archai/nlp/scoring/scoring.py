import numpy as np

from torch import nn

from archai.nlp.tokenizer_utils.vocab_base import VocabBase
from archai.nlp.scoring.text_predictor import TextPredictor
from archai.nlp.scoring.text_prediction_sequence import TextPredictionSequence


def predict_text(in_filepath:str, out_filepath:str, model:nn.Module,
                 tokenizer:VocabBase, bos_token:str,
                 in_file_type='text', # or could be 'smartcompose'
                 max_body_len=100000, # Maximum length of a body to pass to text prediction
                 save_step=100000, # Save file every step predictions
                 min_score=1.0, # Minimum score to return the results
                 max_score=5.0, # Maximum score to check
                 score_step=0.1, # Score step to check
                 current_paragraph_only=False, # Truncate the body to current paragraph only (remove anything before new line)
                 expected_match_rate=0.5, # Match point to estimate parameters at
                 ):
    predictor = TextPredictor(model, tokenizer)
    predictor.MAX_INPUT_TEXT_LEN = max_body_len
    predictor.BOS_TOKEN_ID = tokenizer.token_to_id(bos_token)

    seq = TextPredictionSequence.from_file(in_filepath, in_file_type, predictor)
    # seq.MAX_BODY_LEN = args.max_body_len # Doesn't play well with BOS token
    seq.SAVE_STEP = save_step
    seq.MIN_SCORE = min_score
    seq.CURRENT_PARAGRAPH_ONLY = current_paragraph_only
    seq.predict(out_filepath)
    seq.save(out_filepath)

    min_scores = np.arange(min_score, max_score, score_step).tolist()
    seq.score(min_scores, expected_match_rate)
    seq.save_all(score_output_dir, predict_file=None)

