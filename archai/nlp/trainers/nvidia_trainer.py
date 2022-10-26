# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
"""

class NvidiaTrainer:
    """
    """

    def __init__(
        self,
        model=None,
        args=None,
        data_collator=None,
        tokenizer=None,
        train_dataset=None,
        eval_dataset=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=None,
    ) -> None:
        """"""
        pass

    def load_dataset(self):
        """"""
        
        self.dataset = get_lm_corpus(
            self.args.data,
            self.args.cache_dir,
            self.args.dataset,
            self.args.vocab,
            vocab_size=self.args.vocab_size,
            refresh_cache=self.args.refresh_cache
        )

    def get_dataloader(self, split: str):
        """"""

        return self.dataset.get_iterator(
            split,
            self.args.batch_size,
            self.args.tgt_len,
            self.args.device,
            self.args.ext_len,
            mem_len=self.args.mem_len
        )

    def create_or_load_model(self):
        """"""
        pass

    def create_optimizer_and_scheduler(self):
        """"""
        pass

    def training_step(self):
        """"""
        pass

    def train(self):
        """"""
        pass

    def prediction_step(self):
        """"""
        pass

    def predict(self):
        """"""
        pass

    def evaluate(self):
        """"""
        pass
