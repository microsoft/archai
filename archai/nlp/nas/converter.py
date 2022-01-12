# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Converts from parameters to genes and vice-versa.
"""


model_config_defaults = {'d_head': None,
                         'n_token': 267736,
                         'dropout': 0.1,
                         'dropatt': 0.0,
                         'd_embed': None,
                         'div_val': 4,
                         'pre_lnorm': False,
                         'tgt_len': 192,
                         'ext_len': 0,
                         'mem_len': 192,
                         'same_length': False,
                         'attn_type': 0,
                         'clamp_len': -1,
                         'sample_softmax': -1,
                         'cutoffs': [19997, 39997, 199997],
                         'tie_projs': [False, True, True, True],
                         'tie_weight': True,
                         'dtype': None,
                         'primer_conv': False,
                         'primer_square': False,
                         'use_cache': False}


class Converter:
    def __init__(self, n_layer_choice, d_model_choice, d_inner_choice, n_head_choice, **kwargs):
        self.n_layer_choice = n_layer_choice
        self.d_model_choice = d_model_choice
        self.d_inner_choice = d_inner_choice
        self.n_head_choice = n_head_choice

        self.max_n_layer = self.n_layer_choice[-1]

    def config2gene(self, config):
        gene = []

        sample_n_layer = config['n_layer']

        gene.append(config['d_model'])
        gene.append(sample_n_layer)

        for i in range(max(self.max_n_layer, sample_n_layer)):
            if isinstance(config['d_inner'], list):
                if i < sample_n_layer:
                    gene.append(config['d_inner'][i])
                else:
                    gene.append(config['d_inner'][0])
            else:
                gene.append(config['d_inner'])

        for i in range(max(self.max_n_layer, sample_n_layer)):
            if isinstance(config['n_head'], list):
                if i < sample_n_layer:
                    gene.append(config['n_head'][i])
                else:
                    gene.append(config['n_head'][0])
            else:
                gene.append(config['n_head'])

        return gene

    def gene2config(self, gene):
        config = {'d_model': None,
                  'n_layer': None,
                  'd_inner': None,
                  'n_head': None}

        current_index = 0

        config['d_model'] = gene[current_index]
        current_index += 1

        config['n_layer'] = gene[current_index]
        current_index += 1

        config['d_inner'] = gene[current_index: current_index + config['n_layer']]
        current_index += max(self.max_n_layer, config['n_layer'])

        config['n_head'] = gene[current_index: current_index + config['n_layer']]
        current_index += max(self.max_n_layer, config['n_layer'])

        return config

    def gene2key(self, gene):
        key_list = []

        current_index = 0

        key_list += [gene[current_index]]  # d_model
        current_index += 1

        key_list += [gene[current_index]]  # n_layer
        current_index += 1

        key_list += gene[current_index: current_index + gene[1]]  # d_inner
        current_index += self.max_n_layer

        key_list += gene[current_index: current_index + gene[1]]  # n_head
        current_index += self.max_n_layer

        return ','.join(str(k) for k in key_list)

    def get_gene_choice(self, d_inner_min=None):
        gene_choice = []

        gene_choice.append(self.d_model_choice)
        gene_choice.append(self.n_layer_choice)

        for _ in range(self.max_n_layer):
            if d_inner_min is not None:
                gene_choice.append(list(range(d_inner_min, self.d_inner_choice[-1], 50)))
            else:
                gene_choice.append(self.d_inner_choice)

        for _ in range(self.max_n_layer):
            gene_choice.append(self.n_head_choice)

        return gene_choice
