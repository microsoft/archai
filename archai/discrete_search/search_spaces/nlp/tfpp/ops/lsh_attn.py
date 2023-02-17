from functools import reduce
from operator import mul
from typing import Optional

import torch
from torch import nn

from transformers.models.gptj.modeling_gptj import fixed_pos_embedding, rotate_every_two, apply_rotary_pos_emb
from transformers.models.reformer.modeling_reformer import (
    ReformerConfig, EfficientAttentionMixin, 
    ReverseSort, LSHSelfAttentionOutput,
    _stable_argsort
)

from archai.discrete_search.search_spaces.config import ArchConfig


class HfLSHSelfAttention(nn.Module, EfficientAttentionMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.chunk_length = config.lsh_attn_chunk_length
        self.num_hashes = config.num_hashes
        self.num_buckets = config.num_buckets
        self.num_chunks_before = config.lsh_num_chunks_before
        self.num_chunks_after = config.lsh_num_chunks_after
        self.hash_seed = config.hash_seed
        self.is_decoder = config.is_decoder
        self.max_position_embeddings = config.max_position_embeddings

        self.dropout = config.lsh_attention_probs_dropout_prob

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.hidden_size = config.hidden_size

        # projection matrices
        self.query_key = nn.Linear(self.hidden_size, self.all_head_size, bias=False)
        self.value = nn.Linear(self.hidden_size, self.all_head_size, bias=False)

        # save mask value here. Need fp32 and fp16 mask values
        self.register_buffer("self_mask_value_float16", torch.tensor(-1e3))
        self.register_buffer("self_mask_value_float32", torch.tensor(-1e5))
        self.register_buffer("mask_value_float16", torch.tensor(-1e4))
        self.register_buffer("mask_value_float32", torch.tensor(-1e9))

    def _split_hidden_size_and_add_rotary(self, x, num_attn_heads, attn_head_size, seq_len):
        """
        splits hidden_size dim into attn_head_size and num_attn_heads
        """
        new_x_shape = x.size()[:-1] + (num_attn_heads, attn_head_size)
        x = x.view(*new_x_shape)
        
        # Applies rotary embeddings
        sincos = fixed_pos_embedding(x, 1, seq_len=seq_len)
        x = apply_rotary_pos_emb(x, sincos, offset=0)

        return x.transpose(2, 1)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        num_hashes=None,
        buckets=None,
        past_buckets_states=None,
        use_cache=False,
        output_attentions=False,
        **kwargs,
    ):
        sequence_length = hidden_states.shape[1]
        batch_size = hidden_states.shape[0]

        # num hashes can optionally be overwritten by user
        num_hashes = num_hashes if num_hashes is not None else self.num_hashes

        do_cached_attention = use_cache and past_buckets_states[1] is not None

        # check if cache shall be used and that hidden states are already cached
        if do_cached_attention:
            assert sequence_length == 1, (
                "At the moment, auto-regressive language generation is only possible one word at a time. Make sure"
                f" that input sequence length {sequence_length} equals 1, when `past_buckets_states` is passed."
            )
            past_buckets = past_buckets_states[0]
            past_states = past_buckets_states[1]

            # get query vector
            query_vectors = self.query_key(hidden_states)
            query_vectors = self._split_hidden_size_and_add_rotary(
                query_vectors, self.num_attention_heads, self.attention_head_size, seq_len=sequence_length
            )

            if past_buckets is not None:
                key_value_hidden_states, sorted_bucket_idx, buckets = self._get_relevant_hid_states_and_buckets(
                    query_vectors=query_vectors,
                    attention_mask=attention_mask,
                    num_hashes=num_hashes,
                    hidden_states=hidden_states,
                    past_states=past_states,
                    past_buckets=past_buckets,
                )

                query_key_vectors = self._query_per_attn_head(key_value_hidden_states)
                value_vectors = self._value_per_attn_head(key_value_hidden_states)

                # split key & value vectors by num hashes to apply
                # self attention on each separately
                query_key_vectors = self._split_seq_length_dim_to(
                    query_key_vectors,
                    num_hashes,
                    -1,
                    self.num_attention_heads,
                    self.attention_head_size,
                )
                value_vectors = self._split_seq_length_dim_to(
                    value_vectors,
                    num_hashes,
                    -1,
                    self.num_attention_heads,
                    self.attention_head_size,
                )
                # repeat query vectors across hash dimension
                query_vectors = query_vectors.unsqueeze(2).repeat(1, 1, num_hashes, 1, 1)
            else:
                key_value_hidden_states = torch.cat([past_states, hidden_states], dim=1)

                query_key_vectors = self.query_key(key_value_hidden_states)
                value_vectors = self.value(key_value_hidden_states)

        else:
            # project hidden_states to query_key and value
            query_vectors = None
            query_key_vectors = self.query_key(hidden_states)
            value_vectors = self.value(hidden_states)

        # if query key is not already split
        if not do_cached_attention or past_buckets is None:
            query_key_vectors = self._split_hidden_size_and_add_rotary(
                query_key_vectors, self.num_attention_heads, self.attention_head_size, seq_len=sequence_length
            )
            value_vectors = self._split_hidden_size_dim(
                value_vectors, self.num_attention_heads, self.attention_head_size
            )

        # cache buckets for next incremental decoding
        if do_cached_attention and past_buckets is None and key_value_hidden_states.shape[1] >= self.chunk_length:
            buckets = self._hash_vectors(query_key_vectors, num_hashes, attention_mask)

        # free memory
        del hidden_states

        assert (
            query_key_vectors.shape[-1] == self.attention_head_size
        ), f"last dim of query_key_vectors is {query_key_vectors.shape[-1]} but should be {self.attention_head_size}."
        assert (
            value_vectors.shape[-1] == self.attention_head_size
        ), f"last dim of value_vectors is {value_vectors.shape[-1]} but should be {self.attention_head_size}."

        do_standard_self_attention = (sequence_length <= self.chunk_length) or (
            use_cache and past_buckets_states[1] is not None
        )
        # LSH attention only makes sense if chunked attention should be performed
        if not do_standard_self_attention:
            # set `num_buckets` on the fly, recommended way to do it
            if self.num_buckets is None:
                self._set_num_buckets(sequence_length)

            # use cached buckets for backprop only
            if buckets is None:
                # hash query key vectors into buckets
                buckets = self._hash_vectors(query_key_vectors, num_hashes, attention_mask)
            else:
                # make sure buckets has correct shape for LSH attention
                buckets = buckets.view(batch_size, self.num_attention_heads, num_hashes * sequence_length)

            assert (
                int(buckets.shape[-1]) == num_hashes * sequence_length
            ), f"last dim of buckets is {buckets.shape[-1]}, but should be {num_hashes * sequence_length}"

            sorted_bucket_idx, undo_sorted_bucket_idx = self._get_sorted_bucket_idx_and_undo_sorted_bucket_idx(
                sequence_length, buckets, num_hashes
            )

            # make sure bucket idx is not longer then sequence length
            sorted_bucket_idx_per_hash = sorted_bucket_idx % sequence_length

            # cluster query key value vectors according to hashed buckets
            query_key_vectors = self._gather_by_expansion(query_key_vectors, sorted_bucket_idx_per_hash, num_hashes)
            value_vectors = self._gather_by_expansion(value_vectors, sorted_bucket_idx_per_hash, num_hashes)
            query_key_vectors = self._split_seq_length_dim_to(
                query_key_vectors,
                -1,
                self.chunk_length,
                self.num_attention_heads,
                self.attention_head_size,
            )
            value_vectors = self._split_seq_length_dim_to(
                value_vectors,
                -1,
                self.chunk_length,
                self.num_attention_heads,
                self.attention_head_size,
            )

            if self.chunk_length is None:
                assert self.num_chunks_before == 0 and self.num_chunks_after == 0, (
                    "If `config.chunk_length` is `None`, make sure `config.num_chunks_after` and"
                    " `config.num_chunks_before` are set to 0."
                )
        elif do_cached_attention and past_buckets is not None:
            # use max sequence length
            sorted_bucket_idx_per_hash = sorted_bucket_idx
        else:
            # get sequence length indices
            sorted_bucket_idx_per_hash = torch.arange(sequence_length, device=query_key_vectors.device).repeat(
                batch_size, self.num_attention_heads, 1
            )

        # scale key vectors
        key_vectors = self._len_and_dim_norm(query_key_vectors)

        # set query_vectors to query key vectors if LSH self attention
        query_vectors = query_vectors if query_vectors is not None else query_key_vectors

        # free memory
        del query_key_vectors

        # get attention probs
        out_vectors, logits, attention_probs = self._attend(
            query_vectors=query_vectors,
            key_vectors=key_vectors,
            value_vectors=value_vectors,
            sorted_bucket_idx_per_hash=sorted_bucket_idx_per_hash,
            attention_mask=attention_mask,
            head_mask=head_mask,
            do_standard_self_attention=do_standard_self_attention,
            do_cached_attention=do_cached_attention,
        )

        # free memory
        del key_vectors, value_vectors

        # re-order out_vectors and logits
        if not do_standard_self_attention:
            # sort clusters back to correct ordering
            out_vectors, logits = ReverseSort.apply(out_vectors, logits, sorted_bucket_idx, undo_sorted_bucket_idx)

        if not do_standard_self_attention or (do_cached_attention and past_buckets is not None):
            # sum up all hash rounds
            if num_hashes > 1:
                out_vectors = self._split_seq_length_dim_to(
                    out_vectors,
                    num_hashes,
                    sequence_length,
                    self.num_attention_heads,
                    self.attention_head_size,
                )
                logits = self._split_seq_length_dim_to(
                    logits,
                    num_hashes,
                    sequence_length,
                    self.num_attention_heads,
                    self.attention_head_size,
                ).unsqueeze(-1)

                probs_vectors = torch.exp(logits - torch.logsumexp(logits, dim=2, keepdim=True))
                out_vectors = torch.sum(out_vectors * probs_vectors, dim=2)
                # free memory
                del probs_vectors

            # free memory
            del logits

        assert out_vectors.shape == (
            batch_size,
            self.num_attention_heads,
            sequence_length,
            self.attention_head_size,
        ), (
            "out_vectors have be of shape `[batch_size, config.num_attention_heads, sequence_length,"
            " config.attention_head_size]`."
        )

        out_vectors = self._merge_hidden_size_dims(out_vectors, self.num_attention_heads, self.attention_head_size)

        if output_attentions is False:
            attention_probs = ()

        if buckets is not None:
            buckets = buckets.view(batch_size, self.num_attention_heads, num_hashes, -1)

        return LSHSelfAttentionOutput(hidden_states=out_vectors, attention_probs=attention_probs, buckets=buckets)

    def _query_per_attn_head(self, hidden_states):
        per_head_query_key = self.query_key.weight.reshape(
            self.num_attention_heads, self.attention_head_size, self.hidden_size
        ).transpose(-2, -1)
        # only relevant for inference and no bias => we can use einsum here
        query_key_vectors = torch.einsum("balh,ahr->balr", hidden_states, per_head_query_key)
        return query_key_vectors

    def _value_per_attn_head(self, hidden_states):
        per_head_value = self.value.weight.reshape(
            self.num_attention_heads, self.attention_head_size, self.hidden_size
        ).transpose(-2, -1)
        # only relevant for inference and no bias => we can use einsum here
        value_vectors = torch.einsum("balh,ahr->balr", hidden_states, per_head_value)
        return value_vectors

    def _hash_vectors(self, vectors, num_hashes, attention_mask, increase_num_buckets=False):
        batch_size = vectors.shape[0]

        # See https://arxiv.org/pdf/1509.02897.pdf
        # We sample a different random rotation for each round of hashing to
        # decrease the probability of hash misses.
        if isinstance(self.num_buckets, int):
            assert (
                self.num_buckets % 2 == 0
            ), f"There should be an even number of buckets, but `self.num_buckets`: {self.num_buckets}"
            rotation_size = self.num_buckets
            num_buckets = self.num_buckets
        else:
            # Factorize the hash if self.num_buckets is a list or tuple
            rotation_size, num_buckets = 0, 1
            for bucket_factor in self.num_buckets:
                assert (
                    bucket_factor % 2 == 0
                ), f"The number of buckets should be even, but `num_bucket`: {bucket_factor}"
                rotation_size = rotation_size + bucket_factor
                num_buckets = num_buckets * bucket_factor

        # remove gradient
        vectors = vectors.detach()

        if self.hash_seed is not None:
            # for determinism
            torch.manual_seed(self.hash_seed)

        rotations_shape = (self.num_attention_heads, vectors.shape[-1], num_hashes, rotation_size // 2)
        # create a random self.attention_head_size x num_hashes x num_buckets/2
        random_rotations = torch.randn(rotations_shape, device=vectors.device, dtype=vectors.dtype)
        # Output dim: Batch_Size x Num_Attn_Heads x Num_Hashes x Seq_Len x Num_Buckets/2
        rotated_vectors = torch.einsum("bmtd,mdhr->bmhtr", vectors, random_rotations)

        if isinstance(self.num_buckets, int) or len(self.num_buckets) == 1:
            rotated_vectors = torch.cat([rotated_vectors, -rotated_vectors], dim=-1)
            buckets = torch.argmax(rotated_vectors, dim=-1)
        else:
            # Get the buckets for them and combine.
            buckets, cur_sum, cur_product = None, 0, 1
            for bucket_factor in self.num_buckets:
                rotated_vectors_factor = rotated_vectors[..., cur_sum : cur_sum + (bucket_factor // 2)]
                cur_sum = cur_sum + bucket_factor // 2
                rotated_vectors_factor = torch.cat([rotated_vectors_factor, -rotated_vectors_factor], dim=-1)
                if buckets is None:
                    buckets = torch.argmax(rotated_vectors_factor, dim=-1)
                else:
                    buckets = buckets + (cur_product * torch.argmax(rotated_vectors_factor, dim=-1))

                cur_product = cur_product * bucket_factor

        if attention_mask is not None and (attention_mask.sum().item() < batch_size * attention_mask.shape[-1]):
            # add an extra bucket for padding tokens only
            num_buckets = num_buckets + 1
            # assign padding tokens extra bucket
            buckets_mask = attention_mask.to(torch.uint8)[:, None, None, :].expand(buckets.shape)
            buckets = torch.where(
                buckets_mask, buckets, torch.tensor(num_buckets - 1, dtype=torch.long, device=buckets.device)
            )
        elif increase_num_buckets:
            num_buckets = num_buckets + 1

        # buckets is now (Batch_size x Num_Attn_Heads x Num_Hashes x Seq_Len).
        # Next we add offsets so that bucket numbers from different hashing rounds don't overlap.
        offsets = torch.arange(num_hashes, device=vectors.device)
        offsets = (offsets * num_buckets).view((1, 1, -1, 1))

        # expand to batch size and num attention heads
        offsets = offsets.expand((batch_size, self.num_attention_heads) + offsets.shape[-2:])
        offset_buckets = (buckets + offsets).flatten(start_dim=2, end_dim=3)

        return offset_buckets

    def _get_sorted_bucket_idx_and_undo_sorted_bucket_idx(self, sequence_length, buckets, num_hashes):
        # no gradients are needed
        with torch.no_grad():
            # hash-based sort
            sorted_bucket_idx = _stable_argsort(buckets, dim=-1)

            # create simple indices to scatter to, to have undo sort
            indices = (
                torch.arange(sorted_bucket_idx.shape[-1], device=buckets.device)
                .view(1, 1, -1)
                .expand(sorted_bucket_idx.shape)
            )

            # get undo sort
            undo_sorted_bucket_idx = sorted_bucket_idx.new(*sorted_bucket_idx.size())
            undo_sorted_bucket_idx.scatter_(-1, sorted_bucket_idx, indices)

        return sorted_bucket_idx, undo_sorted_bucket_idx

    def _set_num_buckets(self, sequence_length):
        # `num_buckets` should be set to 2 * sequence_length // chunk_length as recommended in paper
        num_buckets_pow_2 = (2 * (sequence_length // self.chunk_length)).bit_length() - 1
        # make sure buckets are power of 2
        num_buckets = 2**num_buckets_pow_2

        # factorize `num_buckets` if `num_buckets` becomes too large
        num_buckets_limit = 2 * max(
            int((self.max_position_embeddings // self.chunk_length) ** (0.5)),
            self.chunk_length,
        )
        if num_buckets > num_buckets_limit:
            num_buckets = [2 ** (num_buckets_pow_2 // 2), 2 ** (num_buckets_pow_2 - num_buckets_pow_2 // 2)]

        # set num buckets in config to be properly saved
        self.config.num_buckets = num_buckets
        self.num_buckets = num_buckets

    def _attend(
        self,
        query_vectors,
        key_vectors,
        value_vectors,
        sorted_bucket_idx_per_hash,
        attention_mask,
        head_mask,
        do_standard_self_attention,
        do_cached_attention,
    ):
        # look at previous and following chunks if chunked attention
        if not do_standard_self_attention:
            key_vectors = self._look_adjacent(key_vectors, self.num_chunks_before, self.num_chunks_after)
            value_vectors = self._look_adjacent(value_vectors, self.num_chunks_before, self.num_chunks_after)

        # get logits and dots
        # (BS, NumAttn, NumHash x NumChunk, Chunk_L x Hidden),(BS, NumAttn, NumHash x NumChunk, Chunk_L * (Num_bef + Num_aft + 1) x Hidden) -> (BS, NumAttn, NumHash x NumChunk, Chunk_L, Chunk_L * (1 + Num_bef + Num_aft))
        query_key_dots = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))

        # free memory
        del query_vectors, key_vectors

        # if chunked attention split bucket idxs to query and key
        if not do_standard_self_attention:
            query_bucket_idx = self._split_seq_length_dim_to(
                sorted_bucket_idx_per_hash, -1, self.chunk_length, self.num_attention_heads
            )
            key_value_bucket_idx = self._look_adjacent(query_bucket_idx, self.num_chunks_before, self.num_chunks_after)
        elif do_cached_attention and query_key_dots.ndim > 4:
            key_value_bucket_idx = sorted_bucket_idx_per_hash
            query_bucket_idx = (
                key_value_bucket_idx.new_ones(key_value_bucket_idx.shape[:-1] + (1,)) * key_value_bucket_idx.max()
            )
        elif do_cached_attention and query_key_dots.ndim <= 4:
            query_bucket_idx = (query_key_dots.shape[-1] - 1) * torch.ones_like(query_key_dots)[:, :, :, -1]
            key_value_bucket_idx = torch.arange(
                query_key_dots.shape[-1], dtype=torch.long, device=query_key_dots.device
            )[None, None, :].expand(query_bucket_idx.shape[:2] + (-1,))
        else:
            query_bucket_idx = key_value_bucket_idx = sorted_bucket_idx_per_hash

        # get correct mask values depending on precision
        if query_key_dots.dtype == torch.float16:
            self_mask_value = self.self_mask_value_float16.half()
            mask_value = self.mask_value_float16.half()
        else:
            self_mask_value = self.self_mask_value_float32
            mask_value = self.mask_value_float32

        if not do_cached_attention:
            mask = self._compute_attn_mask(
                query_bucket_idx,
                key_value_bucket_idx,
                attention_mask,
                query_key_dots.shape,
                do_standard_self_attention,
            )

            if mask is not None:
                query_key_dots = torch.where(mask, query_key_dots, mask_value)

            # free memory
            del mask

        # Self mask is ALWAYS applied.
        # From the reformer paper (https://arxiv.org/pdf/2001.04451.pdf):
        # " While attention to the future is not allowed, typical implementations of the
        # Transformer do allow a position to attend to itself.
        # Such behavior is undesirable in a shared-QK formulation because the dot-product
        # of a query vector with itself will almost always be greater than the dot product of a
        # query vector with a vector at another position. We therefore modify the masking
        # to forbid a token from attending to itself, except in situations
        # where a token has no other valid attention targets (e.g. the first token in a sequence) "

        self_mask = torch.ne(query_bucket_idx.unsqueeze(-1), key_value_bucket_idx.unsqueeze(-2)).to(
            query_bucket_idx.device
        )

        # apply self_mask
        query_key_dots = torch.where(self_mask, query_key_dots, self_mask_value)

        # free memory
        del self_mask

        logits = torch.logsumexp(query_key_dots, dim=-1, keepdim=True)
        # dots shape is `[batch_size, num_attn_heads, num_hashes * seq_len // chunk_length, chunk_length, chunk_length * (1 + num_chunks_before + num_chunks_after)]`
        attention_probs = torch.exp(query_key_dots - logits)

        # free memory
        del query_key_dots

        # dropout
        attention_probs = nn.functional.dropout(attention_probs, p=self.dropout, training=self.training)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # attend values
        out_vectors = torch.matmul(attention_probs, value_vectors)

        # free memory
        del value_vectors

        # merge chunk length
        if out_vectors.ndim > 4:
            logits = logits.flatten(start_dim=2, end_dim=3).squeeze(-1)
            out_vectors = out_vectors.flatten(start_dim=2, end_dim=3)

        return out_vectors, logits, attention_probs

    def _compute_attn_mask(
        self, query_indices, key_indices, attention_mask, query_key_dot_shape, do_standard_self_attention
    ):
        # attention mask for LSH
        if attention_mask is not None:
            # if chunked attention, the attention mask has to correspond to LSH order
            attention_mask = attention_mask.to(torch.uint8)[:, None, :]
            if not do_standard_self_attention:
                # expand attn_mask to fit with key_value_bucket_idx shape
                attention_mask = attention_mask[:, None, :]
                attention_mask = attention_mask.expand(query_indices.shape[:-1] + (-1,))
                # extract attention mask from LSH sorted key_indices
                attention_mask = torch.gather(attention_mask, -1, key_indices)

            attention_mask = attention_mask.unsqueeze(-2).expand(query_key_dot_shape)

        # Causal mask
        if self.is_decoder is True:
            causal_mask = torch.ge(query_indices.unsqueeze(-1), key_indices.unsqueeze(-2)).to(query_indices.device)

            # add attention mask if not None
            if attention_mask is not None:
                attention_mask = causal_mask * attention_mask
            else:
                attention_mask = causal_mask

        return attention_mask

    def _get_relevant_hid_states_and_buckets(
        self, query_vectors, attention_mask, num_hashes, hidden_states, past_states, past_buckets
    ):
        # concat hidden states
        hidden_states = torch.cat([past_states, hidden_states], dim=1)

        # batch_size hidden
        batch_size = hidden_states.shape[0]
        sequence_length = hidden_states.shape[1]

        # check if cached buckets include pad bucket
        max_bucket = self.num_buckets if isinstance(self.num_buckets, int) else reduce(mul, self.num_buckets)

        # if pad bucket was cached => need to increase num buckets for caching
        increase_num_buckets = past_buckets.max() > num_hashes * max_bucket - 1

        # retrieve query buckets
        query_buckets = self._hash_vectors(
            query_vectors, num_hashes, attention_mask, increase_num_buckets=increase_num_buckets
        )

        # concat buckets
        concat_buckets = torch.cat([past_buckets, query_buckets.unsqueeze(-1)], dim=-1)

        # hash-based sort
        bucket_idx = _stable_argsort(concat_buckets, dim=-1)

        # bucket_idx has shape: BatchSize x NumAttnHeads x NumHashes x SequenceLength
        assert bucket_idx.shape == (
            batch_size,
            self.num_attention_heads,
            num_hashes,
            sequence_length,
        ), (
            f"bucket_idx should have shape {(batch_size, self.num_attention_heads, num_hashes, sequence_length)}, but"
            f" has shape {bucket_idx.shape}."
        )

        # find indices of new bucket indices
        relevant_bucket_idx = (bucket_idx == (bucket_idx.shape[-1] - 1)).nonzero()

        # expand relevant bucket indices to its chunks
        relevant_bucket_idx_chunk = self._expand_to_indices_in_relevant_chunk(relevant_bucket_idx, sequence_length)
        relevant_bucket_idx_chunk = bucket_idx[tuple(relevant_bucket_idx_chunk.transpose(0, 1))]

        # adapt bucket_idx for batch and hidden states for index select
        bucket_idx_batch_offset = sequence_length * (
            batch_size
            * torch.arange(relevant_bucket_idx_chunk.shape[-1], device=hidden_states.device, dtype=torch.long)
            // relevant_bucket_idx_chunk.shape[-1]
        )

        # add batch offset
        relevant_bucket_idx_chunk_all_batch = relevant_bucket_idx_chunk + bucket_idx_batch_offset
        hidden_states = hidden_states.reshape((-1, self.hidden_size))

        # select all relevant hidden states
        relevant_hidden_states = hidden_states.index_select(0, relevant_bucket_idx_chunk_all_batch)

        # reshape hidden states and bucket_idx to correct output
        relevant_hidden_states = relevant_hidden_states.reshape(
            batch_size, self.num_attention_heads, -1, self.hidden_size
        )
        relevant_bucket_idx_chunk = relevant_bucket_idx_chunk.reshape(
            batch_size, self.num_attention_heads, num_hashes, -1
        )

        assert (
            relevant_hidden_states.shape[2]
            == (self.num_chunks_before + self.num_chunks_after + 1) * self.chunk_length * num_hashes
        ), (
            "There should be"
            f" {(self.num_chunks_before + self.num_chunks_after + 1) * self.chunk_length * num_hashes} `hidden_states`,"
            f" there are {relevant_hidden_states.shape[2]} `hidden_states`."
        )

        assert (
            relevant_bucket_idx_chunk.shape[-1]
            == (self.num_chunks_before + self.num_chunks_after + 1) * self.chunk_length
        ), (
            "There should be"
            f" {(self.num_chunks_before + self.num_chunks_after + 1) * self.chunk_length} `hidden_states`, there are"
            f" {relevant_bucket_idx_chunk.shape[-1]} `bucket_idx`."
        )

        return relevant_hidden_states, relevant_bucket_idx_chunk, query_buckets

    def _expand_to_indices_in_relevant_chunk(self, indices, sequence_length):
        # get relevant indices of where chunk starts and its size
        start_indices_chunk = ((indices[:, -1] // self.chunk_length) - self.num_chunks_before) * self.chunk_length
        total_chunk_size = self.chunk_length * (1 + self.num_chunks_before + self.num_chunks_after)

        # expand start indices and add correct chunk offset via arange
        expanded_start_indices = start_indices_chunk.unsqueeze(-1).expand(indices.shape[0], total_chunk_size)
        chunk_sequence_indices = expanded_start_indices + torch.arange(
            total_chunk_size, device=indices.device, dtype=torch.long
        ).unsqueeze(0).expand(indices.shape[0], total_chunk_size)

        # make sure that circular logic holds via % seq len
        chunk_sequence_indices = chunk_sequence_indices.flatten() % sequence_length

        # expand indices and set indices correctly
        indices = indices.unsqueeze(1).expand((indices.shape[0], total_chunk_size, -1)).flatten(0, 1).clone()
        indices[:, -1] = chunk_sequence_indices

        return indices

    def _len_and_dim_norm(self, vectors):
        """
        length and attention head size dim normalization
        """
        vectors = self._len_norm(vectors)
        vectors = vectors * torch.rsqrt(
            torch.tensor(self.attention_head_size, device=vectors.device, dtype=vectors.dtype)
        )
        return vectors

    def _len_norm(self, x, epsilon=1e-6):
        """
        length normalization
        """
        variance = torch.mean(x**2, -1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + epsilon)
        return norm_x

    def _gather_by_expansion(self, vectors, idxs, num_hashes):
        """
        expand dims of idxs and vectors for all hashes and gather
        """
        expanded_idxs = idxs.unsqueeze(-1).expand(-1, -1, -1, self.attention_head_size)
        vectors = vectors.repeat(1, 1, num_hashes, 1)
        return torch.gather(vectors, 2, expanded_idxs)


class LSHAttention(nn.Module):
    def __init__(self, arch_config: ArchConfig, hidden_size: int, total_heads: int,
                 op_heads: int, auto_pick_num_buckets: bool = True, autopad: bool = True, 
                 **kwargs):
        assert hidden_size % total_heads == 0, 'hidden size must be divisible by total heads'
        super().__init__()

        self.hidden_size = hidden_size
        self.total_heads = total_heads
        self.op_heads = op_heads
        self.op_size = (self.hidden_size // self.total_heads) * self.op_heads
        
        self.num_hashes = arch_config.pick('num_hashes')
        self.bucket_size = arch_config.pick('bucket_size')
        self.num_buckets = arch_config.pick('num_buckets') if not auto_pick_num_buckets else None
        self.autopad = autopad

        self.config = ReformerConfig(
            attn_layers=['lsh'],
            hidden_size=hidden_size,
            num_attention_heads=op_heads,
            num_hashes=self.num_hashes,
            lsh_attn_chunk_length=self.bucket_size,
            num_buckets=self.num_buckets,
            attention_head_size=(self.hidden_size // self.total_heads),
            axial_pos_embds=False,
            is_decoder=True,
            use_cache=False
        )

        self.attn = HfLSHSelfAttention(self.config)

    def forward(self, hidden_states, bin_attention_mask: Optional[torch.FloatTensor] = None, 
                past_buckets_states: Optional[torch.Tensor] = None,  use_cache: bool = False,
                *args, **kwargs):
        
        seq_len = hidden_states.size(1)

        # Pads input to be divisible by bucket size
        if self.autopad and seq_len % self.bucket_size != 0:
            pad_size = (self.bucket_size - seq_len % self.bucket_size) % self.bucket_size
            
            # Pads hidden states and attention mask with zeros so attn is not computed for padded tokens
            p_hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, pad_size, 0))
            p_bin_attention_mask = torch.nn.functional.pad(bin_attention_mask, (pad_size, 0))

            # Computes attention with padded input and unpads output
            output = self.attn(p_hidden_states, attention_mask=p_bin_attention_mask)
            
            # add buckets if necessary
            if hasattr(output, "buckets"):
                buckets = output.buckets
            else:
                buckets = None

            return output[0][:, pad_size:], output[1:]
        
        return self.attn(hidden_states, attention_mask=bin_attention_mask)
