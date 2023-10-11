# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import torch
from fairseq import utils
from torch.distributions.beta import Beta
from scipy.stats import beta


class Schedule:
    @staticmethod
    def linear_alpha(N, T):
        """
        Generate a tensor with integers between 1 and T, inclusive.
        
        Parameters:
        - N: Number of elements
        - T: Number of time steps
        
        Returns:
        - Tran_time: Tensor with random integers between 1 and T, inclusive
        """
        Tran_time = torch.randint(1, T + 1, (N,))
        return Tran_time

    @staticmethod
    def linear_lambda(N, T):
        """
        Calculate the transition times of noise values over T steps.
        
        Parameters:
        - N: Number of elements
        - T: Number of time steps
        
        Returns:
        - Tran_time: Transition times for each element
        """
        Noise_index = torch.ones(N, dtype=torch.bool)
        Tran_time = torch.zeros(N, dtype=torch.float32)

        for i in range(T):
            cur_step = T - i - 1
            rate = 1 - (cur_step + 1) / T
            random_row = torch.rand(N)  # Make this 1D
            Tran_A = random_row > rate
            Tran_N = Noise_index & ~Tran_A
            Noise_index &= ~Tran_N
            Tran_time += Tran_N.float() * cur_step

        return Tran_time

    # def cosine(N, T):
    #   k = np.arange(0, T)
    #   weights = np.cos(k / T * np.pi * 0.5) - np.cos((k + 1) / T * np.pi * 0.5)
    #   weights_tensor = torch.tensor(weights, dtype=torch.float32)
    #   # weights_tensor /= weights_tensor.sum()  # normalize weights to sum to 1
    #   Tran_time = torch.multinomial(weights_tensor, N, replacement=True) + 1
    #   return Tran_time
    def cosine(N, T):
      k = np.arange(0, T)
      s = 0.08 
      weights = np.cos(((k / T) + s)/(1+s) * np.pi * 0.5)**2 - np.cos((((k+1) / T) + s)/(1+s) * np.pi * 0.5)**2 
      weights_tensor = torch.tensor(weights, dtype=torch.float32)
      # weights_tensor /= weights_tensor.sum()  # normalize weights to sum to 1
      Tran_time = torch.multinomial(weights_tensor, N, replacement=True) + 1
      return Tran_time
    def Beta(N, T, alpha, beta_val):
      """
      Generate a tensor with integers between 1 and T, inclusive, with weights derived from a Beta distribution.
      
      Parameters:
      - N: Number of elements
      - T: Number of time steps
      - alpha, beta_val: Parameters of the Beta distribution
      
      Returns:
      - Tran_time: Tensor with random integers between 1 and T, inclusive
      """
      x = np.linspace(0, 1, T)
      weights = beta.pdf(x, alpha, beta_val)
      weights_tensor = torch.tensor(weights, dtype=torch.float32)
      weights_tensor /= weights_tensor.sum()  # normalize weights to sum to 1
      Tran_time = torch.multinomial(weights_tensor, N, replacement=True) + 1
      return Tran_time



class DiffusionGenerator(object):
    def __init__(
        self,
        tgt_dict,
        models=None,
        max_iter=10,
        beam_size=1,
        beam_within_length=1,
        retain_dropout=False,
        adaptive=True,
        retain_history=False,
        reranking=False,
        diffusion=None,
        decoder_options=None,
        return_all_cands=False,
        continuous=False,  # ADDED
        continuous_sample=False,  # ADDED
    ):
        """
        Generates translations based on reverse diffusion processes.

        Args:
            tgt_dict: target dictionary
            eos_penalty: if > 0.0, it penalized early-stopping in decoding
            max_iter: maximum number of refinement iterations
            max_ratio: generate sequences of maximum length ax, where x is the source length
            decoding_format: decoding mode in {'unigram', 'ensemble', 'vote', 'dp', 'bs'}
            retain_dropout: retaining dropout in the inference
            adaptive: decoding with early stop
        """
        self.bos = tgt_dict.bos()
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()
        self.vocab_size = len(tgt_dict)
        self.max_iter = max_iter
        self.beam_size = beam_size * beam_within_length
        self.length_beam_size = beam_size
        self.beam_within_length = beam_within_length
        self.reranking = reranking
        self.retain_dropout = retain_dropout
        self.retain_history = retain_history
        self.adaptive = adaptive
        self.models = models
        self.diffusion = diffusion
        self.decoder_options = decoder_options
        self.return_all_cands = return_all_cands

        self.continuous = continuous
        self.continuous_sample = continuous_sample

        self.generate = self.generate_v8
        if continuous_sample:
            self.generate = self.generate_cont

    def generate_batched_itr(
        self,
        data_itr,
        maxlen_a=None,
        maxlen_b=None,
        cuda=False,
        timer=None,
        prefix_size=0,
    ):
        """Iterate over a batched dataset and yield individual translations.

        Args:
            maxlen_a/b: generate sequences of maximum length ax + b,
                where x is the source sentence length.
            cuda: use GPU for generation
            timer: StopwatchMeter for timing generations.
        """

        for sample in data_itr:
            if "net_input" not in sample:
                continue
            if timer is not None:
                timer.start()
            with torch.no_grad():
                hypos = self.generate(
                    self.models,
                    sample,
                    prefix_tokens=sample["target"][:, :prefix_size]
                    if prefix_size > 0
                    else None,
                )
            if timer is not None:
                timer.stop(sample["ntokens"])
            for i, id in enumerate(sample["id"]):
                # remove padding
                src = utils.strip_pad(sample["net_input"]["src_tokens"][i, :], self.pad)
                ref = utils.strip_pad(sample["target"][i, :], self.pad)
                yield id, src, ref, hypos[i]

    @torch.no_grad()
    def generate_cont(self, models, sample, prefix_tokens=None, constraints=None):
        if constraints is not None:
            raise NotImplementedError(
                "Constrained decoding with the IterativeRefinementGenerator is not supported"
            )

        # TODO: iterative refinement generator does not support ensemble for now.
        if not self.retain_dropout:
            for model in models:
                model.eval()

        model, reranker = models[0], None
        if self.reranking:
            assert len(models) > 1, "Assuming the last checkpoint is the reranker"
            assert (
                self.beam_size > 1
            ), "Reranking requires multiple translation for each example"

            reranker = models[-1]
            models = models[:-1]

        if len(models) > 1 and hasattr(model, "enable_ensemble"):
            assert model.allow_ensemble, "{} does not support ensembling".format(
                model.__class__.__name__
            )
            model.enable_ensemble(models)

        # TODO: better encoder inputs?
        src_tokens = sample["net_input"]["src_tokens"]
        src_lengths = sample["net_input"]["src_lengths"]
        bsz, src_len = src_tokens.size()

        # initialize
        encoder_out = model.forward_encoder([src_tokens, src_lengths])
        prev_decoder_out = model.initialize_output_tokens(encoder_out, src_tokens)

        if self.beam_size > 1:
            assert (
                model.allow_length_beam
            ), "{} does not support decoding with length beam.".format(
                model.__class__.__name__
            )

            # regenerate data based on length-beam
            length_beam_order = (
                utils.new_arange(src_tokens, self.beam_size, bsz).t().reshape(-1)
            )
            encoder_out = model.encoder.reorder_encoder_out(
                encoder_out, length_beam_order
            )
            prev_decoder_out = model.regenerate_length_beam(
                prev_decoder_out, self.length_beam_size, self.beam_within_length
            )
            bsz = bsz * self.beam_size

        sent_idxs = torch.arange(bsz).to(src_tokens.device)
        prev_output_tokens = prev_decoder_out.output_tokens.clone()

        if self.retain_history:
            prev_decoder_out = prev_decoder_out._replace(history=[prev_output_tokens])

        finalized = [[] for _ in range(bsz)]

        def finalized_hypos(step, prev_out_token, prev_out_score, prev_out_attn):
            cutoff = prev_out_token.ne(self.pad)
            tokens = prev_out_token[cutoff]
            if prev_out_score is None:
                scores, score = None, None
            else:
                scores = prev_out_score[cutoff]
                score = scores.mean()

            if prev_out_attn is None:
                hypo_attn, alignment = None, None
            else:
                hypo_attn = prev_out_attn[cutoff]
                alignment = hypo_attn.max(dim=1)[1]
            return {
                "steps": step,
                "tokens": tokens,
                "positional_scores": scores,
                "score": score,
                "hypo_attn": hypo_attn,
                "alignment": alignment,
            }
        
        decoder_options = {
            **self.decoder_options,
            "adaptive_decoding": self.adaptive,
            "continuous_sample": self.continuous_sample
        } 
        N = self.max_iter 

        if model.diffusion.continuous_sample == True:
            N = prev_decoder_out.output_tokens.shape[1]
            alphas = torch.ones(N).cuda() * 17.0
            betas = torch.ones(N).cuda() * 4.0
            rand_ind = (torch.rand(N)).cuda()
            dist = Beta(alphas, betas)
            rand_ind = dist.sample() * model.diffusion.num_timesteps

        # previous CMLM fairseq impl. assumes the total number of steps
        # equals max_iters + 1; fix here to avoid consfusion
        for step in range(N):
            if model.diffusion.continuous_sample == True:
                prev_decoder_out = prev_decoder_out._replace(
                    step=step,
                    sampled = rand_ind,
                    max_step=self.max_iter,
                )
            else:
                prev_decoder_out = prev_decoder_out._replace(
                    step=step,
                    max_step=self.max_iter,
                )

            decoder_out = model.forward_decoder(
                prev_decoder_out, encoder_out, **decoder_options
            )

            if self.adaptive:
                # we want shorter sequences to be diffused in fewer time steps.
                # if self.adaptive is flagged as True, we terminate the diffusion process
                # once the time step is larger than or equal to the effective sequence length.
                # the skip stepsize inside each diffusion process is adjusted accordingly.
                non_special_symbols = (
                    decoder_out.output_tokens.ne(self.pad) & 
                    decoder_out.output_tokens.ne(self.bos) & 
                    decoder_out.output_tokens.ne(self.eos)
                ).float()
                effective_seq_len = non_special_symbols.sum(-1)
                terminated = (step + 1) >= effective_seq_len
            else:
                terminated = decoder_out.output_tokens.new_zeros(
                    decoder_out.output_tokens.size(0)
                ).bool()

            if step == (N - 1):  # reach last iteration, terminate
                terminated.fill_(1)

            # collect finalized sentences
            finalized_idxs = sent_idxs[terminated]
            finalized_tokens = decoder_out.output_tokens[terminated]
            finalized_scores = decoder_out.output_scores[terminated]
            finalized_attn = (
                None
                if (decoder_out.attn is None or decoder_out.attn.size(0) == 0)
                else decoder_out.attn[terminated]
            )

            # if self.retain_history:
            #     finalized_history_tokens = [h[terminated] for h in decoder_out.history]

            for i in range(finalized_idxs.size(0)):
                finalized[finalized_idxs[i]] = [
                    finalized_hypos(
                        step,
                        finalized_tokens[i],
                        finalized_scores[i],
                        None if finalized_attn is None else finalized_attn[i],
                    )
                ]

                # if self.retain_history:
                #     finalized[finalized_idxs[i]][0]["history"] = []
                #     for j in range(len(finalized_history_tokens)):
                #         finalized[finalized_idxs[i]][0]["history"].append(
                #             finalized_hypos(
                #                 step, finalized_history_tokens[j][i], None, None
                #             )
                #         )

            # # check if all terminated
            # if terminated.sum() == terminated.size(0):
            #     break

            # # for next step
            # not_terminated = ~terminated
            # prev_decoder_out = decoder_out._replace(
            #     output_tokens=decoder_out.output_tokens[not_terminated],
            #     output_scores=decoder_out.output_scores[not_terminated],
            #     auxiliary_output={k : decoder_out.auxiliary_output[k][not_terminated] for k in decoder_out.auxiliary_output},
            #     attn=decoder_out.attn[not_terminated]
            #     if (decoder_out.attn is not None and decoder_out.attn.size(0) > 0)
            #     else None,
            #     history=[h[not_terminated] for h in decoder_out.history]
            #     if decoder_out.history is not None
            #     else None,
            # )
            # encoder_out = model.encoder.reorder_encoder_out(
            #     encoder_out, not_terminated.nonzero(as_tuple=False).squeeze()
            # )
            # sent_idxs = sent_idxs[not_terminated]
            # prev_output_tokens = prev_decoder_out.output_tokens.clone()

        if self.beam_size > 1:
            if reranker is not None:
                finalized = self.rerank(
                    reranker, finalized, [src_tokens, src_lengths], self.beam_size
                )
            if self.return_all_cands:
                # return all beams
                finalized = [
                    [
                        finalized[self.beam_size * i + j][0]
                        for j in range(self.beam_size)
                    ]
                    for i in range(len(finalized) // self.beam_size)
                ]
            else:
                # aggregate information from length beam
                finalized = [
                    finalized[
                        np.argmax(
                            [
                                finalized[self.beam_size * i + j][0]["score"].cpu().numpy()
                                for j in range(self.beam_size)
                            ]
                        )
                        + self.beam_size * i
                    ]
                    for i in range(len(finalized) // self.beam_size)
                ]

        return finalized

    def generate_v8(self, models, sample, prefix_tokens=None, constraints=None):
        if constraints is not None:
            raise NotImplementedError(
                "Constrained decoding with the IterativeRefinementGenerator is not supported"
            )

        # TODO: iterative refinement generator does not support ensemble for now.
        if not self.retain_dropout:
            for model in models:
                model.eval()

        model, reranker = models[0], None
        if self.reranking:
            assert len(models) > 1, "Assuming the last checkpoint is the reranker"
            assert (
                self.beam_size > 1
            ), "Reranking requires multiple translation for each example"

            reranker = models[-1]
            models = models[:-1]

        if len(models) > 1 and hasattr(model, "enable_ensemble"):
            assert model.allow_ensemble, "{} does not support ensembling".format(
                model.__class__.__name__
            )
            model.enable_ensemble(models)

        # TODO: better encoder inputs?
        src_tokens = sample["net_input"]["src_tokens"]
        src_lengths = sample["net_input"]["src_lengths"]
        bsz, src_len = src_tokens.size()

        # initialize
        encoder_out = model.forward_encoder([src_tokens, src_lengths])
        prev_decoder_out = model.initialize_output_tokens(encoder_out, src_tokens)

        if self.beam_size > 1:
            assert (
                model.allow_length_beam
            ), "{} does not support decoding with length beam.".format(
                model.__class__.__name__
            )

            # regenerate data based on length-beam
            length_beam_order = (
                utils.new_arange(src_tokens, self.beam_size, bsz).t().reshape(-1)
            )
            encoder_out = model.encoder.reorder_encoder_out(
                encoder_out, length_beam_order
            )
            prev_decoder_out = model.regenerate_length_beam(
                prev_decoder_out, self.length_beam_size, self.beam_within_length
            )
            bsz = bsz * self.beam_size

        sent_idxs = torch.arange(bsz).to(src_tokens.device)
        prev_output_tokens = prev_decoder_out.output_tokens.clone()

        finalized = [[] for _ in range(bsz)]

        def finalized_hypos(step, prev_out_token, prev_out_score, prev_out_attn):
            cutoff = prev_out_token.ne(self.pad)
            tokens = prev_out_token[cutoff]
            if prev_out_score is None:
                scores, score = None, None
            else:
                scores = prev_out_score[cutoff]
                score = scores.mean()

            if prev_out_attn is None:
                hypo_attn, alignment = None, None
            else:
                hypo_attn = prev_out_attn[cutoff]
                alignment = hypo_attn.max(dim=1)[1]
            return {
                "steps": step,
                "tokens": tokens,
                "positional_scores": scores,
                "score": score,
                "hypo_attn": hypo_attn,
                "alignment": alignment,
            }
        
        decoder_options = {
            **self.decoder_options,
            "adaptive_decoding": self.adaptive,
            "continuous_sample": self.continuous_sample
        } 

        N = prev_decoder_out.output_tokens.shape[1]
        T = self.max_iter
        ####Get transition time 
        schedule = "Beta"
        if schedule == "linear_lambda":
          Transition_time = Schedule.linear_lambda(N,T).cuda()
        elif schedule == "linear_alpha":
          Transition_time = Schedule.linear_alpha(N,T).cuda()
        elif schedule == "cosine":
          Transition_time = Schedule.cosine(N,T).cuda()
        elif schedule == "Beta":
          # ###Greate for 1000 step
          # Transition_time = Schedule.Beta(N,T,17,4).cuda()
          ##Greate for 50 step
          Transition_time = Schedule.Beta(N,T,2,2).cuda()
        else:
          raise NotImplementedError
        ### Remove duplicates
        unique_time = torch.unique(Transition_time)

        for step in range(unique_time.shape[0]):

            prev_decoder_out = prev_decoder_out._replace(
                step=step,
                history = Transition_time,
                max_step=self.max_iter,
            )

            decoder_out = model.forward_decoder(
                prev_decoder_out, encoder_out, **decoder_options
            )

        # collect finalized sentences
        finalized_idxs = sent_idxs
        finalized_tokens = decoder_out.output_tokens
        finalized_scores = decoder_out.output_scores
        finalized_attn = (
            None
            if (decoder_out.attn is None or decoder_out.attn.size(0) == 0)
            else decoder_out.attn
        )


        for i in range(finalized_idxs.size(0)):
            finalized[finalized_idxs[i]] = [
                finalized_hypos(
                    step,
                    finalized_tokens[i],
                    finalized_scores[i],
                    None if finalized_attn is None else finalized_attn[i],
                )
            ]
        if self.beam_size > 1:
            if reranker is not None:
                finalized = self.rerank(
                    reranker, finalized, [src_tokens, src_lengths], self.beam_size
                )
            if self.return_all_cands:
                # return all beams
                finalized = [
                    [
                        finalized[self.beam_size * i + j][0]
                        for j in range(self.beam_size)
                    ]
                    for i in range(len(finalized) // self.beam_size)
                ]
            else:
                # aggregate information from length beam
                finalized = [
                    finalized[
                        np.argmax(
                            [
                                finalized[self.beam_size * i + j][0]["score"].cpu().numpy()
                                for j in range(self.beam_size)
                            ]
                        )
                        + self.beam_size * i
                    ]
                    for i in range(len(finalized) // self.beam_size)
                ]

        return finalized


    def rerank(self, reranker, finalized, encoder_input, beam_size):
        def rebuild_batch(finalized):
            finalized_tokens = [f[0]["tokens"] for f in finalized]
            finalized_maxlen = max(f.size(0) for f in finalized_tokens)
            final_output_tokens = (
                finalized_tokens[0]
                .new_zeros(len(finalized_tokens), finalized_maxlen)
                .fill_(self.pad)
            )
            for i, f in enumerate(finalized_tokens):
                final_output_tokens[i, : f.size(0)] = f
            return final_output_tokens

        final_output_tokens = rebuild_batch(finalized)
        final_output_tokens[
            :, 0
        ] = self.eos  # autoregressive model assumes starting with EOS

        reranker_encoder_out = reranker.encoder(*encoder_input)
        length_beam_order = (
            utils.new_arange(
                final_output_tokens, beam_size, reranker_encoder_out.encoder_out.size(1)
            )
            .t()
            .reshape(-1)
        )
        reranker_encoder_out = reranker.encoder.reorder_encoder_out(
            reranker_encoder_out, length_beam_order
        )
        reranking_scores = reranker.get_normalized_probs(
            reranker.decoder(final_output_tokens[:, :-1], reranker_encoder_out),
            True,
            None,
        )
        reranking_scores = reranking_scores.gather(2, final_output_tokens[:, 1:, None])
        reranking_masks = final_output_tokens[:, 1:].ne(self.pad)
        reranking_scores = (
            reranking_scores[:, :, 0].masked_fill_(~reranking_masks, 0).sum(1)
        )
        reranking_scores = reranking_scores / reranking_masks.sum(1).type_as(
            reranking_scores
        )

        for i in range(len(finalized)):
            finalized[i][0]["score"] = reranking_scores[i]

        return finalized