import math
import numpy as np
import torch
import torch.distributions as dists
import torch.nn.functional as F
from discrete_diffusions.utils import mean_ds
from discrete_diffusions.discrete_diffusion_base import DiscreteDiffusion

from discrete_diffusions.utils import (
    index_to_log_onehot,
    topk_masking
)
from torch.nn.modules.module import T

class ReparamAbsorbingDiffusion(DiscreteDiffusion):
    def __init__(
            self, 
            num_timesteps,
            mask_id, 
            reweighting_type,
            not_diffusing_special_sym,
            pad_id, bos_id, eos_id,
            continuous=False,
            continuous_sample=True,
        ):
        """
            Reparameterized absorbing diffusion impl. is very similar to that of absorbing diffusion,
            but we implement it in a separate class for reference.
        """
        super().__init__(num_timesteps=num_timesteps)
        # mask the transition probability from normal tokens to special symbols.
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.mask_idx = mask_id
        self.continuous = continuous
        self.continuous_sample = continuous_sample
        self.num_timesteps = num_timesteps
        self.reweighting_type = reweighting_type
        self.not_diffusing_special_sym = not_diffusing_special_sym

    def q_sample_coupled(self, x_0, t1, t2, non_special_sym_mask):
        _t1 = torch.maximum(t1, t2).float().unsqueeze(-1) + 1
        _t2 = torch.minimum(t1, t2).float().unsqueeze(-1) + 1
        if self.continuous == True:
            _t1 = torch.maximum(t1, t2).float().unsqueeze(-1)# + 1/self.num_timesteps
            _t2 = torch.minimum(t1, t2).float().unsqueeze(-1)# + 1/self.num_timesteps


        # first sample q(x_{t1} | x_0)
        # and then sample q(x_{t2} | x_{t1}, x_0)
        # if t1 == t2, then draw an indep. sample.

        select_mask = (_t1 == _t2).float()
        u1 = torch.rand_like(x_0.float())
        # TODO: investigate the effect of such antithetic pairs.
        u2 = torch.rand_like(x_0.float()) # 1. - u1
        if self.continuous == True:
            #print('_t1', _t1, '_t2', _t2)
            mask_t1 = u1 < (_t1)
        else:
            mask_t1 = u1 < (_t1 / self.num_timesteps)
        # for skip steps, the prob. of being **decoded** is
        # p = (_t1 - _t2) / _t1. Therefore, u2 > p indicates
        # the prob. that each token still gets masked.
        _mask_t2_if_neq = u2 > ((_t1 - _t2) / _t1)
        mask_t2_if_neq = torch.bitwise_and(_mask_t2_if_neq, mask_t1)
        if self.continuous == True:
            mask_t2_if_eq = u2 < (_t2)
        else:
            mask_t2_if_eq = u2 < (_t2 / self.num_timesteps)

        mask_t2 = mask_t2_if_neq * (1. - select_mask) + mask_t2_if_eq * select_mask
        mask_t2 = mask_t2.bool()

        # masked out special symbols
        if self.not_diffusing_special_sym:
            mask_t1 = torch.bitwise_and(mask_t1, non_special_sym_mask)
            mask_t2 = torch.bitwise_and(mask_t2, non_special_sym_mask)
            
        x_t1, x_0_ignore_t1 = x_0.clone(), x_0.clone()
        x_t2, x_0_ignore_t2 = x_0.clone(), x_0.clone()
        x_t1[mask_t1] = self.mask_idx
        x_0_ignore_t1[torch.bitwise_not(mask_t1)] = -1
        x_t2[mask_t2] = self.mask_idx
        x_0_ignore_t2[torch.bitwise_not(mask_t2)] = -1
        #print('_t1_t2', torch.max(_t1), torch.max(_t2))
        if self.continuous == True:
            out_t = torch.cat([_t1, _t2], dim=0).squeeze(dim=-1)
        else:
            out_t = torch.cat([_t1, _t2], dim=0).long().squeeze(dim=-1) - 1
        return (torch.cat([x_t1, x_t2], dim=0), 
                torch.cat([x_0_ignore_t1, x_0_ignore_t2], dim=0), 
                torch.cat([mask_t1, mask_t2], dim=0),
                out_t,
        )

    def q_sample(self, x_0, t, non_special_sym_mask):
        raise NotImplementedError
        # samples q(x_t | x_0)
        # randomly set token to mask with probability t/T
        x_t, x_0_ignore = x_0.clone(), x_0.clone()

        if self.continuous == True:
            mask = torch.rand_like(x_0.float()) < (t.float().unsqueeze(-1))
        else:
            mask = torch.rand_like(x_0.float()) < ((t.float().unsqueeze(-1) + 1) / self.num_timesteps)
        if self.not_diffusing_special_sym:
            mask = mask & non_special_sym_mask
        x_t[mask] = self.mask_idx
        x_0_ignore[torch.bitwise_not(mask)] = -1
        return x_t, x_0_ignore, mask

    def compute_loss(self, inputs, **kwargs):
        label_smoothing = kwargs.get("label_smoothing", 0.0)
        x_0_ignore = inputs["x_0_ignore"]
        t = inputs["t"]
        weight_t = inputs["weight_t"]
        decoder_outputs = inputs["decoder_outputs"]
        assert t.dim() == 1
        if inputs["masks"] is None:
            masks = inputs["x_t"].eq(self.unk)
        else:
            masks = inputs["masks"]
        logits = decoder_outputs.transpose(-1, -2)
        # mean over all tokens, even though some unmasked tokens do not produce losses.
        cross_entropy_loss = F.cross_entropy(logits, x_0_ignore, ignore_index=-1, reduction='none').mean(1)
        if self.reweighting_type == "reciprocal":
            # t + 1 here since the passed t ranges from 0 to T-1.
            reweighting_coeff = 1. / (t + 1.)
        elif self.reweighting_type == "linear":
            if self.continuous == True:
                reweighting_coeff = (1 - t)
            else:
                reweighting_coeff = (1 - (t / self.num_timesteps))
        elif self.reweighting_type == "none":
            reweighting_coeff = 1.
        else:
            raise NotImplementedError("reweighting type {} not implemented.".format(self.reweighting_type))
        #print('t', t)
        #print('reweighting_coeff', reweighting_coeff)
        #print('weight_t', weight_t)
        vb_loss = reweighting_coeff * cross_entropy_loss
        diffusion_nll_loss = mean_ds(weight_t * vb_loss)
        if label_smoothing > 0:
            logit_loss = mean_ds(
                weight_t * 
                F.log_softmax(decoder_outputs, dim=-1).mean(dim=-1).masked_fill(~masks, 0.).mean(1)
            )
            diffusion_loss = (
                diffusion_nll_loss * (1 - label_smoothing) - logit_loss * label_smoothing
            )
        else:
            diffusion_loss = diffusion_nll_loss

        output_dict = {
            'diffusion_loss': diffusion_loss,
            'diffusion_nll_loss': diffusion_nll_loss,
        }
        logging_outputs = {
            'loss': vb_loss,
            "t": t,
            "weights": weight_t,
        }
        return output_dict, logging_outputs

    def sample_step(self, decoder_out, denoising_fn, schedule_mode = "linearlambda", topk_mode = "cond", **kwargs):
        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        t = decoder_out.step
        max_step = decoder_out.max_step 

        fake_t = decoder_out.step
        Transition_time = decoder_out.sampled
        unique_time = torch.unique(Transition_time)
                                 
        if self.continuous_sample == True:
            sampled = decoder_out.sampled
            #print(history)
            sorted, indices = torch.sort(sampled)
            max_step = len(sorted)
            #print('sorted', sorted)
            #sorted = decoder_out.sorted
            #indices = decoder_out.indices
        else:
            sorted, indices = torch.sort(unique_time, descending=True)
            t = sorted[fake_t]
            Tran_al = Transition_time >= t    

        argmax_decoding = kwargs.get('argmax_decoding', False)
        #decoding_strategy = kwargs.get('decoding_strategy', "reparam")

        # manually construct a non-special-sym mask.
        non_special_sym_mask = (
            output_tokens.ne(self.pad_id) & 
            output_tokens.ne(self.bos_id) & 
            output_tokens.ne(self.eos_id)
        )
        non_special_sym_mask = kwargs.get('non_special_sym_mask', non_special_sym_mask)

        # int, int
        if self.continuous_sample == True:
            cur_step, mask = self._get_continuous_strategy(t, max_step, sorted, indices)
            cur_step_tensor = torch.full((output_tokens.shape[0],), cur_step, device=output_tokens.device)
        else:
            #cur_step, cur_stepsize = self._get_decoding_strategy(t, decoding_strategy, max_step)
            #cur_step = cur_step - 1
            #cur_step_tensor = torch.full((output_tokens.shape[0],), cur_step/self.num_timesteps*1, device=output_tokens.device)
            cur_step = (t/max_step)#*self.num_timesteps
            cur_step_tensor = torch.full((output_tokens.shape[0],), cur_step, device=output_tokens.device, dtype=torch.long)
        #print('cur_step:', cur_step)
        # minus 1 due to the offset.
        # cur_step = cur_step - 1
        # if self.continuous == True:
        #     cur_step_tensor = torch.full((output_tokens.shape[0],), cur_step/self.num_timesteps * 1, device=output_tokens.device)
        # else:
        #     cur_step_tensor = torch.full((output_tokens.shape[0],), cur_step, device=output_tokens.device)
        #print("cur t : {}, step size: {}, t: {}, max_step: {}, diffusion steps : {}".format(cur_step, cur_stepsize, t, max_step, self.num_timesteps), flush=True)
        # log_x_t = index_to_log_onehot(output_tokens, self.vocab_size) # [b, n, c]
  
        # log_x_t = index_to_log_onehot(output_tokens, self.vocab_size) # [b, n, c]
        # log_x0_recon = denoising_fn(x_t=output_tokens, t=cur_step_tensor) # [b, n, c]
        # log_x0_recon = torch.log_softmax(log_x0_recon.masked_fill(self.special_sym_indicator.bool(), -30), dim=-1)
        # log_x0_recon = torch.where(non_special_sym_mask.unsqueeze(-1), log_x0_recon, log_x_t)

        # denoising_fn(x_t, t, **kwargs)
        scores = denoising_fn(
            x_t=output_tokens,
            t=cur_step_tensor,
        )
        # redistributing probs. to avoid generating unk explicitly.
        scores[..., self.mask_idx] = -math.inf  # apply unk penalty
        scores = torch.log_softmax(scores, dim=-1)
        log_x0_recon = scores
        if argmax_decoding or True:
          cur_scores, cur_tokens = log_x0_recon.max(-1)
        else:
          #zixiang:0.01 is better than argmax
          temperature = 0.01
          cur_tokens = dists.Categorical(logits=log_x0_recon / temperature).sample()
          cur_scores = torch.gather(log_x0_recon, -1, cur_tokens.unsqueeze(-1)).squeeze(-1)
 

        if topk_mode == "real":
          if self.continuous_sample == False:
            pre_Noise_index = decoder_out.auxiliary_output["output_masks"]
            Noise_index_revised = (~Tran_al).repeat(output_scores.size(0), 1)&non_special_sym_mask
            Mask_to_x0 = pre_Noise_index&~Noise_index_revised
          else:
            pre_Noise_index = decoder_out.auxiliary_output["output_masks"]
            Tran_A = (mask.repeat(output_scores.size(0), 1) == 1)
            Noise_index_revised = (~Tran_A)&non_special_sym_mask
            Mask_to_x0 = pre_Noise_index&~Noise_index_revised

              # #### Update all the non noise tokens
              # output_tokens.masked_scatter_(~Noise_index_new, cur_tokens[~Noise_index_new])
              # output_scores.masked_scatter_(~Noise_index_new, cur_scores[~Noise_index_new])
              #### Update all the transition tokens
          #### Update all the transition tokens
          output_tokens.masked_scatter_(Mask_to_x0, cur_tokens[Mask_to_x0])
          output_scores.masked_scatter_(Mask_to_x0, cur_scores[Mask_to_x0])
        elif topk_mode == "cond":
          if self.continuous_sample == False:
            cutoff_len = (
            ((~Tran_al)&non_special_sym_mask).sum(1, keepdim=True).type_as(output_scores)
            ).long()

          else:
            Tran_A = (mask.repeat(output_scores.size(0), 1) == 1) & non_special_sym_mask
            cutoff_len = (
            (~Tran_A&non_special_sym_mask).sum(1, keepdim=True).type_as(output_scores)
            ).long()
            #Since the special_sym_mask will never transition from noise to x0, set them to be 1000 so that never get changed 
          _scores_for_topk = cur_scores.masked_fill(~non_special_sym_mask, 1000.0)
          Noise_index_revised = topk_masking(_scores_for_topk, cutoff_len, stochastic=False)
          #Mask_to_x0 = ~Noise_index_revised
          pre_Noise_index = decoder_out.auxiliary_output["output_masks"]
          Mask_to_x0 = pre_Noise_index&~Noise_index_revised

        
          output_tokens.masked_scatter_(Mask_to_x0, cur_tokens[Mask_to_x0])
          output_scores.masked_scatter_(Mask_to_x0, cur_scores[Mask_to_x0])
          output_tokens.masked_fill_(Noise_index_revised, self.mask_idx)
          output_scores.masked_fill_(Noise_index_revised, -math.inf)


        # return output_tokens, output_scores
        history = decoder_out.history
        
        auxiliary_output = decoder_out.auxiliary_output
        #auxiliary_output["output_masks"] = Noise_index_new
        auxiliary_output["output_masks"] = Noise_index_revised

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            auxiliary_output=auxiliary_output,
            attn=None,
            history=history,
        )

    # def sample_step(self, decoder_out, denoising_fn, **kwargs):
    #     output_tokens = decoder_out.output_tokens
    #     output_scores = decoder_out.output_scores
    #     t = decoder_out.step
    #     max_step = decoder_out.max_step
        
    #     temperature_annealing = kwargs.get('temperature_annealing', False)
    #     decoding_strategy = kwargs.get('decoding_strategy', "linear")
    #     argmax_decoding = kwargs.get('argmax_decoding', False)
    #     decoding_time_difference = kwargs.get('decoding_time_difference', 0.0)
    #     if temperature_annealing:
    #         temperature = -0.05 * (t / (max_step - 1)) + 0.5
    #     else:
    #         temperature = kwargs.get('temperature', 1.0)

    #     cur_step, cur_stepsize = self._get_decoding_strategy(t, decoding_strategy, max_step)
    #     #print('cur_step', cur_step, 'cur_stepsize', cur_stepsize)
    #     # denoising_fn(x_t, t, **kwargs)
    #     if self.continuous == True:
    #         cur_step = cur_step/self.num_timesteps * 1
    #         cur_stepsize = cur_stepsize/self.num_timesteps * 1
    #         cur_step_tensor = torch.full((output_tokens.shape[0],), cur_step, device=output_tokens.device)
    #     else:
    #         cur_step_tensor = torch.full((output_tokens.shape[0],), cur_step, device=output_tokens.device, dtype=torch.long)
        
    #     #print('cur_step_tensor', cur_step_tensor)
    #     scores = denoising_fn(
    #         x_t=output_tokens,
    #         t=cur_step_tensor,
    #     )
    #     # redistributing probs. to avoid generating unk explicitly.
    #     scores[..., self.mask_idx] = -math.inf  # apply unk penalty
    #     scores = torch.log_softmax(scores, dim=-1)

    #     # manually construct a non-special-sym mask, if not passed.
    #     non_special_sym_mask = (
    #         output_tokens.ne(self.pad_id) & 
    #         output_tokens.ne(self.bos_id) & 
    #         output_tokens.ne(self.eos_id)
    #     )
    #     non_special_sym_mask = kwargs.get('non_special_sym_mask', non_special_sym_mask)
    #     if decoding_strategy.startswith("reparam"):
    #         raise NotImplementedError
    #         if argmax_decoding:
    #             cur_scores, cur_tokens = scores.max(-1)
    #         else:
    #             cur_tokens = dists.Categorical(logits=scores / temperature).sample()
    #             cur_scores = torch.gather(scores, -1, cur_tokens.unsqueeze(-1)).squeeze(-1)

    #         # this function modifies output_tokens and output_scores in place.
    #         # see the function for more details.
    #         output_masks = self._reparam_decoding(
    #             output_tokens=output_tokens,
    #             output_scores=output_scores,
    #             cur_tokens=cur_tokens,
    #             cur_scores=cur_scores,
    #             decoding_strategy=decoding_strategy,
    #             xt_neq_x0=decoder_out.auxiliary_output["output_masks"],
    #             non_special_sym_mask=non_special_sym_mask,
    #             t=t,
    #             max_step=max_step,
    #             noise=self.mask_idx,
    #         )
    #     else:
    #         if decoding_time_difference > 0.0:
    #             if self.continuous == True:
    #                 decoding_time_difference = decoding_time_difference/self.num_timesteps
    #             if cur_step <= cur_stepsize:
    #                 cur_step = cur_stepsize
    #             else:
    #                 cur_step = max(cur_step - decoding_time_difference, int(1.5 * cur_stepsize))
    #         # get the mask
    #         # <bos>, <eos> are ignored in this case since
    #         # they are not equal to unk.
    #         #print('cur_step', cur_step)
    #         output_masks = output_tokens.eq(self.mask_idx)
    #         unmask_prob = cur_stepsize / cur_step
    #         # where to unmask
    #         changes = torch.rand(output_tokens.shape, device=output_tokens.device) < unmask_prob
    #         # don't unmask somewhere already unmasked
    #         changes = torch.bitwise_and(changes, output_masks)
    #         #print('unmask_prob', unmask_prob)
    #         #print('output_masks', output_masks)
    #         #print('changes', changes)
    #         if argmax_decoding:
    #             output_scores, new_tokens = scores.max(-1)
    #         else:
    #             new_tokens = dists.Categorical(logits=scores / temperature).sample()
    #             output_scores = torch.gather(scores, -1, new_tokens.unsqueeze(-1)).squeeze(-1)
    #         output_tokens[changes] = new_tokens[changes]
    #     # return output_tokens, output_scores
    #     history = decoder_out.history
    #     # if history is not None:
    #     #     history.append(output_tokens.clone())

    #     auxiliary_output = decoder_out.auxiliary_output
    #     auxiliary_output["output_masks"] = output_masks

    #     return decoder_out._replace(
    #         output_tokens=output_tokens,
    #         output_scores=output_scores,
    #         auxiliary_output=auxiliary_output,
    #         history=history,
    #     )

