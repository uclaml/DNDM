import math
from tkinter import FALSE
import numpy as np
import torch
import torch.distributions as dists
import torch.nn as nn
import torch.nn.functional as F
from inspect import isfunction
from typing import NamedTuple
from discrete_diffusions.utils import (
    index_to_log_onehot,
    log_sample_categorical,
    mean_ds, 
    log_sub_exp,
    log1mexp,
    sample_bernoulli,
)
from discrete_diffusions.discrete_diffusion_base import DiscreteDiffusion
import logging

from discrete_diffusions.utils import (
    topk_masking
)
from torch.nn.modules.module import T
logger = logging.getLogger(__name__)

def exists(x):
    return x is not None

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def get_named_alpha_schedule(timesteps, name='cosine', alpha_min=0.001, alpha_max=1.):
    if name == 'linear':
        alphas_cumprod = np.linspace(1, 0., timesteps + 1)
        alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    elif name in ['cosine', 'cosine_old']:
        """
        cosine schedule
        as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
        """
        steps = timesteps + 1
        s = 0.008
        x = np.linspace(0, steps, steps)
        alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    elif name in ['sqrt']:
        steps = timesteps + 1
        x = np.linspace(0, steps, steps)
        alphas_cumprod = 1 - np.sqrt((x / steps) + 0.0001)
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    alphas = np.clip(alphas, a_min=alpha_min, a_max=alpha_max)
    if name == 'cosine_old':
        # to reproduce multinomial_diffusion numbers.
        alphas = np.sqrt(alphas)
    return alphas

def get_continuous_alpha(t, timesteps, x_shape, name='cosine'):
    b, *_ = t.shape
    if name == 'linear':
        alphas_cumprod = (1 - t/timesteps) if (torch.max(t) > 1) else (1 - t)
    elif name in ['cosine', 'cosine_old']:
        s = 0.008
        x = (t/timesteps) if (torch.max(t) > 1) else t 
        #print('x', x)
        alphas_cumprod = torch.cos((x + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / (np.cos(s/ (1 + s) * np.pi * 0.5) ** 2)
        #print('alphas_cumprod', alphas_cumprod)
        #print('timesteps', timesteps)
    elif name in ['sqrt']:
        x = (t/timesteps) if (torch.max(t) > 1) else t
        alphas_cumprod = 1 - torch.sqrt(x + 0.0001)
        alphas_cumprod = alphas_cumprod / (1 - np.sqrt(0.0001))
    return alphas_cumprod.reshape(b, *((1,) * (len(x_shape) - 1)))


class ReparamMultinomialDiffusion(DiscreteDiffusion):
    def __init__(
            self, 
            num_timesteps,
            vocab_size,
            reweighting_type,
            noise_scheduler_type,
            not_diffusing_special_sym,
            noise_distribution,
            pad_id, bos_id, eos_id,
            vocab_count=None,
            continuous=False,
            continuous_sample=False,
            not_topk=False,
        ):
        super().__init__(num_timesteps)
        self.num_timesteps = num_timesteps
        # with <bos>, <eos>, <pad> removed.
        self.not_diffusing_special_sym = not_diffusing_special_sym
        assert (
            self.not_diffusing_special_sym, 
            "ignoring special symbols and diffusing them as well might lead to drastically worse performance."
        )
        self.reweighting_type = reweighting_type
        # mask the transition probability from normal tokens to special symbols.
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.continuous = continuous
        self.continuous_sample = continuous_sample
        self.not_topk = not_topk
        self.vocab_size = vocab_size
        if noise_distribution == "unigram":
            assert vocab_count is not None and isinstance(vocab_count, list)
            weights = torch.tensor(vocab_count).float() # [V]
        elif noise_distribution == "uniform":
            weights = torch.ones((self.vocab_size)).float()
        weights[self.pad_id] = 0
        weights[self.bos_id] = 0
        weights[self.eos_id] = 0
        log_weight = torch.log(weights.clamp(min=1e-30))
        # divided by some temperature so that the sampling becomes more diverse.
        self.register_buffer('vocab_log_prob', torch.log_softmax(log_weight / 1.5, dim=-1))

        alphas = torch.tensor(get_named_alpha_schedule(self.num_timesteps, name=noise_scheduler_type).astype('float64'))
        log_alpha = np.log(alphas)
        log_cumprod_alpha = np.cumsum(log_alpha)
        log_1_min_alpha = torch.log(1 - log_alpha.exp() + 1e-9)
        log_1_min_cumprod_alpha = torch.log(1 - log_cumprod_alpha.exp() + 1e-9)

        assert torch.logaddexp(log_alpha, log_1_min_alpha).abs().sum().item() < 1.e-5
        assert torch.logaddexp(log_cumprod_alpha, log_1_min_cumprod_alpha).abs().sum().item() < 1e-5
        assert (np.cumsum(log_alpha) - log_cumprod_alpha).abs().sum().item() < 1.e-5

        special_sym_indicator = torch.zeros((self.vocab_size))
        special_sym_indicator[self.pad_id] = 1
        special_sym_indicator[self.bos_id] = 1
        special_sym_indicator[self.eos_id] = 1
        # stored in float format to make it easy to load/operate on model checkpoints.
        self.register_buffer('special_sym_indicator', special_sym_indicator)
        # Convert to float32 and register buffers.
        # here log_alpha[t] denotes the noise schedule at t+1-th time step.
        self.register_buffer('log_alpha', log_alpha.float())
        self.register_buffer('log_1_min_alpha', log_1_min_alpha.float())
        self.register_buffer('log_cumprod_alpha', log_cumprod_alpha.float())
        self.register_buffer('log_1_min_cumprod_alpha', log_1_min_cumprod_alpha.float())

        try:
            with open('cts_config.txt', 'r') as f:
                s = f.read()
                cts_config = eval(s)
                self.continuous = cts_config['continuous']
                self.continuous_sample = cts_config['continuous_sample']
                self.not_topk = cts_config['not_topk']
        except:
            print("NO CONTINUOUS\DISCRETE CONFIG FILE (cts_config.txt) FOUND")
            print('self.continuous: ', self.continuous)
            print('self.continuous_sample: ', self.continuous_sample)
            print('self.not_topk: ', self.not_topk)

        self.topk_mode = 'cond'
        if self.not_topk:
            self.topk_mode = 'real'
        
        self.sample_step = self.sample_step_v8
        if self.continuous_sample:
            self.sample_step = self.sample_step_cont



    def q_sample_coupled(self, x_0, t1, t2, non_special_sym_mask):
        if not self.not_diffusing_special_sym:
            non_special_sym_mask = torch.ones_like(non_special_sym_mask)
        _t1 = torch.maximum(t1, t2)
        _t2 = torch.minimum(t1, t2)
        log_x0 = index_to_log_onehot(x_0, self.vocab_size)
        
        log_x_t1_logits = self.q_xt_given_x0(log_x0, _t1, non_special_sym_mask)
        u1 = torch.rand_like(log_x_t1_logits)
        log_x_t1 = log_sample_categorical(log_x_t1_logits, u1, num_classes=self.vocab_size) # [b, n, c]
        
        log_x_t2_logits_if_eq = self.q_xt_given_x0(log_x0, _t2, non_special_sym_mask)
        xt_eq_x0_mask = log_x_t1.argmax(dim=-1) == x_0
        log_x_t2_logits_if_neq = self.q_tminusk_given_t_x0(
                    log_x_0=log_x0, 
                    log_x_t=log_x_t1, 
                    t=_t1,
                    xt_eq_x0_mask=xt_eq_x0_mask,
                    non_special_sym_mask=non_special_sym_mask,
                    k=_t1 - _t2,
                    )
        select_mask = (_t1 == _t2).float().unsqueeze(-1).unsqueeze(-1)
        log_x_t2_logits = log_x_t2_logits_if_neq * (1. - select_mask) + log_x_t2_logits_if_eq * select_mask

        # TODO: investigate the effect of such antithetic pairs.
        # TODO: check when t1 == t2
        u2 = torch.rand_like(log_x_t2_logits) # 1. - u1
        log_x_t2 = log_sample_categorical(log_x_t2_logits, u2, num_classes=self.vocab_size)
        # first sample q(x_{t1} | x_0)
        # and then sample q(x_{t2} | x_{t1}, x_0)
        # if t1 == t2, then draw an indep. sample.
        return (torch.cat([log_x_t1, log_x_t2], dim=0), torch.cat([_t1, _t2], dim=0))

    def q_sample(self, x_0, t, non_special_sym_mask):
        # here t ranges from 0 to T-1,
        # indexing time steps from 1 to T.
        if not self.not_diffusing_special_sym:
            non_special_sym_mask = torch.ones_like(non_special_sym_mask)
        log_x0 = index_to_log_onehot(x_0, self.vocab_size)
        log_q_xt_x0 = self.q_xt_given_x0(log_x0, t, non_special_sym_mask)
        log_sample = log_sample_categorical(log_q_xt_x0, num_classes=self.vocab_size) # [b, n, c]
        return log_sample
  
    def q_xt_given_x0(self, log_x0, t, non_special_sym_mask):
        if self.continuous == False:
            log_cumprod_alpha_t = extract(self.log_cumprod_alpha, t, log_x0.shape)
            log_1_min_cumprod_alpha = extract(self.log_1_min_cumprod_alpha, t, log_x0.shape)
        elif self.continuous == True:
            ######Check carefully here
            cumprod_alpha_t = get_continuous_alpha(t, self.num_timesteps, log_x0.shape)
            log_cumprod_alpha_t = torch.log(cumprod_alpha_t)
            log_1_min_cumprod_alpha = torch.log(1 - cumprod_alpha_t)

        log_probs = torch.logaddexp(
            log_x0 + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha + self.vocab_log_prob
        )
        # for non-symbol tokens, we mask their porbability getting into symbols;
        # for symbols, we force them to stay in the current state.
        # Might be removed?
        log_probs[..., self.pad_id] = -30
        log_probs[..., self.bos_id] = -30
        log_probs[..., self.eos_id] = -30
        # log_probs = log_x0 * (1. - non_special_sym_mask_float) + (log_probs + self.tok_2_sym_mask) * non_special_sym_mask_float
        log_probs = torch.where(non_special_sym_mask.unsqueeze(-1), log_probs, log_x0)
        return log_probs

    def _q_tminusk_given_t_x0(
        self, 
        log_x_0, 
        log_x_t, 
        t,
        xt_eq_x0_mask,
        non_special_sym_mask,
        k=1, 
    ):
        '''
            compute q(x_{t - k}, b_{t - k} | x_{t}, x_0)
        '''
        ####Check carefully here
        if self.continuous == False:
            log_cumprod_alpha_t = extract(self.log_cumprod_alpha, t, log_x_t.shape)
            log_cumprod_alpha_tminusk = extract(self.log_cumprod_alpha, (t - k).clamp(min=0), log_x_t.shape)
        elif self.continuous == True:
            cumprod_alpha_t = get_continuous_alpha(t, self.num_timesteps, log_x_t.shape)
            log_cumprod_alpha_t = torch.log(cumprod_alpha_t)
            #log_1_min_cumprod_alpha = torch.log(1 - cumprod_alpha_t)
            log_cumprod_alpha_tminusk = torch.log(get_continuous_alpha((t-k).clamp(min=0), self.num_timesteps, log_x_t.shape))

        mask = (t < k).float().unsqueeze(-1).unsqueeze(-1)
        log_cumprod_alpha_tminusk = mask * torch.zeros_like(log_cumprod_alpha_t) + (1. - mask) * log_cumprod_alpha_tminusk
        log_cumprod_alpha_from_tminusk_to_t = log_cumprod_alpha_t - log_cumprod_alpha_tminusk
        log_q_noise_xt = torch.logaddexp(
            log_x_t + log_cumprod_alpha_from_tminusk_to_t,
            self.vocab_log_prob + log1mexp(log_cumprod_alpha_from_tminusk_to_t)
        )
        uniform_noise = torch.rand_like(log_q_noise_xt)
        gumbel_noise = -torch.log(-torch.log(uniform_noise + 1e-10) + 1e-10)
        u = torch.argmax(gumbel_noise + log_q_noise_xt, dim=-1)

        logit_xt = log_sub_exp(log_cumprod_alpha_from_tminusk_to_t, log_cumprod_alpha_t)
        logit_x0 = log_sub_exp(log_cumprod_alpha_tminusk, log_cumprod_alpha_t)
        logit_uniform = log1mexp(log_cumprod_alpha_from_tminusk_to_t) + log1mexp(log_cumprod_alpha_tminusk)

        lambda_t = log_sub_exp(log_cumprod_alpha_tminusk, log_cumprod_alpha_t) - log1mexp(log_cumprod_alpha_t)

        logit_xt_eq_x0 = torch.logaddexp(
            - self.vocab_log_prob + log_cumprod_alpha_t,
            torch.logaddexp(
                logit_xt,
                logit_x0,
            ),
        )
        # if x_t == x_0
        unnormed_logprobs_eq = torch.logaddexp(
            log_x_0 + logit_xt_eq_x0,
            self.vocab_log_prob + logit_uniform,
        )
        # if x_t != x_0
        unnormed_logprobs_neq = torch.logaddexp(
            torch.logaddexp(
                log_x_t + logit_xt,
                log_x_0 + logit_x0,
            ),
            self.vocab_log_prob + logit_uniform,
        )
        # TODO: next step: remove the condition for p_theta; pass an additional argument called mask so that
        # 1) during training, the mask is made by ground truth
        # 2) during decoding, the mask is made by heuristics, such as CMLM style top-k mask, or can be learnt by a 
        # another prediction head.
        unnormed_logprobs = torch.where(
            xt_eq_x0_mask.unsqueeze(-1),
            unnormed_logprobs_eq, 
            unnormed_logprobs_neq
        )
        # for non-symbol tokens, we mask their porbability getting into symbols;
        # for symbols, we force them to stay in the current state.
        unnormed_logprobs = unnormed_logprobs.masked_fill(self.special_sym_indicator.bool(), -30)
        log_probs = torch.log_softmax(unnormed_logprobs, dim=-1)
        # log_probs = log_x0 * (1. - non_special_sym_mask_float) + (log_probs + self.tok_2_sym_mask) * non_special_sym_mask_float
        num_axes = (1,) * (len(log_x_0.size()) - 1)
        t_broadcast = t.view(-1, *num_axes) * torch.ones_like(log_x_0)
        if isinstance(k, torch.Tensor):
            step_size = k.view(-1, *num_axes) * torch.ones_like(log_x_0)
        elif isinstance(k, int):
            step_size = k
        else:
            step_size = int(k)
        log_probs = torch.where(
            (t_broadcast != step_size - 1) & non_special_sym_mask.unsqueeze(-1), 
            log_probs, 
            log_x_0
        )
        return log_probs

    def q_tminusk_given_t_x0(
        self, 
        log_x_0, 
        log_x_t, 
        t,
        xt_eq_x0_mask,
        non_special_sym_mask,
        k=1, 
    ):
        '''
            compute q(x_{t - k} | x_{t}, x_0)
        '''
        #####Check carefully here
        if self.continuous == False:
            log_cumprod_alpha_t = extract(self.log_cumprod_alpha, t, log_x_t.shape)
            log_cumprod_alpha_tminusk = extract(self.log_cumprod_alpha, (t - k).clamp(min=0), log_x_t.shape)
        elif self.continuous == True:
            cumprod_alpha_t = get_continuous_alpha(t, self.num_timesteps, log_x_t.shape)
            log_cumprod_alpha_t = torch.log(cumprod_alpha_t)
            #log_1_min_cumprod_alpha = torch.log(1 - cumprod_alpha_t + 1e-9)
            log_cumprod_alpha_tminusk = torch.log(get_continuous_alpha((t-k).clamp(min=0), self.num_timesteps, log_x_t.shape))
        #print('t', t, 'k', k)
        mask = (t < k).float().unsqueeze(-1).unsqueeze(-1)
        log_cumprod_alpha_tminusk = mask * torch.zeros_like(log_cumprod_alpha_t) + (1. - mask) * log_cumprod_alpha_tminusk
        log_cumprod_alpha_from_tminusk_to_t = log_cumprod_alpha_t - log_cumprod_alpha_tminusk

        logit_xt = log_sub_exp(log_cumprod_alpha_from_tminusk_to_t, log_cumprod_alpha_t)
        logit_x0 = log_sub_exp(log_cumprod_alpha_tminusk, log_cumprod_alpha_t)
        logit_uniform = log1mexp(log_cumprod_alpha_from_tminusk_to_t) + log1mexp(log_cumprod_alpha_tminusk)

        logit_xt_eq_x0 = torch.logaddexp(
            - self.vocab_log_prob + log_cumprod_alpha_t,
            torch.logaddexp(
                logit_xt,
                logit_x0,
            ),
        )
        # if x_t == x_0
        unnormed_logprobs_eq = torch.logaddexp(
            log_x_0 + logit_xt_eq_x0,
            self.vocab_log_prob + logit_uniform,
        )
        # if x_t != x_0
        unnormed_logprobs_neq = torch.logaddexp(
            torch.logaddexp(
                log_x_t + logit_xt,
                log_x_0 + logit_x0,
            ),
            self.vocab_log_prob + logit_uniform,
        )
        # TODO: next step: remove the condition for p_theta; pass an additional argument called mask so that
        # 1) during training, the mask is made by ground truth
        # 2) during decoding, the mask is made by heuristics, such as CMLM style top-k mask, or can be learnt by a 
        # another prediction head.
        unnormed_logprobs = torch.where(
            xt_eq_x0_mask.unsqueeze(-1),
            unnormed_logprobs_eq, 
            unnormed_logprobs_neq
        )
        # for non-symbol tokens, we mask their porbability getting into symbols;
        # for symbols, we force them to stay in the current state.
        unnormed_logprobs = unnormed_logprobs.masked_fill(self.special_sym_indicator.bool(), -30)
        log_probs = torch.log_softmax(unnormed_logprobs, dim=-1)
        # log_probs = log_x0 * (1. - non_special_sym_mask_float) + (log_probs + self.tok_2_sym_mask) * non_special_sym_mask_float
        num_axes = (1,) * (len(log_x_0.size()) - 1)
        t_broadcast = t.view(-1, *num_axes) * torch.ones_like(log_x_0)
        if isinstance(k, torch.Tensor):
            step_size = k.view(-1, *num_axes) * torch.ones_like(log_x_0)
        elif isinstance(k, int):
            step_size = k
        else:
            step_size = int(k)
        log_probs = torch.where(
            (t_broadcast != step_size - 1) & non_special_sym_mask.unsqueeze(-1), 
            log_probs, 
            log_x_0
        )
        return log_probs

    def compute_loss(self, inputs, **kwargs):
        label_smoothing = kwargs.get("label_smoothing", 0.0)
        log_x_t = inputs["log_x_t"] # [b, n, c]
        x_0 = inputs["x_0"]
        t = inputs["t"]
        weight_t = inputs["weight_t"]
        non_special_sym_mask = inputs["non_special_sym_mask"]
        decoder_outputs = inputs["decoder_outputs"]

        log_x_0 = index_to_log_onehot(x_0, self.vocab_size)
        xt_eq_x0_mask = log_x_t.argmax(dim=-1) == x_0
        logit_tilde_x_0 = decoder_outputs
        if not self.not_diffusing_special_sym:
            non_special_sym_mask = torch.ones_like(non_special_sym_mask)
        # for non-symbol tokens, we mask their porbability getting into symbols;
        # for symbols, we force them to stay in the current state.
        logit_tilde_x_0 = logit_tilde_x_0.masked_fill(self.special_sym_indicator.bool(), -30)
        log_x0_recon = F.log_softmax(logit_tilde_x_0, dim=-1) # [b, n, c]
        # retain special symbols in input.
        log_x0_recon = torch.where(non_special_sym_mask.unsqueeze(-1), log_x0_recon, log_x_0)
        # t passed here are indexed in [0, T-1],
        # corresponding to x_1, x_2, \dots, x_T.
        
        kl_loss = (
            F.cross_entropy(
                log_x0_recon.transpose(-1, -2), 
                x_0, 
                reduction='none'
            )
            .masked_fill(xt_eq_x0_mask, 0.0)
            .mean(1)
        )
        # since t here ranges from 0 to T-1
        ###TODO here to deal with "reciprocal"
        if self.reweighting_type == "reciprocal":
            reweighting_coeff = 1. / (t.float() + 1.)
        elif self.reweighting_type == "linear":
            reweighting_coeff = ((1 - (t / self.num_timesteps))) if torch.max(t) > 1 else (1 - t)
            #print('reweighting_coeff', reweighting_coeff)
        elif self.reweighting_type == "none":
            reweighting_coeff = 1.
        else:
            raise NotImplementedError("reweighting type {} not implemented.".format(self.reweighting_type))
        kl_loss = reweighting_coeff * kl_loss # [B]
        #print('weight_t', weight_t)
        diffusion_nll_loss = mean_ds(weight_t * kl_loss)
        if label_smoothing > 0:
            logit_loss = mean_ds(
                weight_t * 
                F.log_softmax(decoder_outputs, dim=-1).mean(dim=-1).masked_fill(xt_eq_x0_mask, 0.).mean(1)
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
            'loss': kl_loss,
            "t": t,
            "weights": weight_t,
        }
        return output_dict, logging_outputs

    # def sample_step(self, decoder_out, denoising_fn, **kwargs):
    #     # continuous = kwargs.get('continuous', self.continuous)
    #     continuous_sample = kwargs.get('continuous_sample', self.continuous_sample)
    #     if continuous_sample:
    #         return self.sample_step_cont(decoder_out, denoising_fn, **kwargs)
    #     return self.sample_step_v8(decoder_out, denoising_fn, **kwargs)

    def sample_step_cont(self, decoder_out, denoising_fn, schedule_mode = "linearlambda", **kwargs):
        topk_mode = self.topk_mode
        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        output_masks = decoder_out.auxiliary_output["output_masks"]
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
        log_x_t = index_to_log_onehot(output_tokens, self.vocab_size) # [b, n, c]
  
        log_x_t = index_to_log_onehot(output_tokens, self.vocab_size) # [b, n, c]
        log_x0_recon = denoising_fn(x_t=output_tokens, t=cur_step_tensor) # [b, n, c]
        log_x0_recon = torch.log_softmax(log_x0_recon.masked_fill(self.special_sym_indicator.bool(), -30), dim=-1)
        log_x0_recon = torch.where(non_special_sym_mask.unsqueeze(-1), log_x0_recon, log_x_t)
        if argmax_decoding or True:
          cur_scores, cur_tokens = log_x0_recon.max(-1)
        else:
          # 0.01 is better than argmax
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
          # output_tokens.masked_fill_(Noise_index_revised, self.mask_idx)
          # output_scores.masked_fill_(Noise_index_revised, -math.inf)
        elif topk_mode == "cond":
          if self.continuous_sample == False:
            cutoff_len = (
            ((~Tran_al)&non_special_sym_mask).sum(1, keepdim=True).type_as(output_scores)
            ).long()

          else:
            Tran_A = (mask.repeat(output_scores.size(0), 1) == 1)# & non_special_sym_mask
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
        #   output_tokens.masked_fill_(Noise_index_revised, self.mask_idx)
        #   output_scores.masked_fill_(Noise_index_revised, -math.inf)


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

    def sample_step_v8(self, decoder_out, denoising_fn, **kwargs):
        topk_mode = self.topk_mode
        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        fake_t = decoder_out.step
        max_step = decoder_out.max_step
        Transition_time = decoder_out.history
        unique_time = torch.unique(Transition_time)
        sorted, indices = torch.sort(unique_time, descending=True)
        t = sorted[fake_t]
        Tran_al = Transition_time >= t
        


        argmax_decoding = kwargs.get('argmax_decoding', False)

        # manually construct a non-special-sym mask.
        non_special_sym_mask = (
            output_tokens.ne(self.pad_id) & 
            output_tokens.ne(self.bos_id) & 
            output_tokens.ne(self.eos_id)
        )
        non_special_sym_mask = kwargs.get('non_special_sym_mask', non_special_sym_mask)


        cur_step = (t/max_step)*self.num_timesteps
        # cur_step_tensor = torch.full((output_tokens.shape[0],), t, device=output_tokens.device, dtype=torch.long)
        cur_step_tensor = torch.full((output_tokens.shape[0],), cur_step, device=output_tokens.device, dtype=torch.long)
        log_x_t = index_to_log_onehot(output_tokens, self.vocab_size) # [b, n, c]
  
        
        log_x0_recon = denoising_fn(x_t=output_tokens, t=cur_step_tensor) # [b, n, c]
        log_x0_recon = torch.log_softmax(log_x0_recon.masked_fill(self.special_sym_indicator.bool(), -30), dim=-1)
        log_x0_recon = torch.where(non_special_sym_mask.unsqueeze(-1), log_x0_recon, log_x_t)

        if argmax_decoding or True:
          cur_scores, cur_tokens = log_x0_recon.max(-1)
        else:
          # 0.01 is better than argmax
          temperature = 0.01
          cur_tokens = dists.Categorical(logits=log_x0_recon / temperature).sample()
          cur_scores = torch.gather(log_x0_recon, -1, cur_tokens.unsqueeze(-1)).squeeze(-1)

    

        if topk_mode == "real":
          pre_Noise_index = decoder_out.auxiliary_output["output_masks"]
          Noise_index_revised = (~Tran_al).repeat(output_scores.size(0), 1)&non_special_sym_mask
          Mask_to_x0 = pre_Noise_index&~Noise_index_revised
          #### Update all the transition tokens
          output_tokens.masked_scatter_(Mask_to_x0, cur_tokens[Mask_to_x0])
          output_scores.masked_scatter_(Mask_to_x0, cur_scores[Mask_to_x0])

        elif topk_mode == "cond":

          cutoff_len = (
          ((~Tran_al)&non_special_sym_mask).sum(1, keepdim=True).type_as(output_scores)
            ).long()
          #Since the special_sym_mask will never transition from noise to x0, set them to be 1000 so that never get changed 
          _scores_for_topk = cur_scores.masked_fill(~non_special_sym_mask, 1000.0)
          Noise_index_revised = topk_masking(_scores_for_topk, cutoff_len, stochastic=False)
          pre_Noise_index = decoder_out.auxiliary_output["output_masks"]
          # Mask_to_x0 = ~Noise_index_revised
          Mask_to_x0 = pre_Noise_index&~Noise_index_revised
          # Mask_to_x0 = non_special_sym_mask&~Noise_index_revised

          
          output_tokens.masked_scatter_(Mask_to_x0, cur_tokens[Mask_to_x0])
          output_scores.masked_scatter_(Mask_to_x0, cur_scores[Mask_to_x0])

       

        # return output_tokens, output_scores
        history = decoder_out.history
        
        auxiliary_output = decoder_out.auxiliary_output
        auxiliary_output["output_masks"] = Noise_index_revised

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            auxiliary_output=auxiliary_output,
            attn=None,
            history=history,
        )
            
#     #Original Code      
#     def sample_step(self, decoder_out, denoising_fn, **kwargs):
#         output_tokens = decoder_out.output_tokens
#         output_scores = decoder_out.output_scores
#         output_masks = decoder_out.auxiliary_output["output_masks"]
#         t = decoder_out.step
#         max_step = decoder_out.max_step

#         temperature_annealing = kwargs.get('temperature_annealing', False)
#         decoding_time_difference = kwargs.get('decoding_time_difference', 0.0)
#         argmax_decoding = kwargs.get('argmax_decoding', False)
#         decoding_strategy = kwargs.get('decoding_strategy', "cmlm-dep")
#         adaptive_decoding = kwargs.get('adaptive_decoding', False)
#         if temperature_annealing:
#             temperature = -0.05 * ((t / (max_step - 1)) if (torch.max(t) > 1) else t) + 0.5
#         else:
#             temperature = kwargs.get('temperature', 1.0)

#         # manually construct a non-special-sym mask.
#         non_special_sym_mask = (
#             output_tokens.ne(self.pad_id) & 
#             output_tokens.ne(self.bos_id) & 
#             output_tokens.ne(self.eos_id)
#         )
#         non_special_sym_mask = kwargs.get('non_special_sym_mask', non_special_sym_mask)
#         if adaptive_decoding:
#             raise NotImplementedError
#             # effective_seq_len = non_special_sym_mask.float().sum(-1)
#             # _max_step = max_step + effective_seq_len - effective_seq_len 
#             # _max_step = torch.minimum(effective_seq_len, _max_step).clamp(min=1)
#             # # tensor, tensor
#             # cur_step_tensor, cur_stepsize = self._get_batched_decoding_strategy(t, _max_step)
#             # cur_step_tensor = cur_step_tensor - 1
#             # #print("cur t : {}, step size: {}, t: {}, max_step: {}, diffusion steps : {}".format(cur_step_tensor, cur_stepsize, t, _max_step, self.num_timesteps), flush=True)
#         else:
#             # int, int
#             cur_step, cur_stepsize = self._get_decoding_strategy(t, decoding_strategy, max_step)
#             # minus 1 due to the offset.
#             cur_step = cur_step - 1
#             cur_step_tensor = torch.full((output_tokens.shape[0],), cur_step, device=output_tokens.device, dtype=torch.long)
#             #print("cur t : {}, step size: {}, t: {}, max_step: {}, diffusion steps : {}".format(cur_step_tensor, cur_stepsize, t, max_step, self.num_timesteps), flush=True)
#             if self.continuous == True:
#                 cur_step_tensor = torch.full((output_tokens.shape[0],), cur_step/self.num_timesteps * 1, device=output_tokens.device)
#                 cur_stepsize = cur_stepsize/self.num_timesteps * 1
#                 cur_step = cur_step/self.num_timesteps * 1
#             # if self.continuous == True:
#             #     cur_step_tensor = cur_step_tensor/self.num_timesteps
#             #     t = t/max_step
#             #     cur_stepsize = cur_stepsize/self.num_timesteps
            
#             # print("cur t : {}, step size: {}, t: {}, max_step: {}, diffusion steps : {}".format(cur_step_tensor, cur_stepsize, t, max_step, self.num_timesteps), flush=True)
#         log_x_t = index_to_log_onehot(output_tokens, self.vocab_size) # [b, n, c]
#         log_x0_recon = denoising_fn(x_t=output_tokens, t=cur_step_tensor) # [b, n, c]
#         log_x0_recon = torch.log_softmax(log_x0_recon.masked_fill(self.special_sym_indicator.bool(), -30), dim=-1)
#         log_x0_recon = torch.where(non_special_sym_mask.unsqueeze(-1), log_x0_recon, log_x_t)
        
#         if decoding_strategy.startswith("reparam"):
#             if argmax_decoding:
#                 cur_scores, cur_tokens = log_x0_recon.max(-1)
#             else:
#                 cur_tokens = dists.Categorical(logits=log_x0_recon / temperature).sample()
#                 cur_scores = torch.gather(log_x0_recon, -1, cur_tokens.unsqueeze(-1)).squeeze(-1)
#             # sample from q_noise(x_t)
#             if self.continuous == False:
#                 log_cumprod_alpha_t = extract(self.log_cumprod_alpha, cur_step_tensor, log_x_t.shape)
#                 log_cumprod_alpha_tminusk = extract(self.log_cumprod_alpha, (cur_step_tensor - cur_stepsize).clamp(min=0), log_x_t.shape)
#             elif self.continuous == True:
#                 cumprod_alpha_t = get_continuous_alpha(cur_step_tensor, self.num_timesteps, log_x_t.shape)
#                 log_cumprod_alpha_t = torch.log(cumprod_alpha_t)
#                 #log_1_min_cumprod_alpha = torch.log(1 - cumprod_alpha_t + 1e-9)
#                 log_cumprod_alpha_tminusk = torch.log(get_continuous_alpha((cur_step_tensor - cur_stepsize).clamp(min=0), self.num_timesteps, log_x_t.shape))

#             mask = (cur_step_tensor < cur_stepsize).float().unsqueeze(-1).unsqueeze(-1)
#             log_cumprod_alpha_tminusk = mask * torch.zeros_like(log_cumprod_alpha_t) + (1. - mask) * log_cumprod_alpha_tminusk
#             log_cumprod_alpha_from_tminusk_to_t = log_cumprod_alpha_t - log_cumprod_alpha_tminusk
#             log_q_noise_xt = torch.logaddexp(
#                 log_x_t + log_cumprod_alpha_from_tminusk_to_t,
#                 self.vocab_log_prob + log1mexp(log_cumprod_alpha_from_tminusk_to_t)
#             )
#             _uniform_noise = torch.rand_like(log_q_noise_xt)
#             gumbel_noise = -torch.log(-torch.log(_uniform_noise + 1e-10) + 1e-10)
#             uniform_noise = torch.argmax(gumbel_noise + log_q_noise_xt, dim=-1)
            
#             # this function modifies output_tokens and output_scores in place.
#             # see the function for more details.
#             output_masks = self._reparam_decoding(
#                 output_tokens=output_tokens,
#                 output_scores=output_scores,
#                 cur_tokens=cur_tokens,
#                 cur_scores=cur_scores,
#                 decoding_strategy=decoding_strategy,
#                 xt_neq_x0=decoder_out.auxiliary_output["output_masks"],
#                 non_special_sym_mask=non_special_sym_mask,
#                 t=t,
#                 max_step=max_step,
#                 noise=uniform_noise,
#             )
#         else:
#             # instead of larger step sizes, we offset the current time instead.
#             # we found this leads to better performance and less noisy translates.
#             new_cur_step_tensor = cur_step_tensor
#             # NOTE we only quse shifted time steps in computing the posterior.
#             xt_neq_x0 = decoder_out.auxiliary_output["output_masks"]
#             if decoding_time_difference > 0:
#                 raise NotImplementedError
#                 # if adaptive_decoding:
#                 #     raise NotImplementedError
#                 #     # new_cur_step_tensor = torch.maximum(cur_step_tensor - decoding_time_difference, (1.5 * cur_stepsize).long())
#                 #     # new_cur_step_tensor = torch.where(cur_step_tensor >= cur_stepsize, new_cur_step_tensor, cur_step_tensor)
#                 # else:
#                 #     decoding_time_difference = decoding_time_difference/max_step * 1
#                 #     print(cur_step, cur_stepsize, decoding_time_difference, cur_step_tensor)
#                 #     if cur_step >= cur_stepqsize:
#                 #         new_cur_step_tensor = (cur_step_tensor - decoding_time_difference).clamp(min=math.floor(1.5 * cur_stepsize))

#             if argmax_decoding:
#                 cur_scores, cur_tokens = log_x0_recon.max(-1)
#             else:
#                 cur_tokens = dists.Categorical(logits=log_x0_recon / temperature).sample()
#                 cur_scores = torch.gather(log_x0_recon, -1, cur_tokens.unsqueeze(-1)).squeeze(-1)

#             xt_neq_x0 = decoder_out.auxiliary_output["output_masks"]
#             if self.continuous == False:
#                 log_cumprod_alpha_t = extract(self.log_cumprod_alpha, new_cur_step_tensor, log_x_t.shape)
#                 log_cumprod_alpha_tminusk = extract(self.log_cumprod_alpha, (new_cur_step_tensor - cur_stepsize).clamp(min=0), log_x_t.shape)
#             elif self.continuous == True:
#                 cumprod_alpha_t = get_continuous_alpha(new_cur_step_tensor, self.num_timesteps, log_x_t.shape)
#                 log_cumprod_alpha_t = torch.log(cumprod_alpha_t)
#                 #log_1_min_cumprod_alpha = torch.log(1 - cumprod_alpha_t + 1e-9)
#                 log_cumprod_alpha_tminusk = torch.log(get_continuous_alpha((new_cur_step_tensor - cur_stepsize).clamp(min=0), self.num_timesteps, log_x_t.shape))
#             mask = (new_cur_step_tensor < cur_stepsize).float().unsqueeze(-1).unsqueeze(-1)
#             log_cumprod_alpha_tminusk = mask * torch.zeros_like(log_cumprod_alpha_t) + (1. - mask) * log_cumprod_alpha_tminusk
#             log_cumprod_alpha_from_tminusk_to_t = log_cumprod_alpha_t - log_cumprod_alpha_tminusk
#             lambda_2 = log_sub_exp(log_cumprod_alpha_tminusk, log_cumprod_alpha_t) - log1mexp(log_cumprod_alpha_t)

#             _temp_zeros = torch.zeros_like(cur_tokens)
#             not_u_t = _temp_zeros.bool()
#             # add correct-shaped tensors for broadcasting
#             not_v_t = ~sample_bernoulli(lambda_2.squeeze(-1) + _temp_zeros)

#             masked_to_noise = (~xt_neq_x0 & not_u_t) | (xt_neq_x0 & not_v_t)

#             # sample from q_noise(x_t)
#             log_q_noise_xt = torch.logaddexp(
#                 log_x_t + log_cumprod_alpha_from_tminusk_to_t,
#                 self.vocab_log_prob + log1mexp(log_cumprod_alpha_from_tminusk_to_t)
#             )
#             uniform_noise = torch.rand_like(log_q_noise_xt)
#             gumbel_noise = -torch.log(-torch.log(uniform_noise + 1e-10) + 1e-10)
#             u = torch.argmax(gumbel_noise + log_q_noise_xt, dim=-1)
#             output_tokens.masked_scatter_(masked_to_noise, u[masked_to_noise])
#             output_scores.masked_fill_(masked_to_noise, -math.inf)

#             masked_to_x0 = xt_neq_x0 & ~not_v_t
#             output_tokens.masked_scatter_(masked_to_x0, cur_tokens[masked_to_x0])
#             output_scores.masked_scatter_(masked_to_x0, cur_scores[masked_to_x0])

#             # 1_x = (1_x & u_t) | v_t
#             # save the NOT output of 1_(x_t = x_0) for the next iteration
#             # NOT_1_x = (NOT_1_x | NOT_u_t) & NOT_v_t
#             output_masks = (xt_neq_x0 | not_u_t) & not_v_t

#         # return output_tokens, output_scores
#         history = decoder_out.history
#         # if history is not None:
#         #     history.append(output_tokens.clone())
        
#         auxiliary_output = decoder_out.auxiliary_output
#         auxiliary_output["output_masks"] = output_masks

#         return decoder_out._replace(
#             output_tokens=output_tokens,
#             output_scores=output_scores,
#             auxiliary_output=auxiliary_output,
#             attn=None,
#             history=history,
#         )

# # if __name__ == "__main__":
# #     torch.set_printoptions(threshold=10_000)
# #     import logging
# #     logger = logging.getLogger(__name__)
# #     num_diffusion_timesteps = 10
# #     mask_id = 1000
# #     vocab_size = 32
# #     seq_len = 8
# #     pad_id = 0
# #     bos_id = 1
# #     eos_id = 2
# #     diffusion = MultinomialDiffusion(
# #         num_diffusion_timesteps, 
# #         vocab_size, 
# #         -1, 
# #         "simple", 
# #         "cosine", 
# #         True,
# #         "elbo",
# #         pad_id=0, bos_id=1, eos_id=2)
# #     batch_size = num_diffusion_timesteps
# #     length_tgt = torch.randint(low=4, high=seq_len, size=(1,)).repeat(batch_size)
# #     idx_length = torch.arange(seq_len)
# #     # TODO here
# #     x_0 = torch.randint(
# #         0,
# #         vocab_size, 
# #         size=(1, seq_len)).repeat(batch_size, 1)
# #     x_0.masked_fill_(
# #         idx_length[None, :] >= length_tgt[:, None], pad_id
# #     )
# #     x_0[:, 0] = bos_id
# #     x_0.scatter_(1, length_tgt[:, None] - 1, eos_id)

# #     non_special_sym_mask = (
# #         x_0.ne(pad_id) & 
# #         x_0.ne(bos_id) & 
# #         x_0.ne(eos_id)
# #     )
# #     print("length : {}, x_0 : {}, non_special_sym_mask : {}".format(length_tgt, x_0, non_special_sym_mask))
# #     # x_0 = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))
# #     t = torch.arange(start=0, end=num_diffusion_timesteps)
# #     weight_t = 0
# #     log_xt = diffusion.q_sample(x_0=x_0, t=t, non_special_sym_mask=non_special_sym_mask)
# #     print("x_t : {}".format(log_xt.exp().max(dim=-1)[1]))
    
# #     # batch_size = (seq_len + num_diffusion_timesteps - 1) ** 2
# #     # x_0 = torch.randint(low=3, high=vocab_size, size=(batch_size, seq_len))
# #     # _t1 = torch.arange(start=1, end=seq_len + num_diffusion_timesteps)
# #     # _t2 = torch.arange(start=1, end=seq_len + num_diffusion_timesteps)
# #     # abs_t = torch.cartesian_prod(_t1, _t2)
# #     # abs_t = (abs_t[:, 0], abs_t[:, 1])
# #     # weight_t = x_0
# #     # x_t, x_0_ignore, mask, abs_t, rel_t, weight_t = diffusion.q_sample_coupled(x_0=x_0,abs_t=abs_t, weight_t=weight_t)
# #     # print("x_t : {}, x_0_ignore : {}, mask : {}, abs_t : {}, rel_t : {}".format(x_t, x_0_ignore, mask, abs_t, rel_t))
# #     decoding_steps = num_diffusion_timesteps // 3
# #     output_tokens = x_0[0:1]
# #     def decoder(normalize, prev_output_tokens, encoder_out, t):
# #         bs, seq_len = prev_output_tokens.shape
# #         return torch.ones(bs, seq_len, vocab_size)
# #     for t in range(decoding_steps):
# #         print("##############################################################################")
# #         output_tokens, _ = diffusion.sample_step(output_tokens, t, None, decoding_steps, decoder)
# #         print("current decoded ====> [t] : {}, output: {}".format(t, output_tokens))
# #         print("##############################################################################")