# Pytorch Implementation of Discrete Non-Markov Diffusion Model (DNDM)

This repository contains the official implementation of paper Fast Sampling via De-randomization for Discrete Diffusion Models.

## Abstract

Diffusion models have emerged as powerful tools for high-quality data generation, such as image generation. Despite its success in continuous spaces, discrete diffusion models, which apply to domains such as texts and natural languages, remain under-studied and often suffer from slow generation speed. In this paper, we propose a novel de-randomized diffusion process, which leads to an accelerated algorithm for discrete diffusion models.  Our technique significantly reduces the number of function evaluations (i.e., calls to the score network), making the sampling process much faster. Furthermore, we introduce a continuous-time (i.e., infinite-step) sampling algorithm that can provide even better sample qualities than its discrete-time (finite-step) counterpart. Extensive experiments on natural language generation and machine translation tasks demonstrate the superior performance of our method in terms of both generation speed and sample quality over existing methods for discrete diffusion models.

## Dependencies

This project uses an older version of [FairSeq](https://github.com/facebookresearch/fairseq). 
This repo is confirmed to work with Python 3.8.10.

For installing the necessary packages for our code, please run the following commands in this order:

```bash
pip install -r requirements1.txt

# Following https://github.com/HKUNLP/reparam-discrete-diffusion/
pip install -e discrete_diffusion
cd fairseq
python3 setup.py build develop  # install the freezed version of fairseq
cd ..

pip install -r requirements2.txt  # Overwriting some package versions
pip install omegaconf==2.1.1  # This package has to be installed separately after hydra-core
```



## Basic Usage of the Discrete-diffusion Library

The code is built based on https://github.com/dtsip/in-context-learning. 
For details about the the Discrete-diffusion Library, information can be found in the above repo.


<details>
  <summary> click to check the implementation details as well as their arguments 👇 </summary>

These diffusion models share the same set of interfaces allowing for external uses. In particular, they are defined as subclasses of `DiscreteDiffusion` class, taking the following form:
```python
class DiscreteDiffusion(nn.Module):
    """
    The parent class for discrete denoising diffusion probabilistic models.

    It supports the following methods:
    - q_sample()
        Sample x_t ~ q(x_t | x_0) to construct noisy Transformer inputs.
    - compute_losses()
        Compute the loss L_t = KL(q||p) at t-th time step.
    - sample_step()
        Sample x_t ~ p(x_{t-1} | x_t, x_0) at t-th time step.
    """
    
    def __init__(self, num_timesteps):
        super().__init__()
        self.num_timesteps = num_timesteps

    def q_sample(self, x_0, t, **kwargs):
        """

        Sample from q(x_t | x_0), which is used as the model inputs.

        Args:
            x_0: token ids with shape [B, N]
            t: current time step, tensor with shape [B]

        Returns:
            return a dict of relevant outputs including x_t.
            
        """

    def compute_losses(self, inputs, **kwargs):
        """
        
        Compute the loss objective KL(q||p) to train our generative process.

        Args:
            inputs: a dict that contains input types specific to different diffusion processes, containing
                - x_t: token ids with shape [B, N]
                - t: scalar timesteps, with shape [B]

        Returns:
            possibly return a dict of relevant outputs, including the loss used for training.
            
        """

    def sample_step(self, decoder_out, denoising_fn, **kwargs):
        """
        Given a time step t, start from x_t and sample x_{t-k} from q(x_{t-k} | x_t).
        
        Args:
            decoder_out: a namedtuple that contains decoding info, including
                - x_t: token ids with shape [B, N]
                - t: scalar timesteps
                - max_steps: the maximum number of decoding steps
                - ...
            
            denoising_fn: a function that takes in x_t and t and returns model logits

            kwargs: other arguments that are used to control decoding.
        
        Returns:
            return a new decoder_out namedtuple.
        """
```

A `DiscreteDiffusion` model can be instantiated by configuring the following:
- Basic attributes, including
    - `--num-diffusion-timesteps <int>` specifies the whole number of diffusion time steps (default: 50)
    - `--diffusion-type <str>` specifies the diffusion model type (choices: `{absorbing, multinomial, reparam-absorbing, reparam-multinomial}`)
    - `--noise-scheduler-type <str>` specifies the noise schedule only in **vanilla/reparam multinomial diffusion** (typical choices: `{linear, cosine}`; default: `cosine`)
- Important arguments specific to the forward sampling routine in `q_sample()`, including
    - `--q-sample-mode <str>` specifies the sampling strategy (choices: `{default, coupled, multi-step, multi-sample}`; default: `default`). We provide various choices for sampling from $q(x_t|x_0)$ to prepare corrupted token sequences for denoising, including
        - `default`: a single sample is drawn as $x_t \sim q(x_t|x_0)$, identical to previous practices;
        - `multi-step`: sample two i.i.d. time steps $s, t$ and draw $x_s \sim q(x_s|x_0)$ and $x_t \sim q(x_t|x_0)$, respectively. We then optimize the average $\frac{1}{2}(\mathcal{L}_s + \mathcal{L}_t)$ for variance reduction;
        - `multi-sample`: sample two i.i.d. samples $x_t \sim q(x_t|x_0)$ and $x_t^{'} \sim q(x_t|x_0)$ at the same step, and compute the loss averaged over these two samples;
        - `coupled`: also known as conditioned training, which is detailed in Appendix F of the paper. This starts with sampling two i.i.d. time steps $s, t$ (assume $s < t$). We draw $x_t \sim q(x_t|x_0)$ as usual, but draw $x_s$ from a distribution conditioned on $x_t$ as $x_s \sim q(x_s|x_t, x_0)$. We then compute the average $\frac{1}{2}(\mathcal{L}_s + \mathcal{L}_t)$ as the objective. This strategy can simulate the backward transition process and help stabilize training. During preliminary experiments, we found the `coupled` sampling mode brings significant improvements for both vanilla multinomial/absorbing diffusion, but the gain is not consistently substantial in reparameterized variants. 
    - `--not-diffusing-special-sym` indicates whether to include special symbols during the diffusion process (default: False)
- Important arguments specific to the loss objective calculation in `compute_losses()`, including
    - `--reweighting-type <str>` specifies the reweighting scheme in our **reparameterized family** (choices: `{linear, reciprocal, none}`; default: `linear`)
    - `--label-smoothing <float>` specifies the rate of label smoothing (default: 0.1)
- Important arguments specific to the decoding routine in `sample_step()`, including
    - `--argmax-decoding` indicates whether to use argmax decoding for the denoised Transformer output $\tilde{x}_0$ (default: False)
    - `--temperature <float>` specifies the temperature $\tau$ for sampling $\tilde{x}_0 \sim \operatorname{Categorical}(f(x_t;\theta)/\tau)$ if the argmax decoding scheme is **not** used. (default: 1.0)
    - `--decoding-strategy <str>` specifies the use of vanilla (`default`) / reparameterized (`reparam-<options>`; see [the details](#decoding-strategies))decoding strategy (choices: `{default, reparam-<options>}`; default: `default`)
    - `--load-ema-weights` indicates whether to load the EMA model weights for generation (default: False)
    - `--iter-decode-max-iter <int>` specifies the maximum number of timesteps for decoding (default: 10)
    - `--iter-decode-with-beam <int>` specifies the beam size for decoding multiple sequences with different lengths in parallel (default: 1)
    - `--iter-decode-force-max-iter` indicates the iterative decoding must run the specified number of iterations and do not exit. Recommended to set this flag to True.

See [here](/fairseq/diffusion_mt/tasks/diffusion_translation_task.py#L23) for a more comprehensive list of arguments.
</details>

### Decoding Strategies

#### Vanilla Sampling Scheme
By passing `--decoding-strategy default`, the vanilla sampling scheme (specific to each discrete diffusion process) is used.

#### Improved Sampling with Reparameterization
A more advanced decoding approach can be invoked by passing `--decoding-strategy reparam-<conditioning-of-v>-<topk_mode>-<schedule>`. This approach is based on the proposed reparameterization in our paper and allows for more effective decoding procedures. The options specify the decoding algorithm via
- `<conditioning-of-v>`: `uncond` or `cond` (default `uncond`): whether to generate the routing variable $v_t$ in a conditional or unconditional manner;
- `<topk_mode>`: `stochastic<float>` or `deterministic` (default `deterministic`): whether to use stochastic or deterministic top-$k$ selection. The float value in `stochastic<float>` specifies the degree of randomness in the stochastic top-$k$ selection;
- `<schedule>`: `linear` or `cosine` (default `cosine`): the schedule for $k$ during our denoising procedure, which is used to control the number of top-$k$ tokens to be denoised for the next decoding step.

See the [implementation](./discrete_diffusion/discrete_diffusions/discrete_diffusion_base.py#L130) for more details about the options.

## Miscellanous
The code is built upon [https://github.com/HKUNLP/reparam-discrete-diffusion](https://github.com/HKUNLP/reparam-discrete-diffusion). More information about the code, such as data preprocessing
can be found in the above repo.


## Machine Translation
The three datasets, including IWLS'14, WMT'14, and WMT'16 datasets, can be used for generation and training. Remember to process the data first.

### Generating
We first get into the `fairseq` folder and then run the following commands to train the models. Basic usages:
```bash
CUDA_VISIBLE_DEVICES=0 bash experiments/mt_generate.sh -a false -c <checkpoint_path> -d <iwslt/wmt14/wmt16> -e True
```

Arguments:
- `-a`: whether to average the last 5 saved checkpoints after training
- `-i`: indicates the number of diffusion steps in the samping process (default 1000).
- `-e`: indicates the end of the script-level arguments.
The following custom arguments can be passed after `-e True` for both training and testing:
- `--continuous`: to enable continuous timesteps (without this argument we are using the discrete accelerated reverse sampling model).
- `--continuous-sample`: to enable continuous timesteps for sampling, including validation (will not work if the model is trained without `--continuous`).
- `--alpha`: indicates the alpha value for the Beta distribution used for discrete sampling (default 3).
- `--beta`: indicates the alpha value for the Beta distribution used for discrete sampling (default 3).
- `--schedule`: indicates the schedule for timesteps in discrete accelerated reverse sampling.
        The best schedule we have explored is 'Beta', so it is also the default.
        The other supported schedules: 'linear_lambda', 'linear_alpha', and 'cosine'.
- `--not-topk`: indicates whether to disable the top-k transition time selection (default False).

When trying to acquire the sampling results of **DNDM-Multi** with continuous timesteps on the IWSLT14 dataset:
```bash
CUDA_VISIBLE_DEVICES=0 bash experiments/mt_generate.sh -a false -c <checkpoint_path> -d wmt -e True --continuous --continuous-sample --not-topk
```
For example, when trying to acquire the sampling results of **DNDM-k-Absorb** at 1000 steps on the WMT16 dataset(with schedule Beta(15, 7) as reported in the appendix):
```bash
CUDA_VISIBLE_DEVICES=0 bash experiments/mt_generate.sh -a false -c <checkpoint_path> -d wmt -i 1000 -e True --alpha 15 --beta 7 --schedule Beta
```



The dataset information is stored in the saved checkpoints of the trained models, so it is only necessary to specify the dataset during training. 

### Training
We first get into the `fairseq` folder and then run the following commands to train the models. Basic usages:
```bash
######## training scripts for IWSLT'14 , WMT'14, and WMT'16 
CUDA_VISIBLE_DEVICES=2 bash experiments/mt_train.sh -m reparam-absorbing -d <iwslt/wmt14/wmt16> -s default -e True --q-sample-mode coupled  --store-ema --label-smoothing 0.1 --reweighting-type linear
CUDA_VISIBLE_DEVICES=3 bash experiments/mt_train.sh -m reparam-multinomial -d <iwslt/wmt14/wmt16> -s default -e True --not-diffusing-special-sym --q-sample-mode coupled --store-ema --label-smoothing 0.1 --reweighting-type linear
```
> **Note**
> - `-s <str>` is used to specify the name of the experiment.






