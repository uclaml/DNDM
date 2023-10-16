# Pytorch Implementation of Discrete Non-Markov Diffusion Model (DNDM)

This repository contains the official implementation of paper Fast Sampling via De-randomization for Discrete Diffusion Models.

## Abstract

Diffusion models have emerged as powerful tools for high-quality data generation, such as image generation. Despite its success in continuous spaces, discrete diffusion models, which apply to domains such as texts and natural languages, remain under-studied and often suffer from slow generation speed. In this paper, we propose a novel de-randomized diffusion process, which leads to an accelerated algorithm for discrete diffusion models.  Our technique significantly reduces the number of function evaluations (i.e., calls to the score network), making the sampling process much faster. Furthermore, we introduce a continuous-time (i.e., infinite-step) sampling algorithm that can provide even better sample qualities than its discrete-time (finite-step) counterpart. Extensive experiments on natural language generation and machine translation tasks demonstrate the superior performance of our method in terms of both generation speed and sample quality over existing methods for discrete diffusion models.

## Dependencies

The codebase is implemented with [FairSeq](https://github.com/facebookresearch/fairseq). 
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
We implement discrete diffusion models in a self-contained library `discrete_diffusion` for general use. The library provides implementations of various typical discrete diffusion models, consisting of
- `(Vanilla/Reparameterized) multinomial diffusion`: diffusion processes that inject `uniform` noise to the token sequence. The implementation of vanilla multinomial diffusion closely follows the [codebase](https://github.com/ehoogeboom/multinomial_diffusion) of the original paper;
- `(Vanilla/Reparameterized) absorbing diffusion`: diffusion processes where tokens within the sequence could get absorbed to the `masking` state, as described in the [D3PM paper](https://arxiv.org/pdf/2107.03006.pdf).


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

## Machine Translation

### Datasets for Maching Translation Tasks
The three datasets can be easily processed before training and evaluation:

#### IWSLT14 DE-EN
We follow the standard pre-processing in [fairseq/examples](https://github.com/facebookresearch/fairseq/tree/main/examples/translation#iwslt14-german-to-english-transformer) to prepare the binarized data:
```bash
# fetch and preprocess the data to BPE codes
cd examples/translation/
bash prepare-iwslt14.sh
cd ../..

# binarize the data
TEXT=examples/translation/iwslt14.tokenized.de-en
fairseq-preprocess --joined-dictionary --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en \
    --workers 20
```

#### WMT14 EN-DE
We use the data released in [fairseq/examples](https://github.com/facebookresearch/fairseq/tree/main/examples/nonautoregressive_translation#dataset) to prepare the dataset:
```bash
wget http://dl.fbaipublicfiles.com/nat/original_dataset.zip
unzip original_dataset.zip
TEXT=wmt14_ende
fairseq-preprocess --joined-dictionary \
    --source-lang en --target-lang de \
    --trainpref $TEXT/train.en-de --validpref $TEXT/valid.en-de --testpref $TEXT/test.en-de \
    --destdir data-bin/wmt14_ende --thresholdtgt 0 --thresholdsrc 0 \
    --workers 20
```

#### WMT16 EN-RO
For this dataset, we use the raw data [wmt16.tar.gz](https://drive.google.com/file/d/1YrAwCEuktG-iDVxtEW-FE72uFTLc5QMl/view?usp=sharing) as pre-processed in [this repository](https://github.com/nyu-dl/dl4mt-nonauto/tree/multigpu).
```bash
gdown --fuzzy https://drive.google.com/file/d/1YrAwCEuktG-iDVxtEW-FE72uFTLc5QMl/view?usp=sharing
tar xzvf wmt16.tar.gz

TEXT=wmt16/en-ro

# move train/ dev/ test/ bpe codes into the $TEXT folder
mv $TEXT/train/corpus.bpe.en $TEXT/train.bpe.en
mv $TEXT/train/corpus.bpe.ro $TEXT/train.bpe.ro
mv $TEXT/dev/dev.bpe.en $TEXT/dev.bpe.en
mv $TEXT/dev/dev.bpe.ro $TEXT/dev.bpe.ro
mv $TEXT/test/test.bpe.en $TEXT/test.bpe.en
mv $TEXT/test/test.bpe.ro $TEXT/test.bpe.ro

# binarize the data
fairseq-preprocess --joined-dictionary \
    --source-lang en --target-lang ro \
    --trainpref $TEXT/train.bpe --validpref $TEXT/dev.bpe --testpref $TEXT/test.bpe \
    --destdir data-bin/wmt16_enro --thresholdtgt 0 --thresholdsrc 0 \
    --workers 20
```

### Training
We first get into the `fairseq` folder and then run the following commands to train the models.
```bash
######## training scripts for IWSLT'14 , WMT'14, and WMT'16 
# first cd to fairseq
# we use 1 GPU for IWSLT'14, 4 GPUs for WMT'14 and 2 GPUs for WMT'16 datasets respectively.
CUDA_VISIBLE_DEVICES=0 bash experiments/mt_train.sh -m absorbing -d <iwslt/wmt14/wmt16> -s default -e True --store-ema --label-smoothing 0.1
CUDA_VISIBLE_DEVICES=1 bash experiments/mt_train.sh -m multinomial -d <iwslt/wmt14/wmt16> -s default -e True --not-diffusing-special-sym --store-ema --label-smoothing 0.0
CUDA_VISIBLE_DEVICES=2 bash experiments/mt_train.sh -m reparam-absorbing -d <iwslt/wmt14/wmt16> -s default -e True --q-sample-mode coupled  --store-ema --label-smoothing 0.1 --reweighting-type linear
CUDA_VISIBLE_DEVICES=3 bash experiments/mt_train.sh -m reparam-multinomial -d <iwslt/wmt14/wmt16> -s default -e True --not-diffusing-special-sym --q-sample-mode coupled --store-ema --label-smoothing 0.1 --reweighting-type linear
```

> **Note**
> - `-s <str>` is used to specify the name of the experiment.
> - We could pass custom arguments that might be specific to training by appending them after `-e True`.

### Generation & Evaluation
The evaluation pipeline is handled by `experiments/mt_generate.sh`. The script will generate the translation results and evaluate the BLEU score.
```bash
########### IWLS'14, WMT'14, and WMT'16 datasets
# we recommend putting each checkpoint into a separate folder
# since the script will put the decoded results into a file under the same folder of each checkpoint.
CUDA_VISIBLE_DEVICES=0 bash experiments/mt_generate.sh -a false -c <checkpoint_path> -d <iwslt/wmt14/wmt16> 
```
Arguments:
- `-a`: whether to average multiple checkpoints
- `-c`: indicates the location of the checkpoint.
        If `-a false` (not to average checkpoints), pass the checkpoint **path**; 
        if `-a true`, pass the **directory** that stores multiple checkpoints at different training steps for averaging.
- `-d`: the dataset name

### Trained Model Checkpoints

We also provide the checkpoints of our trained models.

| Dataset | Model | Checkpoint link |
| --- | --- | :---: |
| IWSLT'14 | Multinomial | [link](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/linzheng_connect_hku_hk/EpAzao9L5XBMsef5LNZ1iXkB36Mp9V2gQGOwbopgPaOTVA?e=OraA81) |
| IWSLT'14 | Absorbing | [link](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/linzheng_connect_hku_hk/Eg1tqijPqkpNvc0Lai-BDE0Btc8L4UIJ-7oedCp4MXDPKw?e=liuASC) |
| IWSLT'14 | Reparam-multinomial | [link](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/linzheng_connect_hku_hk/EmCnWDgoj8JKmji1QE8UlkMB-3ow1aI8Bdo78-C7LqU_hA?e=DNahYn) |
| IWSLT'14 | Reparam-absorbing | [link](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/linzheng_connect_hku_hk/EmvmYZemCIRMsKQF-GNitzQB1lRUYj5MSow9jyxHZ4BCUg?e=nS81rB) |
| WMT'14 | Multinomial | [link](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/linzheng_connect_hku_hk/Ehgx0Ur0fbdJgY0zreg4KbABrN21txHM-sisbR9xZ6unDQ?e=T1vnJL) |
| WMT'14 | Absorbing | [link](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/linzheng_connect_hku_hk/EtO0Hft6GmhKogahr4V1hnQB4Odt5MUcjSUXawg_lH_0wg?e=Ikzs3R) |
| WMT'14 | Reparam-multinomial | [link](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/linzheng_connect_hku_hk/EtfgIjc9g2tEh3F9IpcvFoUBmIkcihy_tpVezr845fEDtQ?e=uTYJYF) |
| WMT'14 | Reparam-absorbing | [link](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/linzheng_connect_hku_hk/EniOmBTtL2dDtk1GNBw-kg4BsJ3SWTGmGASNdjRjSCP27w?e=Ona4qx) |
| WMT'16 | Multinomial | [link](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/linzheng_connect_hku_hk/EiBNFip8De5Nk-kimmyQ3UYBftUH3Cz74RsiA9IfoIryBQ?e=tzswtp) |
| WMT'16 | Absorbing | [link](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/linzheng_connect_hku_hk/EiFkp1Ros4VCsl4w-Feez7oB_h2zLEV61dHwsaFGxk7ioQ?e=96xT6h) |
| WMT'16 | Reparam-multinomial | [link](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/linzheng_connect_hku_hk/Em4byDij7zJIl1SY6nIcVeABbAEQZvsb1O8LdlS4i6t92A?e=0QQZaA) |
| WMT'16 | Reparam-absorbing | [link](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/linzheng_connect_hku_hk/Ep5D3LYr7FJLiWOrPbm3T3YBWtloPcdlNOmh5k9nM6CuzA?e=7pC43S) |


## Citation
```bibtex
@article{zheng2023rdm,
  title={A Reparameterized Discrete Diffusion Model for Text Generation},
  author={Zheng, Lin and Yuan, Jianbo and Yu, Lei and Kong, Lingpeng},
  journal={arXiv preprint arXiv:2302.05737},
  year={2023}
}
```
