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



## Basic Usage of the Discrete-diffusion
The code is built upon [https://github.com/HKUNLP/reparam-discrete-diffusion](https://github.com/HKUNLP/reparam-discrete-diffusion). More information about the code, such as data preprocessing can be found in the above repo.

#### Vanilla Sampling Scheme
By passing `--decoding-strategy default`, the vanilla sampling scheme (specific to each discrete diffusion process) is used.

#### Improved Sampling with Reparameterization
A more advanced decoding approach can be invoked by passing `--decoding-strategy reparam-<conditioning-of-v>-<topk_mode>-<schedule>`. This approach is based on the proposed reparameterization in our paper and allows for more effective decoding procedures. The options specify the decoding algorithm via
- `<conditioning-of-v>`: `uncond` or `cond` (default `uncond`): whether to generate the routing variable $v_t$ in a conditional or unconditional manner;
- `<topk_mode>`: `stochastic<float>` or `deterministic` (default `deterministic`): whether to use stochastic or deterministic top-$k$ selection. The float value in `stochastic<float>` specifies the degree of randomness in the stochastic top-$k$ selection;
- `<schedule>`: `linear` or `cosine` (default `cosine`): the schedule for $k$ during our denoising procedure, which is used to control the number of top-$k$ tokens to be denoised for the next decoding step.

See the [implementation](./discrete_diffusion/discrete_diffusions/discrete_diffusion_base.py#L130) for more details about the options.




## Machine Translation
The three datasets, including IWLS'14, WMT'14, and WMT'16 datasets, can be used for generation and training. Remember to process the data first.

### Generating
We first get into the `fairseq` folder and then run the following commands to train the models. Basic usages:
```bash
CUDA_VISIBLE_DEVICES=0 bash experiments/mt_generate.sh -a false -c <checkpoint_path> -d <iwslt/wmt14/wmt16> -e True
```

Arguments:
- `-a`: whether to average the last 5 saved checkpoints after training (Default is false especially if the checkpoint is loaded)
- `-c`: the path to saved model checkpoint file (if `-a True`, the directory of the saved models) 
- `-i`: indicating the number of diffusion steps in the samping process (default 1000).
- `-e`: indicating the end of the script-level arguments.
The following custom arguments can be passed after `-e True` for both sampling and training:
- `--continuous`: to enable continuous timesteps (without this argument we are using the discrete accelerated reverse sampling model).
- `--continuous-sample`: to enable continuous timesteps for sampling, including validation (will not work if the model is trained without `--continuous`).
- `--alpha`: for discrete sampling, indicating the alpha value for the Beta distribution schedule; for continuous sampling, indicating the alpha value of the Beta distribution from which the transition timestamps are sampled (default 3);
- `--beta`: for discrete sampling, indicating the beta value for the Beta distribution schedule; for continuous sampling, indicating the beta value of the Beta distribution from which the transition timestamps are sampled (default 3);
- `--schedule`: indicating the schedule for timesteps in discrete accelerated reverse sampling.
        The best schedule we have explored is 'Beta', so it is also the default.
        The other supported schedules: 'linear_lambda', 'linear_alpha', and 'cosine'.
- `--not-topk`: indicating whether to disable the top-k transition time selection (default False).

For example, when trying to acquire the sampling results of **DNDM-Multi** (without top-k) with continuous timesteps on the IWSLT14 dataset (with timestamps sampled from Beta(17,4) as reported):
```bash
CUDA_VISIBLE_DEVICES=0 bash experiments/mt_generate.sh -a false -c <checkpoint_path> -d wmt -e True --continuous --continuous-sample --alpha 17 --beta 4 --not-topk
```

When trying to acquire the sampling results of **DNDM-k-Absorb** at 1000 steps on the WMT16 dataset(with schedule Beta(15,7) as reported):
```bash
CUDA_VISIBLE_DEVICES=0 bash experiments/mt_generate.sh -a false -c <checkpoint_path> -d wmt -i 1000 -e True --alpha 15 --beta 7 --schedule Beta
```

For the sampling processes of discrete **DNDM** with or without top-k transition time selection, the results can be replicated using the trained model checkpoints provided by [https://github.com/HKUNLP/reparam-discrete-diffusion](https://github.com/HKUNLP/reparam-discrete-diffusion) (the links to the Reparam-multinomial and Reparam-absorbing models within the README).

If you want to try the continuous **DNDM-C** (with or without top-k) sampling, it is necessary to train a new model with "--continuous" and "--continuous-sample". For training either the continuous **DNDM-C** or the discrete **DNDM** from scratch, please refer to the following subsection.

### Training
We first get into the `fairseq` folder and then run the following commands to train the models. Basic usages:
```bash
######## training scripts for IWSLT'14 , WMT'14, and WMT'16 
CUDA_VISIBLE_DEVICES=2 bash experiments/mt_train.sh -m reparam-absorbing -d <iwslt/wmt14/wmt16> -s default -e True --q-sample-mode coupled  --store-ema --label-smoothing 0.1 --reweighting-type linear
CUDA_VISIBLE_DEVICES=3 bash experiments/mt_train.sh -m reparam-multinomial -d <iwslt/wmt14/wmt16> -s default -e True --not-diffusing-special-sym --q-sample-mode coupled --store-ema --label-smoothing 0.1 --reweighting-type linear
```
The additional custom arguments available to be passed after `-e True` is the same as the Sampling section above.






