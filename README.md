# BabyLlama-2

[![arXiv](https://img.shields.io/badge/arXiv-2409.17312-b31b1b.svg)](https://arxiv.org/abs/2409.17312)

Training code for BabyLlama 2, our submission to the "strict-small" track of the 2024 edition of the [BabyLM Challenge](https://babylm.github.io/) and the highest-performing decoder model within this track.

## How to run

### Install dependencies

You can use pyenv-virtualenv (shown below), Conda, and probably other environment managers too.
```sh
$ pyenv virtualenv 3.11.9 babyllama
$ pyenv activate babyllama
$ pip install -r requirements.txt
$ pip install -e .
```
Make sure that the correct environment is also activated for notebooks.

> [!NOTE]
> If you want to enable logging using W&B, enter `$ wandb login`. Otherwise, set `wandb: False` under `logging` in the YAML model config files.

### Data preparation

- Download the files `train_10M.zip` and `dev.zip` from the [BabyLM OSF directory](https://osf.io/ad7qg/) (they are located in the `text_data` subfolder).<details><summary>Note about 100M dataset</summary>The training hyperparameters have been optimized for the 10M training dataset. If you want to train on the 100M dataset (`train_100M.zip`) instead, you will likely need to adjust them (in particular, the number of epochs, maximum learning rate and weight decay).</details>
- Unzip the files. (We recommend placing the resulting `train_10M` and `dev` directories within a `data` subdirectory located at the root of the present repository, since default paths will point there.)
- Run the notebook `cleaning_and_tokenization.ipynb` to clean up and tokenize the dataset, adjusting the paths if needed.
- Split the `train_10M` dataset into training and test splits, by running the notebook `split_dataset.ipynb` (adjusting paths if needed.)
- Finally, check that the `{train,eval,test}_path` in the model config files point to the correct location.

#### (Optional) Installing the evaluation pipeline

> [!NOTE]
> If you don’t install the evaluation pipeline, add the `--skip_eval` flag to the training scripts.

We recommend cloning the evaluation pipeline to the home folder. If cloning to a different location, you will need to pass `--eval_dir=PATH/TO/evaluation-pipeline-2024` to the training scripts.
```
$ git clone https://github.com/babylm/evaluation-pipeline-2024
$ cd evaluation-pipeline-2024
$ pip install -e .
```

Then, download the `evaluation_data` folder from [the OSF directory](https://osf.io/ad7qg/), and place it in the root directory of the `evaluation-pipeline-2024` repository.

See the [upstream repository](https://github.com/babylm/evaluation-pipeline-2024) for further instructions.

### Training the teacher models

Train each teacher model by running the following command from the `scripts` subdirectory:
```sh
$ python train_and_eval.py --config ../config/llama-Smol-distillation.yaml
```
By default, this will automatically run the BLiMP evals after training. To avoid this behavior, pass the `--skip_eval` option. If the `evaluation-pipeline-2024` repository is not cloned to the home folder, specify its path using the `--eval_dir` argument.

The architecture, training and distillation hyperparameters are read from the config file, where you can edit them as needed.

> [!TIP]
> If using Slurm, you can pass the job ID by appending `--job_id $SLURM_JOB_ID`, otherwise a random UUID will be used to uniquely identify each training run.

### Training the student model

After training one or more teacher models, a student model can be trained by running the following command from the `scripts` subdirectory:
```
python distill_from_multiple_teachers.py \
    --config ../config/llama-Smol-distillation.yaml \
    --teachers ../models/<OUTPUT_DIR/TEACHER_1>,../models/<OUTPUT_DIR/TEACHER_2>
```
where `--teachers` should be a comma-separated list of paths to the directories containing the checkpoints of the teacher models (as `.safetensors` files). At least one teacher model should be specified, but there is no hard upper limit. You can pass additional options like `--eval_dir`, `--skip_eval` or `--job_id` if needed. If training succeeds, this script will by default run the BLiMP evals on the student model.

## Additional scripts

### Hyperparameter sweep

To perform a hyperparameter sweep for the non-distilled model using W&B, we provide a modified training script `sweep.py` and matching configuration files for the model and the sweep under `config/SmolLlama-345M-sweep/`. Create a sweep using `$ wandb sweep ../config/SmolLlama-345M-sweep/sweep.yaml` from the `scripts` subdirectory, then start as many agents as needed.

### Manually computing the loss

By default, the validation and test losses are already logged at the end of training. If you want to manually compute the loss on a different dataset or for another model, you can use the script `compute_test_loss.py`.
