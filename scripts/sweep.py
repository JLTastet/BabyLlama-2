print('starting imports')
import torch
torch.backends.cuda.matmul.allow_tf32 = True
from transformers import (
    LlamaConfig, LlamaForCausalLM,
)
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import GPT2TokenizerFast
from pathlib import Path
import yaml
import json
import os
import argparse
import math
from datetime import datetime
from uuid import uuid4
import wandb
wandb.require("core")

from babyllama2.babylm_dataset import BabylmDataset
from babyllama2.scoring import skim_results, flatten_dictionary

print('imports are done')

starting_time = datetime.now()

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=None, help="Configuration file path")
parser.add_argument("--job_id", type=str, default=None, help="(Optional) Job ID")
parser.add_argument("--eval_dir", type=str, default=str(Path.home() / "evaluation-pipeline-2024"), help="Path to the evaluation pipeline directory")
parser.add_argument("--skip_eval", action='store_true', help="Do not run the evals")
args = parser.parse_args()

if args.config is None:
    raise(ValueError('Missing configuration file. Pass it using the --config option.'))
else:
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
if 'training' not in config:
    # Add an empty dictionary to avoid a KeyError later
    config['training'] = dict()

MAX_PER_DEVICE_BATCH_SIZE = config['training'].get('max_per_device_batch_size', 128)

run = wandb.init(name=config['model']['name'], config=config)
del config # Make sure we only access the config through wandb.config from now on

print(f"Num epochs: {wandb.config['num_epochs']}")

SEQ_LENGTH = wandb.config['data']['seq_length']

tokenizer_path = wandb.config['data']['tokenizer_path']
tokenizer = GPT2TokenizerFast(tokenizer_file= str(tokenizer_path))
tokenizer.bos_token = "<s>"
tokenizer.eos_token = "</s>"
tokenizer.pad_token = "<pad>"

train_dataset = BabylmDataset(wandb.config['data']['train_path'], SEQ_LENGTH, tokenizer=tokenizer, random_chunk=True)
eval_dataset = BabylmDataset(wandb.config['data']['eval_path'], SEQ_LENGTH, tokenizer=tokenizer, offset=0)
test_dataset = BabylmDataset(wandb.config['data']['test_path'], SEQ_LENGTH, tokenizer=tokenizer, offset=0)

eval_dir = Path(args.eval_dir)
if not args.skip_eval:
    assert eval_dir.exists()

# We tokenize the whole dataset and then set the max length
tokenizer.model_max_length = SEQ_LENGTH

model_config = LlamaConfig(
    vocab_size=tokenizer.vocab_size,
    max_position_embeddings=2*tokenizer.model_max_length,
    hidden_size=wandb.config['model']['hidden_size'],
    intermediate_size=wandb.config['model']['intermediate_size'],
    num_hidden_layers=wandb.config['model']['n_layer'],
    num_attention_heads=wandb.config['model']['n_head'],
    num_key_value_heads=wandb.config['model'].get('n_KV', wandb.config['model']['n_head']),
    tie_word_embeddings=wandb.config['model'].get('tie_word_embeddings', False),
    pad_token_id=tokenizer.convert_tokens_to_ids("<pad>"),
    attention_dropout=wandb.config['attention_dropout'],
)


model = LlamaForCausalLM(model_config)

PATH_SUBSTITUTIONS = {
    '{NAME}': wandb.config['model']['name'],
    '{ID}': run.id,
}
if 'output_dir' in wandb.config['logging'] and 'output_path' not in wandb.config['logging']:
    # Use a subdirectory named <model name>-<unique ID>
    output_dir = Path(wandb.config['logging']['output_dir']) / ( wandb.config['model']['name'] + '-' + run.id )
elif 'output_path' in wandb.config['logging'] and 'output_dir' not in wandb.config['logging']:
    # Expand {NAME} and {ID} in the path
    output_path_str = wandb.config['logging']['output_path']
    for (variable, substitution) in PATH_SUBSTITUTIONS.items():
        output_path_str = output_path_str.replace(variable, substitution)
    output_dir = Path(output_path_str)
else:
    raise(ValueError('Please specify exactly one of "output_dir" or "output_path". For the latter, you can use the variables {NAME} and {ID}.'))

print(f"Model has {model.num_parameters()} parameters")
print(f'Saving model to {output_dir}')

# Use gradient accumulation is the batch size is too large
batch_size = wandb.config['batch_size']
if batch_size > MAX_PER_DEVICE_BATCH_SIZE:
    accumulation_steps = math.ceil(batch_size / MAX_PER_DEVICE_BATCH_SIZE)
    per_device_bsz = batch_size // accumulation_steps
    batch_size = per_device_bsz * accumulation_steps
else:
    accumulation_steps = 1
    per_device_bsz = batch_size
wandb.config.update({'actual_batch_size': batch_size})
del batch_size # Make sure we don’t use the logical batch size by mistake

# Compute the actual Adam beta parameters
wandb.config.update({
    'adam_beta1': 1 - wandb.config['one_minus_adam_beta1'],
    'adam_beta2': 1 - wandb.config['one_minus_adam_beta2'],
})

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    save_strategy = "epoch",
    eval_strategy = "epoch",
    num_train_epochs=wandb.config['num_epochs'],
    gradient_accumulation_steps=accumulation_steps,
    per_device_train_batch_size=per_device_bsz,
    per_device_eval_batch_size=per_device_bsz,
    save_total_limit=1,  # Set to zero to avoid saving
    warmup_steps=wandb.config['num_warmup_steps'], 
    lr_scheduler_type=wandb.config['lr_scheduler_type'],
    learning_rate=wandb.config['lr'],
    adam_beta1=wandb.config['adam_beta1'],
    adam_beta2=wandb.config['adam_beta2'],
    adam_epsilon=wandb.config['adam_epsilon'],
    max_grad_norm=wandb.config['max_grad_norm'],
    weight_decay=wandb.config['weight_decay'],
    logging_steps=20,
    fp16=wandb.config['training'].get('fp16', False),
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    torch_compile = wandb.config['training'].get('torch_compile', False),
)


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

if __name__ == "__main__":
    import gc

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    eval_dataloader = trainer.get_eval_dataloader(eval_dataset)
    eval_output = trainer.evaluation_loop(eval_dataloader, 'Best checkpoint', prediction_loss_only=True, metric_key_prefix='eval/best_ckpt')
    wandb.log(eval_output.metrics)
    with (output_dir / 'eval_output.json').open('w') as fd:
        json.dump(eval_output.metrics, fd, indent=4)

    test_dataloader = trainer.get_eval_dataloader(test_dataset)
    test_output = trainer.evaluation_loop(test_dataloader, 'Best checkpoint', prediction_loss_only=True, metric_key_prefix='test/best_ckpt')
    wandb.log(test_output.metrics)
    with (output_dir / 'test_output.json').open('w') as fd:
        json.dump(test_output.metrics, fd, indent=4)

    del model
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print(torch.cuda.memory_summary())

    if not args.skip_eval:

        import subprocess
        import os

        # Assuming you're in the notebook directory
        notebook_dir = os.getcwd()
        model_path = output_dir.resolve().absolute()
        results_path = model_path / 'results/blimp/blimp_results.json'

        # Change to the evaluation pipeline directory
        os.chdir(eval_dir)

        # Run the evaluation script
        command = f"""
    python -m lm_eval --model hf \
        --model_args pretrained={model_path} \
        --tasks blimp_filtered,blimp_supplement \
        --device cuda:0 \
        --batch_size {per_device_bsz} \
        --output_path {results_path} \
    """

        # Capture the output
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        # Change back to the notebook directory
        os.chdir(notebook_dir)

        # Save the output to a file
        output_file = output_dir / "eval_blimp_results.txt"
        with open(output_file, "w") as f:
            f.write(result.stdout)

        # Print the output
        print(result.stdout)

        # Read and log the scores
        if results_path.exists():
            with results_path.open() as fd:
                results = json.load(fd)
            results = skim_results(results)
            wandb.log(flatten_dictionary(results))
            
            blimp_filtered_score = results['results']['groups']['blimp_filtered']['acc']
            blimp_supplement_score = results['results']['groups']['blimp_supplement']['acc']
            # For backward compatibility
            wandb.log({"blimp_filtered": blimp_filtered_score})
            wandb.log({"blimp_supplement": blimp_supplement_score})
            wandb.log({"blimp_averaged": (blimp_supplement_score + blimp_filtered_score)/2.0})
            print(f"Logged to wandb: blimp_supplement = {blimp_supplement_score}, blimp_filtered = {blimp_filtered_score}")
        else:
            print(f'Error: could not find results file {results_path}')

        # Check for errors
        if result.returncode != 0:
            print("Error occurred:")
            print(result.stderr)

        wandb.finish()
