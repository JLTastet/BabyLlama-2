print('starting imports')
from transformers import (
    LlamaConfig, LlamaForCausalLM,
)
from transformers import DataCollatorForLanguageModeling
from transformers import GPT2TokenizerFast
from pathlib import Path
import yaml
import json
import argparse
from datetime import datetime
from uuid import uuid4

from babyllama2.babylm_dataset import BabylmDataset
from babyllama2.distillation import DistillationTrainer, DistillationTrainingArguments
from babyllama2.scoring import skim_results, flatten_dictionary

print('imports are done')

RUN_ID = str(uuid4())

starting_time = datetime.now()

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="../config/llama-16M.yaml", help="Configuration file path")
parser.add_argument("--lr", type=float, default=None, help="Learning rate")
parser.add_argument("--model_name", type=str, default=None, help="Model name")
parser.add_argument("--num_epochs", type=int, default=None, help="Number of training epochs")
parser.add_argument("--job_id", type=str, default=None, help="(Optional) Job ID")
parser.add_argument("--eval_dir", type=str, default=str(Path.home() / "evaluation-pipeline-2024"), help="Path to the evaluation pipeline directory")
parser.add_argument("--skip_eval", action='store_true', help="Do not run the evals")
parser.add_argument("--teachers", type=str, default=None, required=True, help="Comma-separated list of paths to the teacher models")
args = parser.parse_args()

if args.teachers is None:
    raise(ValueError('Please provide the paths to teacher models as a comma-separated list'))
teacher_paths = args.teachers.split(',')

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

if args.job_id:
    config['job_id'] = args.job_id
if 'distillation' not in config:
    config['distillation'] = dict()
config['distillation'].update({
    'num_teachers': len(teacher_paths),
    **{f'teacher{i+1}': path for (i, path) in enumerate(teacher_paths)}
})

teachers = [LlamaForCausalLM.from_pretrained(path) for path in teacher_paths]

# Override config parameters if provided as command-line arguments
if args.lr:
    config['training']['lr'] = args.lr
if args.model_name:
    config['model']['name'] = args.model_name
if args.num_epochs:
    config['training']['num_epochs'] = args.num_epochs

print(f'Num epochs: {config["training"]["num_epochs"]}')

SEQ_LENGTH = config['data']['seq_length']

tokenizer_path = config['data']['tokenizer_path']
tokenizer = GPT2TokenizerFast(tokenizer_file= str(tokenizer_path))
tokenizer.bos_token = "<s>"
tokenizer.eos_token = "</s>"
tokenizer.pad_token = "<pad>"

train_dataset = BabylmDataset(config['data']['train_path'], SEQ_LENGTH, tokenizer=tokenizer, random_chunk=True)
eval_dataset = BabylmDataset(config['data']['eval_path'], SEQ_LENGTH, tokenizer=tokenizer, offset=0)
test_dataset = BabylmDataset(config['data']['test_path'], SEQ_LENGTH, tokenizer=tokenizer, offset=0)

eval_dir = Path(args.eval_dir)
if not args.skip_eval:
    assert eval_dir.exists()

# We tokenize the whole dataset and then set the max length
tokenizer.model_max_length = SEQ_LENGTH

model_config = LlamaConfig(
    vocab_size=tokenizer.vocab_size,
    max_position_embeddings=2*tokenizer.model_max_length,
    hidden_size=config['model']['hidden_size'],
    intermediate_size=config['model']['intermediate_size'],
    num_hidden_layers=config['model']['n_layer'],
    num_attention_heads=config['model']['n_head'],
    num_key_value_heads=config['model'].get('n_KV', config['model']['n_head']),
    tie_word_embeddings=config['model'].get('tie_word_embeddings', False),
    pad_token_id=tokenizer.convert_tokens_to_ids("<pad>"),
)

model = LlamaForCausalLM(model_config)

if args.job_id:
    unique_id = args.job_id
else:
    unique_id = starting_time.strftime('%y%m%d-%H%M%S')

PATH_SUBSTITUTIONS = {
    '{NAME}': config['model']['name'],
    '{ID}': unique_id,
}
if 'output_dir' in config['logging'] and 'output_path' not in config['logging']:
    # Use a subdirectory named <model name>-<unique ID>
    output_dir = Path(config['logging']['output_dir']) / ( config['model']['name'] + '-' + unique_id )
elif 'output_path' in config['logging'] and 'output_dir' not in config['logging']:
    # Expand {NAME} and {ID} in the path
    output_path_str = config['logging']['output_path']
    for (variable, substitution) in PATH_SUBSTITUTIONS.items():
        output_path_str = output_path_str.replace(variable, substitution)
    output_dir = Path(output_path_str)
else:
    raise(ValueError('Please specify exactly one of "output_dir" or "output_path". For the latter, you can use the variables {NAME} and {ID}.'))

print(f"Model has {model.num_parameters()} parameters")
print(f'Saving model to {output_dir}')

accumulation_steps = config['training']['gradient_accumulation_steps']
per_device_bsz = config['training']['batch_size'] // accumulation_steps

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

training_args = DistillationTrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    save_strategy = "epoch",
    evaluation_strategy = "epoch",
    num_train_epochs=config['training']['num_epochs'],
    gradient_accumulation_steps=accumulation_steps,
    per_device_train_batch_size=per_device_bsz,
    per_device_eval_batch_size=per_device_bsz,
    save_total_limit=1,  # Set to zero to avoid saving
    warmup_steps=config['training']['warmup_steps'], 
    lr_scheduler_type="cosine",
    learning_rate=float(config['training']['lr']),
    adam_beta1=float(config['training'].get('adam_beta1', 0.9)),
    adam_beta2=float(config['training'].get('adam_beta2', 0.999)),
    adam_epsilon=float(config['training'].get('adam_epsilon', 1e-8)),
    max_grad_norm=float(config['training'].get('max_grad_norm', 1.0)),
    weight_decay=float(config['training'].get('weight_decay', 0.0)),
    alpha=float(config['distillation']['alpha']),
    temperature=float(config['distillation']['temperature']),
    logging_steps=20,
    fp16=config['training']['fp16'],
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    torch_compile = config['training'].get('torch_compile', False),
)

trainer = DistillationTrainer(
    model=model,
    args=training_args,
    teacher_models=teachers,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

if __name__ == "__main__":
    import gc
    import torch
    import wandb
    wandb.require("core")
    # wandb.login()
    wandb.init(project= config['logging']['project'], name=config['model']['name'], id=RUN_ID, config=config)

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
