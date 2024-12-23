# %%
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from pathlib import Path
import json
from babyllama2.babylm_dataset import BabylmDataset
from argparse import ArgumentParser

# %%
parser = ArgumentParser()
parser.add_argument('model_path', type=str, help='Path to the model to evaluate')
parser.add_argument('--test_path', type=str, default='../data/dev_clean')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--seq_length', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--tokenizer_suffix', type=str, default=None, help='Used to distinguish the cached tokenized files')
args = parser.parse_args()

# %%
MODEL_PATH = Path(args.model_path)
assert MODEL_PATH.exists()
TEST_PATH = Path(args.test_path)
assert TEST_PATH.exists()
SEQ_LENGTH = args.seq_length
DEVICE = args.device

# %%
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True).to(DEVICE)

# %%
test_dataset = BabylmDataset(
    TEST_PATH, SEQ_LENGTH, tokenizer=tokenizer, offset=0, tokenizer_suffix=args.tokenizer_suffix)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)
training_args = TrainingArguments(output_dir=MODEL_PATH / 'output', per_device_eval_batch_size=args.batch_size)
trainer = Trainer(model=model, args=training_args, data_collator=data_collator)
test_dataloader = trainer.get_eval_dataloader(test_dataset)

# %%
test_output = trainer.evaluation_loop(
    test_dataloader,
    'Test loss',
    prediction_loss_only=True,
    metric_key_prefix='test/best_ckpt'
)

# %%
with (MODEL_PATH / 'test_output.json').open('w') as fd:
        json.dump(test_output.metrics, fd, indent=4)
