import os
import torch
import pickle
from tqdm import tqdm
from omegaconf import OmegaConf
from datasets import load_dataset
from sip_lib.make_llm import make_language_model_and_tokenizer
from sip_lib.data.tokenize import make_raw_dataset_and_dataloader
from sip_lib.data.get_data import make_data

def main(args):
    flags  = OmegaConf.create({})
    for k, v in vars(args).items():
        setattr(flags, k, v)

    lm_model, tokenizer = make_language_model_and_tokenizer(**flags)

    print("Loading data from sonnets_dataset.csv...")
    dataset = load_dataset('csv', data_files='Data/sonnets_dataset.csv')['train']
    # It selects all rows from the second row (index 1) to the end, skipping the title at index 0.
    dataset = dataset.select(range(1, len(dataset)))

    dataset, dataloader = make_raw_dataset_and_dataloader(dataset, tokenizer, 'text', flags.batch_size, flags.batch_size, num_proc=1, max_token_length=flags.max_token_length)

    print("🚀 Start gathering activations...")
    n_layers = len(eval(flags.hook_layers))
    hiddens = None
    idx = 0
    with tqdm(enumerate(dataloader), total=len(dataloader)) as pbar:
        pbar.set_description("Gathering activation")
        for step, batch in pbar:
            # THIS IS THE CRITICAL FIX: Move data to the GPU
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to("cuda:0")

            with torch.no_grad():
                outputs = lm_model(batch['input_ids'], labels=batch['input_ids'], attention_mask=batch['attention_mask'],  output_hidden_states=True)
                if hiddens is None:
                    size = (n_layers, len(dataloader.dataset),  *outputs.hidden_states[-1].shape[1:])
                    print('Pre allocated memory of size... ', size)
                    hiddens = torch.zeros(*size)
                    flags.hidden_size = size

                for i in range(outputs.logits.shape[0]):
                    for j, layer in enumerate(eval(flags.hook_layers)):
                        hiddens[j, idx, ...] = outputs.hidden_states[layer][i].detach().cpu()
                    idx+=1

    if not os.path.exists(flags.save_dir):
        os.makedirs(flags.save_dir)
    with open(os.path.join(flags.save_dir, 'hiddens.pkl'), 'wb') as f:
            pickle.dump(hiddens, f, pickle.HIGHEST_PROTOCOL)
    OmegaConf.save(flags, os.path.join(flags.save_dir, 'config.yaml'))
    print(f"Activations are saved at: {os.path.join(flags.save_dir, 'hiddens.pkl')}")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--lm_model")
parser.add_argument("--lm_size")
parser.add_argument("--lm_cache_dir")
parser.add_argument("--num_gpus", type=int)
parser.add_argument("--data")
parser.add_argument("--data_cache_dir")
parser.add_argument("--hook_layers", type=str)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--max_labels", type=int)
parser.add_argument("--max_token_length", type=int, default=-1)
parser.add_argument("--save_dir")

args = parser.parse_args()
main(args)
