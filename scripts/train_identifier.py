import os
import torch
import pickle
import datetime
import numpy as np
import gc
import copy
import random
from tqdm import tqdm
from omegaconf import OmegaConf
from transformers import get_scheduler
from sip_lib.identifiers.get_identifier import make_identifier
from sip_lib.utils.train_helper import TrainHelper
from sip_lib.utils.seed_data import split_indices
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset

def train(identifier, num_outputs, x, y, flags, **kwargs):
    class TorchDataset(Dataset):
        def __init__(self, x, y, train, seed_data, split, **kwargs):
            train_indices, test_indices = split_indices(len(x), seed_data, split)
            if train:
                self.x = x[train_indices]
                self.y = y[train_indices]
            else:
                self.x = x[test_indices]
                self.y = y[test_indices]
            print("size:", self.x.size(), self.y.size())
        def __len__(self):
            return len(self.x)
        def __getitem__(self, idx):
            x = self.x[idx].float()
            y = self.y[idx]
            return x, y

    # Convert input sequences according to label type
    if flags.source_label_type == "unigram":
        new_x = x.reshape(-1, x.size(-1))
        new_y = y.repeat_interleave(x.size(1))
    elif flags.source_label_type == "bigram":
        first_gram = x[:, :-1, ...]
        second_gram = x[:, 1:, ...]
        new_x = torch.cat([first_gram, second_gram], dim=-1)
        new_x = new_x.reshape(-1, new_x.size(-1))
        new_y = y.repeat_interleave(x.size(1)-1)
    elif flags.source_label_type == "trigram":
        first_gram = x[:, :-2, ...]
        second_gram = x[:, 1:-1, ...]
        third_gram = x[:, 2:, ...]
        new_x = torch.cat([first_gram, second_gram, third_gram], dim=-1)
        new_x = new_x.reshape(-1, new_x.size(-1))
        new_y = y.repeat_interleave(x.size(1)-2)
    else:
        raise ValueError(f"not implemented source_label_type {flags.source_label_type}")

    print("label data:", new_x.shape, new_y.shape)

    gpt_hidden_size = x.shape[-1]
    flags.gpt_hidden_size = gpt_hidden_size
    flags.num_outputs = num_outputs
    identifier_model = make_identifier(identifier,  **flags)
    torch.save(identifier_model.state_dict(), os.path.join(flags.save_dir,  'model.pt'))

    # DataLoaders
    train_dataset = TorchDataset(new_x, new_y, True, flags.seed_data, flags.split)
    test_dataset  = TorchDataset(new_x, new_y, False, flags.seed_data, flags.split)
    train_dataloader =  DataLoader(train_dataset, batch_size=flags.batch_size, shuffle=True)
    eval_dataloader =  DataLoader(test_dataset, batch_size=flags.batch_size, shuffle=False)
    train_dataloader_for_eval = DataLoader(train_dataset, batch_size=flags.batch_size, shuffle=False)

    th = TrainHelper(num_steps_per_epoch=len(train_dataloader), num_eval=5, num_save=1, num_epochs=flags.num_epochs)
    loss_fn = torch.nn.CrossEntropyLoss()

    if flags.optim == "adam":
        optimizer = torch.optim.Adam(identifier_model.parameters(), lr=flags.lr)
    elif flags.optim == "sgd":
        optimizer = torch.optim.SGD(identifier_model.parameters(), lr=flags.lr)
    elif flags.optim == "rmsprop":
        optimizer = torch.optim.RMSprop(identifier_model.parameters(), lr=flags.lr)

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=1,
        num_training_steps=th.num_train_steps,
    )

    summary_writer = SummaryWriter(log_dir=os.path.join(flags.save_dir))
    device = flags.device
    identifier_model.to(device)
    target_holder = torch.zeros(flags.batch_size, num_outputs).to(flags.device)

    OmegaConf.save(flags, os.path.join(flags.save_dir, "config.yaml"))

    with tqdm(range(th.num_train_steps)) as pbar:
        for epoch in range(th.num_epochs):
            for batch in train_dataloader:
                pbar.update(1)
                th.update_global_step()

                x_batch = batch[0].to(device)
                y_batch = batch[1].to(device)

                target = target_holder[:len(y_batch), ...]
                target.fill_(0)
                for i, label in enumerate(y_batch.int()):
                    target[i, label] = 1.0

                y_hat = identifier_model(x_batch)
                loss = loss_fn(y_hat, y_batch)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(identifier_model.parameters(), 5.0)
                optimizer.step()
                lr_scheduler.step()
                pbar.set_postfix({"loss": f"{loss.item():.3f}"})

                if th.global_step % 100 == 0:
                    summary_writer.add_scalar("in_training/loss", loss.item(), th.global_step)

                if th.is_eval_step():
                    identifier_model.eval()
                    for name, dataloader in zip(['train', 'eval'], [train_dataloader_for_eval, eval_dataloader]):
                        losses = []
                        count = 0
                        eq = 0
                        for step, batch in enumerate(dataloader):
                            with torch.no_grad():
                                x_eval = batch[0].to(device)
                                y_eval = batch[1].to(device)
                                y_hat_eval = identifier_model(x_eval)

                                target_eval = target_holder[:len(y_eval), ...]
                                target_eval.fill_(0)
                                for i, label in enumerate(y_eval.int()):
                                    target_eval[i, label] = 1.0

                                loss_eval = loss_fn(y_hat_eval, target_eval)
                                eq += (y_hat_eval.argmax(dim=-1) == y_eval).sum().item()
                                losses.append(loss_eval)
                                count += y_eval.size(0)
                        eq = eq / count
                        loss_eval = (torch.sum(torch.tensor(losses)).item() / count)
                        print(f"[EVAL-{name}] step:{th.global_step} | acc:{eq:.3f} | loss:{loss_eval:.3f}")
                        summary_writer.add_scalar(f"{name}/loss", loss_eval, th.global_step)
                        summary_writer.add_scalar(f"{name}/acc", eq, th.global_step)
                    identifier_model.train()

    torch.save(identifier_model.state_dict(), os.path.join(flags.save_dir, 'model.pt'))
    flags.done = True
    OmegaConf.save(flags, os.path.join(flags.save_dir, "config.yaml"))

def main(args):
    flags  = OmegaConf.create({})
    flags.done = False
    flags.datetime = datetime.datetime.now().strftime("%m%d_%H%M%S")

    for k, v in vars(args).items():
        setattr(flags, k, v)

    random.seed(flags.seed)
    np.random.seed(flags.seed)
    torch.manual_seed(flags.seed)
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)

    # Load dataset
    print("Loading data from sonnets_dataset.csv...")
    dataset = load_dataset('csv', data_files='Data/sonnets_dataset.csv')['train']
    dataset = dataset.select(range(1, len(dataset)))
    min_label = min(dataset['source_label'])
    max_label = max(dataset['source_label'])
    num_outputs = max_label - min_label + 1

    # Load hidden states
    a_flags  = OmegaConf.load(os.path.join(flags.hidden_dir, 'config.yaml'))
    hiddens  = pickle.load(open(os.path.join(flags.hidden_dir, 'hiddens.pkl'), 'rb'))

    if flags.cut_labels != -1:
        hiddens_temp = hiddens[:, :len(dataset['source_label']), ...]
        del hiddens
        gc.collect()
        hiddens = hiddens_temp

    print("🚀 Start training identification...")
    flags.activation_gather_config = a_flags
    identifier = flags.identifier_model
    base_dir = copy.deepcopy(flags.save_dir)

    hook_layers = a_flags.hook_layers
    if isinstance(hook_layers, str):
        hook_layers = eval(hook_layers)

    for hidden_index, hidden_layer in enumerate(hook_layers):
        hidden_layer_int = int(hidden_layer)
        flags.save_dir = os.path.join(base_dir, f"layer_{hidden_layer_int}")
        flags.hook_layer = hidden_layer_int
        os.makedirs(flags.save_dir, exist_ok=True)

        x = hiddens[hidden_index]
        y = torch.tensor(dataset['source_label']).long()
        y = y - min_label
        assert x.size(0) == y.size(0)
        print("📌 X (Samples, Sequences, Hiddens):", x.size())
        print("📌 Y (Samples):", y.size())
        print("📌 #documents:", len(y.unique()))
        train(identifier, num_outputs, x, y, flags)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--identifier_model")
parser.add_argument("--lm_model")
parser.add_argument("--lm_size")
parser.add_argument("--seed_data", type=int)
parser.add_argument("--seed", type=int)
parser.add_argument("--data")
parser.add_argument("--data_cache_dir")
parser.add_argument("--hidden_dir")
parser.add_argument("--cut_labels", type=int, default=-1)

parser.add_argument("--source_label_type", type=str)
parser.add_argument("--save_dir")
parser.add_argument("--test", action='store_true')

parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--lr",  type=float),
parser.add_argument("--split",  type=float),
parser.add_argument("--optim", type=str),
parser.add_argument("--device", type=str),
parser.add_argument("--linear_hidden_size"),
parser.add_argument("--linear_activation", default='relu', type=str),

args = parser.parse_args()

if __name__ == "__main__":
    main(args)
