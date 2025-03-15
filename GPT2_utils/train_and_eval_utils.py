import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from GPT2_utils.data_processing import *
from GPT2_utils.GPT_model_blocks import *

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def calculate_batch_loss(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)

    # we want to calculate loss over tokens to predict which is the next token
    # so we flatten along the batch and sequence length dimensions
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calculate_loss_loader(data_loader, model, device, num_batches=None):
    # either calculate loss over all batches in the dataloader, or a few
    # specified batches
    total_loss = 0
    if len(data_loader) == 0:
        return float('nan')
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    # in our implementation, all inputs and targets are the same length in the dataloader
    # in practice, it is better to train LLMs with batches of varying sequence lengths
    # for better generalization
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            batch_loss = calculate_batch_loss(input_batch, target_batch, model, device)
            total_loss = total_loss + batch_loss
        else:
            break

    return total_loss/num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()  # --> disables dropout 
    with torch.no_grad():
        train_loss = calculate_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calculate_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_length = model.pos_emb.weight.shape[0]
    input_token_ids = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        output_ids = generate_text_simple(model, input_token_ids, context_length,
                                          num_tokens_to_generate=50)
    output_text = token_ids_to_text(output_ids, tokenizer)
    print(output_text.replace("\n", " ")) # --> replace new lines with space in generated text
    model.train()
    return

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, num_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in tqdm(range(num_epochs)):
        model.train()
        for i, (input_batch, target_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = calculate_batch_loss(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            tokens_seen = tokens_seen + input_batch.numel()
            global_step = global_step + 1
            
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader,
                                                      device, eval_iter)
                train_losses.append(train_loss.cpu().numpy().item())
                val_losses.append(val_loss.cpu().numpy().item())
                num_tokens_seen.append(tokens_seen)

                print(f"Ep {epoch + 1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}")

        generate_and_print_sample(model, tokenizer, device, start_context)
    return train_losses, val_losses, num_tokens_seen

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    ax.plot(epochs_seen, train_losses, label='Training Losses')
    ax.plot(epochs_seen, val_losses, linestyle='-.', label='Validation Losses')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend(loc='upper right')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax1 = ax.twiny()
    ax1.plot(tokens_seen, train_losses, alpha=0) # --> alpha = 0 => transparent plot 
    # to align with train losses vs epochs plot
    ax1.set_xlabel('Tokens seen')
    fig.tight_layout()
    plt.show()
    return

## greedy sampling once model weights are fixed 
## --> same tokens are generated everytime for the same prompt

## text generation -> sampling

def softmax_with_temperature(logits, temperature):
    # --> logits shape: (vocab_size, )
    logits_scaled = logits/temperature
    return torch.softmax(logits_scaled, dim=0)

def generate(model, input_ids, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        input_ids_ = input_ids[:, -context_size:]  # --> take the last text window that fits in the context size for long texts

        with torch.no_grad():
            last_logits = model(input_ids_)[:, -1, :]

        if top_k is not None:
            top_k_logits, _ = torch.topk(last_logits, top_k)
            min_top_k_logit = top_k_logits[:, -1]

        conditioned_logits = torch.where(last_logits<min_top_k_logit, torch.tensor(float('-inf')).to(last_logits.device), 
                                         last_logits)

        if temperature > 0.0:
            probs = torch.softmax(conditioned_logits/temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        else:
            next_id = torch.argmax(conditioned_logits, dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_id], dim=1)
        if next_id == eos_id:
            break
    return input_ids

def save_model_and_optimizer(name, model, optimizer):
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
                f'./models/{name}-model_and_optimizer.pth')
    return

def load_model_and_optimizer(filepath, device, config):
    ckpt = torch.load(filepath, map_location=device)
    model = GPT2Model(config)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    model.train()
    return model, optimizer


### functions for loading pretrained weights
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight, 
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias, 
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, 
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight, 
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias, 
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale, 
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift, 
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale, 
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift, 
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])
    