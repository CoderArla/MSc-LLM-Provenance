import os
import re
import datetime
import argparse
import torch
import matplotlib
from omegaconf import OmegaConf
from sip_lib.identifiers.get_identifier import make_identifier
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import html 

# ------------------ Arguments ------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", required=True, help="Hugging Face model identifier (e.g., 'meta-llama/Llama-2-7b-chat-hf')")
parser.add_argument("--path", required=True, help="Path to your trained identifier folder")
parser.add_argument("--prompt", default="ChatGPT is ", help="Text prompt to start generation")
parser.add_argument("--logit_filter", default=0.5, type=float, help="Threshold for showing labels")
parser.add_argument("--hf_token", required=True, help="Hugging Face access token")
parser.add_argument("--use_gpu_identifier", action="store_true", help="Run identifier on GPU if available")
parser.add_argument("--quantization", choices=["none", "4bit"], default="none", help="Quantization mode")
args = parser.parse_args()

# ------------------ Save directory ------------------
current_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
save_dir = os.path.join(args.path, "generated", current_time)
os.makedirs(save_dir, exist_ok=True)

# ------------------ Load config ------------------
flags = OmegaConf.load(os.path.join(args.path, "config.yaml"))

# ------------------ Load Language Model ------------------
model_name = args.model_name
print(f"Loading model ({model_name}) with {args.quantization} quantization...")

bnb_config = None
if args.quantization == "4bit":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

# --- NOTE: Using 'token' is the modern and preferred argument over 'use_auth_token' ---
tokenizer = AutoTokenizer.from_pretrained(model_name, token=args.hf_token)

lm_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16 if args.quantization == "none" else None,
    quantization_config=bnb_config,
    token=args.hf_token
)

device = next(lm_model.parameters()).device
print("LM device:", device)

# ------------------ Load identifier ------------------
# This part remains the same, but you must provide the correct path for the loaded model.
identifier_model = make_identifier(flags.identifier_model, **flags)
identifier_model.load_state_dict(torch.load(os.path.join(args.path, "model.pt"), map_location="cpu"))
identifier_device = "cuda" if args.use_gpu_identifier and torch.cuda.is_available() else "cpu"
identifier_model.to(identifier_device)
identifier_model.eval()


# ------------------ Generate text ------------------ 
input_ids = tokenizer([args.prompt], return_tensors="pt").to(device) 
with torch.no_grad(): 
    output = lm_model.generate( 
        **input_ids, 
        max_length=100, 
        do_sample=False, 
        pad_token_id=tokenizer.eos_token_id 
    ) 
text = tokenizer.batch_decode(output, skip_special_tokens=True)[0] 
print("Generated text:", text) 

# ------------------ Gather hidden states ------------------ 
input_ids_h = tokenizer([text], return_tensors="pt").to(device) 
with torch.no_grad(): 
    hiddens = lm_model.forward(**input_ids_h, output_hidden_states=True).hidden_states[flags.hook_layer] 
hiddens = hiddens.to(torch.float32).to(identifier_device) 

# ------------------ Prepare identifier input ------------------ 
if flags.source_label_type == "unigram": 
    new_x = hiddens 
elif flags.source_label_type == "bigram": 
    new_x = torch.cat([hiddens[:, :-1, ...], hiddens[:, 1:, ...]], dim=-1) 
elif flags.source_label_type == "trigram": 
    new_x = torch.cat([hiddens[:, :-2, ...], hiddens[:, 1:-1, ...], hiddens[:, 2:, ...]], dim=-1) 

# ------------------ Run identifier ------------------ 
with torch.no_grad(): 
    output_id = identifier_model(new_x) 

# ------------------ Decode tokens and labels ------------------ 
id_to_vocab = {v: k for k, v in tokenizer.get_vocab().items()} 
tokenized = tokenizer(text)["input_ids"] 
decoded_tokens = [id_to_vocab[idx] for idx in tokenized] 

labels = output_id.argmax(dim=-1).squeeze(0) + 1 
logits = torch.softmax(output_id, dim=-1).max(dim=-1)[0].squeeze(0) 
labels[logits < args.logit_filter] = -1 

decoded_tokens_for_print = [re.sub("<0x0A>", "<br>", d) for d in decoded_tokens] 
decoded_tokens_for_print = [re.sub("<s>", "", d) for d in decoded_tokens_for_print] 
words = decoded_tokens_for_print 

# ------------------ DYNAMIC VISUALIZATION  ------------------ 

raw_labels = labels.detach().cpu().numpy() 
unique_labels_in_output = sorted(list(set(raw_labels))) 

# Create a dynamic color map for only the labels present in this specific output 
color_map = {} 
cmap = matplotlib.colormaps.get_cmap("tab20") 
color_idx = 0 
for label in unique_labels_in_output: 
    if label == -1: 
        color_map[label] = "#CCCCCC"  # Grey for No Source 
    else: 
        rgb = cmap(color_idx % cmap.N)[:3] 
        color_map[label] = matplotlib.colors.to_hex(rgb) 
        color_idx += 1 

# --- 1. Build the HTML for the main text body --- 
text_html = "" 
for word, label in zip(words, raw_labels): 
    # Get the dynamically assigned color from our run-specific map 
    color = color_map.get(label, "#FFFFFF") # Default to white if label not in map 
    safe_word = html.escape(word) 
    text_html += f'<span class="token source-{label}" style="background-color:{color};">{safe_word}</span>' 

# --- 2. Build the HTML for the legend --- 
legend_html = "" 
for label in unique_labels_in_output: 
    color = color_map.get(label) 
    label_text = "No Source / Low Confidence" if label == -1 else f"Sonnet {label}" 
    legend_html += f'<span class="legend-item" data-source="{label}" style="background-color:{color};">{label_text}</span>' 

# --- 3. Define the CSS and JavaScript for interactivity --- 
css_style = """ 
    body { font-family: sans-serif; line-height: 1.6; } 
    h3 { margin-bottom: 5px; } 
    #text-container { border: 1px solid #eee; padding: 10px; border-radius: 5px; } 
    .token { padding: 1px 2px; margin: 0 1px; border-radius: 3px; } 
    #legend-container { margin-top: 20px; } 
    .legend-item { padding: 3px 8px; margin: 2px 4px; border-radius: 4px; cursor: pointer; display: inline-block; border: 2px solid transparent; } 
    .highlight { border: 2px solid #000000; font-weight: bold; } 
""" 

javascript_code = """ 
    const legendItems = document.querySelectorAll('.legend-item'); 
    legendItems.forEach(item => { 
        const sourceId = item.getAttribute('data-source'); 
        const correspondingTokens = document.querySelectorAll(`.token.source-${sourceId}`); 

        item.addEventListener('mouseenter', () => { 
            correspondingTokens.forEach(token => token.classList.add('highlight')); 
        }); 

        item.addEventListener('mouseout', () => { 
            correspondingTokens.forEach(token => token.classList.remove('highlight')); 
        }); 
    }); 
""" 

# --- 4. Assemble the final HTML document --- 
final_html = f""" 
<!DOCTYPE html> 
<html> 
<head> 
    <title>Interactive Token-Level Provenance</title> 
    <style>{css_style}</style> 
</head> 
<body> 
    <h3>Interactive Provenance Visualization</h3> 
    <p>Hover over a legend item to highlight the corresponding tokens in the text.</p> 
    <div id="text-container">{text_html}</div> 
    <div id="legend-container"> 
        <strong>Legend:</strong><br> 
        {legend_html} 
    </div> 
    <script>{javascript_code}</script> 
</body> 
</html> 
""" 
# ------------------  DYNAMIC VISUALIZATION CODE ------------------ 

# ------------------ Save HTML and labels ------------------ 
with open(os.path.join(save_dir, "source_labels.html"), "w", encoding="utf-8") as f: 
    f.write(final_html) 

with open(os.path.join(save_dir, "source_labels.txt"), "w") as f: 
    f.write("> Labels:\n") 
    f.write(str(labels) + "\n\n") 
    f.write("> Text:\n") 
    f.write(text + "\n\n") 
    f.write("> Predicted Labels\n") 
    for label in unique_labels_in_output: 
        if label != -1: 
            f.write(f"label:{label}\n") 

print("---------------------------------------------------") 
print("Predicted Argmax Labels (1-oriented):", labels) 
print(f"🚀 Done! See interactive results in: {save_dir}")
