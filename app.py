import gradio as gr
#import spaces
import librosa
import soundfile as sf
import wavio
import os
import subprocess
import pickle
import torch
import torch.nn as nn
from transformers import T5Tokenizer
from transformer_model import Transformer
from miditok import REMI, TokenizerConfig
from pathlib import Path
from huggingface_hub import hf_hub_download

repo_id = "amaai-lab/text2midi"
# Download the model.bin file
model_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
# Download the vocab_remi.pkl file
tokenizer_path = hf_hub_download(repo_id=repo_id, filename="vocab_remi.pkl")
# Download the soundfont file
soundfont_path = "FluidR3_GM.sf2"


def save_wav(filepath):
    # Extract the directory and the stem (filename without extension)
    directory = os.path.dirname(filepath)
    stem = os.path.splitext(os.path.basename(filepath))[0]

    # Construct the full paths for MIDI and WAV files
    midi_filepath = os.path.join(directory, f"{stem}.mid")
    wav_filepath = os.path.join(directory, f"{stem}.wav")

    # Run the fluidsynth command to convert MIDI to WAV
    # f"fluidsynth -r 16000 soundfont.sf2 -g 1.0 --quiet --no-shell {midi_filepath} -T wav -F {wav_filepath} > /dev/null",
    process = subprocess.Popen(
        f'fluidsynth -ni -F "{wav_filepath}" -r 16000 "{soundfont_path}" "{midi_filepath}"',
        shell=True
    )
    process.wait()

    return wav_filepath


# def post_processing(input_midi_path: str, output_midi_path: str):
#     # Define tokenizer configuration
#     config = TokenizerConfig(
#         pitch_range=(21, 109),
#         beat_res={(0, 4): 8, (4, 12): 4},
#         num_velocities=32,
#         special_tokens=["PAD", "BOS", "EOS", "MASK"],
#         use_chords=True,
#         use_rests=False,
#         use_tempos=True,
#         use_time_signatures=False,
#         use_programs=True
#     )

#     # Initialize tokenizer
#     tokenizer = REMI(config)

#     # Tokenize the input MIDI
#     tokens = tokenizer(Path(input_midi_path))

#     # Remove notes in the first bar
#     modified_tokens = []
#     bar_count = 0
#     bars_after = 2
#     for token in tokens.tokens:
#         if token == "Bar_None":
#             bar_count += 1
#         if bar_count > bars_after:
#             modified_tokens.append(token)

#     # Decode tokens back into MIDI
#     modified_midi = tokenizer(modified_tokens)
#     modified_midi.dump_midi(Path(output_midi_path))


def generate_midi(caption, temperature=0.9, max_len=500):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    artifact_folder = 'artifacts'

    # tokenizer_filepath = os.path.join(artifact_folder, "vocab_remi.pkl")
    # Load the tokenizer dictionary
    with open(tokenizer_path, "rb") as f:
        r_tokenizer = pickle.load(f)

    # Get the vocab size
    vocab_size = len(r_tokenizer)
    print("Vocab size: ", vocab_size)
    model = Transformer(vocab_size, 768, 8, 2048, 18, 1024, False, 8, device=device)
    # model_path = os.path.join("amaai-lab/text2midi", "pytorch_model.bin")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

    inputs = tokenizer(caption, return_tensors='pt', padding=True, truncation=True)
    input_ids = nn.utils.rnn.pad_sequence(inputs.input_ids, batch_first=True, padding_value=0)
    input_ids = input_ids.to(device)
    attention_mask =nn.utils.rnn.pad_sequence(inputs.attention_mask, batch_first=True, padding_value=0) 
    attention_mask = attention_mask.to(device)
    output = model.generate(input_ids, attention_mask, max_len=max_len,temperature = temperature)
    output_list = output[0].tolist()
    generated_midi = r_tokenizer.decode(output_list)
    generated_midi.dump_midi("output.mid")
    # post_processing("output.mid", "output.mid")


#@spaces.GPU(duration=120)
def gradio_generate(prompt, temperature, max_length):
    # Generate midi
    max_length = int(max_length)
    generate_midi(prompt, temperature, max_length)

    # Convert midi to wav
    midi_filename = "output.mid"
    save_wav(midi_filename)
    wav_filename = midi_filename.replace(".mid", ".wav")

    # Read the generated WAV file
    output_wave, samplerate = sf.read(wav_filename, dtype='float32')
    temp_wav_filename = "temp.wav"
    wavio.write(temp_wav_filename, output_wave, rate=16000, sampwidth=2)
    
    return temp_wav_filename  # Return both WAV and MIDI file paths



#description_text = ""
# Gradio input and output components
input_text = gr.Textbox(lines=2, label="Prompt")
output_audio = gr.Audio(label="Generated Music", type="filepath")
output_midi = gr.File(label="Download MIDI File")
temperature = gr.Slider(minimum=0.8, maximum=1.1, value=0.9, step=0.1, label="Temperature", interactive=True)
max_length = gr.Number(value=500, label="Max Length", minimum=500, maximum=2000, step=100)

# CSS styling for the Duplicate button


# Gradio interface
gr_interface = gr.Interface(
    fn=gradio_generate,
    inputs=[
        gr.Textbox(label="Describe Your Track"),
        gr.Slider(0.1, 1.5, value=0.8, label="Creativity"),
        gr.Number(value=512, label="Length")
    ],
    outputs=gr.Audio(label="Generated Track", type="filepath"),
    title="CoverComposer 🎵",
    description="Generate AI-powered instrumental music.",
    allow_flagging="never",
    cache_examples=False
)

gr_interface.launch()

