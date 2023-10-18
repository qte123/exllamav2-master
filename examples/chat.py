
import sys, os, re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav2 import(
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
    model_init,
)

import argparse
import torch

from exllamav2.generator import (
    ExLlamaV2StreamingGenerator,
    ExLlamaV2Sampler
)

from chat_formatting import CodeBlockFormatter
from chat_prompts import prompt_formats
prompt_formats_list = list(prompt_formats.keys())

import time

# Options

parser = argparse.ArgumentParser(description = "Simple Llama2 chat example for ExLlamaV2")
parser.add_argument("-modes", "--modes", action = "store_true", help = "List available modes and exit.")
parser.add_argument("-mode", "--mode", choices = prompt_formats_list, help = "Chat mode. Use llama for Llama 1/2 chat finetunes.")
parser.add_argument("-un", "--username", type = str, default = "User", help = "Username when using raw chat mode")
parser.add_argument("-bn", "--botname", type = str, default = "Chatbort", help = "Bot name when using raw chat mode")
parser.add_argument("-sp", "--system_prompt", type = str, help = "Use custom system prompt")

parser.add_argument("-temp", "--temperature", type = float, default = 0.95, help = "Sampler temperature, default = 0.95 (1 to disable)")
parser.add_argument("-topk", "--top_k", type = int, default = 50, help = "Sampler top-K, default = 50 (0 to disable)")
parser.add_argument("-topp", "--top_p", type = float, default = 0.8, help = "Sampler top-P, default = 0.8 (0 to disable)")
parser.add_argument("-typical", "--typical", type = float, default = 0.0, help = "Sampler typical threshold, default = 0.0 (0 to disable)")
parser.add_argument("-repp", "--repetition_penalty", type = float, default = 1.1, help = "Sampler repetition penalty, default = 1.1 (1 to disable)")
parser.add_argument("-maxr", "--max_response_tokens", type = int, default = 1000, help = "Max tokens per response, default = 1000")
parser.add_argument("-resc", "--response_chunk", type = int, default = 250, help = "Space to reserve in context for reply, default = 250")
parser.add_argument("-ncf", "--no_code_formatting", action = "store_true", help = "Disable code formatting/syntax highlighting")

# Arrrgs

model_init.add_args(parser)
args = parser.parse_args()

# Prompt templates/modes

if args.modes:
    print(" -- Available formats:")
    for k, v in prompt_formats.items():
        print(f" --   {k:12} : {v().description}")
    sys.exit()

username = args.username
botname = args.botname
system_prompt = args.system_prompt

if args.mode is None:
    print(" ## Error: No mode specified.")
    sys.exit()

prompt_format = prompt_formats[args.mode]()
prompt_format.botname = botname
prompt_format.username = username
if system_prompt is None: system_prompt = prompt_format.default_system_prompt()

# Initialize model and tokenizer

model_init.check_args(args)
model_init.print_options(args)
model, tokenizer = model_init.init(args)

# Create cache

cache = ExLlamaV2Cache(model)


# Chat context

def format_prompt(user_prompt, first):
    global system_prompt, prompt_format

    if first:
        return prompt_format.first_prompt() \
            .replace("<|system_prompt|>", system_prompt) \
            .replace("<|user_prompt|>", user_prompt)
    else:
        return prompt_format.subs_prompt() \
            .replace("<|user_prompt|>", user_prompt)

def encode_prompt(text):
    global tokenizer, prompt_format

    add_bos, add_eos, encode_special_tokens = prompt_format.encoding_options()
    return tokenizer.encode(text, add_bos = add_bos, add_eos = add_eos, encode_special_tokens = encode_special_tokens)

user_prompts = []
responses_ids = []

def get_tokenized_context(max_len):
    global user_prompts, responses_ids

    while True:

        context = torch.empty((1, 0), dtype=torch.long)

        for turn in range(len(user_prompts)):

            up_text = format_prompt(user_prompts[turn], context.shape[-1] == 0)
            up_ids = encode_prompt(up_text)
            context = torch.cat([context, up_ids], dim=-1)

            if turn < len(responses_ids):
                context = torch.cat([context, responses_ids[turn]], dim=-1)

        if context.shape[-1] < max_len: return context

        # If the context is too long, remove the first Q/A pair and try again. The system prompt will be moved to
        # the first entry in the truncated context

        user_prompts = user_prompts[1:]
        responses_ids = responses_ids[1:]


# Generator

generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)

settings = ExLlamaV2Sampler.Settings()
settings.temperature = args.temperature
settings.top_k = args.top_k
settings.top_p = args.top_p
settings.typical = args.typical
settings.token_repetition_penalty = args.repetition_penalty

max_response_tokens = args.max_response_tokens
min_space_in_context = args.response_chunk

# Stop conditions

generator.set_stop_conditions(prompt_format.stop_conditions(tokenizer))

# ANSI color codes

col_default = "\u001b[0m"
col_user = "\u001b[33;1m"  # Yellow
col_bot = "\u001b[34;1m"  # Blue
col_error = "\u001b[31;1m"  # Magenta
col_sysprompt = "\u001b[37;1m"  # Grey

# Code block formatting

codeblock_formatter = None if args.no_code_formatting else CodeBlockFormatter()
in_code_block = False

delim_overflow = ""

# Main loop

print(f" -- Prompt format: {args.mode}")
print(f" -- System prompt:")
print()
print(col_sysprompt + system_prompt.strip() + col_default)

while True:

    # Get user prompt

    print()
    up = input(col_user + username + ": " + col_default).strip()
    print()

    # Add to context

    user_prompts.append(up)

    # Send tokenized context to generator

    active_context = get_tokenized_context(model.config.max_seq_len - min_space_in_context)
    generator.begin_stream(active_context, settings)

    # Stream response

    if prompt_format.print_bot_name():

        print(col_bot + botname + ": " + col_default, end = "")

    response_tokens = 0
    response_text = ""
    responses_ids.append(torch.empty((1, 0), dtype = torch.long))

    while True:

        # Get response stream

        chunk, eos, tokens = generator.stream()
        if len(response_text) == 0: chunk = chunk.lstrip()
        response_text += chunk
        responses_ids[-1] = torch.cat([responses_ids[-1], tokens], dim = -1)

        # Check for code block delimiters
        # Let formatter suppress text as long as it may be part of delimiter
        chunk, codeblock_delimiter = (chunk, False) if codeblock_formatter is None else codeblock_formatter.process_delimiter(chunk)

        # Enter code block

        if not in_code_block:

            # Start of codeblock
            if codeblock_delimiter:
                codeblock_formatter.begin()
                print("\n")
                in_code_block = True
                codeblock_delimiter = False

        # Print

        if in_code_block:

            # Print unformatted
            codeblock_formatter.print_code_block(chunk)

        else:

            # Print formatted
            print(chunk, end = "")

        # Exit code block

        if in_code_block:

            # End of code block
            if codeblock_delimiter:

                # Edge case when we get EOS right after code block
                if eos: codeblock_formatter.print_code_block("\n")

                print("\033[0m")  # Reset block color to be certain
                in_code_block = False
                codeblock_delimiter = False

        sys.stdout.flush()
        # time.sleep(1)

        # If model has run out of space, rebuild the context and restart stream

        if generator.full():

            active_context = get_tokenized_context(model.config.max_seq_len - min_space_in_context)
            generator.begin_stream(active_context, settings)

        # If response is too long, cut it short, and append EOS if that was a stop condition

        response_tokens += 1
        if response_tokens == max_response_tokens:

            if tokenizer.eos_token_id in generator.stop_tokens:
                responses_ids[-1] = torch.cat([responses_ids[-1], tokenizer.single_token(tokenizer.eos_token_id)], dim = -1)

            print()
            print(col_error + f" !! Response exceeded {max_response_tokens} tokens and was cut short." + col_default)
            break

        # EOS signal returned

        if eos:

            if prompt_format.print_extra_newline():
                print()

            break

