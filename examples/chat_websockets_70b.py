from typing import List
import logging
import json
from fastapi import FastAPI, WebSocket
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import websockets
from typing import Optional
import time
from chat_prompts import prompt_formats
from chat_formatting import CodeBlockFormatter
import sys
import os
import re
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from exllamav2.generator import ExLlamaV2StreamingGenerator, ExLlamaV2Sampler
import pandas as pd
import torch
import gc
import argparse
from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
    model_init,
)

import matplotlib.pyplot as plt
from datetime import datetime

prompt_formats_list = list(prompt_formats.keys())


# Options

parser = argparse.ArgumentParser(description="Simple Llama2 chat example for ExLlamaV2")
parser.add_argument(
    "-modes", "--modes", action="store_true", help="List available modes and exit."
)
parser.add_argument(
    "-mode",
    "--mode",
    choices=prompt_formats_list,
    help="Chat mode. Use llama for Llama 1/2 chat finetunes.",
)
parser.add_argument(
    "-un",
    "--username",
    type=str,
    default="User",
    help="Username when using raw chat mode",
)
parser.add_argument(
    "-bn",
    "--botname",
    type=str,
    default="Chatbort",
    help="Bot name when using raw chat mode",
)
parser.add_argument("-sp", "--system_prompt", type=str, help="Use custom system prompt")

parser.add_argument(
    "-temp",
    "--temperature",
    type=float,
    default=0.95,
    help="Sampler temperature, default = 0.95 (1 to disable)",
)
parser.add_argument(
    "-topk",
    "--top_k",
    type=int,
    default=50,
    help="Sampler top-K, default = 50 (0 to disable)",
)
parser.add_argument(
    "-topp",
    "--top_p",
    type=float,
    default=0.8,
    help="Sampler top-P, default = 0.8 (0 to disable)",
)
parser.add_argument(
    "-typical",
    "--typical",
    type=float,
    default=0.0,
    help="Sampler typical threshold, default = 0.0 (0 to disable)",
)
parser.add_argument(
    "-repp",
    "--repetition_penalty",
    type=float,
    default=1.1,
    help="Sampler repetition penalty, default = 1.1 (1 to disable)",
)
parser.add_argument(
    "-maxr",
    "--max_response_tokens",
    type=int,
    default=1000,
    help="Max tokens per response, default = 1000",
)
parser.add_argument(
    "-resc",
    "--response_chunk",
    type=int,
    default=250,
    help="Space to reserve in context for reply, default = 250",
)
parser.add_argument(
    "-ncf",
    "--no_code_formatting",
    action="store_true",
    help="Disable code formatting/syntax highlighting",
)

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
if system_prompt is None:
    system_prompt = prompt_format.default_system_prompt()

# Initialize model and tokenizer

model_init.check_args(args)
model_init.print_options(args)
model, tokenizer = model_init.init(args)

# Create cache

cache = ExLlamaV2Cache(model)


# Chat context


def format_prompt(user_prompt, system_prompt, first):
    global prompt_format

    if first:
        return (
            prompt_format.first_prompt()
            .replace("<|system_prompt|>", system_prompt)
            .replace("<|user_prompt|>", user_prompt)
        )
    else:
        return prompt_format.subs_prompt().replace("<|user_prompt|>", user_prompt)


def encode_prompt(text):
    global tokenizer, prompt_format

    add_bos, add_eos, encode_special_tokens = prompt_format.encoding_options()
    return tokenizer.encode(
        text,
        add_bos=add_bos,
        add_eos=add_eos,
        encode_special_tokens=encode_special_tokens,
    )


user_prompts = []
responses_ids = []


def get_tokenized_context(max_len, system_prompt):
    global user_prompts, responses_ids

    while True:
        context = torch.empty((1, 0), dtype=torch.long)

        for turn in range(len(user_prompts)):
            up_text = format_prompt(
                user_prompts[turn], system_prompt, context.shape[-1] == 0
            )
            up_ids = encode_prompt(up_text)
            context = torch.cat([context, up_ids], dim=-1)

            if turn < len(responses_ids):
                context = torch.cat([context, responses_ids[turn]], dim=-1)

        if context.shape[-1] < max_len:
            return context

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
# print(f" -- System prompt:")
# print()
# print(col_sysprompt + system_prompt.strip() + col_default)
# print(system_prompt.strip() + col_default)

"""
def getChat(username, user_prompt, botname, system_prompt):
    start_time = time.time()  # 获取开始时间戳
    chunks = []
    msg="success"
    # Get user prompt
    up = user_prompt.strip()
    # up = input(username + ": " + col_default).strip()
    # Add to context

    user_prompts.append(up)

    # Send tokenized context to generator

    active_context = get_tokenized_context(
        model.config.max_seq_len - min_space_in_context, system_prompt)
    generator.begin_stream(active_context, settings)

    # Stream response

    # if prompt_format.print_bot_name():

    #     print(col_bot + botname + ": " + col_default, end = "")

    response_tokens = 0
    response_text = ""
    responses_ids.append(torch.empty((1, 0), dtype=torch.long))

    while True:

        # Get response stream

        chunk, eos, tokens = generator.stream()
        if len(response_text) == 0:
            chunk = chunk.lstrip()
        response_text += chunk
        responses_ids[-1] = torch.cat([responses_ids[-1], tokens], dim=-1)

        # Check for code block delimiters
        # Let formatter suppress text as long as it may be part of delimiter
        chunk, codeblock_delimiter = (
            chunk, False) if codeblock_formatter is None else codeblock_formatter.process_delimiter(chunk)

        # Enter code block

        # if not in_code_block:

        #     # Start of codeblock
        #     if codeblock_delimiter:
        #         codeblock_formatter.begin()
        #         print("\n")
        #         in_code_block = True
        #         codeblock_delimiter = False

        # Print

        # if in_code_block:

        #     # Print unformatted
        #     codeblock_formatter.print_code_block(chunk)

        # else:

        #     # Print formatted
        chunks.append(chunk)
        # print(chunk, end="")

        # Exit code block

        # if in_code_block:

        #     # End of code block
        #     if codeblock_delimiter:

        #         # Edge case when we get EOS right after code block
        #         if eos: codeblock_formatter.print_code_block("\n")

        #         print("\033[0m")  # Reset block color to be certain
        #         in_code_block = False
        #         codeblock_delimiter = False

        sys.stdout.flush()
        # time.sleep(1)

        # If model has run out of space, rebuild the context and restart stream

        if generator.full():

            active_context = get_tokenized_context(
                model.config.max_seq_len - min_space_in_context)
            generator.begin_stream(active_context, settings)

        # If response is too long, cut it short, and append EOS if that was a stop condition

        response_tokens += 1
        if response_tokens == max_response_tokens:

            if tokenizer.eos_token_id in generator.stop_tokens:
                responses_ids[-1] = torch.cat(
                    [responses_ids[-1], tokenizer.single_token(tokenizer.eos_token_id)], dim=-1)

            
            msg=col_error + f" !! Response exceeded {max_response_tokens} tokens and was cut short." + col_default
            break

        # EOS signal returned

        if eos:

            # if prompt_format.print_extra_newline():
            #     print()
            break
    end_time = time.time()  # 获取结束时间戳
    elapsed_time = end_time - start_time  # 计算经过的时间
    return chunks, elapsed_time,msg
"""


#####################分割线##########################
# 计算token效率
def calculate_token_efficiency(chunks, elapsed_time):
    # 计算输出token数量
    output_token_count = len(chunks)
    # 计算token效率
    token_efficiency = output_token_count / elapsed_time
    return token_efficiency




# 计算token的平均效率
def calculate_token_aver_efficiency(token_efficiency_list):
    token_efficiency_arr = np.array(token_efficiency_list)
    number = token_efficiency_arr.size  # 使用数组的size属性获取长度
    num_efficiency = np.sum(token_efficiency_arr)
    token_aver_efficiency = np.mean(token_efficiency_arr)
    return token_aver_efficiency

# 计算TPS
def get_token_per_second(chunks, elapsed_time):
    # 计算输出token数量
    output_token_count = len(chunks)
    # 计算TPS
    token_per_second = output_token_count / elapsed_time
    return token_per_second

# 计算WPS
def get_word_per_second(chunks, elapsed_time):
    # 计算总单词长度
    total_words_len = sum(len(chunk) for chunk in chunks)
    # 计算单词每秒数
    word_per_second = total_words_len / elapsed_time
    return word_per_second


# 数据处理和曲线绘制
def create_plot(outputs, max_token, name,is_gpu=True):
    # 指定 data 文件夹的路径
    data_folder_plt = "data/plt"
    # 生成带有时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"output_{timestamp}"
    outputs = pd.DataFrame(outputs)
    # 计算除第一行外的数据行数量
    list_length = len(outputs)

    # 根据数据行数量选择 token_length 的长度
    token_length = outputs["token_length"][0 : list_length + 1]

    # 提取数据列
    spend_times = outputs["spend_time"][0 : list_length + 1]
    token_per_second = outputs["token_per_second"][0 : list_length + 1]
    word_per_second = outputs["word_per_second"][0 : list_length + 1]

    # 获取数据中的最小值和最大值
    min_token_length = np.min(token_length)
    max_token_length = np.max(token_length)

    # 设置x轴的范围
    x_length = [min_token_length, max_token]
    # 设置y轴的范围
    y_time_length = [0,150]
    y_tps_length=[0,30]
    y_wps_length=[0,50]

    # 创建图像
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    # 指定需要标注的 x 坐标
    x_ticks = np.arange(0, max_token + 1, 100)
    #指定需要标注的x的坐标
    x_ticks_ = np.arange(0, max_token + 1, 200)
    # 使用 numpy 的 isin 函数筛选出满足条件的数据点下标
    indices = np.isin(token_length, x_ticks)
    selected_x = token_length[indices]
    indices_ = np.isin(token_length, x_ticks_)
    selected_x_ = token_length[indices_]
    # 绘制 "spend_time" 的曲线图
    ax1.plot(token_length, spend_times, marker="", linestyle="-", color="b")
    # 获取满足条件的 x 坐标及对应的 y 坐标
    selected_y_time = spend_times[indices]
    selected_y_time_ = spend_times[indices_]
    #设置表格
    x_tables = np.append(selected_x, [token_length.iloc[-1]])
    y_tables_time = np.append(selected_y_time, [spend_times.iloc[-1]])
    data1 = {
    "Token Length": x_tables,
    "Spend Time": y_tables_time
    }
    df1 = pd.DataFrame(data1)
    for x, y in zip(selected_x, selected_y_time):
        ax1.annotate(
            f"{y:.2f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )
        ax1.scatter(x, y, color="black")  # 添加散点图，用红色表示
    # 标注曲线的最后一个点
    ax1.annotate(
        f"{spend_times.iloc[-1]:.2f}",
        (token_length.iloc[-1], spend_times.iloc[-1]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
    )
    
    # # 显示表格数据 1
    # cell_text1 = []
    # for row in range(len(df1)):
    #     rounded_values = df1.iloc[row].round(2)  # 对每一行数据四舍五入保留两位小数
    #     cell_text1.append(rounded_values.values)
    # ax1.table(cellText=cell_text1, colLabels=df1.columns, cellLoc='center', loc='bottom', bbox=[0, -0.55, 1, 0.3])
    
    ax1.scatter(token_length.iloc[-1], spend_times.iloc[-1], color="black")
    ax1.set_xlabel("Token Length")
    ax1.set_ylabel("Spend Time")
    ax1.set_title("Spend Time by Token Length")
    ax1.grid(True)
    # 设置 x 轴范围
    ax1.set_xlim(x_length)
    # 设置 x 轴刻度的间隔
    ax1.set_xticks(x_ticks)
    # ax1.set_ylim(y_time_length)

    # 绘制 "token_efficiency" 的曲线图
    ax2.plot(token_length, token_per_second, marker="", linestyle="-", color="r")
    selected_y_efficiencies = token_per_second[indices]
    selected_y_efficiencies_ = token_per_second[indices_]
    y_tables_efficiencies=np.append(selected_y_efficiencies,[token_per_second.iloc[-1]])
    data2 = {
    "Token Length": x_tables,
    "Token Efficiency": y_tables_efficiencies
    }
    df2=pd.DataFrame(data2)
    for x, y in zip(selected_x, selected_y_efficiencies):
        ax2.annotate(
            f"{y:.2f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )
        ax2.scatter(x, y, color="black")  # 添加散点图，用红色表示
    ax2.annotate(
    f"{token_per_second.iloc[-1]:.2f}",
    (token_length.iloc[-1], token_per_second.iloc[-1]),
    textcoords="offset points",
    xytext=(0, 10),
    ha="center",
    )
    
    # # 显示表格数据 2
    # cell_text2 = []
    # for row in range(len(df2)):
    #     rounded_values = df2.iloc[row].round(2)  # 对每一行数据四舍五入保留两位小数
    #     cell_text2.append(rounded_values.values)
    # ax2.table(cellText=cell_text2, colLabels=df2.columns, cellLoc='center', loc='bottom', bbox=[0, -0.55, 1, 0.3])

    ax2.scatter(token_length.iloc[-1], token_per_second.iloc[-1], color="black")
    ax2.set_xlabel("Token Length")
    ax2.set_ylabel("Token per second (TPS)")
    ax2.set_title("Token per second (TPS) by Token Length")
    ax2.grid(True)
    # 设置 x 轴范围
    ax2.set_xlim(x_length)
    # 设置 x 轴刻度的间隔
    ax2.set_xticks(x_ticks)
    # ax2.set_ylim(y_tps_length)

    # 绘制 "token_aver_efficiency" 的曲线图
    ax3.plot(token_length, word_per_second, marker="", linestyle="-", color="g")
    selected_y_aver_efficiencies = word_per_second[indices]
    selected_y_aver_efficiencies_ = word_per_second[indices_]
    #设置表格
    y_tables_aver_efficiencies=np.append(selected_y_aver_efficiencies,[word_per_second.iloc[-1]])
    data3 = {
    "Token Length": x_tables,
    "Token Average Efficiency": y_tables_aver_efficiencies
    }
    df3 = pd.DataFrame(data3)
    for x, y in zip(selected_x, selected_y_aver_efficiencies):
        ax3.annotate(
            f"{y:.2f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )
        ax3.scatter(x, y, color="black")  # 添加散点图，用红色表示
    ax3.annotate(
    f"{word_per_second.iloc[-1]:.2f}",
    (token_length.iloc[-1], word_per_second.iloc[-1]),
    textcoords="offset points",
    xytext=(0, 10),
    ha="center",
    )

    # # 显示表格数据 3
    # cell_text3 = []
    # for row in range(len(df3)):
    #     rounded_values = df3.iloc[row].round(2)  # 对每一行数据四舍五入保留两位小数
    #     cell_text3.append(rounded_values.values)
    # ax3.table(cellText=cell_text3, colLabels=df3.columns, cellLoc='center', loc='bottom', bbox=[0, -0.55, 1, 0.3])

    ax3.scatter(token_length.iloc[-1], word_per_second.iloc[-1], color="black")
    ax3.set_xlabel("Token Length")
    ax3.set_ylabel("Word per second (WPS)")
    ax3.set_title("Word per second (WPS) by Token Length")
    ax3.grid(True)
    # 设置 x 轴范围
    ax3.set_xlim(x_length)
    # 设置 x 轴刻度的间隔
    ax3.set_xticks(x_ticks)
    # ax3.set_ylim(y_wps_length)
    

    plt.subplots_adjust(wspace=0.4)
    # 设置总标题
    if is_gpu:
        num_gpus = torch.cuda.device_count()
        fig.suptitle(
        f"{name}(GPUS{num_gpus}) Generation Efficiency",
        fontsize=16,
        fontweight="bold",
    )
    else:
        fig.suptitle(
            f"{name}(ONLY CPU) Generation Efficiency",
            fontsize=16,
            fontweight="bold",
        )

    # 生成带有时间戳的文件名
    png_filename = f"{name}_{filename}_gpus{num_gpus}.png"
    png_filepath = os.path.join(data_folder_plt, png_filename)
    # plt.tight_layout()
    # 保存 PNG 文件
    plt.savefig(png_filepath)
    # 关闭图形
    plt.close(fig)
    # 显示图像
    # plt.show()


# 导入FastAPI,HTTPException库
# 创建FastAPI应用
app = FastAPI()
# 配置日志记录
logging.basicConfig(level=logging.INFO)  # 设置日志级别为 INFO

# 配置跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该更加严格地配置
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 定义请求模型
class ChatRequest(BaseModel):
    user_prompt: str
    username: str = "User"
    botname: str = "Chatbort"
    system_prompt: str = "You are a helpful assistant"


async def chat_websocket(websocket: WebSocket, request: ChatRequest):
    try:
        average_start_time = time.time()
        chunks = []
        # 在这里执行聊天逻辑，包括处理用户提示、调用模型生成响应等
        # 替换下面的示例代码为你的实际逻辑
        user_prompt = request.user_prompt
        username = request.username
        botname = request.botname
        system_prompt = request.system_prompt
        ##############################

        # Get user prompt
        up = user_prompt.strip()
        # up = input(username + ": " + col_default).strip()
        # Add to context

        user_prompts.append(up)

        # Send tokenized context to generator

        active_context = get_tokenized_context(
            model.config.max_seq_len - min_space_in_context, system_prompt
        )
        generator.begin_stream(active_context, settings)

        # Stream response

        # if prompt_format.print_bot_name():

        #     print(col_bot + botname + ": " + col_default, end = "")

        response_tokens = 0
        response_text = ""
        responses_ids.append(torch.empty((1, 0), dtype=torch.long))
        character_length = 0
        token_efficiency_list = []
        outputs = []
        while True:
            single_start_time = time.time()
            # Get response stream

            chunk, eos, tokens = generator.stream()
            if len(response_text) == 0:
                chunk = chunk.lstrip()
            response_text += chunk
            responses_ids[-1] = torch.cat([responses_ids[-1], tokens], dim=-1)

            # Check for code block delimiters
            # Let formatter suppress text as long as it may be part of delimiter
            chunk, codeblock_delimiter = (
                (chunk, False)
                if codeblock_formatter is None
                else codeblock_formatter.process_delimiter(chunk)
            )
            chunks.append(chunk)
            # Enter code block

            # if not in_code_block:

            #     # Start of codeblock
            #     if codeblock_delimiter:
            #         codeblock_formatter.begin()
            #         print("\n")
            #         in_code_block = True
            #         codeblock_delimiter = False

            # Print

            # if in_code_block:

            #     # Print unformatted
            #     codeblock_formatter.print_code_block(chunk)

            # else:

            #     # Print formatted

            # print(chunk, end="")

            # Exit code block

            # if in_code_block:

            #     # End of code block
            #     if codeblock_delimiter:

            #         # Edge case when we get EOS right after code block
            #         if eos: codeblock_formatter.print_code_block("\n")

            #         print("\033[0m")  # Reset block color to be certain
            #         in_code_block = False
            #         codeblock_delimiter = False

            sys.stdout.flush()
            # time.sleep(1)

            # If model has run out of space, rebuild the context and restart stream

            if generator.full():
                active_context = get_tokenized_context(
                    model.config.max_seq_len - min_space_in_context
                )
                generator.begin_stream(active_context, settings)

            # If response is too long, cut it short, and append EOS if that was a stop condition

            response_tokens += 1
            if response_tokens == max_response_tokens:
                if tokenizer.eos_token_id in generator.stop_tokens:
                    responses_ids[-1] = torch.cat(
                        [
                            responses_ids[-1],
                            tokenizer.single_token(tokenizer.eos_token_id),
                        ],
                        dim=-1,
                    )

                msg = (
                    col_error
                    + f" !! Response exceeded {max_response_tokens} tokens and was cut short."
                    + col_default
                )
                break

            # EOS signal returned

            if eos:
                # if prompt_format.print_extra_newline():
                #     print()
                break
            end_time = time.time()  # 获取结束时间戳
            token_length = len(chunks)
            character_length += len(chunk)
            spend_time = end_time - average_start_time  # 计算平均输出时间
            # single_elapsed_time = end_time - single_start_time  # 计算单个输出时间
            token_efficiency = calculate_token_efficiency(
                chunks, spend_time
            )  # 计算平均token输出效率
            character_efficiency = calculate_token_efficiency(
                "".join(chunks), spend_time
            )  # 计算单个字符输出效率
            token_efficiency_list.append(token_efficiency)
            token_aver_efficiency = calculate_token_aver_efficiency(
                token_efficiency_list
            )
            token_per_second=get_token_per_second(chunks, spend_time)
            word_per_second=get_word_per_second(chunks, spend_time)
            # 逐个chunk发送
            message = {
                "type": "text",
                "username": username,
                "user_prompt": user_prompt,
                "botname": botname,
                "system_prompt": system_prompt,
                "chunk": chunk,
                "chunks": chunks,
                "token_length": token_length,
                "character_length": character_length,
                "spend_time": spend_time,
                "token_efficiency": token_efficiency,
                "character_efficiency": character_efficiency,
            }

            output = {
                "token_length": token_length,
                "spend_time": spend_time,
                "token_per_second": token_per_second,
                "word_per_second": word_per_second,
            }
            outputs.append(output)
            await websocket.send_json(message)
        num_gpus = torch.cuda.device_count()
        if num_gpus > 0:
            create_plot(outputs, 1000, "Llama2-70B-chat-exl2",is_gpu=True)
        else:
            create_plot(outputs, 1000, "Llama2-70B-chat-exl2",is_gpu=False)
    except websockets.ConnectionClosedOK:
        print(f"WebSocket connection closed for {username}")
        ##############################


# 使用一个列表来管理WebSocket连接
websocket_connections: List[WebSocket] = []
# 创建 WebSocket 聊天路由


@app.websocket("/ws/chat/")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    websocket_connections.append(websocket)

    try:
        request_data = await websocket.receive_json()
        logging.info(f"Received message from {username}: {request_data}")  # 记录接收的消息
        request = ChatRequest(**request_data)
        await chat_websocket(websocket, request)
    except Exception as e:
        print(f"WebSocket error: {e}")
        # 手动执行垃圾回收
        # 执行一些操作后清除缓存
        torch.cuda.empty_cache()
        gc.collect()
    finally:
        await websocket.close()  # 使用await等待连接关闭操作完成
        # 连接关闭时从列表中移除
        websocket_connections.remove(websocket)
        # 执行一些操作后清除缓存
        torch.cuda.empty_cache()
        gc.collect()
        pass


# 启动FastAPI应用
if __name__ == "__main__":
    # 检查系统中可用的GPU数量
    num_gpus = torch.cuda.device_count()
    logging.info(num_gpus)
    import uvicorn

    host = "0.0.0.0"
    post = 19324
    uvicorn.run(app, host=host, port=post)
