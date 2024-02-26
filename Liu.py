import tkinter as tk
from tkinter import scrolledtext
import threading
from nltk.translate.bleu_score import sentence_bleu
import math
from tool.DataTool import *
import torch.nn.functional as F
from model.Transformer import Transformer
from model.transformerWithExperts import Transformer as TransformerWithExperts
from controlNN.model import gateModel
import time
import queue
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Original_Time = 0
Expert_Time = 0
Original_score = 0
Expert_score = 0
Time_Queue = queue.Queue()
Score_Queue = queue.Queue()


# 束搜索
def beamSearch_Original(model, enc_id2vocab, enc_vocab2id, dec_id2vocab, dec_vocab2id, source_sentence, k: int):
    source_sentence = source_sentence.split(char_space)
    dec_max_len = len(source_sentence) * 1.5

    enc_input = []
    for w in source_sentence:
        enc_input.append(enc_vocab2id[w])

    dec_input = []
    dec_input.append(dec_vocab2id[char_start])

    need_search = []
    scores = []

    final_result = []
    final_scores = []

    for _ in range(k):
        need_search.append(dec_input.copy())
        scores.append(0)

    input_1 = torch.tensor([enc_input]).to(device)
    input_2 = torch.tensor([dec_input]).to(device)
    enc_output,output = model(input_1, input_2)

    # 初始化,获取概率最大的k个单词的 id
    proba = F.softmax(output[-1],dim=-1)
    idxs = proba.argsort(descending=True).view(-1)[:k]
    for i in range(k):
        max_id = idxs.data[i].item()
        need_search[i].append(max_id)
        scores[i] = math.log(proba[max_id].item())

    has_end = 0

    while has_end < k:
        temp_h = []
        temp_score = []

        for i in range(k - has_end):
            dec_input = need_search[i]
            sc = scores[i]

            # 判断该序列是否有必要继续搜索
            sentence_len = len(dec_input)
            last_word_id = dec_input[len(dec_input) - 1]
            last_word_vocab = dec_id2vocab[last_word_id]

            if last_word_vocab == char_end or sentence_len >= dec_max_len:
                has_end += 1
                final_result.append(dec_input)
                final_scores.append(sc)
                continue

            # 对该序列进行搜索
            _,output = model(torch.tensor([enc_input]).to(device), torch.tensor([dec_input]).to(device),enc_output)

            # 获取概率最大的k-hasend个单词的 id
            output = F.softmax(output[-1],dim=-1)
            idxs = output.argsort(descending=True).view(-1)[:k - has_end]

            for i in range(k - has_end):
                # print(idxs.data[i].item())
                max_id = idxs.data[i].item()
                sentence = dec_input.copy()
                sentence.append(max_id)
                temp_h.append(sentence)
                temp_score.append(sc + math.log(output[max_id].item()))

        # 如果k个句子都已经预测完成（达到最大长度或者结束符）
        if has_end == k:
            break

        # 从temp_score中选择k-hasend个最大的，放入need_search中继续搜索
        temp_score_ids = np.argsort(temp_score)[::-1]
        for i in range(k - has_end):
            max_s_id = temp_score_ids[i]
            need_search[i] = temp_h[max_s_id].copy()
            scores[i] = temp_score[max_s_id]

    return final_scores, final_result

# 束搜索
def beamSearch_Expert(model, enc_id2vocab, enc_vocab2id, dec_id2vocab, dec_vocab2id, source_sentence, k: int, expertModule, expertList):
    source_sentence = source_sentence.split(char_space)
    dec_max_len = len(source_sentence) * 1.5

    enc_input = []
    for w in source_sentence:
        enc_input.append(enc_vocab2id[w])

    dec_input = []
    dec_input.append(dec_vocab2id[char_start])

    need_search = []
    scores = []

    final_result = []
    final_scores = []

    for _ in range(k):
        need_search.append(dec_input.copy())
        scores.append(0)

    input_1 = torch.tensor([enc_input]).to(device)
    input_2 = torch.tensor([dec_input]).to(device)
    enc_output,output = model(input_1, input_2)

    # 初始化,获取概率最大的k个单词的 id
    proba = F.softmax(output[-1],dim=-1)
    idxs = proba.argsort(descending=True).view(-1)[:k]
    for i in range(k):
        max_id = idxs.data[i].item()
        need_search[i].append(max_id)
        scores[i] = math.log(proba[max_id].item())

    has_end = 0

    while has_end < k:
        temp_h = []
        temp_score = []

        for i in range(k - has_end):
            dec_input = need_search[i]
            sc = scores[i]

            # 判断该序列是否有必要继续搜索
            sentence_len = len(dec_input)
            last_word_id = dec_input[len(dec_input) - 1]
            last_word_vocab = dec_id2vocab[last_word_id]

            if last_word_vocab == char_end or sentence_len >= dec_max_len:
                has_end += 1
                final_result.append(dec_input)
                final_scores.append(sc)
                continue

            # 对该序列进行搜索
            _,output = model(torch.tensor([enc_input]).to(device), torch.tensor([dec_input]).to(device),enc_output, expertModule, expertList)

            # 获取概率最大的k-hasend个单词的 id
            output = F.softmax(output[-1],dim=-1)
            idxs = output.argsort(descending=True).view(-1)[:k - has_end]

            for i in range(k - has_end):
                # print(idxs.data[i].item())
                max_id = idxs.data[i].item()
                sentence = dec_input.copy()
                sentence.append(max_id)
                temp_h.append(sentence)
                temp_score.append(sc + math.log(output[max_id].item()))

        # 如果k个句子都已经预测完成（达到最大长度或者结束符）
        if has_end == k:
            break

        # 从temp_score中选择k-hasend个最大的，放入need_search中继续搜索
        temp_score_ids = np.argsort(temp_score)[::-1]
        for i in range(k - has_end):
            max_s_id = temp_score_ids[i]
            need_search[i] = temp_h[max_s_id].copy()
            scores[i] = temp_score[max_s_id]

    return final_scores, final_result


def translate_Original(test_size, text_area, time_label,bleu_label, k=3):
    # 使用和原始代码相同的模型和词汇表处理
    global model, enc_id2vocab, enc_vocab2id, dec_id2vocab, dec_vocab2id
    # 清除文本区域
    text_area.delete(1.0, tk.END)

    # 从文件中读取输入数据
    test_s = open(test_file_path, 'r', encoding='utf-8').readlines()

    bleu_score_1 = 0
    bleu_score_2 = 0
    bleu_score_3 = 0
    bleu_score_4 = 0
    # 根据test_size进行测试
    test_size = int(test_size)
    
    start_time = time.time() # 记录开始时间
    
    for line in test_s[:test_size]:
        source_sentence = line.split('\t')[0]
        reference_sentence = line.split("\t")[1]

        # 替换<e>
        source_sentence_dieplay = source_sentence.replace("<e>", "")
        reference_sentence = reference_sentence.replace(" ", "")
        reference_sentence = reference_sentence.replace("<e>", " ")

        text_area.insert(tk.END, f'原文：{source_sentence_dieplay}\n')
        text_area.insert(tk.END, f'参考译文：{reference_sentence}\n')

        enc_input = char_start + char_space + source_sentence + char_space + char_end
        search_scores, search_results = beamSearch_Original(model, enc_id2vocab, enc_vocab2id, dec_id2vocab, dec_vocab2id, enc_input, k)
        
        # 将搜索结果打印到文本区域
        for i in range(k):
            dec_input = search_results[i]
            sent = ''
            for w in dec_input:
                sent += dec_id2vocab[w] + ' '
            sent = sent.replace(" ", "")
            sent = sent.replace("<e>", " ")
            sent = sent.replace(char_start, "")
            sent = sent.replace(char_end, "")
            bleu_score_1 += sentence_bleu([reference_sentence.split(char_space)], sent.split(char_space),
                                              weights=(1, 0, 0, 0))
            bleu_score_2 += sentence_bleu([reference_sentence.split(char_space)], sent.split(char_space),
                                              weights=(0, 1, 0, 0))
            bleu_score_3 += sentence_bleu([reference_sentence.split(char_space)], sent.split(char_space),
                                              weights=(0, 0, 1, 0))
            bleu_score_4 += sentence_bleu([reference_sentence.split(char_space)], sent.split(char_space),
                                              weights=(0, 0, 0, 1))
            text_area.insert(tk.END, '{:.3f},{}\n'.format(search_scores[i], sent))
        text_area.insert(tk.END, "\n")
    bleu_score_1 = bleu_score_1 / test_size / k
    bleu_score_2 = bleu_score_2 / test_size / k
    bleu_score_3 = bleu_score_3 / test_size / k
    bleu_score_4 = bleu_score_4 / test_size / k
    mean_bleu = (bleu_score_1 + bleu_score_2 + bleu_score_3 + bleu_score_4) / 4
    Original_score = mean_bleu
    # Score_Queue.put(mean_bleu)
    end_time = time.time() # 记录结束时间
    duration = end_time - start_time # 计算时间差
    Original_Time = duration
    # Time_Queue.put(duration)
    if Score_Queue.empty():
        Expert_score = 0
        Score_Queue.put(Original_score)
    else:
        Expert_score = Score_Queue.get()
        Score_Queue.put(Original_score)
    if Time_Queue.empty():
        Expert_Time = 0
        Time_Queue.put(Original_Time)
    else:
        Expert_Time = Time_Queue.get()
        Time_Queue.put(Original_Time)
    time_label.config(text=f'Original Model:Translation took {duration} seconds') # 打印输出时间
    time_label.update_idletasks()  # 确保标签的更新
    bleu_label.config(text=f'Original Model:Mean BLEU:{mean_bleu:.3f}')
    bleu_label.update_idletasks()
    # Expert_score = Score_Queue.get()
    # Expert_Time = Time_Queue.get()
    if Original_score != 0:
    #     score_accelerator_lable.config(text=f'speed-up ratio: 0 ')
    # else:
        score_accelerator_lable.config(text=f'accuracy rate: {Expert_score/Original_score:.4f} ') 
        score_accelerator_lable.update_idletasks()  # 确保标签的更新
    if Expert_Time != 0:
    #     time_accelerator_lable.config(text=f'accuracy rate: 0 ')
    # else:
        time_accelerator_lable.config(text=f'speed-up ratio: {Original_Time/Expert_Time:.4f} ')
        time_accelerator_lable.update_idletasks()  # 确保标签的更新

def run_translation_Original(entry, text_area, time_label,bleu_label):
    test_size = entry.get()
    if test_size == '':
        test_size = 10
    global Expert_score, Original_score, Expert_Time, Original_Time
    thread = threading.Thread(target=translate_Original, args=(test_size, text_area, time_label,bleu_label))
    thread.start()
    # thread.join()

def translate_Expert(test_size, text_area, time_label,bleu_label, k=3):
    # 使用和原始代码相同的模型和词汇表处理
    global model_expert, enc_id2vocab, enc_vocab2id, dec_id2vocab, dec_vocab2id,expertModule, expertList
    # 清除文本区域
    text_area.delete(1.0, tk.END)

    # 从文件中读取输入数据
    test_s = open(test_file_path, 'r', encoding='utf-8').readlines()

    bleu_score_1 = 0
    bleu_score_2 = 0
    bleu_score_3 = 0
    bleu_score_4 = 0
    # 根据test_size进行测试
    test_size = int(test_size)
    
    start_time = time.time() # 记录开始时间
    
    for line in test_s[:test_size]:
        source_sentence = line.split('\t')[0]
        reference_sentence = line.split("\t")[1]

        # 替换<e>
        source_sentence_dieplay = source_sentence.replace("<e>", "")
        reference_sentence = reference_sentence.replace(" ", "")
        reference_sentence = reference_sentence.replace("<e>", " ")

        text_area.insert(tk.END, f'原文：{source_sentence_dieplay}\n')
        text_area.insert(tk.END, f'参考译文：{reference_sentence}\n')

        enc_input = char_start + char_space + source_sentence + char_space + char_end
        search_scores, search_results = beamSearch_Expert(model_expert, enc_id2vocab, enc_vocab2id, dec_id2vocab, dec_vocab2id, enc_input, k,expertModule, expertList)
        
        # 将搜索结果打印到文本区域
        for i in range(k):
            dec_input = search_results[i]
            sent = ''
            for w in dec_input:
                sent += dec_id2vocab[w] + ' '
            sent = sent.replace(" ", "")
            sent = sent.replace("<e>", " ")
            sent = sent.replace(char_start, "")
            sent = sent.replace(char_end, "")
            bleu_score_1 += sentence_bleu([reference_sentence.split(char_space)], sent.split(char_space),
                                              weights=(1, 0, 0, 0))
            bleu_score_2 += sentence_bleu([reference_sentence.split(char_space)], sent.split(char_space),
                                              weights=(0, 1, 0, 0))
            bleu_score_3 += sentence_bleu([reference_sentence.split(char_space)], sent.split(char_space),
                                              weights=(0, 0, 1, 0))
            bleu_score_4 += sentence_bleu([reference_sentence.split(char_space)], sent.split(char_space),
                                              weights=(0, 0, 0, 1))
            text_area.insert(tk.END, '{:.3f},{}\n'.format(search_scores[i], sent))
        text_area.insert(tk.END, "\n")
    bleu_score_1 = bleu_score_1 / test_size / k
    bleu_score_2 = bleu_score_2 / test_size / k
    bleu_score_3 = bleu_score_3 / test_size / k
    bleu_score_4 = bleu_score_4 / test_size / k
    mean_bleu = (bleu_score_1 + bleu_score_2 + bleu_score_3 + bleu_score_4) / 4
    
    
    Expert_score = mean_bleu
    end_time = time.time() # 记录结束时间
    duration = end_time - start_time # 计算时间差
    Expert_Time = duration
    time_label.config(text=f'Expert Model:Translation took {duration} seconds') # 打印输出时间
    time_label.update_idletasks()  # 确保标签的更新
    bleu_label.config(text=f'Expert Model:Mean BLEU:{mean_bleu:.3f}')
    bleu_label.update_idletasks()
    # 队列为空放入数据，否则取出数据
    if Score_Queue.empty():
        Expert_score = 0
        Score_Queue.put(Expert_score)
    else:
        Original_score = Score_Queue.get()
        Score_Queue.put(Expert_score)
    if Time_Queue.empty():
        Original_Time = 0
        Time_Queue.put(Expert_Time)
    else:
        Original_Time = Time_Queue.get()
        Time_Queue.put(Expert_Time)


    # Original_score = Score_Queue.get()
    # Original_Time = Time_Queue.get()
    if Original_score != 0:
    #     score_accelerator_lable.config(text=f'speed-up ratio: 0 ')
    # else:
        score_accelerator_lable.config(text=f'accuracy rate: {Expert_score/Original_score:.4f} ') 
        score_accelerator_lable.update_idletasks()  # 确保标签的更新
    if Expert_Time != 0:
    #     time_accelerator_lable.config(text=f'accuracy rate: 0 ')
    # else:
        time_accelerator_lable.config(text=f'speed-up ratio: {Original_Time/Expert_Time:.4f} ')
        time_accelerator_lable.update_idletasks()  # 确保标签的更新

def run_translation_Expert(entry, text_area, time_label,bleu_label):
    test_size = entry.get()
    if test_size == '':
        test_size = 10
    global Expert_score, Original_score, Expert_Time, Original_Time
    thread = threading.Thread(target=translate_Expert, args=(test_size, text_area, time_label,bleu_label))
    thread.start()
    # thread.join()

if __name__ == '__main__':
    # ... （和原始代码相同的初始化代码）
    device = torch.device("cuda")
    encoder_chars, decoder_chars, max_enc_seq_length, max_dec_seq_length = calculate_data()
    enc_vocab2id = {word: i for i, word in enumerate(encoder_chars)}
    enc_id2vocab = {i: word for i, word in enumerate(encoder_chars)}

    dec_vocab2id = {word: i for i, word in enumerate(decoder_chars)}
    dec_id2vocab = {i: word for i, word in enumerate(decoder_chars)}


    model = Transformer(len(encoder_chars), len(decoder_chars), d_model, d_ff, num_layers, num_heads, device, 0, 0, 0.1)
    m_state_dict = torch.load('./save/de2en_2k_0020.pt', map_location="cuda:{}".format(map_gpu_index))
    model.load_state_dict(m_state_dict)
    model.to(device)
    model.eval()

    model_expert = TransformerWithExperts(len(encoder_chars), len(decoder_chars), d_model, d_ff, num_layers, num_heads, device, 0, 0, 0.1)
    k = 8
    model_expert.load_state_dict(m_state_dict)
    model_expert.to(device)
    model_expert.eval()
    w0f = open("Expert/index0.txt", "r")
    w0Index = w0f.readlines()
    w0Index = [int(x) for x in w0Index]
    expertList = [[] for i in range(k)]
    for i in range(len(w0Index)):
        expertList[w0Index[i]].append(i)

    expertModule = gateModel.gateNet(k)
    expertModule.load_state_dict(torch.load("controlNN/weights/best.pth"))
    expertModule.to(device)

    root = tk.Tk()
    # 设置大小
    root.geometry('900x700')

    # 设置标题？
    root.title("Transformer MOE accelerator")
    label = tk.Label(root, text="请输入test_size") # 创建一个Label组件，显示"请输入你的姓名"
    label.pack() # 放置Label组件
    entry = tk.Entry(root)
    # 设置entry的文本提示
    # entry.insert(0, "test_size")
    entry.pack()
    num = entry.get()
    if num:
        test_size = int(num)
    else:
        test_size = 5
    time_label = tk.Label(root, text="Original Model:Translation took 0 seconds")
    time_label.pack()
    time_label_expert = tk.Label(root, text="Expert Model:Translation took 0 seconds")
    time_label_expert.pack()
    bleu_label = tk.Label(root, text="Original Model:Mean BLEU:0")
    bleu_label.pack()
    bleu_label_expert = tk.Label(root, text="Expert Model:Mean BLEU:0")
    bleu_label_expert.pack()
    time_accelerator_lable = tk.Label(root, text="speed-up ratio:0")
    time_accelerator_lable.pack()
    score_accelerator_lable = tk.Label(root, text="accuracy rate:0")
    score_accelerator_lable.pack()
    
    translate_button = tk.Button(root, text="Original Model Translate", command=lambda: run_translation_Original(entry, text_area, time_label,bleu_label))
    translate_button.pack()
    translate_button_expert = tk.Button(root, text="Expert Model Translate", command=lambda: run_translation_Expert(entry, text_area, time_label_expert,bleu_label_expert))
    translate_button_expert.pack()
    text_area = scrolledtext.ScrolledText(root)
    # 设置文本区域的大小
    text_area['width'] = 400
    text_area['height'] = 600
    text_area.pack()
    if Original_score != 0:
    #     score_accelerator_lable.config(text=f'speed-up ratio: 0 ')
    # else:
        score_accelerator_lable.config(text=f'accuracy rate: {Expert_score/Original_score:.4f} ') 
        score_accelerator_lable.update_idletasks()  # 确保标签的更新
    if Expert_Time != 0:
    #     time_accelerator_lable.config(text=f'accuracy rate: 0 ')
    # else:
        time_accelerator_lable.config(text=f'speed-up ratio: {Original_Time/Expert_Time:.4f} ')
        time_accelerator_lable.update_idletasks()  # 确保标签的更新
    
    root.mainloop()
