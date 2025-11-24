from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QLabel, QLineEdit, QPushButton, QTextEdit)
from PyQt6.QtCore import QObject, pyqtSignal as Signal
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
import os
from dotenv import load_dotenv
load_dotenv()
# 从环境变量获取设备，如果没有设置则使用默认值
device_str = os.getenv("DEVICE")
if device_str is None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(device_str)
print("running on device: ", device)

Original_Time = 0
Expert_Time = 0
Original_score = 0
Expert_score = 0
Time_Queue = queue.Queue()
Score_Queue = queue.Queue()


# 信号类，用于线程安全的GUI更新
class TranslationSignals(QObject):
    clear_text = Signal()
    append_text = Signal(str)
    update_time_label = Signal(str)
    update_bleu_label = Signal(str)
    update_time_accelerator = Signal(str)
    update_score_accelerator = Signal(str)


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
    _, idxs = torch.topk(proba, k, dim=-1)
    idxs = idxs.view(-1)
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
            _, idxs = torch.topk(output, k - has_end, dim=-1)
            idxs = idxs.view(-1)

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
    _, idxs = torch.topk(proba, k, dim=-1)
    idxs = idxs.view(-1)
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
            _, idxs = torch.topk(output, k - has_end, dim=-1)
            idxs = idxs.view(-1)

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


def translate_Original(test_size, signals, k=3):
    # 使用和原始代码相同的模型和词汇表处理
    global model, enc_id2vocab, enc_vocab2id, dec_id2vocab, dec_vocab2id
    # 清除文本区域
    signals.clear_text.emit()

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

        text_area.append(f'原文：{source_sentence_dieplay}')
        text_area.append(f'参考译文：{reference_sentence}')

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
            text_area.append('{:.3f},{}'.format(search_scores[i], sent))
        text_area.append("")
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
    signals.update_time_label.emit(f'Original Model:Translation took {duration} seconds') # 打印输出时间
    signals.update_bleu_label.emit(f'Original Model:Mean BLEU:{mean_bleu:.3f}')
    # Expert_score = Score_Queue.get()
    # Expert_Time = Time_Queue.get()
    if Original_score != 0:
    #     score_accelerator_lable.setText(f'speed-up ratio: 0 ')
    # else:
        signals.update_score_accelerator.emit(f'accuracy rate: {Expert_score/Original_score:.4f} ') 
    if Expert_Time != 0:
    #     time_accelerator_lable.setText(f'accuracy rate: 0 ')
    # else:
        signals.update_time_accelerator.emit(f'speed-up ratio: {Original_Time/Expert_Time:.4f} ')

def run_translation_Original(entry, signals):
    test_size = entry.text()
    if test_size == '':
        test_size = 10
    global Expert_score, Original_score, Expert_Time, Original_Time
    thread = threading.Thread(target=translate_Original, args=(test_size, signals))
    thread.start()
    # thread.join()

def translate_Expert(test_size, signals, k=3):
    # 使用和原始代码相同的模型和词汇表处理
    global model_expert, enc_id2vocab, enc_vocab2id, dec_id2vocab, dec_vocab2id,expertModule, expertList
    # 清除文本区域
    signals.clear_text.emit()

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

        signals.append_text.emit(f'原文：{source_sentence_dieplay}')
        signals.append_text.emit(f'参考译文：{reference_sentence}')

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
            signals.append_text.emit('{:.3f},{}'.format(search_scores[i], sent))
        signals.append_text.emit("")
    bleu_score_1 = bleu_score_1 / test_size / k
    bleu_score_2 = bleu_score_2 / test_size / k
    bleu_score_3 = bleu_score_3 / test_size / k
    bleu_score_4 = bleu_score_4 / test_size / k
    mean_bleu = (bleu_score_1 + bleu_score_2 + bleu_score_3 + bleu_score_4) / 4
    
    
    Expert_score = mean_bleu
    end_time = time.time() # 记录结束时间
    duration = end_time - start_time # 计算时间差
    Expert_Time = duration
    signals.update_time_label.emit(f'Expert Model:Translation took {duration} seconds') # 打印输出时间
    signals.update_bleu_label.emit(f'Expert Model:Mean BLEU:{mean_bleu:.3f}')
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
    #     score_accelerator_lable.setText(f'speed-up ratio: 0 ')
    # else:
        signals.update_score_accelerator.emit(f'accuracy rate: {Expert_score/Original_score:.4f} ') 
    if Expert_Time != 0:
    #     time_accelerator_lable.setText(f'accuracy rate: 0 ')
    # else:
        signals.update_time_accelerator.emit(f'speed-up ratio: {Original_Time/Expert_Time:.4f} ')

def run_translation_Expert(entry, signals):
    test_size = entry.text()
    if test_size == '':
        test_size = 10
    global Expert_score, Original_score, Expert_Time, Original_Time
    thread = threading.Thread(target=translate_Expert, args=(test_size, signals))
    thread.start()
    # thread.join()

if __name__ == '__main__':
    # ... （和原始代码相同的初始化代码）
    # device 已经在文件开头定义，这里不需要重新定义
    # 如果需要在主函数中使用特定的设备，可以取消下面的注释
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_chars, decoder_chars, max_enc_seq_length, max_dec_seq_length = calculate_data()
    enc_vocab2id = {word: i for i, word in enumerate(encoder_chars)}
    enc_id2vocab = {i: word for i, word in enumerate(encoder_chars)}

    dec_vocab2id = {word: i for i, word in enumerate(decoder_chars)}
    dec_id2vocab = {i: word for i, word in enumerate(decoder_chars)}


    model = Transformer(len(encoder_chars), len(decoder_chars), d_model, d_ff, num_layers, num_heads, device, 0, 0, 0.1)
    # 根据实际设备设置 map_location
    if torch.cuda.is_available():
        m_state_dict = torch.load('./save/de2en_2k_0020.pt', map_location="cuda:{}".format(map_gpu_index))
    else:
        m_state_dict = torch.load('./save/de2en_2k_0020.pt', map_location=torch.device('cpu'))
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
    # 根据实际设备设置 map_location
    if torch.cuda.is_available():
        expert_state_dict = torch.load("controlNN/weights/best.pth", map_location="cuda:{}".format(map_gpu_index))
    else:
        expert_state_dict = torch.load("controlNN/weights/best.pth", map_location=torch.device('cpu'))
    expertModule.load_state_dict(expert_state_dict)
    expertModule.to(device)

    app = QApplication([])
    window = QMainWindow()
    window.setWindowTitle("Transformer MOE accelerator")
    window.resize(900, 700)
    
    # 创建中央widget和布局
    central_widget = QWidget()
    window.setCentralWidget(central_widget)
    layout = QVBoxLayout()
    central_widget.setLayout(layout)
    
    # 创建标签和输入框
    label = QLabel("请输入test_size")
    layout.addWidget(label)
    
    entry = QLineEdit()
    layout.addWidget(entry)
    
    num = entry.text()
    if num:
        test_size = int(num)
    else:
        test_size = 5
    
    # 创建时间标签
    time_label = QLabel("Original Model:Translation took 0 seconds")
    layout.addWidget(time_label)
    time_label_expert = QLabel("Expert Model:Translation took 0 seconds")
    layout.addWidget(time_label_expert)
    
    # 创建BLEU标签
    bleu_label = QLabel("Original Model:Mean BLEU:0")
    layout.addWidget(bleu_label)
    bleu_label_expert = QLabel("Expert Model:Mean BLEU:0")
    layout.addWidget(bleu_label_expert)
    
    # 创建加速器标签
    time_accelerator_lable = QLabel("speed-up ratio:0")
    layout.addWidget(time_accelerator_lable)
    score_accelerator_lable = QLabel("accuracy rate:0")
    layout.addWidget(score_accelerator_lable)
    
    # 创建文本区域（需要在按钮之前创建，因为按钮的lambda会引用它）
    text_area = QTextEdit()
    text_area.setReadOnly(True)
    layout.addWidget(text_area)
    
    # 创建信号对象用于Original模型
    signals_original = TranslationSignals()
    signals_original.clear_text.connect(text_area.clear)
    signals_original.append_text.connect(text_area.append)
    signals_original.update_time_label.connect(time_label.setText)
    signals_original.update_bleu_label.connect(bleu_label.setText)
    signals_original.update_time_accelerator.connect(time_accelerator_lable.setText)
    signals_original.update_score_accelerator.connect(score_accelerator_lable.setText)
    
    # 创建信号对象用于Expert模型
    signals_expert = TranslationSignals()
    signals_expert.clear_text.connect(text_area.clear)
    signals_expert.append_text.connect(text_area.append)
    signals_expert.update_time_label.connect(time_label_expert.setText)
    signals_expert.update_bleu_label.connect(bleu_label_expert.setText)
    signals_expert.update_time_accelerator.connect(time_accelerator_lable.setText)
    signals_expert.update_score_accelerator.connect(score_accelerator_lable.setText)
    
    # 创建按钮
    translate_button = QPushButton("Original Model Translate")
    translate_button.clicked.connect(lambda: run_translation_Original(entry, signals_original))
    layout.addWidget(translate_button)
    
    translate_button_expert = QPushButton("Expert Model Translate")
    translate_button_expert.clicked.connect(lambda: run_translation_Expert(entry, signals_expert))
    layout.addWidget(translate_button_expert)
    
    if Original_score != 0:
    #     score_accelerator_lable.setText(f'speed-up ratio: 0 ')
    # else:
        score_accelerator_lable.setText(f'accuracy rate: {Expert_score/Original_score:.4f} ') 
    if Expert_Time != 0:
    #     time_accelerator_lable.setText(f'accuracy rate: 0 ')
    # else:
        time_accelerator_lable.setText(f'speed-up ratio: {Original_Time/Expert_Time:.4f} ')
    
    window.show()
    app.exec()
