import torch
from nltk.translate.bleu_score import sentence_bleu
import math
from tool.DataTool import *
from tool.Global import *
import torch.nn.functional as F
from model.model4Rate import Transformer
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")
# 束搜索
def beamSearch(model, enc_id2vocab, enc_vocab2id, dec_id2vocab, dec_vocab2id, source_sentence, k: int, expChoice, labelIndex):
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
    finalInput = None
    finalOutput = None
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
            decInput = torch.tensor([dec_input]).to(device)
            _, output, inputS, actiRate= model(torch.tensor([enc_input]).to(device), decInput , enc_output = enc_output, expChoice = expChoice)
            finalInput = inputS[0]
            finalOutput = actiRate
            # actiRate = actiRate[0]
            # print(actiRate)
            # for i in range(inputS.shape[0]):
            #     xName = "data4Gate/x/" + str(labelIndex) + ".pt"
            #     torch.save()
            #     labelIndex += 1
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
            if finalInput is not None:
                for j in range(finalInput.shape[0]):
                    inputName = "data4Gate/input/" + str(labelIndex) + ".pt"
                    outputName = "data4Gate/label/" + str(labelIndex) + ".pt"
                    torch.save(finalInput[j], inputName)
                    torch.save(finalOutput[j], outputName)
                    labelIndex += 1
            break

        # 从temp_score中选择k-hasend个最大的，放入need_search中继续搜索
        temp_score_ids = np.argsort(temp_score)[::-1]
        for i in range(k - has_end):
            max_s_id = temp_score_ids[i]
            need_search[i] = temp_h[max_s_id].copy()
            scores[i] = temp_score[max_s_id]

    return final_scores, final_result, labelIndex

if __name__ == '__main__':
    device = torch.device("cuda")
    encoder_chars, decoder_chars, max_enc_seq_length, max_dec_seq_length = calculate_data()
    print('encoder_chars:', len(encoder_chars))
    print('decoder_chars:', len(decoder_chars))
    print('max_enc_seq_length:', max_enc_seq_length)
    print('max_dec_seq_length:', max_dec_seq_length)

    enc_vocab2id = {word: i for i, word in enumerate(encoder_chars)}
    enc_id2vocab = {i: word for i, word in enumerate(encoder_chars)}

    dec_vocab2id = {word: i for i, word in enumerate(decoder_chars)}
    dec_id2vocab = {i: word for i, word in enumerate(decoder_chars)}

    print('-----------------')
    print(enc_vocab2id[char_space])
    print(dec_vocab2id[char_space])
    print('-----------------')
    model = Transformer(len(encoder_chars), len(decoder_chars), d_model, d_ff, num_layers, num_heads, device, 0, 0, 0.1)
    m_state_dict = torch.load('./save/de2en_2k.pt', map_location="cuda:{}".format(map_gpu_index))
    model.load_state_dict(m_state_dict)
    w0f = open("./Expert/index0.txt","r")
    w0Index = w0f.readlines()
    w0Index = [int(x) for x in w0Index]
    k = 8
    labelIndex = 0
    expChoice = [[] for i in range(k)]
    for i in range(len(w0Index)):
        expChoice[w0Index[i]].append(i)
    # print(expChoice)
    model = model.to(device)
    model.eval()
    bleu_score_1 = 0
    bleu_score_2 = 0
    bleu_score_3 = 0
    bleu_score_4 = 0
    num = 0
    with torch.no_grad():
        # 已知序列
        test_s = open(test_file_path, 'r', encoding='utf-8').readlines()
        # 取前50条进行测试
        # 生成PT文件
        test_size = 800
        for line in test_s[:test_size]:
            num += 1
            print("{} / {}".format(num, test_size))
            print("dataNum : {}".format(labelIndex))
            enc_input = line.split('\t')[0]
            enc_pre_1 = enc_input.replace(" ", "")
            enc_pre_1 = enc_pre_1.replace("<e>", " ")

            target_sentence = line.split("\t")[1]
            target_sentence = target_sentence.replace(" ", "")
            target_sentence = target_sentence.replace("<e>", " ")
            k = 3
            enc_input = char_start + char_space + enc_input + char_space + char_end
            search_sources, search_result, labelIndex = beamSearch(model, enc_id2vocab, enc_vocab2id, dec_id2vocab, dec_vocab2id,
                                                       enc_input, k, expChoice, labelIndex)





