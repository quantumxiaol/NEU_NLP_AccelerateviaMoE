# *_*coding:utf-8 *_*
"""
束搜索（Beam Search）工具函数
提供统一的beamSearch实现，支持多种变体
"""
import math
import numpy as np
import torch
import torch.nn.functional as F
from tool.DataTool import *
import os
from dotenv import load_dotenv

def _get_device():
    """
    获取device
    从环境变量获取设备，如果没有设置则使用默认值
    """
    load_dotenv()
    device_str = os.getenv("DEVICE")
    if device_str is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(device_str)


def beamSearch(model, enc_id2vocab, enc_vocab2id, dec_id2vocab, dec_vocab2id, 
                source_sentence, k: int, 
                expertModule=None, expertList=None, 
                expChoice=None, labelIndex=None,
                graphCal=False, eChoice=False, eList=None,
                use_topk=False):
    """
    统一的束搜索函数，支持多种变体
    
    Args:
        model: 模型
        enc_id2vocab: 编码器id到词汇的映射
        enc_vocab2id: 编码器词汇到id的映射
        dec_id2vocab: 解码器id到词汇的映射
        dec_vocab2id: 解码器词汇到id的映射
        source_sentence: 源句子
        k: beam size
        expertModule: 专家模块（可选，用于Expert版本）
        expertList: 专家列表（可选，用于Expert版本）
        expChoice: 专家选择（可选，用于Label版本）
        labelIndex: 标签索引（可选，用于Label版本）
        graphCal: 是否计算图（可选，用于Graph版本）
        eChoice: 专家选择标志（可选，用于Graph版本）
        eList: 专家列表（可选，用于Graph版本）
        use_topk: 是否使用torch.topk（Liu.py的Original版本使用）
    
    Returns:
        根据不同的变体返回不同的值：
        - 基础版本: (final_scores, final_result)
        - Expert版本: (final_scores, final_result)
        - Label版本: (final_scores, final_result, labelIndex)
        - Graph版本: (final_scores, final_result, graphM)
    """
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

    device = _get_device()
    input_1 = torch.tensor([enc_input]).to(device)
    input_2 = torch.tensor([dec_input]).to(device)
    
    # 根据不同的模型类型调用不同的forward方法
    if graphCal:
        enc_output, output = model(input_1, input_2, graphCal=False, eChoice=eChoice, eList=eList)
    elif expChoice is not None:
        enc_output, output = model(input_1, input_2)
    elif expertModule is not None and expertList is not None:
        enc_output, output = model(input_1, input_2)
    else:
        enc_output, output = model(input_1, input_2)

    # 初始化,获取概率最大的k个单词的 id
    proba = F.softmax(output[-1], dim=-1)
    if use_topk:
        # Original版本使用torch.topk
        _, idxs = torch.topk(proba, k, dim=-1)
        idxs = idxs.view(-1)
    else:
        # 其他版本使用argsort
        idxs = proba.argsort(descending=True).view(-1)[:k]
    
    for i in range(k):
        max_id = idxs.data[i].item()
        need_search[i].append(max_id)
        scores[i] = math.log(proba[max_id].item())

    has_end = 0
    finalInput = None
    finalOutput = None
    graphM = None
    
    # Label版本需要确保目录存在
    if expChoice is not None and labelIndex is not None:
        os.makedirs("data4Gate/input", exist_ok=True)
        os.makedirs("data4Gate/label", exist_ok=True)
    
    # Graph版本需要初始化graphM
    if graphCal:
        from tool.Global import d_ff
        graphM = torch.zeros([3, d_ff, d_ff]).to(device)

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
            dec_input_tensor = torch.tensor([dec_input]).to(device)
            
            # 根据不同的模型类型调用不同的forward方法
            if graphCal:
                _, output, graphTmp = model(torch.tensor([enc_input]).to(device), dec_input_tensor, 
                                           enc_output, graphCal=True, eChoice=eChoice, eList=eList)
                graphM += graphTmp
            elif expChoice is not None:
                _, output, inputS, actiRate = model(torch.tensor([enc_input]).to(device), dec_input_tensor, 
                                                    enc_output=enc_output, expChoice=expChoice)
                finalInput = inputS[0]
                finalOutput = actiRate
            elif expertModule is not None and expertList is not None:
                _, output = model(torch.tensor([enc_input]).to(device), dec_input_tensor, 
                                 enc_output, expertModule, expertList)
            else:
                _, output = model(torch.tensor([enc_input]).to(device), dec_input_tensor, enc_output)

            # 获取概率最大的k-hasend个单词的 id
            output = F.softmax(output[-1], dim=-1)
            if use_topk:
                _, idxs = torch.topk(output, k - has_end, dim=-1)
                idxs = idxs.view(-1)
            else:
                idxs = output.argsort(descending=True).view(-1)[:k - has_end]

            for j in range(k - has_end):
                max_id = idxs.data[j].item()
                sentence = dec_input.copy()
                sentence.append(max_id)
                temp_h.append(sentence)
                temp_score.append(sc + math.log(output[max_id].item()))

        # 如果k个句子都已经预测完成（达到最大长度或者结束符）
        if has_end == k:
            # Label版本需要保存数据
            if expChoice is not None and labelIndex is not None and finalInput is not None:
                for j in range(finalInput.shape[0]):
                    inputName = "data4Gate/input/" + str(labelIndex) + ".pt"
                    outputName = "data4Gate/label/" + str(labelIndex) + ".pt"
                    torch.save(finalInput[j], inputName)
                    torch.save(finalOutput[j], outputName)
                    labelIndex += 1
            break
        
        # Label版本在每次循环中保存数据（如果finalInput不为None）
        if expChoice is not None and labelIndex is not None and finalInput is not None:
            for j in range(finalInput.shape[0]):
                inputName = "data4Gate/input/" + str(labelIndex) + ".pt"
                outputName = "data4Gate/label/" + str(labelIndex) + ".pt"
                torch.save(finalInput[j], inputName)
                torch.save(finalOutput[j], outputName)
                labelIndex += 1
            finalInput = None
            finalOutput = None

        # 从temp_score中选择k-hasend个最大的，放入need_search中继续搜索
        temp_score_ids = np.argsort(temp_score)[::-1]
        for i in range(k - has_end):
            max_s_id = temp_score_ids[i]
            need_search[i] = temp_h[max_s_id].copy()
            scores[i] = temp_score[max_s_id]

    # 根据不同的变体返回不同的值
    if expChoice is not None and labelIndex is not None:
        return final_scores, final_result, labelIndex
    elif graphCal:
        return final_scores, final_result, graphM
    else:
        return final_scores, final_result


def beamSearch_Original(model, enc_id2vocab, enc_vocab2id, dec_id2vocab, dec_vocab2id, 
                         source_sentence, k: int):
    """
    使用torch.topk而不是argsort
    """
    return beamSearch(model, enc_id2vocab, enc_vocab2id, dec_id2vocab, dec_vocab2id, 
                     source_sentence, k, use_topk=True)


def beamSearch_Expert(model, enc_id2vocab, enc_vocab2id, dec_id2vocab, dec_vocab2id, 
                      source_sentence, k: int, expertModule, expertList):
    """
    Expert版本beamSearch
    支持expertModule和expertList
    """
    return beamSearch(model, enc_id2vocab, enc_vocab2id, dec_id2vocab, dec_vocab2id, 
                     source_sentence, k, expertModule=expertModule, expertList=expertList, 
                     use_topk=True)

