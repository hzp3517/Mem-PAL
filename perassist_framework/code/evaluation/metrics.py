import jieba
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import sys
sys.path.append('/data5/hzp/MemPAL/perassist_framework/code')


def calculate_single_bleu(reference, candidate):
    '''
    https://zhuanlan.zhihu.com/p/659633044

    reference: str
    candidate: str
    '''
    # 使用 jieba 进行中文分词
    reference_tokenized = [list(jieba.cut(reference))]
    candidate_tokenized = list(jieba.cut(candidate))
    
    # 设定平滑函数
    smoothie = SmoothingFunction().method1

    # 计算不同n-gram长度的BLEU分数
    bleu_1 = sentence_bleu(reference_tokenized, candidate_tokenized, weights=(1, 0, 0, 0), smoothing_function=smoothie)
    bleu_2 = sentence_bleu(reference_tokenized, candidate_tokenized, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
    bleu_3 = sentence_bleu(reference_tokenized, candidate_tokenized, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)
    bleu_4 = sentence_bleu(reference_tokenized, candidate_tokenized, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
    
    return {'bleu-1': bleu_1, 'bleu-2': bleu_2, 'bleu-3': bleu_3, 'bleu-4': bleu_4}


if __name__ == '__main__':
    reference = "用户需要一种既能满足工作需求又能持续改善健康状况的综合计划，重点在于如何在不影响职业发展的前提下逐步建立健康的生活习惯。"
    candidate = "用户希望获得一个全面的健康管理方案，包括饮食调整、适合当前身体状况的运动建议以及有效的压力管理方法，以帮助其在忙碌的工作生活中保持良好的身体和精神状态。"
    print(calculate_single_bleu(reference, candidate))
    

