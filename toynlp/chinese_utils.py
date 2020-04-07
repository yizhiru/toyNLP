import zhconv
from Stemmer import Stemmer
from zhon.cedict import all

stemmer = Stemmer('english')


def char_word_tokenize(text):
    """分词器、中文单独成词，英文单词、连续数字作为一个词"""
    # 大写转小写，繁体转简体
    text = zhconv.convert(text.lower(), 'zh-cn')
    # 全角转半角
    text = full_to_half(text)
    tokenized_chs = []
    text_len = len(text)
    i = 0
    while i < text_len:
        ch = text[i]
        # 中文字符
        if ch in all:
            tokenized_chs.append(ch)
            i += 1
        # 数字或英文字母
        elif ch.isdigit() or ch.islower():
            word = ch
            j = i + 1
            while j < text_len:
                tch = text[j]
                if tch.isdigit() or tch.islower():
                    word += tch
                    j += 1
                else:
                    break
            i = j
            # 抽取词干API有错误，暂弃
            # tokenized_chs.append(stemmer.stemWord(word))
            tokenized_chs.append(word)
        else:
            i += 1
    return tokenized_chs


def full_to_half(text):
    results = []
    for ch in text:
        rstring = ""
        for uchar in ch:
            inside_code = ord(uchar)
            # 全角空格
            if inside_code == 12288:
                inside_code = 32
            # 全角字符（除空格）
            elif 65281 <= inside_code <= 65374:
                inside_code -= 65248
            rstring += chr(inside_code)
        results.append(rstring)
    return ''.join(results)
