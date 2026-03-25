import zhconv
from zhon.cedict import all


def char_word_tokenize(text):
    """分词器、中文单独成词，英文单词、连续数字作为一个词"""
    text = zhconv.convert(text.lower(), 'zh-cn')
    text = full_to_half(text)
    tokenized_chs = []
    text_len = len(text)
    i = 0
    while i < text_len:
        ch = text[i]
        if ch in all:
            tokenized_chs.append(ch)
            i += 1
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
            tokenized_chs.append(word)
        else:
            i += 1
    return tokenized_chs


def full_to_half(text):
    """全角字符转半角"""
    results = []
    for ch in text:
        code = ord(ch)
        if code == 0x3000:
            code = 0x0020
        elif 0xFF01 <= code <= 0xFF5E:
            code -= 0xFEE0
        results.append(chr(code))
    return ''.join(results)
