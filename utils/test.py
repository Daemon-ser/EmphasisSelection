import re
def word2features(word):
    features=[]
    if word[0].isupper() and not word.isupper():
        features.append(0)
    elif not word[0].isupper():
        features.append(1)
    elif word.isupper():
        features.append(2)

    if re.search('\w',word) is None:
        features.append(0)
    elif re.search('\W',word) is None:
        features.append(1)
    else:
        features.append(2)
    return features

def read_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    word_upp,word_upp_cnt=0,0
    word_spec, word_spec_cnt=0,0
    word_stat, word_stat_cnt=0,0
    for line in lines:
        if not line.isspace():
            id, word, bio, freq, prob = line.strip().split()
            prob = float(prob)
            _, sid, lid, wid = id.split('-')
            if word.isupper():
                word_upp+=prob
                word_upp_cnt+=1
            elif not word.isupper() and word[0].isupper():
                word_stat+=prob
                word_stat_cnt+=1
            if re.search('\W',word) is not None:
                word_spec+=prob
                word_spec_cnt+=1
    print(word_upp/word_upp_cnt)
    print(word_spec/word_stat_cnt)
    print(word_stat/word_spec_cnt)



read_data('datasets/all.txt')