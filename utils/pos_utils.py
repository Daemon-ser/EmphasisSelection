from stanfordcorenlp import StanfordCoreNLP

def gen_data():
    nlp = StanfordCoreNLP(r'utils/stanford-corenlp-4.1.0', lang="en")
    with open('datasets/train.txt','r',encoding='utf-8') as f:
        data=f.readlines()

    sep_pos=['$','\'\'',',','.',':','``']
    all_pos=['UNK']
    with open("datasets/data_pos/train.txt",'w',encoding='utf-8') as f:
        for line in data:
            if not line.isspace():
                id, word, bio, freq, prob = line.strip().split()
                word_pos =nlp.pos_tag(word)
                if len(word_pos)==0:
                    pos="UNK"
                else:
                    _,pos=word_pos[0]
                f.write(line.strip()+"\t"+pos+"\n")
                all_pos.append(pos)
            else:
                f.write(line)

    #    with open("datasets/data_pos/pos.txt",'w',encoding='utf-8') as f:
    #        f.write("\n".join(list(set(all_pos))))

    nlp.close()

def gen_index_data():
    nlp = StanfordCoreNLP(r'utils/stanford-corenlp-4.1.0', lang="en")
    with open("datasets/data_pos/pos.txt",'r',encoding='utf-8') as f:
        all_pos=[line.strip() for line in f.readlines()]
    with open('datasets/test_data.txt','r',encoding='utf-8') as f:
        data=f.readlines()
    with open("datasets/data_pos/test_data.txt",'w',encoding='utf-8') as f:
        for line in data:
            if not line.isspace():
                id, word = line.strip().split()
                word_pos =nlp.pos_tag(word)
                if len(word_pos)==0:
                    pos_index=all_pos.index('UNK')
                else:
                    _,pos=word_pos[0]
                    try:
                        pos_index=all_pos.index(pos)
                    except:
                        pos_index=all_pos.index('UNK')
                f.write(line.strip()+"\t"+str(pos_index)+"\n")
            else:
                f.write(line)
    nlp.close()

gen_index_data()
