import numpy as np
import en_vectors_web_lg, random, re, json


def shuffle_list(ans_list):
    random.shuffle(ans_list)


# ------------------------------
# ---- Initialization Utils ----
# ------------------------------

def tokenize(stat_ques_list, use_glove):
    token_to_ix = {
        'PAD': 0,
        'UNK': 1,
    }

    i2w = {}
    i2w[0] = 'PAD'
    i2w[1] = 'UNK'
    
    spacy_tool = None
    pretrained_emb = []
    if use_glove:
        spacy_tool = en_vectors_web_lg.load()
        pretrained_emb.append(spacy_tool('PAD').vector)
        pretrained_emb.append(spacy_tool('UNK').vector)

    for ques in stat_ques_list:
        #print("ques: ", ques)

        words_from_questions = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            ques['question'].lower()
        ).replace('-', ' ').replace('/', ' ').split()
        
        words_from_answers = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            ques['answer'].lower()
        ).replace('-', ' ').replace('/', ' ').split()

        words = words_from_questions + words_from_answers


        for word in words:
            if word not in token_to_ix:
                token_to_ix[word] = len(token_to_ix)
                i2w[len(token_to_ix)] = word
                if use_glove:
                    pretrained_emb.append(spacy_tool(word).vector)

    pretrained_emb = np.array(pretrained_emb)

    return token_to_ix, pretrained_emb, i2w



# ------------------------------------
# ---- Real-Time Processing Utils ----
# ------------------------------------

def proc_img_feat(img_feat, img_feat_pad_size):

    #(2,2048), 100
    if img_feat.shape[0] > img_feat_pad_size:
        img_feat = img_feat[:img_feat_pad_size]

    img_feat = np.pad(
        img_feat,
        ((0, img_feat_pad_size - img_feat.shape[0]), (0, 0)),
        mode='constant',
        constant_values=0
    )

    return img_feat


def proc_ques(ques, token_to_ix, max_token, type):
    ques_ix = np.zeros(max_token, np.int64)

    if type == 'question':
        words = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            ques['question'].lower()
        ).replace('-', ' ').replace('/', ' ').split()
    elif type == 'answer':
        words = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            ques['answer'].lower()
        ).replace('-', ' ').replace('/', ' ').split()

    for ix, word in enumerate(words):
        if word in token_to_ix:
            ques_ix[ix] = token_to_ix[word]
        else:
            ques_ix[ix] = token_to_ix['UNK']

        if ix + 1 == max_token:
            break

    return ques_ix


def get_score(occur):
    if occur == 0:
        return .0
    elif occur == 1:
        return .3
    elif occur == 2:
        return .6
    elif occur == 3:
        return .9
    else:
        return 1.



