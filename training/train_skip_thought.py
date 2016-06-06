# coding: utf-8

import vocab
import train
import tools
import numpy as np

with open("../../wikipedia_txt/result_wakati.txt") as f:
    fdata = [line.rstrip() for i, line in enumerate(f)]
print '# lines: ', len(fdata)

worddict, wordcount = vocab.build_dictionary(fdata)
vocab.save_dictionary(worddict, wordcount, "word_dict")
print '# vocab: ', len(worddict)

train.trainer(fdata, dictionary="word_dict", saveFreq=100, saveto="model", reload_=True, n_words=40000)

model = tools.load_model()
vectors = tools.encode(model, fdata, use_norm=False)
np.savez('vecs.npz', vectors)

