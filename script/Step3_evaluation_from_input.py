import argparse
import torch
import numpy as np
import json, io
from nltk import word_tokenize
import nltk
import os
from nltk.tag import StanfordPOSTagger
from keras.utils import to_categorical
import xml.etree.ElementTree as ET

"""fastText: https://github.com/facebookresearch/fastText"""
from fastText import load_model


class Model(torch.nn.Module):
    def __init__(self, gen_emb, domain_emb, num_classes=3, dropout=0.5, crf=False, tag=False):
        super(Model, self).__init__()
        self.tag_dim = 45 if tag else 0

        self.gen_embedding = torch.nn.Embedding(gen_emb.shape[0], gen_emb.shape[1])
        self.gen_embedding.weight=torch.nn.Parameter(torch.from_numpy(gen_emb), requires_grad=False)
        self.domain_embedding = torch.nn.Embedding(domain_emb.shape[0], domain_emb.shape[1])
        self.domain_embedding.weight=torch.nn.Parameter(torch.from_numpy(domain_emb), requires_grad=False)
    
        self.conv1=torch.nn.Conv1d(gen_emb.shape[1]+domain_emb.shape[1]+self.tag_dim, 128, 5, padding=2 )
        self.conv2=torch.nn.Conv1d(gen_emb.shape[1]+domain_emb.shape[1]+self.tag_dim, 128, 3, padding=1 )
        self.dropout=torch.nn.Dropout(dropout)

        self.conv3=torch.nn.Conv1d(256, 256, 5, padding=2)
        self.conv4=torch.nn.Conv1d(256, 256, 5, padding=2)
        self.conv5=torch.nn.Conv1d(256, 256, 5, padding=2)
        self.linear_ae=torch.nn.Linear(256, num_classes)
        self.crf_flag=crf
        if self.crf_flag:
            from allennlp.modules import ConditionalRandomField
            self.crf=ConditionalRandomField(num_classes)            
          
    def forward(self, x, x_len, x_mask, x_tag, y=None, testing=False):
        if self.tag_dim == 0:
            x_emb=torch.cat((self.gen_embedding(x), self.domain_embedding(x) ), dim=2)  # shape = [batch_size (128), sentence length (83), embedding output size (300+100+tag_num)]     
        else:
            x_emb=torch.cat((self.gen_embedding(x), self.domain_embedding(x), x_tag ), dim=2)
        x_emb=self.dropout(x_emb).transpose(1, 2)  # shape = [batch_size (128), embedding output size (300+100+tag_num) , sentence length (83)]
        x_conv=torch.nn.functional.relu(torch.cat((self.conv1(x_emb), self.conv2(x_emb)), dim=1) )  # shape = [batch_size, 128+128, 83]
        x_conv=self.dropout(x_conv)
        x_conv=torch.nn.functional.relu(self.conv3(x_conv) )
        x_conv=self.dropout(x_conv)
        x_conv=torch.nn.functional.relu(self.conv4(x_conv) )
        x_conv=self.dropout(x_conv)
        x_conv=torch.nn.functional.relu(self.conv5(x_conv) )
        x_conv=x_conv.transpose(1, 2) # shape = [batch_size, 83, 256]
        x_logit=self.linear_ae(x_conv) # shape = [batch_size, 83, num_classes]
        if testing:
            if self.crf_flag:
                score=self.crf.viterbi_tags(x_logit, x_mask)
            else:
                x_logit=x_logit.transpose(2, 0)
                score=torch.nn.functional.log_softmax(x_logit).transpose(2, 0)
        else:
            if self.crf_flag:
                score=-self.crf(x_logit, y, x_mask)
            else:
                x_logit=torch.nn.utils.rnn.pack_padded_sequence(x_logit, x_len, batch_first=True)
                score=torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(x_logit.data), y.data)
        return score

# This is for color the output text
class bcolors:
    ONGREEN = '\x1b[6;30;42m'
    ONYELLOW = '\x1b[6;30;46m'
    ONRED = '\x1b[6;30;41m'
    ONPURPLE = '\x1b[6;30;45m'
    END = '\x1b[0m'

def build_emb_dictionary(fn):
    words=[]
    vectors=[]
    with open(fn) as f:
        for l in f:
            t=l.rstrip().split(' ')
            words.append(t[0])
            vectors.append(list(map(float, t[1:])))
        wordvecs = np.array(vectors, dtype=np.double)
        word2id = {word:i for i, word in enumerate(words)}
    return wordvecs,word2id

def prep_emb(fn, gen_emb, domain_emb, prep_dir, gen_dim=300, domain_dim=100):
    text = []
    with open(fn) as f:
        for line in f:
            token = word_tokenize(line)
            text=text+token
    vocab = sorted(set(text))
    word_idx = {}
    if os.path.exists(prep_dir+'word_idx.json'):
        with io.open(prep_dir+'word_idx.json') as f:
            prev_word = json.load(f)
    else:
        prev_word = {}
    wx = 0
    new_word = []
    for word in vocab:
        if word not in prev_word:
            wx = wx+1
            new_word.append(word)
            word_idx[word] = wx+len(prev_word)
    prev_word.update(word_idx)          
    if new_word == []:
        return
    # create embedding
    embedding_gen=np.zeros((len(prev_word)+2, gen_dim) )
    embedding_domain=np.zeros((len(prev_word)+2, domain_dim) )    
    if os.path.exists(prep_dir+'gen.vec.npy'):
        gen_emb_prev=np.load(prep_dir+"gen.vec.npy")
        embedding_gen[:gen_emb_prev.shape[0],:] = gen_emb_prev
    if os.path.exists(prep_dir+'gen.vec.npy'):
        domain_emb_prev=np.load(prep_dir+'restaurant_emb.vec.npy')
        embedding_domain[:domain_emb_prev.shape[0],:] = domain_emb_prev
    with open(gen_emb) as f:
        # read the embedding .vec file
        for l in f:
            rec=l.rstrip().split(' ')
            if len(rec)==2: #skip the first line.
                continue 
            # if the word in word_idx, fill the embedding
            if rec[0] in new_word:
                embedding_gen[prev_word[rec[0]]] = np.array([float(r) for r in rec[1:] ])
    with open(domain_emb) as f:
        # read the embedding .vec file
        for l in f:
            # for each line, get the word and its vector
            rec=l.rstrip().split(' ')
            if len(rec)==2: #skip the first line.
                continue 
            # if the word in word_idx, fill the embedding
            if rec[0] in new_word:
                embedding_domain[prev_word[rec[0]]] = np.array([float(r) for r in rec[1:] ])
    ftmodel = load_model(domain_emb+".bin")
    for w in new_word:
        if embedding_domain[word_idx[w] ].sum()==0.:
            embedding_domain[word_idx[w] ] = ftmodel.get_word_vector(w)
    with io.open(prep_dir+'word_idx.json', 'w') as outfile:
        outfile.write(json.dumps(prev_word)) 
    np.save(prep_dir+'gen.vec.npy', embedding_gen.astype('float32') )
    np.save(prep_dir+'restaurant_emb.vec.npy', embedding_domain.astype('float32') )    
        
def prep_text(fn, POSdir, prep_dir):
        # map part-of-speech tag to int
    pos_tag_list = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS','NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP','SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB',',','.',':','$','#',"``","''",'(',')']
    tag_to_num = {tag:i+1 for i, tag in enumerate(sorted(pos_tag_list))}

    with io.open(prep_dir+'word_idx.json') as f:
        word_idx=json.load(f)
    sentence_size = [-1, 130]
    with open(fn) as f:
        count = 0
        for l in f:
            token = word_tokenize(l)
            if len(token) > 0:
                count = count + 1
                if len(token) > sentence_size[1]:
                    sentence_size[1]=len(token)
        sentence_size[0] = count
        X = np.zeros((sentence_size[0], sentence_size[1]), np.int16)
        X_tag = np.zeros((sentence_size[0], sentence_size[1]), np.int16)
     
    with open(fn) as f:   
        count = -1
        raw_X=[]
        for l in f:
            token = word_tokenize(l)
            jar = POSdir+'stanford-postagger.jar'
            model = POSdir+'models/english-left3words-distsim.tagger'
            pos_tagger = StanfordPOSTagger(model, jar, encoding='utf8')
            pos_tag_stf = [tag_to_num[tag] for (_,tag) in pos_tagger.tag(token)]
            if len(token) > 0:
                count = count + 1
                raw_X.append(token)   
                # write word index and tag in train_X and train_X_tag
                for wx, word in enumerate(token):
                    X[count, wx] = word_idx[word]
                    X_tag[count, wx] = pos_tag_stf[wx]
    return raw_X, X, X_tag

def output_text(fn, pred_y, out_fn=None):
    count = -1
    with open(fn) as f:
        for l in f:
            token = word_tokenize(l)
            if len(token) > 0:
                count = count + 1
                for wx, word in enumerate(token):
                    if pred_y[count, wx] == 1:
                        print(bcolors.ONGREEN + word + bcolors.END, end=" ")
                    elif pred_y[count, wx] == 2:
                        print(bcolors.ONYELLOW + word + bcolors.END, end=" ")
                    else:
                        print(word, end=" ")
                print('\n')
            else:
                print('\n') 

    
def test(model, test_X, test_X_tag, raw_X, domain, batch_size=128, crf=False, tag=False):
    pred_y=np.zeros((test_X.shape[0], test_X.shape[1]), np.int16)
    model.eval()
    for offset in range(0, test_X.shape[0], batch_size):
        batch_test_X_len=np.sum(test_X[offset:offset+batch_size]!=0, axis=1)
        batch_idx=batch_test_X_len.argsort()[::-1]
        batch_test_X_len=batch_test_X_len[batch_idx]
        batch_test_X_mask=(test_X[offset:offset+batch_size]!=0)[batch_idx].astype(np.uint8)
        batch_test_X=test_X[offset:offset+batch_size][batch_idx]
        batch_test_X_mask=torch.autograd.Variable(torch.from_numpy(batch_test_X_mask).long() )
        batch_test_X = torch.autograd.Variable(torch.from_numpy(batch_test_X).long() )
        if tag:
            batch_test_X_tag = test_X_tag[offset:offset+batch_size][batch_idx]
            batch_test_X_tag_onehot = to_categorical(batch_test_X_tag, num_classes=45+1)[:,:,1:]
            batch_test_X_tag_onehot = torch.autograd.Variable(torch.from_numpy(batch_test_X_tag_onehot).type(torch.FloatTensor) )
        else:
            batch_test_X_tag_onehot = None
        batch_pred_y=model(batch_test_X, batch_test_X_len, batch_test_X_mask, batch_test_X_tag_onehot, testing=True)
        r_idx=batch_idx.argsort()
        if crf:
            batch_pred_y=[batch_pred_y[idx] for idx in r_idx]
            for ix in range(len(batch_pred_y) ):
                for jx in range(len(batch_pred_y[ix]) ):
                    pred_y[offset+ix,jx]=batch_pred_y[ix][jx]
        else:
            batch_pred_y=batch_pred_y.data.cpu().numpy().argmax(axis=2)[r_idx]
            pred_y[offset:offset+batch_size,:batch_pred_y.shape[1]]=batch_pred_y
    model.train()
    assert len(pred_y)==len(test_X)    
    return pred_y

def demo_by_input(demo_dir, demo_fn, model_fn, POSdir, domain, gen_emb, domain_emb, runs, gen_dim, domain_dim, prep_dir, crf=False, tag=True):           
    fn = demo_dir + demo_fn
    prep_emb(fn, gen_emb, domain_emb, prep_dir, gen_dim, domain_dim)
    raw_X, X, X_tag = prep_text(fn, POSdir, prep_dir)
    
    for run in range(runs):
        model_fn_run=model_fn+str(run)
        model=torch.load(model_fn_run, map_location=lambda storage, loc: storage)
        embedding_gen=np.load(prep_dir+"gen.vec.npy")
        embedding_domain=np.load(prep_dir+"restaurant_emb.vec.npy")
        model.gen_embedding = torch.nn.Embedding(embedding_gen.shape[0], embedding_gen.shape[1])
        model.gen_embedding.weight=torch.nn.Parameter(torch.from_numpy(embedding_gen).type(torch.FloatTensor), requires_grad=False)
        model.domain_embedding = torch.nn.Embedding(embedding_domain.shape[0], embedding_domain.shape[1])
        model.domain_embedding.weight=torch.nn.Parameter(torch.from_numpy(embedding_domain).type(torch.FloatTensor), requires_grad=False)        
        if not tag:
            model.tag_dim = 0
        
        pred_y = test(model, X, X_tag, raw_X, domain, crf=crf, tag=tag)
        output_text(fn, pred_y)
    
parser = argparse.ArgumentParser()
parser.add_argument('--runs', type=int, default=5)
parser.add_argument('--emb_dir', type=str, default="data/embedding/")
parser.add_argument('--demo_dir', type=str, default="demo/")
parser.add_argument('--prep_dir', type=str, default="demo/prep/")
parser.add_argument('--model_fn', type=str, default="model/restaurant")
parser.add_argument('--gen_emb', type=str, default="gen.vec")
parser.add_argument('--restaurant_emb', type=str, default="restaurant_emb.vec")
parser.add_argument('--domain', type=str, default="restaurant")
parser.add_argument('--PoStag', type=bool, default=True)
parser.add_argument('--crf', type=bool, default=False)
parser.add_argument('--StanfordPOSTag_dir', type=str, default="stanford-postagger-full/")
parser.add_argument('--demo_fn', type=str, default='Yelp_review_short.txt')

args = parser.parse_args()

# wordvecs,word2id = build_emb_dictionary("data/embedding/gen.vec")    
demo_by_input(args.demo_dir,args.demo_fn, args.model_fn,args.StanfordPOSTag_dir,args.domain, args.emb_dir+args.gen_emb, args.emb_dir+args.restaurant_emb, args.runs, 300, 100, args.prep_dir, crf=args.crf, tag=args.PoStag)
