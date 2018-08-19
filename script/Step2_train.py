import argparse
import torch
import json
import numpy as np
import random
import os
import pickle
from keras.utils import to_categorical

CUDA = torch.cuda.is_available()

np.random.seed(1337)
random.seed(1337)
torch.manual_seed(1337)
if CUDA:
    torch.cuda.manual_seed(1337)


def batch_generator(X, y, X_tag, batch_size=128, return_idx=False, CUDA=False, crf=False, tag=False):
    for offset in range(0, X.shape[0], batch_size):
        # Count sentence length (none zero element in each row of X)
        batch_X_len = np.sum(X[offset:offset+batch_size]!=0, axis=1)
        # Order batch_X, batch_y by sentence length (descending)
        batch_idx = batch_X_len.argsort()[::-1]
        batch_X_len = batch_X_len[batch_idx]
        batch_X_mask = (X[offset:offset+batch_size]!=0)[batch_idx].astype(np.uint8) 
        batch_X = X[offset:offset+batch_size][batch_idx] 
        batch_y = y[offset:offset+batch_size][batch_idx]
        if CUDA:
            batch_X = torch.autograd.Variable(torch.from_numpy(batch_X).long().cuda() )
            batch_X_mask=torch.autograd.Variable(torch.from_numpy(batch_X_mask).long().cuda() )
            batch_y = torch.autograd.Variable(torch.from_numpy(batch_y).long().cuda() )
        else:
            batch_X = torch.autograd.Variable(torch.from_numpy(batch_X).long() )
            batch_X_mask=torch.autograd.Variable(torch.from_numpy(batch_X_mask).long() )
            batch_y = torch.autograd.Variable(torch.from_numpy(batch_y).long() )        
        if tag:
            batch_X_tag = X_tag[offset:offset+batch_size][batch_idx]
            batch_X_tag_onehot = to_categorical(batch_X_tag, num_classes=45+1)[:,:,1:]
            if CUDA:
                batch_X_tag_onehot = torch.autograd.Variable(torch.from_numpy(batch_X_tag_onehot).type(torch.FloatTensor).cuda() )
            else:
                batch_X_tag_onehot = torch.autograd.Variable(torch.from_numpy(batch_X_tag_onehot).type(torch.FloatTensor) )
        else:
            batch_X_tag_onehot = None
        
        if len(batch_y.size() )==2 and not crf:
            # packing is used for seq to seq models with variable lengths
            batch_y=torch.nn.utils.rnn.pack_padded_sequence(batch_y, batch_X_len, batch_first=True)
        if return_idx: #in testing, need to sort back.
            yield (batch_X, batch_y, batch_X_len, batch_X_mask, batch_X_tag_onehot, batch_idx)
        else:
            yield (batch_X, batch_y, batch_X_len, batch_X_mask, batch_X_tag_onehot)

class Model(torch.nn.Module):
    def __init__(self, gen_emb, domain_emb, num_classes=3, dropout=0.5, crf=False, tag=False):
        super(Model, self).__init__()
        self.tag_dim = 45 if tag else 0

        self.gen_embedding = torch.nn.Embedding(gen_emb.shape[0], gen_emb.shape[1])
        self.gen_embedding.weight=torch.nn.Parameter(torch.from_numpy(gen_emb), requires_grad=False)
        self.domain_embedding = torch.nn.Embedding(domain_emb.shape[0], domain_emb.shape[1])
        self.domain_embedding.weight=torch.nn.Parameter(torch.from_numpy(domain_emb), requires_grad=False)
    
        self.conv1=torch.nn.Conv1d(gen_emb.shape[1]+domain_emb.shape[1], 128, 5, padding=2 )
        self.conv2=torch.nn.Conv1d(gen_emb.shape[1]+domain_emb.shape[1], 128, 3, padding=1 )
        self.dropout=torch.nn.Dropout(dropout)

        self.conv3=torch.nn.Conv1d(256, 256, 5, padding=2)
        self.conv4=torch.nn.Conv1d(256, 256, 5, padding=2)
        self.conv5=torch.nn.Conv1d(256, 256, 5, padding=2)
        self.linear_ae1=torch.nn.Linear(256+self.tag_dim+domain_emb.shape[1], 50)
        self.linear_ae2=torch.nn.Linear(50, num_classes)
        self.crf_flag=crf
        if self.crf_flag:
            from allennlp.modules import ConditionalRandomField
            self.crf=ConditionalRandomField(num_classes)            
          
    def forward(self, x, x_len, x_mask, x_tag, y=None, testing=False):
        x_emb=torch.cat((self.gen_embedding(x), self.domain_embedding(x) ), dim=2)  # shape = [batch_size (128), sentence length (83), embedding output size (300+100)]       
        x_emb=self.dropout(x_emb).transpose(1, 2)  # shape = [batch_size (128), embedding output size (300+100+tag_num) , sentence length (83)]
        x_conv=torch.nn.functional.relu(torch.cat((self.conv1(x_emb), self.conv2(x_emb)), dim=1) )  # shape = [batch_size, 128+128, 83]
        x_conv=self.dropout(x_conv)
        x_conv=torch.nn.functional.relu(self.conv3(x_conv) )
        x_conv=self.dropout(x_conv)
        x_conv=torch.nn.functional.relu(self.conv4(x_conv) )
        x_conv=self.dropout(x_conv)
        x_conv=torch.nn.functional.relu(self.conv5(x_conv) )
        x_conv=x_conv.transpose(1, 2) # shape = [batch_size, 83, 256]
        x_logit=torch.nn.functional.relu(self.linear_ae1(torch.cat((x_conv, x_tag, self.domain_embedding(x)), dim=2) ) ) # shape = [batch_size, 83, 20]
        x_logit=self.linear_ae2(x_logit)
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

def valid_loss(model, valid_X, valid_X_tag, valid_y, CUDA=False, crf=False, tag=False):
    """calculate loss"""
    # set to evaluation mode (dropout layer will be treated differently)
    model.eval()
    losses=[]
    for batch in batch_generator(valid_X, valid_y, valid_X_tag, CUDA=CUDA, crf=crf, tag=tag):
        batch_valid_X, batch_valid_y, batch_valid_X_len, batch_valid_X_mask, batch_valid_X_tag = batch
        loss=model(batch_valid_X, batch_valid_X_len, batch_valid_X_mask, batch_valid_X_tag, batch_valid_y)  # loss for mini-batch
        losses.append(loss.data[0])
    # set back to train mode
    model.train()
    return sum(losses)/len(losses)  # average loss for mini-batch

def train(train_X, train_X_tag, train_y, valid_X, valid_X_tag, valid_y, model, model_fn, optimizer, parameters, epochs=200, batch_size=128, CUDA=False, crf=False, tag=False):
    best_loss=float("inf") 
    valid_history=[]
    train_history=[]
    for epoch in range(epochs):
        # iterate all batches and do grad descent
        for batch in batch_generator(train_X, train_y, train_X_tag, batch_size, CUDA=CUDA, crf=crf, tag=tag):
            batch_train_X, batch_train_y, batch_train_X_len, batch_train_X_mask, batch_train_X_tag=batch
            loss=model(batch_train_X, batch_train_X_len, batch_train_X_mask, batch_train_X_tag, batch_train_y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(parameters, 1.)
            optimizer.step()
        # train loss
        loss=valid_loss(model, train_X, train_X_tag, train_y, CUDA=CUDA, crf=crf, tag=tag)
        train_history.append(loss)
        # valid loss, print
        loss=valid_loss(model, valid_X, valid_X_tag, valid_y, CUDA=CUDA, crf=crf, tag=tag)
        valid_history.append(loss)
        print("Epoch: %d, Loss: %f" %(epoch,loss))        
        # save to a model file every time the new loss gets smaller than best record
        if loss<best_loss:
            best_loss=loss
            torch.save(model, model_fn)
        # shuffle train data
        shuffle_idx=np.random.permutation(len(train_X) )
        train_X=train_X[shuffle_idx]
        train_y=train_y[shuffle_idx]
        if tag:
            train_X_tag=train_X_tag[shuffle_idx,:]
    model=torch.load(model_fn) 
    return train_history, valid_history

def run(domain, data_dir, model_dir, valid_split, runs, epochs, lr, dropout, CUDA=False, batch_size=128, crf=False, tag=False):
    gen_emb=np.load(data_dir+"gen.vec.npy")
    domain_emb=np.load(data_dir+domain+"_emb.vec.npy")
    ae_data=np.load(data_dir+domain+"Train.npz")
    
    valid_X=ae_data['train_X'][-valid_split:]
    train_X=ae_data['train_X'][:-valid_split]
    if tag:
        valid_X_tag=ae_data['train_X_tag'][-valid_split:]
        train_X_tag=ae_data['train_X_tag'][:-valid_split]
    else:
        valid_X_tag, train_X_tag = None, None
    
    valid_y=ae_data['train_y'][-valid_split:]    
    train_y=ae_data['train_y'][:-valid_split]

    """Model will run args.runs (5) times and optimal model for each run will be saved"""
    for r in range(runs):
        print("Run: %d" %r)
        model=Model(gen_emb, domain_emb, 3, dropout=dropout, crf=crf, tag=tag)
        if CUDA:
            model.cuda()
        parameters = [p for p in model.parameters() if p.requires_grad]  # all parameters in the model that requires grad
        optimizer=torch.optim.Adam(parameters, lr=lr)
        train_history, valid_history=train(train_X, train_X_tag, train_y, valid_X, valid_X_tag, valid_y, model, model_dir+domain+str(r), optimizer, parameters, epochs, CUDA=CUDA, crf=crf, tag=tag)
        # Saving the loss history:
        with open(model_dir+domain+str(r)+'_info.pkl', 'wb') as f: 
            pickle.dump([train_history, valid_history], f)  # use pickle.load(f) for getting history
    
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, default="model/")
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=200) 
parser.add_argument('--runs', type=int, default=5)
parser.add_argument('--domain', type=str, default="restaurant")
parser.add_argument('--data_dir', type=str, default="data/prep_data/")
parser.add_argument('--valid', type=int, default=150) #number of validation data.
parser.add_argument('--lr', type=float, default=0.0001) 
parser.add_argument('--dropout', type=float, default=0.55) 
parser.add_argument('--crf', type=bool, default=False) 
parser.add_argument('--PoStag', type=bool, default=True) 

args = parser.parse_args()

run(args.domain, args.data_dir, args.model_dir, args.valid, args.runs, args.epochs, args.lr, args.dropout, CUDA, args.batch_size, args.crf, args.PoStag)

