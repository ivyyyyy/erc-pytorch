import numpy as np
np.random.seed(1234)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim

import argparse
import time
import pickle

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score,\
                        classification_report, precision_recall_fscore_support

from model import BiModel, BiModelCRF, Model, MaskedNLLLoss, WithUserSpecificCRF
from dataloader import IEMOCAPDataset

def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])

def get_IEMOCAP_loaders(path, batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = IEMOCAPDataset(path=path)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset(path=path, train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader

def train_loop(model, loss_function, dataloader, epoch, optimizer=None, train=False):
    
    losses = []
    preds = []
    labels = []
    masks = []
    alphas, alphas_f, alphas_b, vids = [], [], [], []
    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()
    for data in dataloader:
        if train:
            optimizer.zero_grad()
        # import ipdb;ipdb.set_trace()
        textf, visuf, acouf, qmask, umask, label =\
                [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        labels_intput = label.T # seq_len, batch
        #log_prob = model(torch.cat((textf,acouf,visuf),dim=-1), qmask,umask,att2=True) # seq_len, batch, n_classes
        log_prob, alpha, alpha_f, alpha_b, loss = model(textf, qmask, umask, labels_intput, att2=True) # seq_len, batch, n_classes
        lp_ = log_prob.transpose(0,1).contiguous().view(-1,log_prob.size()[2]) # batch*seq_len, n_classes
        labels_ = label.view(-1)
        # labels_intput = label.view(log_prob.size()[0],-1) # seq_len, batch
        # umask_ = umask.T.byte()

        # loss = model.crf(log_prob, labels_intput, umask_)
        # loss = loss_function(lp_, labels_, umask)

        pred_ = torch.ones_like(label)
        qmask_1, qmask_2 = qmask[:, :, 0], qmask[:, :, 1]

        log_prob_1 = torch.zeros_like(log_prob)
        log_prob_2 = torch.zeros_like(log_prob)

        umask_1 = torch.zeros_like(umask)
        umask_2 = torch.zeros_like(umask)

        # label_1 = torch.zeros_like(label)
        # label_2 = torch.zeros_like(label)

        for x, xx in enumerate(qmask_1.T):
            count1=0
            count2=0
            for y, yy in enumerate(xx):
                if yy:
                    log_prob_1[count1,x] = log_prob[y,x]
                    umask_1[x,count1] = 1
                    # label_1[count1, x] = label[y,x]
                    count1+=1
                else:
                    log_prob_2[count2, x] = log_prob[y,x]
                    umask_2[x, count2] =1
                    # label_2[count2, x] = label[y,x]
                    count2+=1

        pred_1 = torch.Tensor(model.crf_1.decode(log_prob_1, umask_1.T.bool())).view(-1)
        pred_2 = torch.Tensor(model.crf_2.decode(log_prob_2, umask_1.T.bool())).view(-1)

        pred_.reshape(-1)[qmask_1.bool().T.reshape(-1)] = pred_1
        pred_.reshape(-1)[qmask_2.bool().T.reshape(-1)] = pred_2
        # pred_ = torch.argmax(lp_,1) # batch*seq_len
        
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())    

        losses.append(loss.item()*masks[-1].sum())
        if train:
            loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()
        else:
            alphas += alpha
            alphas_f += alpha_f
            alphas_b += alpha_b
            vids += data[-1]

    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks  = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'),[]

    avg_loss = round(np.sum(losses)/np.sum(masks),4)
    avg_accuracy = round(accuracy_score(labels,preds,sample_weight=masks)*100,2)
    avg_fscore = round(f1_score(labels,preds,sample_weight=masks,average='weighted')*100,2)
    return avg_loss, avg_accuracy, labels, preds, masks,avg_fscore, [alphas, alphas_f, alphas_b, vids]

def train_or_eval_model(model, loss_function, dataloader, epoch, optimizer=None, train=False):
    
    losses = []
    preds = []
    labels = []
    masks = []
    alphas, alphas_f, alphas_b, vids = [], [], [], []
    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()
    for data in dataloader:
        if train:
            optimizer.zero_grad()
        # import ipdb;ipdb.set_trace()
        textf, visuf, acouf, qmask, umask, label =\
                [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        labels_intput = label.T # seq_len, batch
        #log_prob = model(torch.cat((textf,acouf,visuf),dim=-1), qmask,umask,att2=True) # seq_len, batch, n_classes
        log_prob, alpha, alpha_f, alpha_b, loss = model(textf, qmask, umask, labels_intput, att2=True) # seq_len, batch, n_classes
        lp_ = log_prob.transpose(0,1).contiguous().view(-1,log_prob.size()[2]) # batch*seq_len, n_classes
        labels_ = label.view(-1)
        # labels_intput = label.view(log_prob.size()[0],-1) # seq_len, batch
        # umask_ = umask.T.byte()

        # loss = model.crf(log_prob, labels_intput, umask_)
        # loss = loss_function(lp_, labels_, umask)

        pred_ = torch.Tensor(model.crf.decode(log_prob)).view(-1)
        # pred_ = torch.argmax(lp_,1) # batch*seq_len
        
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())
        

        losses.append(loss.item()*masks[-1].sum())
        if train:
            loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()
        else:
            alphas += alpha
            alphas_f += alpha_f
            alphas_b += alpha_b
            vids += data[-1]

    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks  = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'),[]

    avg_loss = round(np.sum(losses)/np.sum(masks),4)
    avg_accuracy = round(accuracy_score(labels,preds,sample_weight=masks)*100,2)
    avg_fscore = round(f1_score(labels,preds,sample_weight=masks,average='weighted')*100,2)
    return avg_loss, avg_accuracy, labels, preds, masks,avg_fscore, [alphas, alphas_f, alphas_b, vids]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2',
                        help='L2 regularization weight')
    parser.add_argument('--rec-dropout', type=float, default=0.1,
                        metavar='rec_dropout', help='rec_dropout rate')
    parser.add_argument('--dropout', type=float, default=0.1, metavar='dropout',
                        help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=30, metavar='BS',
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=60, metavar='E',
                        help='number of epochs')
    parser.add_argument('--class-weight', action='store_true', default=True,
                        help='class weight')
    parser.add_argument('--active-listener', action='store_true', default=False,
                        help='active listener')
    parser.add_argument('--attention', default='general', help='Attention type')
    parser.add_argument('--tensorboard', action='store_true', default=False,
                        help='Enables tensorboard log')
    args = parser.parse_args()

    print(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    if args.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()

    batch_size = args.batch_size
    n_classes  = 6
    cuda       = args.cuda
    n_epochs   = args.epochs

    D_m = 100 # utterance representation
    D_g = 500 # global state
    D_p = 500 # patry state
    D_e = 300 # emotion state
    D_h = 300 # hidden state

    D_a = 100 # concat attention

    model = BiModel(D_m, D_g, D_p, D_e, D_h,
                    n_classes=n_classes,
                    listener_state=args.active_listener,
                    context_attention=args.attention,
                    dropout_rec=args.rec_dropout,
                    dropout=args.dropout)
    if cuda:
        model.cuda()
    loss_weights = torch.FloatTensor([
                                        1/0.086747,
                                        1/0.144406,
                                        1/0.227883,
                                        1/0.160585,
                                        1/0.127711,
                                        1/0.252668,
                                        ])
    if args.class_weight:
        loss_function  = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
    else:
        loss_function = MaskedNLLLoss()


    # crf_params = list(map(id, model.crf_1.parameters())) + list(map(id, model.crf_2.parameters()))
    # base_params = filter(lambda p: id(p) not in crf_params, model.parameters())
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.l2)

    # optimizer = optim.Adam([
    #                         {'params': base_params},
    #                         {'params': model.crf_2.parameters(), 'lr':args.lr * 10000},
    #                         {'params': model.crf_1.parameters(), 'lr':args.lr * 10000}
    #                     ], lr=args.lr, weight_decay=args.l2)

    train_loader, valid_loader, test_loader =\
            get_IEMOCAP_loaders('./IEMOCAP_features/IEMOCAP_features_raw.pkl',
                                valid=0.0,
                                batch_size=batch_size,
                                num_workers=2)

    best_loss, best_label, best_pred, best_mask = None, None, None, None

    for e in range(n_epochs):
        start_time = time.time()
        train_loss, train_acc, _,_,_,train_fscore,_= train_or_eval_model(model, loss_function,
                                               train_loader, e, optimizer, True)
        valid_loss, valid_acc, _,_,_,val_fscore,_= train_or_eval_model(model, loss_function, valid_loader, e)
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, attentions = train_or_eval_model(model, loss_function, test_loader, e)

        if best_loss == None or best_loss > test_loss:
            best_loss, best_label, best_pred, best_mask, best_attn =\
                    test_loss, test_label, test_pred, test_mask, attentions

        if args.tensorboard:
            writer.add_scalar('test: accuracy/loss',test_acc/test_loss,e)
            writer.add_scalar('train: accuracy/loss',train_acc/train_loss,e)
        print('epoch {} train_loss {} train_acc {} train_fscore{} valid_loss {} valid_acc {} val_fscore{} test_loss {} test_acc {} test_fscore {} time {}'.\
                format(e+1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, val_fscore,\
                        test_loss, test_acc, test_fscore, round(time.time()-start_time,2)))
    if args.tensorboard:
        writer.close()

    print('Test performance..')
    print('Loss {} accuracy {}'.format(best_loss,
                                     round(accuracy_score(best_label,best_pred,sample_weight=best_mask)*100,2)))
    print(classification_report(best_label,best_pred,sample_weight=best_mask,digits=4))
    print(confusion_matrix(best_label,best_pred,sample_weight=best_mask))
    # with open('best_attention.p','wb') as f:
    #     pickle.dump(best_attn+[best_label,best_pred,best_mask],f)
