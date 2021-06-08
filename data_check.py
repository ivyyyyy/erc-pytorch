from os import path
import pickle
import torch
from train_IEMOCAP import get_IEMOCAP_loaders

path = './IEMOCAP_features/IEMOCAP_features_raw.pkl'

# videoIDs: 每个video编号及其包含的utterance编号 {video编号:[utterance编号]} 
# videoSpeakers: {video编号:[M or F]}
# videoLabels: {video编号:[0~5]}
# videoText: {video编号: 100维特征向量}
# videoAudio: {video编号: 100维特征向量}
# videoVisual: {video编号: 100维特征向量}
# videoSentence: {video编号: ['utterance自然语言]}
# trainVid: {120个video编号}
# trainVid: {31个video编号}
videoIDs, videoSpeakers, videoLabels, videoText, videoAudio, \
    videoVisual, videoSentence, trainVid, testVid = pickle.load(open(path, 'rb'), encoding='latin1')

train_keys = [x for x in trainVid]
test_keys = [x for x in testVid]

train_len = len(train_keys)
test_len = len(test_keys)

vid = train_keys[0]
# print(vid)
# print(videoText[vid][0])
# print(len(videoText[vid][0]))
# print(videoSpeakers[vid])
# print(len(videoSpeakers[vid]))
# print(videoLabels[vid])
# print(len(videoLabels[vid]))
# print(videoSentence[vid])
# print(len(videoSentence[vid]))

train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(path, valid=0.0, batch_size=32)
for data in train_loader:
    # textf, visuf, acouf, qmask, umask, label = data[:-1]
    print(len(data[:-1]))
    print('------------')
