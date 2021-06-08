from os import path
import pickle

from train_IEMOCAP import get_IEMOCAP_loaders


path = './IEMOCAP_features/IEMOCAP_features_raw.pkl'
# x = pickle.load(open(path, 'rb'), encoding='latin1')

videoIDs, videoSpeakers, videoLabels, videoText, videoAudio, videoVisual, videoSentence, trainVid, testVid = pickle.load(open(path, 'rb'), encoding='latin1')


train_loader, valid_loader, test_loader = get_IEMOCAP_loaders('./IEMOCAP_features/IEMOCAP_features_raw.pkl', valid=0.0, batch_size=32, num_workers=2)
        