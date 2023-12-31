# -*- coding: utf-8 -*-
import os
import json
import struct

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"]="1" #"0,1,2,3"

import random as rnd
from scipy import sparse
from lightfm import LightFM

import numpy as np

rnd.seed(0)

def save(keys, feats, out_fname): # keys와 feats를 binary file로 저장
    ''' Method used to save a list of element ids and the corresponing vectos
    '''
    feats = np.array(feats, dtype=np.float32)
    with open(out_fname + '.tmp', 'wb') as fout: # .tmp: 임시 파일, wb: write binary
        fout.write(b' '.join([k.encode() for k in keys])) # encode로 key를 binary로 변환
        fout.write(b'\n')
        R, C = feats.shape
        fout.write(struct.pack('qq', *(R, C))) # number of rows, columns를 binary representation으로 pack하여 write
        fout.write(feats.tostring())
    os.rename(out_fname + '.tmp', out_fname)


def load_feats(feat_fname, meta_only=False, nrz=False): # binary file로부터 data를 읽어옴
    ''' Method used to load element ids and the corresponing vectos
    '''
    with open(feat_fname, 'rb') as fin: # rb: read binary
        keys = fin.readline().strip().split()
        R, C = struct.unpack('qq', fin.read(16)) # 다음 16bytes 읽어서 R,C
        if meta_only: # actual data 말고 key와 data shape만 return
            return keys, (R, C)
        feat = np.fromstring(fin.read(), count=R * C, dtype=np.float32) # 나머지 file을 읽어서 numpy array로 만든다 (1D)
        feat = feat.reshape((R, C)) # 1D array를 RxC 2D array로 바꾼다
        if nrz: # normalization
            feat = feat / np.sqrt((feat ** 2).sum(-1) + 1e-8)[..., np.newaxis]
    return keys, feat

'''
def train_mf(impl_train_data, item_ids, user_ids, item_features_file, user_features_file, dims=200, epochs=50, max_sampled=10, lr=0.05):

    model = LightFM(loss='warp', no_components=dims, max_sampled=max_sampled, learning_rate=lr, random_state=42)
    model = model.fit(impl_train_data, epochs=epochs, num_threads=24)

    user_biases, user_embeddings = model.get_user_representations() # (M,), (M, 200)
    item_biases, item_embeddings = model.get_item_representations() # (N,), (N, 200)
    item_vec = np.concatenate((item_embeddings, np.reshape(item_biases, (1, -1)).T), axis=1) # (N,200) + (N,1) -> (N,201)
    user_vec = np.concatenate((user_embeddings, np.ones((1, user_biases.shape[0])).T), axis=1) # (M,200) + (M,1) -> (M,201)

    print("USER FEAT:", user_vec.shape)
    print("ITEM FEAT:", item_vec.shape)
    save(item_ids, item_vec, item_features_file)
    save(user_ids, user_vec, user_features_file)
    return user_vec, item_vec
'''

def load_data(train_file, model_folder):
    ''' In this method we create the splits for training and testing
    '''
    train_playlists = json.load(open(train_file, encoding='utf-8'))
    track_counter = {}
    test_playlists = {}
    for plalist in train_playlists: # playlist 내의 track count
        for track in plalist['songs']:
            if track not in track_counter:
                track_counter[track] = 0
            track_counter[track] += 1
        if rnd.uniform(0,1) < 0.1: # 10%의 playlist는 test dataset으로
            test_playlists[plalist['id']] = 1

    rows= {'train': [], '1': [], '5': [], '8': []}
    cols= {'train': [], '1': [], '5': [], '8': []}
    data= {'train': [], '1': [], '5': [], '8': []}
    track_ids = {'train': [], '1': [], '5': [], '8': []}
    dict_trac_ids = {'train': {}, '1': {}, '5': {}, '8': {}}
    playlists_ids = []
    test_tracks = {'1': [], '5': [], '8': []} # APC-test1,2,3 (cold-startness에 따른 구분)
    playlists_test = {'train': {}, '1': {}, '5': {}, '8': {}}
    playlists_test_cf = {'train': {}, '1': {}, '5': {}, '8': {}}
    for plalist in train_playlists:
        curr_playlist = []
        curr_playlist_tail = {'1': [], '5': [], '8': []}
        for track in plalist['songs']:
            if track_counter[track] > 9: # track의 등장횟수 10이상->training set
                curr_playlist.append(track)
            if 7 < track_counter[track] < 10: # 8
                curr_playlist_tail['8'].append(track)
                test_tracks['8'].append(track)
            if 4 < track_counter[track] < 8: # 5~7
                curr_playlist_tail['5'].append(track)
                test_tracks['5'].append(track)
            if 1 < track_counter[track] < 5: # 2~4
                curr_playlist_tail['1'].append(track)
                test_tracks['1'].append(track)
                
                #191481 -> 171440 -> 155,000
                #55772 -> 46384 -> 46069
                #17069 -> 17069 -> 17042

        if len(curr_playlist) > 4 : # playlist length 5 이상인 것만 취급
            for track in curr_playlist:
                if plalist['id'] in test_playlists:
                    if track not in dict_trac_ids['train']:
                        dict_trac_ids['train'][track] = len(track_ids['train'])
                        track_ids['train'].append(str(track))
                        cols['train'].append(dict_trac_ids['train'][track])
                        rows['train'].append(len(playlists_ids))
                        data['train'].append(1)
                    elif rnd.uniform(0,1) > 0.3:
                        cols['train'].append(dict_trac_ids['train'][track])
                        rows['train'].append(len(playlists_ids))
                        data['train'].append(1)
                    else:
                        if plalist['id'] not in playlists_test['train']:
                            playlists_test['train'][plalist['id']] = []
                        playlists_test['train'][plalist['id']].append(track)
                else:
                    if track not in dict_trac_ids['train']:
                        dict_trac_ids['train'][track] = len(track_ids['train'])
                        track_ids['train'].append(str(track))
                    cols['train'].append(dict_trac_ids['train'][track])
                    rows['train'].append(len(playlists_ids))
                    data['train'].append(1)
            for split in ['8', '5', '1']:
                for track in curr_playlist_tail[split]:
                    if track not in dict_trac_ids[split]:
                        dict_trac_ids[split][track] = len(track_ids[split])
                        track_ids[split].append(str(track))
                        cols[split].append(dict_trac_ids[split][track])
                        rows[split].append(len(playlists_ids))
                        data[split].append(1)

                    elif rnd.uniform(0,1) > 0.3:
                        cols[split].append(dict_trac_ids[split][track])
                        rows[split].append(len(playlists_ids))
                        data[split].append(1)
                    else:
                        if plalist['id'] not in playlists_test_cf[split]:
                            playlists_test_cf[split][plalist['id']] = []
                        playlists_test_cf[split][plalist['id']].append(track)
                    if plalist['id'] not in playlists_test[split]:
                        playlists_test[split][plalist['id']] = []
                    playlists_test[split][plalist['id']].append(track)
            playlists_ids.append(str(plalist['id']))
    
    # Now we save all the splits
    for split in ['train', '8', '5', '1']:
        if split == 'train':
            tr_ids = track_ids['train']
            curr_data = data['train']
            curr_rows = rows['train']
            curr_cols = cols['train']
        else:
            tr_ids =  track_ids['train'] +  track_ids[split]
            curr_data = data['train'] + data[split]
            curr_rows = rows['train'] + rows[split]
            total_train = len(track_ids['train'])
            curr_cols= cols['train'] + [i+total_train for i in cols[split]]
        train_coo = sparse.coo_matrix((curr_data, (curr_rows, curr_cols)), dtype=np.float32) # train_coo: playlist by track binary sparse matrix
        sparse.save_npz(os.path.join(model_folder, 'train_cf_{}.npz'.format(split)), train_coo)
        json.dump(playlists_test[split], open(os.path.join(model_folder, 'test_playlists_{}.json'.format(split)), 'w')) # json.dump: python object를 json으로 저장하기
        if split != 'train':
            json.dump(playlists_test_cf[split], open(os.path.join(model_folder, 'test_cf_playlists_{}.json'.format(split)), 'w'))
        json.dump(tr_ids, open(os.path.join(model_folder, 'track_ids_{}.json'.format(split)), 'w'))
        json.dump(playlists_ids, open(os.path.join(model_folder, 'playlists_ids_{}.json'.format(split)), 'w'))
    
    lists = test_tracks.values()
    for l in lists:
        print(len(set(l)))

if __name__ == '__main__':

    dims = "300"
    epochs = '50'
    max_sampled = '40'
    lr = '0.05'
    # Specify the location of the dataset:
    train_file = "data/train.json"
    model_folder = 'models_split'

    # Create the splits for training and testing
    load_data(train_file, model_folder)

    for split in ['train', '8', '5', '1']:
        train_data = sparse.load_npz(os.path.join(model_folder, 'train_cf_{}.npz'.format(split))).tocsr()

        tracks_ids = json.load(open(os.path.join(model_folder, 'track_ids_{}.json'.format(split)), 'r'))
        playlists_ids = json.load(open(os.path.join(model_folder, 'playlists_ids_{}.json'.format(split)), 'r'))

        item_features_file = os.path.join(model_folder, 'cf_item_{}_{}.feats'.format(dims, split))
        user_features_file = os.path.join(model_folder, 'cf_playlist_{}_{}.feats'.format(dims, split))

        print ("Start training", split)
        
        '''
        # Train the MF model using the training set, also we train the MF model for the evaluations set to measure the upper bound
        playlists_vects, track_orig_vects = train_mf(train_data,  tracks_ids, playlists_ids, item_features_file, user_features_file,  dims=int(dims),
                                                    epochs=int(epochs), max_sampled=int(max_sampled), lr=float(lr))
        print ("Finished training")
        '''