# Implement Paper's random baseline

import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import notebook
import json
from collections import Counter
from plotnine import *
import plotnine

# -*- coding: utf-8 -*-
import copy
import random

import fire

from arena_util import load_json
from arena_util import write_json

class ArenaSplitter:
    # discard playlists with less than 5 tracks
    def _discard(self, playlists):
        for playlist in playlists:
            if len(playlist["songs"]) < 5:
                playlists.remove(playlist)
        return playlists
    
    def _mapping_function(self, data, col1, col2):
        # 플레이리스트 아이디(col1)와 수록곡(col2) 추출
        plylst_song_map = data[[col1, col2]]

        # unnest col2
        plylst_song_map_unnest = np.dstack(
            (
                np.repeat(plylst_song_map[col1].values, list(map(len, plylst_song_map[col2]))),
                np.concatenate(plylst_song_map[col2].values)
            )
        )

        # unnested 데이터프레임 생성 : plylst_song_map
        plylst_song_map = pd.DataFrame(data = plylst_song_map_unnest[0], columns = plylst_song_map.columns)
        plylst_song_map[col1] = plylst_song_map[col1].astype(str)
        plylst_song_map[col2] = plylst_song_map[col2].astype(str)

        # unnest 객체 제거
        del plylst_song_map_unnest
        return plylst_song_map
    
    ''' track의 playlist 등장횟수에 따라 구분하는 코드
    def _split_data(self, playlists, more, less):
        training_pl = copy.deepcopy(playlists)
        test_pl = copy.deepcopy(playlist)
        
        for playlist in playlists:
            for song in playlist["songs"]:
                if song in set(more):
                    test_pl["songs"].remove(song)
                elif song in set(less):
                    training_pl["songs"].remove(song)
        
    

        return train, val
    '''
    
    def _split_data(self, playlists):
        tot = len(playlists)
        train = playlists[:int(tot*0.90)]
        val = playlists[int(tot*0.90):]

        return train, val

    def _mask(self, playlists, mask_cols, del_cols):
        q_pl = copy.deepcopy(playlists)
        a_pl = copy.deepcopy(playlists)

        for i in range(len(playlists)):
            for del_col in del_cols: # question generation위해 값 없애기
                q_pl[i][del_col] = []
                if del_col == 'songs':
                    a_pl[i][del_col] = a_pl[i][del_col][:100]
                elif del_col == 'tags':
                    a_pl[i][del_col] = a_pl[i][del_col][:10]

            for col in mask_cols:
                mask_len = len(playlists[i][col])
                mask = np.full(mask_len, False)
                mask[:mask_len//2] = True
                np.random.shuffle(mask)
                
                q_pl[i][col] = list(np.array(q_pl[i][col])[mask]) # 각 q_pl의 playlist column을 앞의 절반만 남기기
                a_pl[i][col] = list(np.array(a_pl[i][col])[np.invert(mask)]) # 정답은 뒤의 절반 남기기

        return q_pl, a_pl

    def _mask_data(self, playlists):
        playlists = copy.deepcopy(playlists)
        tot = len(playlists)
        song_only = playlists[:int(tot * 0.3)]
        song_and_tags = playlists[int(tot * 0.3):int(tot * 0.8)]
        tags_only = playlists[int(tot * 0.8):int(tot * 0.95)]
        title_only = playlists[int(tot * 0.95):]

        print(f"Total: {len(playlists)}, "
              f"Song only: {len(song_only)}, "
              f"Song & Tags: {len(song_and_tags)}, "
              f"Tags only: {len(tags_only)}, "
              f"Title only: {len(title_only)}")

        song_q, song_a = self._mask(song_only, ['songs'], ['tags']) # tags는 다 없애고 songs는 절반 남기기
        songtag_q, songtag_a = self._mask(song_and_tags, ['songs', 'tags'], []) # songs, tags 절반
        tag_q, tag_a = self._mask(tags_only, ['tags'], ['songs']) # songs 다 없애고 tags 절반 남기기
        title_q, title_a = self._mask(title_only, [], ['songs', 'tags']) # songs, tags 다 없애고 title만 남기기

        q = song_q + songtag_q + tag_q + title_q # 총 문제 모음
        a = song_a + songtag_a + tag_a + title_a # 총 정답 모음

        shuffle_indices = np.arange(len(q))
        np.random.shuffle(shuffle_indices)

        q = list(np.array(q)[shuffle_indices])
        a = list(np.array(a)[shuffle_indices])

        return q, a

    def run(self, fname):
        random.seed(777)

        print("Reading data...\n")
        playlists = load_json(fname)
        random.shuffle(playlists)
        print(f"Total playlists: {len(playlists)}")

        print("Discard playlists with less than 5 tracks...\n")
        playlists = self._discard(playlists)
        
        playlst_song_map = self._mapping_function(playlists, 'id', 'songs')
        agg = pd.DataFrame(playlst_song_map['songs'].value_counts()).reset_index()
        agg.columns = ['곡', '플레이리스트내의 등장횟수']
        
        morethan_10_songs = agg[agg['플레이리스트내의 등장횟수'] >= 10]['곡']
        lessthan_10_songs = agg[agg['플레이리스트내의 등장횟수'] < 10]['곡'] # type: pd.series
        
        print("Splitting data...")
        #train, val = self._split_data(playlists, morethan_10_songs, lessthan_10_songs)
        train, val = self._split_data(playlists)

        print("Original train...")
        write_json(train, "orig/train.json")
        print("Original val...")
        write_json(val, "orig/val.json")

        print("Masked val...")
        val_q, val_a = self._mask_data(val)
        write_json(val_q, "questions/val.json")
        write_json(val_a, "answers/val.json")


if __name__ == "__main__":
    fire.Fire(ArenaSplitter)