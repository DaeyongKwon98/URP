# -*- coding: utf-8 -*-
import fire
from tqdm import tqdm
import numpy as np
import sklearn
from sklearn.decomposition import TruncatedSVD

from arena_util import load_json
from arena_util import write_json
from arena_util import remove_seen
from arena_util import most_popular


class MFmax:
    def _generate_answers(self, song_meta_json, train, questions):
        ###
        num_playlist = len(train)
        num_songs = len(song_meta_json)
        num_val = len(questions)

        # binary playlist_by_track 만들기 (train + val)
        playlist_by_track = np.zeros((num_playlist+num_val,num_songs), dtype=int)

        for i, row in enumerate(train):
            songs = row["songs"]
            for songid in songs:
                if songid < num_songs:
                    playlist_by_track[i][songid] = 1

        for i, row in enumerate(questions):
            songs = row["songs"]
            for songid in songs:
                if songid < num_songs:
                    playlist_by_track[num_playlist+i][songid] = 1
        
        # playlist_by_track에 SVD로 reconstructed_matrix 만들기
        svd = TruncatedSVD(n_components=128)
        tsvd = svd.fit_transform(playlist_by_track)
        reconstructed_matrix = np.dot(tsvd, svd.components_)

        # 기존의 1인 값들 없애기
        one_index = [np.where(row > 0)[0] for row in playlist_by_track]

        mask = np.zeros_like(reconstructed_matrix, dtype=bool)

        for i, index in enumerate(one_index):
            mask[i, index] = True
            
        reconstructed_matrix[mask] = 0

        # 남은 값들중 TOP 100 index(songid) list
        largest_indices = np.argsort(-reconstructed_matrix, axis=1)[:, :100]
        ###
             
        answers = []
        i = len(train)
        for q in tqdm(questions):
            answers.append({
                "id": q["id"],
                "songs": list(largest_indices[i]),
                "tags": [_ for _ in range(10)],
            })
            i+=1

        return answers

    def run(self, song_meta_fname, train_fname, question_fname):
        print("Loading song meta...")
        song_meta_json = load_json(song_meta_fname)

        print("Loading train file...")
        train_data = load_json(train_fname)

        print("Loading question file...")
        questions = load_json(question_fname)

        print("Writing answers...")
        answers = self._generate_answers(song_meta_json, train_data, questions)
        write_json(answers, "results/results.json")


if __name__ == "__main__":
    fire.Fire(MFmax)
