# -*- coding: utf-8 -*-
import fire
import numpy as np

from arena_util import load_json


class ArenaEvaluator:
    def _idcg(self, l):
        return sum((1.0 / np.log(i + 2) for i in range(l)))

    def __init__(self):
        self._idcgs = [self._idcg(i) for i in range(101)]

    def _ndcg(self, gt, rec, k):
        gt = gt[:k]
        rec = rec[:k]
        
        dcg = 0.0
        for i, r in enumerate(rec):
            if r in gt:
                dcg += 1.0 / np.log(i + 2)

        return dcg / self._idcgs[len(gt)]

    ###
    def _ap(self, gt, rec, k):
        gt = gt[:k]
        rec = rec[:k]
        
        precision = []
        p = 0.0
        true_count = 0
        for i in range(1,len(rec)+1):
            if rec[i-1] in gt:
                true_count += 1
                p = p*(i-1)/i+1/i
                precision.append(p)
            else:
                p = p*(i-1)/i    
        
        if true_count == 0:
            return 0
        
        ap = sum(precision)/len(precision)

        return ap
    ###
    
    def _eval(self, gt_fname, rec_fname):
        gt_playlists = load_json(gt_fname)
        gt_dict = {g["id"]: g for g in gt_playlists} # gt_dict: {playlistid: playlist_dictionary}
        rec_playlists = load_json(rec_fname)

        gt_ids = set([g["id"] for g in gt_playlists])
        rec_ids = set([r["id"] for r in rec_playlists])

        if gt_ids != rec_ids:
            raise Exception("결과의 플레이리스트 수가 올바르지 않습니다.")

        rec_song_counts = [len(p["songs"]) for p in rec_playlists]
        rec_tag_counts = [len(p["tags"]) for p in rec_playlists]

        if set(rec_song_counts) != set([100]):
            raise Exception("추천 곡 결과의 개수가 맞지 않습니다.")

        if set(rec_tag_counts) != set([10]):
            raise Exception("추천 태그 결과의 개수가 맞지 않습니다.")

        rec_unique_song_counts = [len(set(p["songs"])) for p in rec_playlists]
        rec_unique_tag_counts = [len(set(p["tags"])) for p in rec_playlists]

        if set(rec_unique_song_counts) != set([100]):
            raise Exception("한 플레이리스트에 중복된 곡 추천은 허용되지 않습니다.")

        if set(rec_unique_tag_counts) != set([10]):
            raise Exception("한 플레이리스트에 중복된 태그 추천은 허용되지 않습니다.")

        music_ndcg = 0.0
        tag_ndcg = 0.0

        ndcg_k = 10
        
        for rec in rec_playlists: # rec: {id:1, songs:[s1,s2], tags:[t1,t2]}
            gt = gt_dict[rec["id"]] # gt: id별 dictionary
            music_ndcg += self._ndcg(gt["songs"], rec["songs"][:100],ndcg_k) # gt["songs"]: 실제 playlist의 songs list
            tag_ndcg += self._ndcg(gt["tags"], rec["tags"][:10],ndcg_k)

        music_ndcg = music_ndcg / len(rec_playlists)
        tag_ndcg = tag_ndcg / len(rec_playlists)
        #score = music_ndcg * 0.85 + tag_ndcg * 0.15

        ###
        music_map = 0.0
        tag_map = 0.0
        
        map_k = 10
        
        for rec in rec_playlists:
            gt = gt_dict[rec["id"]]
            music_map += self._ap(gt["songs"], rec["songs"][:100],map_k)
            tag_map += self._ap(gt["tags"], rec["tags"][:10],map_k)

        music_map = music_map / len(rec_playlists)
        tag_map = tag_map / len(rec_playlists)
        #map_score = music_map * 0.85 + tag_map * 0.15
        
        ###
        
        #return music_ndcg, tag_ndcg, music_map, tag_map
        return music_ndcg, music_map

    def evaluate(self, gt_fname, rec_fname):
        try:
            #music_ndcg, tag_ndcg, music_map, tag_map = self._eval(gt_fname, rec_fname)
            music_ndcg, music_map = self._eval(gt_fname, rec_fname)
            print(f"Music nDCG: {music_ndcg:.6}")
            #print(f"Tag nDCG: {tag_ndcg:.6}")
            #print(f"nDCG Score: {score:.6}")
            print(f"Music MAP: {music_map:.6}")
            #print(f"Tag MAP: {tag_map:.6}")
            #print(f"MAP Score: {map_score:.6}")
        except Exception as e:
            print(e)


if __name__ == "__main__":
    fire.Fire(ArenaEvaluator)
