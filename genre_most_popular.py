# -*- coding: utf-8 -*-
from collections import Counter

import fire
from tqdm import tqdm

from arena_util import load_json
from arena_util import write_json
from arena_util import remove_seen
from arena_util import most_popular

class GenreMostPopular:
    def _song_mp_per_genre(self, song_meta, global_mp):
        res = {}
        for sid, song in song_meta.items():
            for genre in song['song_gn_gnr_basket']:
                res.setdefault(genre, []).append(sid) # res dictionary의 genre key를 가진 value list에 sid를 append. 없으면 empty list.
                # res: {genre1: [songid1,songid2,...], genre2: [songid3,songid4,...]}
        for genre, sids in res.items():
            res[genre] = Counter({k: global_mp.get(int(k), 0) for k in sids}) # res[genre]: {songid1: count, songid2: 0, ...}
            # res[genre]의 value인 song list를 counter로 변경 {songid: count}
            res[genre] = [k for k, v in res[genre].most_common(200)] # res[genre]의 가장 유명한 200개 songid list

        return res # res: {genre1: [song1,song2,...]} (유명한 순서대로)

    def _generate_answers(self, song_meta_json, train, questions):
        song_meta = {int(song["id"]): song for song in song_meta_json} # song_meta: {song_id: song(dictionary)}
        song_mp_counter, song_mp = most_popular(train, "songs", 200) # song_mp: 제일 많이 나온 200개의 songs의 list
        tag_mp_counter, tag_mp = most_popular(train, "tags", 100) # tag_mp: 제일 많이 나온 100개의 tags의 list
        song_mp_per_genre = self._song_mp_per_genre(song_meta, song_mp_counter) 
        # song_mp_counter: Counter({'song': count}), tag_mp_counter: Counter({'tag': count})
        answers = []
        for q in tqdm(questions):
            genre_counter = Counter()

            for sid in q["songs"]:
                for genre in song_meta[sid]["song_gn_gnr_basket"]:
                    genre_counter.update({genre: 1})

            top_genre = genre_counter.most_common(1) # 주어진 songs중 가장 많이 등장한 장르

            if len(top_genre) != 0:
                cur_songs = song_mp_per_genre[top_genre[0][0]] # 가장 많이 나온 genre의 song list
            else:
                cur_songs = song_mp # genre가 없으므로 most_popular처럼

            answers.append({
                "id": q["id"],
                "songs": remove_seen(q["songs"], cur_songs)[:100],
                "tags": remove_seen(q["tags"], tag_mp)[:10]
            })

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
    fire.Fire(GenreMostPopular)
