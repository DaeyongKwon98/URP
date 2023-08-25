# -*- coding: utf-8 -*-
import fire
from tqdm import tqdm

from arena_util import load_json
from arena_util import write_json
from arena_util import remove_seen
from arena_util import most_popular


class MostPopular:
    def _generate_answers(self, train, questions):
        _, song_mp = most_popular(train, "songs", 200) # song_mp: 제일 많이 나온 200개의 songs의 list
        _, tag_mp = most_popular(train, "tags", 100) # tag_mp: 제일 많이 나온 100개의 tags의 list

        answers = []

        for q in tqdm(questions):
            answers.append({
                "id": q["id"],
                "songs": remove_seen(q["songs"], song_mp)[:100], # 제일 많이 나온 songs, tags중 중복만 제외하고 정답으로 제출
                "tags": remove_seen(q["tags"], tag_mp)[:10],
            })

        return answers

    def run(self, train_fname, question_fname):
        print("Loading train file...")
        train = load_json(train_fname)

        print("Loading question file...")
        questions = load_json(question_fname)

        print("Writing answers...")
        answers = self._generate_answers(train, questions)
        write_json(answers, "results/results.json")


if __name__ == "__main__":
    fire.Fire(MostPopular)
