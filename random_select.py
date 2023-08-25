# Random selection

# -*- coding: utf-8 -*-
import random
import fire
from tqdm import tqdm

from arena_util import load_json
from arena_util import write_json
from arena_util import remove_seen
from arena_util import random_select

class RandomSelect:
    def _generate_answers(self, train, questions):
        song_list = random_select(train, "songs", 200)
        tag_list = random_select(train, "tags", 100)
        
        answers = []

        for q in tqdm(questions):
            answers.append({
                "id": q["id"],
                "songs": remove_seen(q["songs"], song_list)[:100],
                "tags": remove_seen(q["tags"], tag_list)[:10],
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
    fire.Fire(RandomSelect)
