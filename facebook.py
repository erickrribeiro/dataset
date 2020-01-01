
import shutil
import os
import sys
import requests
import zipfile
import io
import pickle

import spacy
from tqdm import tqdm
from git import Repo
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords


print("Carregando spacy [english]...")
nlp_en = spacy.load('en', disable=['ner', 'parser', 'textcat'])

print("Carregando spacy [spanish]...")
nlp_es = spacy.load('es', disable=['ner', 'parser', 'textcat'])

print("Carregando stopwords [english]")
stop_words_en = set(stopwords.words('english'))

print("Carregando stopwords [spanish]")
stop_words_es = set(stopwords.words('spanish'))
BASE = os.path.dirname(os.path.realpath(__file__))


class FacebookDataset:
    def __init__(self):
        url = "https://fb.me/multilingual_task_oriented_data"

        if not os.path.exists(f"{BASE}/multilingual_task_oriented_data.zip"):
            # Streaming, so we can iterate over the response.
            r = requests.get(url, stream=True)

            # Total size in bytes.
            total_size = int(r.headers.get("content-length", 0))
            block_size = 1024  # 1 Kibibyte
            t = tqdm(total=total_size, unit="iB", unit_scale=True)
            with open(f"{BASE}/multilingual_task_oriented_data.zip", "wb") as f:
                for data in r.iter_content(block_size):
                    t.update(len(data))
                    f.write(data)
            t.close()

            if total_size != 0 and t.n != total_size:
                print("ERROR, something went wrong")
        print("Complete download.")

        z = zipfile.ZipFile(f"{BASE}/multilingual_task_oriented_data.zip")
        print("unzip files")
        z.extractall(f"{BASE}/multilingual_task_oriented_data")

    def parse(self):
        types = [
            ("valid", "en"),
            ("valid", "es"),
            ("train", "en"),
            ("train", "es"),
            ("test", "en"),
            ("test", "es"),
        ]

        for mode, lang in types:
            self.parse_(mode, lang)
            self.format_atis(mode, lang)
            self.remove_stopword(mode, lang)
            self.apply_stemmer(mode, lang)
            self.apply_lemmatizer(mode, lang)

        print("Finish FacebookDataset")

        if os.path.exists(f"{BASE}/fb.pickle"):
            return

        paths = [
            f"{BASE}/fb_en/valid/fb.valid.iob",
            f"{BASE}/fb_en/valid/fb.stopwords.valid.iob",
            f"{BASE}/fb_en/valid/fb.stem.valid.iob",
            f"{BASE}/fb_en/valid/fb.lemma.valid.iob",
            f"{BASE}/fb_en/test/fb.test.iob",
            f"{BASE}/fb_en/test/fb.stopwords.test.iob",
            f"{BASE}/fb_en/test/fb.stem.test.iob",
            f"{BASE}/fb_en/test/fb.lemma.test.iob",
            f"{BASE}/fb_en/train/fb.train.iob",
            f"{BASE}/fb_en/train/fb.stopwords.train.iob",
            f"{BASE}/fb_en/train/fb.stem.train.iob",
            f"{BASE}/fb_en/train/fb.lemma.train.iob",
            f"{BASE}/fb_es/valid/fb.valid.iob",
            f"{BASE}/fb_es/valid/fb.stopwords.valid.iob",
            f"{BASE}/fb_es/valid/fb.stem.valid.iob",
            f"{BASE}/fb_es/valid/fb.lemma.valid.iob",
            f"{BASE}/fb_es/test/fb.test.iob",
            f"{BASE}/fb_es/test/fb.stopwords.test.iob",
            f"{BASE}/fb_es/test/fb.stem.test.iob",
            f"{BASE}/fb_es/test/fb.lemma.test.iob",
            f"{BASE}/fb_es/train/fb.train.iob",
            f"{BASE}/fb_es/train/fb.stopwords.train.iob",
            f"{BASE}/fb_es/train/fb.stem.train.iob",
            f"{BASE}/fb_es/train/fb.lemma.train.iob",
        ]

        datas = {}
        for path in paths:
            datas[path] = self.statistic(path)

        pickle.dump(datas, open(f"{BASE}/fb.pickle", "wb"))

    def parse_(self, mode, lang):
        modes = {
            "valid": {
                "original": f"{BASE}/multilingual_task_oriented_data/{lang}/eval-{lang}.conllu",
                "dir": f"{BASE}/fb_{lang}/valid",
                "path_words": f"{BASE}/fb_{lang}/valid/seq.in",
                "path_slots": f"{BASE}/fb_{lang}/valid/seq.out",
                "path_intents": f"{BASE}/fb_{lang}/valid/label",
            },
            "train": {
                "original": f"{BASE}/multilingual_task_oriented_data/{lang}/train-{lang}.conllu",
                "dir": f"{BASE}/fb_{lang}/train",
                "path_words": f"{BASE}/fb_{lang}/train/seq.in",
                "path_slots": f"{BASE}/fb_{lang}/train/seq.out",
                "path_intents": f"{BASE}/fb_{lang}/train/label",
            },
            "test": {
                "original": f"{BASE}/multilingual_task_oriented_data/{lang}/test-{lang}.conllu",
                "dir": f"{BASE}/fb_{lang}/test",
                "path_words": f"{BASE}/fb_{lang}/test/seq.in",
                "path_slots": f"{BASE}/fb_{lang}/test/seq.out",
                "path_intents": f"{BASE}/fb_{lang}/test/label",
            }
        }

        mode = modes[mode]

        if not os.path.exists(mode["dir"]):
            os.makedirs(mode["dir"])

        mapped = {"NoLabel": "O"}

        with open(mode["original"]) as infile:
            lines = infile.read().split("\n\n")

            in_writer = open(mode["path_words"], "w")
            out_writer = open(mode["path_slots"], "w")
            label_writer = open(mode["path_intents"], "w")
            for line in lines:
                if not line:
                    continue
                data = line.split("\n")
                intent = data[1].split()[2]
                words = []
                slots = []
                for x in data[3:]:
                    _, word, _, slot = x.split("\t")
                    slot = mapped.get(slot, slot)

                    words.append(word)
                    slots.append(slot)

                words = " ".join(words)
                slots = " ".join(slots)

                in_writer.write(f"{words}\n")
                out_writer.write(f"{slots}\n")
                label_writer.write(f"{intent}\n")
            in_writer.close()
            out_writer.close()
            label_writer.close()

    def format_atis(self, mode, lang):
        modes = {
            "valid": {
                "in": f"{BASE}/fb_{lang}/valid/seq.in",
                "out": f"{BASE}/fb_{lang}/valid/seq.out",
                "label": f"{BASE}/fb_{lang}/valid/label",
                "format_atis": f"{BASE}/fb_{lang}/valid/fb.valid.iob",
            },
            "train": {
                "in": f"{BASE}/fb_{lang}/train/seq.in",
                "out": f"{BASE}/fb_{lang}/train/seq.out",
                "label": f"{BASE}/fb_{lang}/train/label",
                "format_atis": f"{BASE}/fb_{lang}/train/fb.train.iob",
            },
            "test": {
                "in": f"{BASE}/fb_{lang}/test/seq.in",
                "out": f"{BASE}/fb_{lang}/test/seq.out",
                "label": f"{BASE}/fb_{lang}/test/label",
                "format_atis": f"{BASE}/fb_{lang}/test/fb.test.iob",
            }
        }

        mode = modes[mode]
        if os.path.exists(mode["format_atis"]):
            return

        with open(mode["in"]) as infile:
            utterances = infile.readlines()

        with open(mode["label"]) as infile:
            intents = infile.readlines()

        with open(mode["out"]) as infile:
            slots = infile.readlines()

        with open(mode["format_atis"], "w") as outfile:
            for x, y, z in zip(utterances, slots, intents):
                words = ['BOS'] + x.split() + ['EOS']
                tags = ['O'] + y.split() + [z.strip()]

                words = " ".join(words)
                tags = " ".join(tags)
                outfile.write(f"{words}\t{tags}\n")

    def remove_stopword(self, mode, lang):
        modes = {
            "valid": {
                "origin": f"{BASE}/fb_{lang}/valid/fb.valid.iob",
                "destination": f"{BASE}/fb_{lang}/valid/fb.stopwords.valid.iob",
            },
            "train": {
                "origin": f"{BASE}/fb_{lang}/train/fb.train.iob",
                "destination": f"{BASE}/fb_{lang}/train/fb.stopwords.train.iob",
            },
            "test": {
                "origin": f"{BASE}/fb_{lang}/test/fb.test.iob",
                "destination": f"{BASE}/fb_{lang}/test/fb.stopwords.test.iob",
            }
        }

        mode = modes[mode]
        if os.path.exists(mode["destination"]):
            return

        replaced = {
            "i'd": "i",
            "i'll": "i",
            "what's": "what",
            "i'm": "i",
            "we're": "we",
            "i've": "i",
            "that's": "that",
            "what're": "what",
            "let's": "let",
            "american's": "american",
            "wednesday's": "wednesday",
            "delta's": "delta",
            "atlanta's": "atlanta",
            "sunday's": "sunday",
            "one's": "one",
            "york's": "york",
            "friday's": "friday"
        }

        if lang == "en":
            stop_words = stop_words_en
        elif lang == "es":
            stop_words = stop_words_en

        writer = open(mode["destination"], "w")
        with open(mode["origin"]) as infile:
            data = infile.readlines()
            for line in data:
                words, tags = line.split("\t")

                words = words.split()
                tags = tags.split()

                new_words = []
                new_tags = []
                for w, t in zip(words, tags):
                    w = replaced[w] if w in replaced else w
                    if w not in stop_words:
                        new_words += [w]
                        new_tags += [t]

                new_words = " ".join(new_words)
                new_tags = " ".join(new_tags)

                writer.write("{}\t{}\n".format(new_words, new_tags))
            writer.close()

    def apply_stemmer(self, mode, lang):
        modes = {
            "valid": {
                "origin": f"{BASE}/fb_{lang}/valid/fb.valid.iob",
                "destination": f"{BASE}/fb_{lang}/valid/fb.stem.valid.iob",
            },
            "train": {
                "origin": f"{BASE}/fb_{lang}/train/fb.train.iob",
                "destination": f"{BASE}/fb_{lang}/train/fb.stem.train.iob",
            },
            "test": {
                "origin": f"{BASE}/fb_{lang}/test/fb.test.iob",
                "destination": f"{BASE}/fb_{lang}/test/fb.stem.test.iob",
            }
        }

        mode = modes[mode]

        if os.path.exists(mode["destination"]):
            return

        replaced = {
            "i'd": "i",
            "i'll": "i",
            "what's": "what",
            "i'm": "i",
            "we're": "we",
            "i've": "i",
            "that's": "that",
            "what're": "what",
            "let's": "let",
            "american's": "american",
            "wednesday's": "wednesday",
            "delta's": "delta",
            "atlanta's": "atlanta",
            "sunday's": "sunday",
            "one's": "one",
            "york's": "york",
            "friday's": "friday"
        }

        if lang == "en":
            stemmer = SnowballStemmer('english')
        elif lang == "es":
            stemmer = SnowballStemmer('spanish')

        writer = open(mode["destination"], "w")
        with open(mode["origin"]) as infile:
            data = infile.readlines()

        for line in data:
            words, tags = line.split('\t')

            new_words = ["BOS"]
            for w in words.split()[1:-1]:
                w = replaced[w] if w in replaced else w
                new_words.append(stemmer.stem(w.lower()))
            new_words.append("EOS")

            words = " ".join(new_words)
            writer.write("{}\t{}".format(words, tags))
        writer.close()

    def apply_lemmatizer(self, mode, lang):
        modes = {
            "valid": {
                "origin": f"{BASE}/fb_{lang}/valid/fb.valid.iob",
                "destination": f"{BASE}/fb_{lang}/valid/fb.lemma.valid.iob",
            },
            "train": {
                "origin": f"{BASE}/fb_{lang}/train/fb.train.iob",
                "destination": f"{BASE}/fb_{lang}/train/fb.lemma.train.iob",
            },
            "test": {
                "origin": f"{BASE}/fb_{lang}/test/fb.test.iob",
                "destination": f"{BASE}/fb_{lang}/test/fb.lemma.test.iob",
            }
        }

        mode = modes[mode]

        if os.path.exists(mode["destination"]):
            return
        replaced = {
            "i'd": "i",
            "i'll": "i",
            "what's": "what",
            "i'm": "i",
            "we're": "we",
            "i've": "i",
            "that's": "that",
            "what're": "what",
            "let's": "let",
            "american's": "american",
            "wednesday's": "wednesday",
            "delta's": "delta",
            "atlanta's": "atlanta",
            "sunday's": "sunday",
            "one's": "one",
            "york's": "york",
            "friday's": "friday"
        }

        if lang == "en":
            nlp = nlp_en
        elif lang == "es":
            nlp = nlp_es

        def replace(w):
            return replaced[w] if w in replaced else w

        def parser(arr):
            def arr2text(n): return " ".join([w for w, t in n])

            text = arr2text(arr)
            doc = nlp(text)
            n = [(t.text, t.lemma_) for t in doc]

            try:
                pivot_n = 0
                pivot_x = 0
                result = []

                while pivot_x < len(arr):
                    check_size = len(n[pivot_n][0]) <= len(arr[pivot_x][0])
                    while check_size and n[pivot_n][0] in arr[pivot_x][0]:
                        w = n[pivot_n][0]
                        w_lema = n[pivot_n][1]
                        t = arr[pivot_x][1]

                        if w_lema in ['i', 'you', 'he', 'she', 'it', 'we', 'they']:
                            result.append(('-PRON-', t))
                        else:
                            result.append((w_lema, t))

                        #print(n[pivot_n][0]+": "+n[pivot_n][1], '->', arr[pivot_x][0], '->', arr[pivot_x][1])
                        pivot_n += 1

                    pivot_x += 1
            except IndexError:
                pass
            return result

        writer = open(mode["destination"], "w")
        with open(mode["origin"]) as infile:
            data = infile.readlines()
            for line in data:
                words, tags = line.split("\t")

                words = words.split()
                words = [replace(w) for w in words]
                tags = tags.split()

                tuples = []
                for w, t in zip(words, tags):
                    tuples.append((w, t))

                tuples = parser(tuples)

                two_lists = list(map(list, zip(*tuples)))
                words = " ".join(two_lists[0])
                tags = " ".join(two_lists[1])
                writer.write("{}\t{}\n".format(words, tags))
            writer.close()

    def statistic(self, path):
        print(f"Analizando {path}")
        with open(path) as infile:
            data = infile.readlines()

            samples = 0
            vocabulary = set()
            slots = set()

            for line in data:
                samples += 1
                words, tags = line.split("\t")

                words = set(words.strip().split()[1:-1])
                tags = set(tags.strip().split()[1:-1])

                vocabulary = vocabulary.union(words)
                slots = slots.union(tags)

            print("samples:", samples)
            print("words:", len(vocabulary))
            print("slots:", len(slots))
            return {
                "samples": samples,
                "words": len(vocabulary),
                "slots": len(slots)
            }


class SNIPSDataset:
    def __init__(self):
        if not os.path.exists(f"{BASE}/SF-ID-Network-For-NLU"):
            Repo.clone_from(
                "https://github.com/ZephyrChenzf/SF-ID-Network-For-NLU.git",
                f"{BASE}/SF-ID-Network-For-NLU")

        if not os.path.exists(f"{BASE}/snips"):
            shutil.copytree(
                f"{BASE}/SF-ID-Network-For-NLU/data/snips",
                f"{BASE}/snips")

    def parse(self):
        types = [
            "valid",
            "train",
            "test",
        ]

        for mode in types:
            self.format_atis(mode)
            self.remove_stopword(mode)
            self.apply_stemmer(mode)
            self.apply_lemmatizer(mode)

        print("Finish SNIPSDataset")
        if os.path.exists(f"{BASE}/snips.pickle"):
            return

        paths = [
            f"{BASE}/snips/valid/snips.valid.iob",
            f"{BASE}/snips/valid/snips.stopwords.valid.iob",
            f"{BASE}/snips/valid/snips.stem.valid.iob",
            f"{BASE}/snips/valid/snips.lemma.valid.iob",
            f"{BASE}/snips/test/snips.test.iob",
            f"{BASE}/snips/test/snips.stopwords.test.iob",
            f"{BASE}/snips/test/snips.stem.test.iob",
            f"{BASE}/snips/test/snips.lemma.test.iob",
            f"{BASE}/snips/train/snips.train.iob",
            f"{BASE}/snips/train/snips.stopwords.train.iob",
            f"{BASE}/snips/train/snips.stem.train.iob",
            f"{BASE}/snips/train/snips.lemma.train.iob",
        ]

        datas = {}
        for path in paths:
            datas[path] = self.statistic(path)

        pickle.dump(datas, open("snips.pickle", "wb"))

    def format_atis(self, mode):
        modes = {
            "valid": {
                "in": f"{BASE}/snips/valid/seq.in",
                "out": f"{BASE}/snips/valid/seq.out",
                "label": f"{BASE}/snips/valid/label",
                "format_atis": f"{BASE}/snips/valid/snips.valid.iob",
            },
            "train": {
                "in": f"{BASE}/snips/train/seq.in",
                "out": f"{BASE}/snips/train/seq.out",
                "label": f"{BASE}/snips/train/label",
                "format_atis": f"{BASE}/snips/train/snips.train.iob",
            },
            "test": {
                "in": f"{BASE}/snips/test/seq.in",
                "out": f"{BASE}/snips/test/seq.out",
                "label": f"{BASE}/snips/test/label",
                "format_atis": f"{BASE}/snips/test/snips.test.iob",
            }
        }

        mode = modes[mode]
        if os.path.exists(mode["format_atis"]):
            return

        with open(mode["in"]) as infile:
            utterances = infile.readlines()

        with open(mode["label"]) as infile:
            intents = infile.readlines()

        with open(mode["out"]) as infile:
            slots = infile.readlines()

        with open(mode["format_atis"], "w") as outfile:
            for x, y, z in zip(utterances, slots, intents):
                words = ['BOS'] + x.split() + ['EOS']
                tags = ['O'] + y.split() + [z.strip()]

                words = " ".join(words)
                tags = " ".join(tags)
                outfile.write(f"{words}\t{tags}\n")

    def remove_stopword(self, mode):
        modes = {
            "valid": {
                "origin": f"{BASE}/snips/valid/snips.valid.iob",
                "destination": f"{BASE}/snips/valid/snips.stopwords.valid.iob",
            },
            "train": {
                "origin": f"{BASE}/snips/train/snips.train.iob",
                "destination": f"{BASE}/snips/train/snips.stopwords.train.iob",
            },
            "test": {
                "origin": f"{BASE}/snips/test/snips.test.iob",
                "destination": f"{BASE}/snips/test/snips.stopwords.test.iob",
            }
        }

        mode = modes[mode]
        if os.path.exists(mode["destination"]):
            return

        replaced = {
            "i'd": "i",
            "i'll": "i",
            "what's": "what",
            "i'm": "i",
            "we're": "we",
            "i've": "i",
            "that's": "that",
            "what're": "what",
            "let's": "let",
            "american's": "american",
            "wednesday's": "wednesday",
            "delta's": "delta",
            "atlanta's": "atlanta",
            "sunday's": "sunday",
            "one's": "one",
            "york's": "york",
            "friday's": "friday"
        }

        stop_words = stop_words_en

        writer = open(mode["destination"], "w")
        with open(mode["origin"]) as infile:
            data = infile.readlines()
            for line in data:
                words, tags = line.split("\t")

                words = words.split()
                tags = tags.split()

                new_words = []
                new_tags = []
                for w, t in zip(words, tags):
                    w = replaced[w] if w in replaced else w
                    if w not in stop_words:
                        new_words += [w]
                        new_tags += [t]

                new_words = " ".join(new_words)
                new_tags = " ".join(new_tags)

                writer.write("{}\t{}\n".format(new_words, new_tags))
            writer.close()

    def apply_stemmer(self, mode):
        modes = {
            "valid": {
                "origin": f"{BASE}/snips/valid/snips.valid.iob",
                "destination": f"{BASE}/snips/valid/snips.stem.valid.iob",
            },
            "train": {
                "origin": f"{BASE}/snips/train/snips.train.iob",
                "destination": f"{BASE}/snips/train/snips.stem.train.iob",
            },
            "test": {
                "origin": f"{BASE}/snips/test/snips.test.iob",
                "destination": f"{BASE}/snips/test/snips.stem.test.iob",
            }
        }

        mode = modes[mode]

        if os.path.exists(mode["destination"]):
            return

        replaced = {
            "i'd": "i",
            "i'll": "i",
            "what's": "what",
            "i'm": "i",
            "we're": "we",
            "i've": "i",
            "that's": "that",
            "what're": "what",
            "let's": "let",
            "american's": "american",
            "wednesday's": "wednesday",
            "delta's": "delta",
            "atlanta's": "atlanta",
            "sunday's": "sunday",
            "one's": "one",
            "york's": "york",
            "friday's": "friday"
        }

        stemmer = SnowballStemmer('english')
        writer = open(mode["destination"], "w")
        with open(mode["origin"]) as infile:
            data = infile.readlines()

        for line in data:
            words, tags = line.split('\t')

            new_words = ["BOS"]
            for w in words.split()[1:-1]:
                w = replaced[w] if w in replaced else w
                new_words.append(stemmer.stem(w.lower()))
            new_words.append("EOS")

            words = " ".join(new_words)
            writer.write("{}\t{}".format(words, tags))
        writer.close()

    def apply_lemmatizer(self, mode):
        modes = {
            "valid": {
                "origin": f"{BASE}/snips/valid/snips.valid.iob",
                "destination": f"{BASE}/snips/valid/snips.lemma.valid.iob",
            },
            "train": {
                "origin": f"{BASE}/snips/train/snips.train.iob",
                "destination": f"{BASE}/snips/train/snips.lemma.train.iob",
            },
            "test": {
                "origin": f"{BASE}/snips/test/snips.test.iob",
                "destination": f"{BASE}/snips/test/snips.lemma.test.iob",
            }
        }

        mode = modes[mode]

        if os.path.exists(mode["destination"]):
            return
        replaced = {
            "i'd": "i",
            "i'll": "i",
            "what's": "what",
            "i'm": "i",
            "we're": "we",
            "i've": "i",
            "that's": "that",
            "what're": "what",
            "let's": "let",
            "american's": "american",
            "wednesday's": "wednesday",
            "delta's": "delta",
            "atlanta's": "atlanta",
            "sunday's": "sunday",
            "one's": "one",
            "york's": "york",
            "friday's": "friday"
        }

        nlp = nlp_en

        def replace(w):
            return replaced[w] if w in replaced else w

        def parser(arr):
            def arr2text(n): return " ".join([w for w, t in n])

            text = arr2text(arr)
            doc = nlp(text)
            n = [(t.text, t.lemma_) for t in doc]

            try:
                pivot_n = 0
                pivot_x = 0
                result = []

                while pivot_x < len(arr):
                    check_size = len(n[pivot_n][0]) <= len(arr[pivot_x][0])
                    while check_size and n[pivot_n][0] in arr[pivot_x][0]:
                        w = n[pivot_n][0]
                        w_lema = n[pivot_n][1]
                        t = arr[pivot_x][1]

                        if w_lema in ['i', 'you', 'he', 'she', 'it', 'we', 'they']:
                            result.append(('-PRON-', t))
                        else:
                            result.append((w_lema, t))

                        #print(n[pivot_n][0]+": "+n[pivot_n][1], '->', arr[pivot_x][0], '->', arr[pivot_x][1])
                        pivot_n += 1

                    pivot_x += 1
            except IndexError:
                pass
            return result

        writer = open(mode["destination"], "w")
        with open(mode["origin"]) as infile:
            data = infile.readlines()
            for line in data:
                words, tags = line.split("\t")

                words = words.split()
                words = [replace(w) for w in words]
                tags = tags.split()

                tuples = []
                for w, t in zip(words, tags):
                    tuples.append((w, t))

                tuples = parser(tuples)

                two_lists = list(map(list, zip(*tuples)))
                words = " ".join(two_lists[0])
                tags = " ".join(two_lists[1])
                writer.write("{}\t{}\n".format(words, tags))
            writer.close()

    def statistic(self, path):
        print(f"Analizando {path}")
        with open(path) as infile:
            data = infile.readlines()

            samples = 0
            vocabulary = set()
            slots = set()

            for line in data:
                samples += 1
                words, tags = line.split("\t")

                words = set(words.strip().split()[1:-1])
                tags = set(tags.strip().split()[1:-1])

                vocabulary = vocabulary.union(words)
                slots = slots.union(tags)

            print("samples:", samples)
            print("words:", len(vocabulary))
            print("slots:", len(slots))
            return {
                "samples": samples,
                "words": len(vocabulary),
                "slots": len(slots)
            }


class ATISDataset:
    def __init__(self):
        if not os.path.exists(f"{BASE}/SF-ID-Network-For-NLU"):
            Repo.clone_from(
                "https://github.com/ZephyrChenzf/SF-ID-Network-For-NLU.git",
                f"{BASE}/SF-ID-Network-For-NLU")

        if not os.path.exists(f"{BASE}/atis"):
            shutil.copytree(
                f"{BASE}/SF-ID-Network-For-NLU/data/atis",
                f"{BASE}/atis")

    def parse(self):
        types = [
            "valid",
            "train",
            "test",
        ]

        for mode in types:
            self.format_atis(mode)
            self.remove_stopword(mode)
            self.apply_stemmer(mode)
            self.apply_lemmatizer(mode)

        print("Finish ATISDataset")
        if os.path.exists(f"{BASE}/atis.pickle"):
            return

        paths = [
            f"{BASE}/atis/valid/atis.valid.iob",
            f"{BASE}/atis/valid/atis.stopwords.valid.iob",
            f"{BASE}/atis/valid/atis.stem.valid.iob",
            f"{BASE}/atis/valid/atis.lemma.valid.iob",
            f"{BASE}/atis/test/atis.test.iob",
            f"{BASE}/atis/test/atis.stopwords.test.iob",
            f"{BASE}/atis/test/atis.stem.test.iob",
            f"{BASE}/atis/test/atis.lemma.test.iob",
            f"{BASE}/atis/train/atis.train.iob",
            f"{BASE}/atis/train/atis.stopwords.train.iob",
            f"{BASE}/atis/train/atis.stem.train.iob",
            f"{BASE}/atis/train/atis.lemma.train.iob",
        ]

        datas = {}
        for path in paths:
            datas[path] = self.statistic(path)

        pickle.dump(datas, open("atis.pickle", "wb"))

    def format_atis(self, mode):
        modes = {
            "valid": {
                "in": f"{BASE}/atis/valid/seq.in",
                "out": f"{BASE}/atis/valid/seq.out",
                "label": f"{BASE}/atis/valid/label",
                "format_atis": f"{BASE}/atis/valid/atis.valid.iob",
            },
            "train": {
                "in": f"{BASE}/atis/train/seq.in",
                "out": f"{BASE}/atis/train/seq.out",
                "label": f"{BASE}/atis/train/label",
                "format_atis": f"{BASE}/atis/train/atis.train.iob",
            },
            "test": {
                "in": f"{BASE}/atis/test/seq.in",
                "out": f"{BASE}/atis/test/seq.out",
                "label": f"{BASE}/atis/test/label",
                "format_atis": f"{BASE}/atis/test/atis.test.iob",
            }
        }

        mode = modes[mode]
        if os.path.exists(mode["format_atis"]):
            return

        with open(mode["in"]) as infile:
            utterances = infile.readlines()

        with open(mode["label"]) as infile:
            intents = infile.readlines()

        with open(mode["out"]) as infile:
            slots = infile.readlines()

        with open(mode["format_atis"], "w") as outfile:
            for x, y, z in zip(utterances, slots, intents):
                words = ['BOS'] + x.split() + ['EOS']
                tags = ['O'] + y.split() + [z.strip()]

                words = " ".join(words)
                tags = " ".join(tags)
                outfile.write(f"{words}\t{tags}\n")

    def remove_stopword(self, mode):
        modes = {
            "valid": {
                "origin": f"{BASE}/atis/valid/atis.valid.iob",
                "destination": f"{BASE}/atis/valid/atis.stopwords.valid.iob",
            },
            "train": {
                "origin": f"{BASE}/atis/train/atis.train.iob",
                "destination": f"{BASE}/atis/train/atis.stopwords.train.iob",
            },
            "test": {
                "origin": f"{BASE}/atis/test/atis.test.iob",
                "destination": f"{BASE}/atis/test/atis.stopwords.test.iob",
            }
        }

        mode = modes[mode]
        if os.path.exists(mode["destination"]):
            return

        replaced = {
            "i'd": "i",
            "i'll": "i",
            "what's": "what",
            "i'm": "i",
            "we're": "we",
            "i've": "i",
            "that's": "that",
            "what're": "what",
            "let's": "let",
            "american's": "american",
            "wednesday's": "wednesday",
            "delta's": "delta",
            "atlanta's": "atlanta",
            "sunday's": "sunday",
            "one's": "one",
            "york's": "york",
            "friday's": "friday"
        }

        stop_words = stop_words_en

        writer = open(mode["destination"], "w")
        with open(mode["origin"]) as infile:
            data = infile.readlines()
            for line in data:
                words, tags = line.split("\t")

                words = words.split()
                tags = tags.split()

                new_words = []
                new_tags = []
                for w, t in zip(words, tags):
                    w = replaced[w] if w in replaced else w
                    if w not in stop_words:
                        new_words += [w]
                        new_tags += [t]

                new_words = " ".join(new_words)
                new_tags = " ".join(new_tags)

                writer.write("{}\t{}\n".format(new_words, new_tags))
            writer.close()

    def apply_stemmer(self, mode):
        modes = {
            "valid": {
                "origin": f"{BASE}/atis/valid/atis.valid.iob",
                "destination": f"{BASE}/atis/valid/atis.stem.valid.iob",
            },
            "train": {
                "origin": f"{BASE}/atis/train/atis.train.iob",
                "destination": f"{BASE}/atis/train/atis.stem.train.iob",
            },
            "test": {
                "origin": f"{BASE}/atis/test/atis.test.iob",
                "destination": f"{BASE}/atis/test/atis.stem.test.iob",
            }
        }

        mode = modes[mode]

        if os.path.exists(mode["destination"]):
            return

        replaced = {
            "i'd": "i",
            "i'll": "i",
            "what's": "what",
            "i'm": "i",
            "we're": "we",
            "i've": "i",
            "that's": "that",
            "what're": "what",
            "let's": "let",
            "american's": "american",
            "wednesday's": "wednesday",
            "delta's": "delta",
            "atlanta's": "atlanta",
            "sunday's": "sunday",
            "one's": "one",
            "york's": "york",
            "friday's": "friday"
        }

        stemmer = SnowballStemmer('english')
        writer = open(mode["destination"], "w")
        with open(mode["origin"]) as infile:
            data = infile.readlines()

        for line in data:
            words, tags = line.split('\t')

            new_words = ["BOS"]
            for w in words.split()[1:-1]:
                w = replaced[w] if w in replaced else w
                new_words.append(stemmer.stem(w.lower()))
            new_words.append("EOS")

            words = " ".join(new_words)
            writer.write("{}\t{}".format(words, tags))
        writer.close()

    def apply_lemmatizer(self, mode):
        modes = {
            "valid": {
                "origin": f"{BASE}/atis/valid/atis.valid.iob",
                "destination": f"{BASE}/atis/valid/atis.lemma.valid.iob",
            },
            "train": {
                "origin": f"{BASE}/atis/train/atis.train.iob",
                "destination": f"{BASE}/atis/train/atis.lemma.train.iob",
            },
            "test": {
                "origin": f"{BASE}/atis/test/atis.test.iob",
                "destination": f"{BASE}/atis/test/atis.lemma.test.iob",
            }
        }

        mode = modes[mode]

        if os.path.exists(mode["destination"]):
            return
        replaced = {
            "i'd": "i",
            "i'll": "i",
            "what's": "what",
            "i'm": "i",
            "we're": "we",
            "i've": "i",
            "that's": "that",
            "what're": "what",
            "let's": "let",
            "american's": "american",
            "wednesday's": "wednesday",
            "delta's": "delta",
            "atlanta's": "atlanta",
            "sunday's": "sunday",
            "one's": "one",
            "york's": "york",
            "friday's": "friday"
        }

        nlp = nlp_en

        def replace(w):
            return replaced[w] if w in replaced else w

        def parser(arr):
            def arr2text(n): return " ".join([w for w, t in n])

            text = arr2text(arr)
            doc = nlp(text)
            n = [(t.text, t.lemma_) for t in doc]

            try:
                pivot_n = 0
                pivot_x = 0
                result = []

                while pivot_x < len(arr):
                    check_size = len(n[pivot_n][0]) <= len(arr[pivot_x][0])
                    while check_size and n[pivot_n][0] in arr[pivot_x][0]:
                        w = n[pivot_n][0]
                        w_lema = n[pivot_n][1]
                        t = arr[pivot_x][1]

                        if w_lema in ['i', 'you', 'he', 'she', 'it', 'we', 'they']:
                            result.append(('-PRON-', t))
                        else:
                            result.append((w_lema, t))

                        #print(n[pivot_n][0]+": "+n[pivot_n][1], '->', arr[pivot_x][0], '->', arr[pivot_x][1])
                        pivot_n += 1

                    pivot_x += 1
            except IndexError:
                pass
            return result

        writer = open(mode["destination"], "w")
        with open(mode["origin"]) as infile:
            data = infile.readlines()
            for line in data:
                words, tags = line.split("\t")

                words = words.split()
                words = [replace(w) for w in words]
                tags = tags.split()

                tuples = []
                for w, t in zip(words, tags):
                    tuples.append((w, t))

                tuples = parser(tuples)

                two_lists = list(map(list, zip(*tuples)))
                words = " ".join(two_lists[0])
                tags = " ".join(two_lists[1])
                writer.write("{}\t{}\n".format(words, tags))
            writer.close()

    def statistic(self, path):
        print(f"Analizando {path}")
        with open(path) as infile:
            data = infile.readlines()

            samples = 0
            vocabulary = set()
            slots = set()

            for line in data:
                samples += 1
                words, tags = line.split("\t")

                words = set(words.strip().split()[1:-1])
                tags = set(tags.strip().split()[1:-1])

                vocabulary = vocabulary.union(words)
                slots = slots.union(tags)

            print("samples:", samples)
            print("words:", len(vocabulary))
            print("slots:", len(slots))
            return {
                "samples": samples,
                "words": len(vocabulary),
                "slots": len(slots)
            }


if __name__ == "__main__":
    fb = FacebookDataset()
    fb.parse()

    snips = SNIPSDataset()
    snips.parse()

    atis = ATISDataset()
    atis.parse()

# RUN pip install -U spacy
# RUN pip install -U spacy-lookups-data
# RUN python -m spacy download es_core_news_sm