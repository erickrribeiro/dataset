import shutil
import os

BASE = os.path.dirname(os.path.realpath(__file__))


def export_as_atis(dataset, split, transform, name):
    if transform == "normal":
        mode = "atis"
    elif transform == "stem":
        mode = "atis.stem"
    elif transform == "stopwords":
        mode = "atis.stopwords"
    elif transform == "lemma":
        mode = "atis.lemma"
    else:
        raise Exception(
            "You must choose one this values (normal, stem, stopwords, lemma)."
        )

    mode = {
        "origin": f"{BASE}/{dataset}/{split}/{mode}.{split}.iob",
        "destination": f"{BASE}/dataset/{name}"
    }

    path = os.path.dirname(mode["destination"])
    if not os.path.exists(path):
        os.makedirs(path)

    res = shutil.copyfile(mode["origin"], mode["destination"])
    print(res)


def export_as_snips(dataset, split, transform):
    if transform == "normal":
        mode = "atis"
    elif transform == "stem":
        mode = "atis.stem"
    elif transform == "stopwords":
        mode = "atis.stopwords"
    elif transform == "lemma":
        mode = "atis.lemma"
    else:
        raise Exception(
            "You must choose one this values (normal, stem, stopwords, lemma)."
        )

    mode = {
        "dir": f"{BASE}/dataset/{split}",
        "origin": f"{BASE}/{dataset}/{split}/{mode}.{split}.iob",
    }

    if not os.path.exists(mode["dir"]):
        os.makedirs(mode["dir"])

    in_write = open(f"{mode['dir']}/seq.in", "w")
    out_write = open(f"{mode['dir']}/seq.out", "w")
    label_write = open(f"{mode['dir']}/label.out", "w")

    with open(mode["origin"]) as infile:
        data = infile.readlines()
        for line in data:
            words, tags = line.split("\t")
            words = words.strip().split()[1:-1]

            tags = tags.strip().split()
            label = tags[-1]
            tags = tags[1:-1]

            in_write.write(" ".join(words) + "\n")
            out_write.write(" ".join(tags) + "\n")
            label_write.write(label + "\n")

        in_write.close()
        out_write.close()
        label_write.close()

        if len(words) != len(tags):
            print(len(words), len(tags))
            raise Exception("The lengths must be equals.")


if __name__ == "__main__":
    # export_as_atis("atis", "train", "stem", "atis-2.train.w-intent.iob")
    # export_as_atis("atis", "test", "stem", "atis-2.test.w-intent.iob")
    # export_as_atis("atis", "valid", "stem", "atis-2.dev.w-intent.iob")

    # export_as_atis("atis", "train", "stopwords", "atis-2.train.w-intent.iob")
    # export_as_atis("atis", "test", "stopwords", "atis-2.test.w-intent.iob")
    # export_as_atis("atis", "valid", "stopwords", "atis-2.dev.w-intent.iob")

    # export_as_atis("atis", "train", "lemma", "atis-2.train.w-intent.iob")
    # export_as_atis("atis", "test", "lemma", "atis-2.test.w-intent.iob")
    # export_as_atis("atis", "valid", "lemma", "atis-2.dev.w-intent.iob")

    # export_as_atis("atis", "train", "normal", "atis-2.train.w-intent.iob")
    # export_as_atis("atis", "test", "normal", "atis-2.test.w-intent.iob")
    # export_as_atis("atis", "valid", "normal", "atis-2.dev.w-intent.iob")

    export_as_snips("atis", "valid", "lemma")
    export_as_snips("atis", "test", "lemma")
    export_as_snips("atis", "train", "lemma")