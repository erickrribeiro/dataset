import shutil
import os
import argparse

BASE = os.path.dirname(os.path.realpath(__file__))


def export_as_atis(dataset, split, transform, name):
    if transform == "normal":
        mode = f"{dataset}"
    elif transform == "stem":
        mode = f"{dataset}.stem"
    elif transform == "stopwords":
        mode = f"{dataset}.stopwords"
    elif transform == "lemma":
        mode = f"{dataset}.lemma"
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
        mode = f"{dataset}"
    elif transform == "stem":
        mode = f"{dataset}.stem"
    elif transform == "stopwords":
        mode = f"{dataset}.stopwords"
    elif transform == "lemma":
        mode = f"{dataset}.lemma"
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
    label_write = open(f"{mode['dir']}/label", "w")

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

    parser = argparse.ArgumentParser()
    parser.add_argument("--format", 
        choices=["atis", "snips"],
        help="Choice a text format.")
    
    parser.add_argument("--dataset", 
        choices=["atis", "snips", "fb_en", "fb_es"],
        help="Choice a dataset.")
    parser.add_argument("--transform",
        choices=["normal", "stem", "stopwords", "lemma"],
        help="Choice a type of text transform.")
    args = parser.parse_args()
    
    if args.format == "atis" and args.dataset == "atis" and args.transform == "normal":
        if os.path.exists(f"{BASE}/dataset/"):
             shutil.rmtree(f"{BASE}/dataset/")
        
        export_as_atis("atis", "train", "normal", "atis-2.train.w-intent.iob")
        export_as_atis("atis", "test", "normal", "atis-2.test.w-intent.iob")
        export_as_atis("atis", "valid", "normal", "atis-2.dev.w-intent.iob")
    elif args.format == "atis" and args.dataset == "atis" and args.transform == "stem":
        if os.path.exists(f"{BASE}/dataset/"):
             shutil.rmtree(f"{BASE}/dataset/")
        
        export_as_atis("atis", "train", "stem", "atis.stem.train.w-intent.iob")
        export_as_atis("atis", "test", "stem", "atis.stem.test.w-intent.iob")
        export_as_atis("atis", "valid", "stem", "atis.stem.dev.w-intent.iob")
    elif args.format == "atis" and args.dataset == "atis" and args.transform == "stopwords":
        if os.path.exists(f"{BASE}/dataset/"):
             shutil.rmtree(f"{BASE}/dataset/")
        
        export_as_atis("atis", "train", "stopwords", "atis.stopwords.train.w-intent.iob")
        export_as_atis("atis", "test", "stopwords", "atis.stopwords.test.w-intent.iob")
        export_as_atis("atis", "valid", "stopwords", "atis.stopwords.dev.w-intent.iob")
    elif args.format == "atis" and args.dataset == "atis" and args.transform == "lemma":
        if os.path.exists(f"{BASE}/dataset/"):
             shutil.rmtree(f"{BASE}/dataset/")
        
        export_as_atis("atis", "train", "lemma", "atis.lemma.train.w-intent.iob")
        export_as_atis("atis", "test", "lemma", "atis.lemma.test.w-intent.iob")
        export_as_atis("atis", "valid", "lemma", "atis.lemma.dev.w-intent.iob")
    elif args.format == "atis" and args.dataset == "snips" and args.transform == "normal":
        if os.path.exists(f"{BASE}/dataset/"):
             shutil.rmtree(f"{BASE}/dataset/")
        
        export_as_atis("snips", "train", "normal", "snips.train.w-intent.iob")
        export_as_atis("snips", "test", "normal", "snips.test.w-intent.iob")
        export_as_atis("snips", "valid", "normal", "snips.dev.w-intent.iob")
    elif args.format == "atis" and args.dataset == "snips" and args.transform == "stem":
        if os.path.exists(f"{BASE}/dataset/"):
             shutil.rmtree(f"{BASE}/dataset/")
        
        export_as_atis("snips", "train", "stem", "snips.stem.train.w-intent.iob")
        export_as_atis("snips", "test", "stem", "snips.stem.test.w-intent.iob")
        export_as_atis("snips", "valid", "stem", "snips.stem.dev.w-intent.iob")
    elif args.format == "atis" and args.dataset == "snips" and args.transform == "lemma":
        if os.path.exists(f"{BASE}/dataset/"):
             shutil.rmtree(f"{BASE}/dataset/")
        
        export_as_atis("snips", "train", "lemma", "snips.lemma.train.w-intent.iob")
        export_as_atis("snips", "test", "lemma", "snips.lemma.test.w-intent.iob")
        export_as_atis("snips", "valid", "lemma", "snips.lemma.dev.w-intent.iob")
    elif args.format == "atis" and args.dataset == "snips" and args.transform == "stopwords":
        if os.path.exists(f"{BASE}/dataset/"):
             shutil.rmtree(f"{BASE}/dataset/")
        
        export_as_atis("snips", "train", "stopwords", "snips.stopwords.train.w-intent.iob")
        export_as_atis("snips", "test", "stopwords", "snips.stopwords.test.w-intent.iob")
        export_as_atis("snips", "valid", "stopwords", "snips.stopwords.dev.w-intent.iob")
    elif args.format == "atis" and args.dataset == "fb_en" and args.transform == "normal":
        if os.path.exists(f"{BASE}/dataset/"):
             shutil.rmtree(f"{BASE}/dataset/")
        
        export_as_atis("fb_en", "train", "normal", "fb_en.train.w-intent.iob")
        export_as_atis("fb_en", "test", "normal", "fb_en.test.w-intent.iob")
        export_as_atis("fb_en", "valid", "normal", "fb_en.dev.w-intent.iob")
    elif args.format == "atis" and args.dataset == "fb_en" and args.transform == "stem":
        if os.path.exists(f"{BASE}/dataset/"):
             shutil.rmtree(f"{BASE}/dataset/")
        
        export_as_atis("fb_en", "train", "stem", "fb_en.stem.train.w-intent.iob")
        export_as_atis("fb_en", "test", "stem", "fb_en.stem.test.w-intent.iob")
        export_as_atis("fb_en", "valid", "stem", "fb_en.stem.dev.w-intent.iob")
    elif args.format == "atis" and args.dataset == "fb_en" and args.transform == "lemma":
        if os.path.exists(f"{BASE}/dataset/"):
             shutil.rmtree(f"{BASE}/dataset/")
        
        export_as_atis("fb_en", "train", "lemma", "fb_en.lemma.train.w-intent.iob")
        export_as_atis("fb_en", "test", "lemma", "fb_en.lemma.test.w-intent.iob")
        export_as_atis("fb_en", "valid", "lemma", "fb_en.lemma.dev.w-intent.iob")
    elif args.format == "atis" and args.dataset == "fb_en" and args.transform == "stopwords":
        if os.path.exists(f"{BASE}/dataset/"):
             shutil.rmtree(f"{BASE}/dataset/")
        
        export_as_atis("fb_en", "train", "stopwords", "fb_en.stopwords.train.w-intent.iob")
        export_as_atis("fb_en", "test", "stopwords", "fb_en.stopwords.test.w-intent.iob")
        export_as_atis("fb_en", "valid", "stopwords", "fb_en.stopwords.dev.w-intent.iob")

    elif args.format == "atis" and args.dataset == "fb_es" and args.transform == "normal":
        if os.path.exists(f"{BASE}/dataset/"):
             shutil.rmtree(f"{BASE}/dataset/")
        
        export_as_atis("fb_es", "train", "normal", "fb_es.train.w-intent.iob")
        export_as_atis("fb_es", "test", "normal", "fb_es.test.w-intent.iob")
        export_as_atis("fb_es", "valid", "normal", "fb_es.dev.w-intent.iob")
    elif args.format == "atis" and args.dataset == "fb_es" and args.transform == "stem":
        if os.path.exists(f"{BASE}/dataset/"):
             shutil.rmtree(f"{BASE}/dataset/")
        
        export_as_atis("fb_es", "train", "stem", "fb_es.stem.train.w-intent.iob")
        export_as_atis("fb_es", "test", "stem", "fb_es.stem.test.w-intent.iob")
        export_as_atis("fb_es", "valid", "stem", "fb_es.stem.dev.w-intent.iob")
    elif args.format == "atis" and args.dataset == "fb_es" and args.transform == "lemma":
        if os.path.exists(f"{BASE}/dataset/"):
             shutil.rmtree(f"{BASE}/dataset/")
        
        export_as_atis("fb_es", "train", "lemma", "fb_es.lemma.train.w-intent.iob")
        export_as_atis("fb_es", "test", "lemma", "fb_es.lemma.test.w-intent.iob")
        export_as_atis("fb_es", "valid", "lemma", "fb_es.lemma.dev.w-intent.iob")
    elif args.format == "atis" and args.dataset == "fb_es" and args.transform == "stopwords":
        if os.path.exists(f"{BASE}/dataset/"):
             shutil.rmtree(f"{BASE}/dataset/")
        
        export_as_atis("fb_es", "train", "stopwords", "fb_es.stopwords.train.w-intent.iob")
        export_as_atis("fb_es", "test", "stopwords", "fb_es.stopwords.test.w-intent.iob")
        export_as_atis("fb_es", "valid", "stopwords", "fb_es.stopwords.dev.w-intent.iob")
    
    if args.format == "snips":
        if os.path.exists(f"{BASE}/dataset/"):
             shutil.rmtree(f"{BASE}/dataset/")

        export_as_snips(args.dataset, "valid", args.transform)
        export_as_snips(args.dataset, "test", args.transform)
        export_as_snips(args.dataset, "train", args.transform)    
        
    
    
    

    