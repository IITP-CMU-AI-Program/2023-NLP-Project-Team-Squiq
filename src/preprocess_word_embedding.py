import os
from collections import defaultdict


def preprocess_word_embedding():
    if os.path.isfile("../raw/node_info_new.txt"):
        os.remove("../raw/node_info_new.txt")

    nodelist = defaultdict(int)
    word2embidx = {}

    with open("../raw/node_info.txt", "r") as f:
        for line in f.readlines():
            vidx, nodename = line.rstrip().split("\t")
            nodes = nodename.split(" ")
            if len(nodes) <= 3:
                for node in nodes:
                    nodelist[node] += 1

    with open("../raw/node_info.txt", "r") as f:
        for line in f.readlines():
            vidx, nodename = line.rstrip().split("\t")
            vidx = int(vidx)
            nodes = nodename.split(" ")
            if len(nodes) <= 3:
                for node in nodes:
                    if nodelist[node] < 2:
                        word2embidx[node] = vidx

    with open("../raw/node_info_new.txt", "w") as f:
        for word, embidx in word2embidx.items():
            f.write(f"{word}\t{embidx}\n")


if __name__ == "__main__":
    preprocess_word_embedding()
