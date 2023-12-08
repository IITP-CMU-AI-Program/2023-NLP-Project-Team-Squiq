import wikipedia
import json


def extract_corpus(num_corpus):
    corpus = []
    num = 0

    while num < num_corpus:
        title = wikipedia.random()

        try:
            summary = wikipedia.summary(title)
        except Exception:
            continue

        print(f"Cumulative corpus num: {num}. Title: {title}")

        sentences = summary.replace("\n", " ").strip().split(". ")  # remove " "

        if len(sentences) < 2:
            continue

        num += len(sentences) - 1

        if num >= num_corpus:
            sentences = sentences[: -(num - num_corpus + 1)]

        for idx in range(len(sentences) - 1):
            corpus.append(
                {"input": sentences[idx] + ".", "label": sentences[idx + 1] + "."}
            )

    return corpus


def save_corpus(corpus, filepath):
    with open(filepath, "w") as f:
        json.dump(corpus, f, indent=4)


if __name__ == "__main__":
    corpus = extract_corpus(100000)
    save_corpus(corpus, "../data/wiki.json")
