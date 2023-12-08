import openai
import os
from tqdm import tqdm
import numpy as np

n_tests = 29

with open("pretrain_gen/gen.txt", "r", encoding="utf-8") as f:
    generated = list(f)

assert len(generated) == n_tests
for item in generated:
    assert type(item) is str

parsed_generated = []

for text in generated:
    start_index = text.index("sos")
    temp = text[start_index + 4 :]
    generation_start_index = temp.index("| ")
    parsed_text = temp[:generation_start_index] + temp[generation_start_index + 2 :]
    parsed_text = parsed_text.replace("<eol>", "\n")
    parsed_generated.append(parsed_text)


def perplexity(text, modelname):
    """Compute the perplexity of the provided text."""
    completion = openai.Completion.create(
        model=modelname,
        prompt=text,
        logprobs=0,
        max_tokens=0,
        temperature=1.0,
        echo=True,
    )
    token_logprobs = completion["choices"][0]["logprobs"]["token_logprobs"]
    ll = np.mean([i for i in token_logprobs if i is not None])
    ppl = np.exp(-ll)
    return ppl


# Add you API key here to get perplexity. However, delete the key from the notebook before creating the handin.
# REMEMBER: ALWAYS KEEP YOUR API KEYS AND SECRETS SECURE.

openai.api_key = "Your API key here"
modelname = "text-embedding-ada-002"

perps = [perplexity(text, modelname) for text in tqdm(parsed_generated)]
avg_perp = np.mean(perps)

# Report this number when running the makefile to create the handin
print("Your mean perplexity for generated sequences: {}".format(avg_perp))
