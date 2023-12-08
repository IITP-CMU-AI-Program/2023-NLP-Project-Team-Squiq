import pandas as pd
import numpy as np
import json
import torch
from torch import nn
import math
import os
from transformers import AutoTokenizer, BertModel
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertConfig
from transformers import AdamW
from torch.nn import CrossEntropyLoss
from collections import defaultdict


class CustomDataset(Dataset):
    def __init__(self, p_info, c_info, d_info):
        self.patient_ques = p_info
        self.context_seq = c_info
        self.doc_ans = d_info

    def __len__(self):
        return len(self.patient_ques)

    def __getitem__(self, idx):
        input_seq = tokenizer(
            self.patient_ques[idx],
            self.context_seq[idx],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )
        target_seq = tokenizer(
            self.doc_ans[idx],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )["input_ids"]

        return input_seq, target_seq


def make_generation_text(inp, pred):
    outputs = ""
    for i in range(len(inp)):
        outputs += "Input | Output #{}: {} | {}\n".format(i, inp[i], pred[i])
    return outputs


if os.path.isfile("logs/loss_pretrained.txt"):
    os.remove("logs/loss_pretrained.txt")
if os.path.isfile("logs/pred_pretrained.txt"):
    os.remove("logs/pred_pretrained.txt")

with open("../../data/english-train.json", "r") as json_file:
    english_train = json.load(json_file)
with open("../../data/english-dev.json", "r") as json_file:
    english_dev = json.load(json_file)
with open("../../data/english-test.json", "r") as json_file:
    english_test = json.load(json_file)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Device: ", device)

context_seq = []
patient_ques = []
docter_ans = []

context_seq.extend([i["description"] for i in english_train])
patient_ques.extend([i["utterances"][0] for i in english_train])
docter_ans.extend([i["utterances"][1] for i in english_train])

context_seq.extend([i["description"] for i in english_dev])
patient_ques.extend([i["utterances"][0] for i in english_dev])
docter_ans.extend([i["utterances"][1] for i in english_dev])

context_seq = ["SOS " + i + " EOS" for i in context_seq]
patient_ques = ["SOS " + i[9:] + " EOS" for i in patient_ques]
docter_ans = ["SOS " + i[8:] + " EOS" for i in docter_ans]

tot_data = []
all_patient_ques = []
for i in range(len(patient_ques)):
    all_patient_ques.append(patient_ques[i] + " SEP " + context_seq[i])

tot_data.extend(all_patient_ques)
tot_data.extend(docter_ans)

# Changed
context_seq_test = []
patient_ques_test = []
docter_ans_test = []

context_seq_test.extend([i["description"] for i in english_test])
patient_ques_test.extend([i["utterances"][0] for i in english_test])
docter_ans_test.extend([i["utterances"][1] for i in english_test])

context_seq_test = ["SOS " + i + " EOS" for i in context_seq_test]
patient_ques_test = ["SOS " + i[9:] + " EOS" for i in patient_ques_test]
docter_ans_test = ["SOS " + i[8:] + " EOS" for i in docter_ans_test]

tot_data_test = []
all_patient_ques_test = []

for i in range(len(patient_ques_test)):
    all_patient_ques_test.append(patient_ques_test[i] + " SEP " + context_seq_test[i])

tot_data_test.extend(all_patient_ques_test)
tot_data_test.extend(docter_ans_test)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

patient_input = [
    tokenizer(
        patient_ques[idx],
        context_seq[idx],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )
    for idx in range(len(patient_ques))
]
docter_input = [
    tokenizer(
        docter_ans[idx],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )["input_ids"]
    for idx in range(len(patient_ques))
]

# Create dataset
dataset = CustomDataset(patient_ques, context_seq, docter_ans)

# Create DataLoader
batch_size = 8  # You can adjust this
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for inputs, targets in dataloader:
    break
input_seq = tokenizer(
    patient_ques[1:5],
    context_seq[1:5],
    truncation=True,
    padding="max_length",
    max_length=512,
    return_tensors="pt",
)

inputs["input_ids"] = inputs["input_ids"].squeeze(1)
inputs["token_type_ids"] = inputs["token_type_ids"].squeeze(1)
inputs["attention_mask"] = inputs["attention_mask"].squeeze(1)
test_dataset = CustomDataset(patient_ques_test, context_seq_test, docter_ans_test)

# Create DataLoader
batch_size = 8  # You can adjust this
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the optimizer
config = BertConfig.from_pretrained("bert-base-uncased")
model = BertModel(config)
model = model.to(device)

# Replace Word Embedding by Graph Embedding -----------------------------------------------
wordembeddings = torch.from_numpy(np.load("../emb/node_embedding_768.npy"))

word2embidx = {}
with open("../rawKG/node_info_new.txt", "r") as f:
    for line in f.readlines():
        word, embidx = line.rstrip().split("\t")
        embidx = int(embidx)
        word2embidx[word] = embidx

replace_count = 0
for word in word2embidx:
    if word in tokenizer.vocab.keys():
        embidx = word2embidx[word]
        tokenidx = tokenizer.vocab[word]
        model.embeddings.word_embeddings.weight.data[tokenidx] = wordembeddings[
            embidx
        ].to(device)
        replace_count += 1

print(replace_count, len(word2embidx))
# Done  ----------------------------------------------------------------------------------

linear_layer = torch.nn.Linear(config.hidden_size, config.vocab_size)
linear_layer.weight = model.embeddings.word_embeddings.weight  # weight tying
linear_layer = linear_layer.to(device)

# Training loop
optimizer = AdamW(model.parameters(), lr=5e-5)
losses = defaultdict(list)

num_epochs = 1000
for epoch in range(num_epochs):
    running_loss = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        inputs["input_ids"] = inputs["input_ids"].squeeze(1)
        inputs["token_type_ids"] = inputs["token_type_ids"].squeeze(1)
        inputs["attention_mask"] = inputs["attention_mask"].squeeze(1)

        outputs = model(**inputs)
        logits = linear_layer(outputs.last_hidden_state)

        # Compute the loss
        loss_fn = nn.CrossEntropyLoss(reduction="mean", ignore_index=0)
        loss = loss_fn(logits.view(-1, config.vocab_size), targets.view(-1))
        running_loss += loss
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print("====================Training===================")
    print("Patient:")
    print(tokenizer.decode(inputs["input_ids"].squeeze(1)[0], skip_special_tokens=True))
    print("Docter True:")
    print(tokenizer.decode(targets[0].squeeze(0), skip_special_tokens=True))
    print("Docter Pred:")
    print(tokenizer.decode(torch.argmax(logits, dim=-1)[0], skip_special_tokens=True))
    print("===============================================")

    total_loss_test = 0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.squeeze(1)
            inputs["input_ids"] = inputs["input_ids"].squeeze(1)
            inputs["token_type_ids"] = inputs["token_type_ids"].squeeze(1)
            inputs["attention_mask"] = inputs["attention_mask"].squeeze(1)

            outputs = model(**inputs)
            logits = linear_layer(outputs.last_hidden_state)

            ans_seq_len = targets.shape[1]
            logits = logits[:, :ans_seq_len, :]  # Changed

            loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            total_loss_test += loss.item()

            if i == 0:
                print("====================Testing===================")
                print("Patient:")
                print(
                    tokenizer.decode(
                        inputs["input_ids"].squeeze(1)[0], skip_special_tokens=True
                    )
                )
                print("Docter True:")
                print(tokenizer.decode(targets[0], skip_special_tokens=True))
                print("Docter Pred:")
                print(
                    tokenizer.decode(
                        torch.argmax(logits, dim=-1)[0], skip_special_tokens=True
                    )
                )
                print("===============================================")

            if epoch % 100 == 0 and i == 1:
                with open("pred_pretrained.txt", "+a") as f:
                    f.write("[Epoch] " + str(epoch + 1) + "\n")
                    f.write(
                        "Patient      :"
                        + str(
                            tokenizer.decode(
                                inputs["input_ids"].squeeze(1)[0],
                                skip_special_tokens=True,
                            )
                        )
                        + "\n"
                    )
                    f.write(
                        "Doctor(targ) :"
                        + str(tokenizer.decode(targets[0], skip_special_tokens=True))
                        + "\n"
                    )
                    f.write(
                        "Doctor(pred) :"
                        + str(
                            tokenizer.decode(
                                torch.argmax(logits, dim=-1)[0],
                                skip_special_tokens=True,
                            )
                        )
                        + "\n"
                    )

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {running_loss/len(dataloader)}")
    print(
        f"Epoch {epoch+1}/{num_epochs}, Test  Loss: {total_loss_test/len(testloader)}"
    )

    with open("logs/pretrain_log.txt", "+a") as f:
        f.write(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {running_loss/len(dataloader)}\n"
        )
        f.write(
            f"Epoch {epoch+1}/{num_epochs}, Test  Loss: {total_loss_test/len(testloader)}\n"
        )
