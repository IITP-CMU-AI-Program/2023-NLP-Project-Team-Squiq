import pandas as pd
import numpy as np
import json
import torch
from torch import nn
import math
import os


def make_generation_text(inp, pred):
    outputs = ""
    for i in range(len(inp)):
        outputs += "Input | Output #{}: {} | {}\n".format(i, inp[i], pred[i])
    return outputs


if os.path.isfile("./bert/loss.txt"):
    os.remove("./bert/loss.txt")
if os.path.isfile("./bert/pred.txt"):
    os.remove("./bert/pred.txt")

with open("../data/english-train.json", "r") as json_file:
    english_train = json.load(json_file)
with open("../data/english-dev.json", "r") as json_file:
    english_dev = json.load(json_file)
with open("../data/english-test.json", "r") as json_file:
    english_test = json.load(json_file)

context_seq = []
patient_ques = []
docter_ans = []

context_seq.extend([i["description"] for i in english_train])
patient_ques.extend([i["utterances"][0] for i in english_train])
docter_ans.extend([i["utterances"][1] for i in english_train])

context_seq.extend([i["description"] for i in english_dev])
patient_ques.extend([i["utterances"][0] for i in english_dev])
docter_ans.extend([i["utterances"][1] for i in english_dev])

# Changed
# context_seq.extend([i['description'] for i in english_test])
# patient_ques.extend([i['utterances'][0] for i in english_test])
# docter_ans.extend([i['utterances'][1] for i in english_test])

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

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize the tokenizer
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
dialogs = tot_data
# Fit the tokenizer on the texts
# tokenizer.fit_on_texts(dialogs)
tokenizer.fit_on_texts(dialogs + tot_data_test)  # Changed

# Convert texts to sequences of integers
all_patient_ques = tokenizer.texts_to_sequences(all_patient_ques)
patient_ques = tokenizer.texts_to_sequences(patient_ques)
docter_ans = tokenizer.texts_to_sequences(docter_ans)

# Pad the sequences to have equal length
# patient_ques = pad_sequences(patient_ques, padding='post')
# docter_ans = pad_sequences(docter_ans, padding='post')

# print("Word Index = " , tokenizer.word_index)
# print("Sequences = " , patient_ques)
# print("Padded Sequences:")
# print(padded_sequences)

# Changed
all_patient_ques_test = tokenizer.texts_to_sequences(all_patient_ques_test)
patient_ques_test = tokenizer.texts_to_sequences(patient_ques_test)
docter_ans_test = tokenizer.texts_to_sequences(docter_ans_test)

# print("Sequences = " , patient_ques_test)

segmented_embedding = []
sep_token_id = tokenizer.word_index["sep"]
for sequence in all_patient_ques:
    segment_ids = []
    current_segment_id = 1
    for token_id in sequence:
        segment_ids.append(current_segment_id)
        if token_id == sep_token_id:
            current_segment_id = 2
    segmented_embedding.append(segment_ids)

# Changed
segmented_embedding_test = []
sep_token_id = tokenizer.word_index["sep"]
for sequence in all_patient_ques_test:
    segment_ids = []
    current_segment_id = 1
    for token_id in sequence:
        segment_ids.append(current_segment_id)
        if token_id == sep_token_id:
            current_segment_id = 2
    segmented_embedding_test.append(segment_ids)


class BERTEmbeddingBlock(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, max_seq_len=512):
        super(BERTEmbeddingBlock, self).__init__()

        self.embedding_dim = embedding_dim
        self.pos_units = [
            10000 ** (2 * i / self.embedding_dim)
            for i in range(self.embedding_dim // 2)
        ]

        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim
        )
        self.segment_embedding = SegmentEmbedding(self.embedding_dim)

        pos = torch.zeros((max_seq_len, self.embedding_dim))
        for p in range(max_seq_len):
            for i in range(0, self.embedding_dim, 2):
                pos[p, i] = torch.sin(torch.tensor(p) / self.pos_units[i // 2])
                pos[p, i + 1] = torch.cos(torch.tensor(p) / self.pos_units[i // 2])
        self.register_buffer("pos", pos)

    def forward(self, x, segment_info):
        out = self.embedding(x)

        # 미리 계산된 Positional Embedding 불러오기
        seq_len = x.size(1)
        pos = self.pos[:seq_len, :]

        out += pos.unsqueeze(0).expand_as(out)  # Batch size에 맞게 확장
        out += self.segment_embedding(segment_info)

        return out


class EmbeddingBlock(nn.Module):
    """
    Arguments:
        num_embeddings : the number of word types
        embedding_dim : the dimension of embedding vector
    """

    def __init__(self, num_embeddings, embedding_dim):
        super(EmbeddingBlock, self).__init__()
        max_seq_len = 512
        self.embedding_dim = embedding_dim
        self.pos_units = [
            10000 ** (2 * i / self.embedding_dim)
            for i in range(self.embedding_dim // 2)
        ]

        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim
        )

        pos = torch.zeros((max_seq_len, self.embedding_dim))
        for p in range(max_seq_len):
            for i in range(0, self.embedding_dim, 2):
                pos[p, i] = torch.sin(torch.tensor(p) / self.pos_units[i // 2])
                pos[p, i + 1] = torch.cos(torch.tensor(p) / self.pos_units[i // 2])
        self.register_buffer("pos", pos)

    def forward(self, x):
        """
        input : indexed words (batch_size, num_words)
        output : word embeddings (batch_size, num_words, embedding_dim)
        """

        out = self.embedding(x)
        seq_len = x.size(1)
        pos = self.pos[:seq_len, :]
        out += pos.unsqueeze(0).expand_as(out)  # Batch size에 맞게 확장
        # pos = torch.zeros(out.shape)

        # for p in range(pos.shape[1]):
        #     for i in range(0, pos.shape[2], 2):
        #         pos[:, p, i] = torch.sin(torch.Tensor([p/self.pos_units[i//2]]))
        #         pos[:, p, i+1] = torch.cos(torch.Tensor([p/self.pos_units[i//2]]))
        # out += pos

        return out


class AttentionBlock(nn.Module):
    """
    Arguments:
        in_channel : the dimension of embedding vector
        out_channel : the dimension of query/key/value vector


    Variables:
        in_channel : d_model
        out_channel : d_k
    """

    def __init__(self, in_channel, out_channel):
        super(AttentionBlock, self).__init__()

        self.in_channel = in_channel

        self.fc_q = nn.Linear(in_channel, out_channel)  # W^Q
        self.fc_k = nn.Linear(in_channel, out_channel)  # W^K
        self.fc_v = nn.Linear(in_channel, out_channel)  # W^V

        self.softmax = nn.Softmax(dim=1)

    def forward(self, Q, K, V):
        """
        input : embedded words (batch_size, query_dim, key_dim, value_dim)
        output : attention score (batch_size, query_dim)
        """
        out_q = self.fc_q(Q)
        out_k = self.fc_k(K)
        out_v = self.fc_v(V)

        out = self.softmax(out_q @ out_k.transpose(1, 2) / math.sqrt(self.in_channel))

        out = out @ out_v

        return out


class MultiHeadAttentionBlock(nn.Module):
    """
    Arguments:
        in_channel : the dimension of embedding vector
        num_attention : the number of attention heads
        hidden_channel : the number of hidden channels in Position-wise Feed-Forward Networks

    Variables:
        in_channel : d_model
        inner_channel : d_ff
        num_attention : h
    """

    def __init__(self, in_channel, num_attention, hidden_channel):
        super(MultiHeadAttentionBlock, self).__init__()

        self.num_attention = num_attention

        self.heads = nn.ModuleList(
            [
                AttentionBlock(in_channel, in_channel // self.num_attention)
                for _ in range(num_attention)
            ]
        )
        self.flatten = nn.Flatten()

        self.fc = nn.Linear(in_channel, in_channel)  # W^O

        self.ln1 = nn.LayerNorm((in_channel))

        self.ffc = nn.Sequential(
            nn.Linear(
                in_channel, hidden_channel
            ),  # Position-wise Feed-Forward Networks
            nn.ReLU(),
            nn.Linear(hidden_channel, in_channel),
        )

        self.ln2 = nn.LayerNorm((in_channel))

    def forward(self, x):
        """
        input : indexed words (batch_size, num_words)
        output : processed attention scores (batch_size, embedding_dim)
        """
        outs = [self.heads[i](x, x, x) for i in range(self.num_attention)]
        out = torch.cat(outs, dim=2)
        out = self.fc(out)

        out = self.ln1(out + x)

        out = self.ln2(out + self.ffc(out))

        return out


class TransformerEncoder(nn.Module):
    """
    Arguments:
        num_embeddings : the number of word types
        num_enc_layers : the number of encoder stack
        embedding_dim : the dimension of embedding vector
        num_attention : the number of attention heads
        hidden_channel : the number of hidden channels in Position-wise Feed-Forward Networks
        use_embedding : Transformer embedding enabled or not
    """

    def __init__(
        self,
        num_embeddings,
        num_enc_layers=6,
        embedding_dim=512,
        num_attention=8,
        hidden_channel=2048,
        use_embedding=True,
    ):
        super(TransformerEncoder, self).__init__()

        self.num_enc_layers = num_enc_layers
        self.embedding_dim = embedding_dim
        self.num_attention = num_attention
        self.hidden_channel = hidden_channel
        self.use_embedding = use_embedding

        if use_embedding:
            self.embedding = EmbeddingBlock(num_embeddings, embedding_dim)

        self.multihead_attention_blocks = nn.ModuleList(
            [
                MultiHeadAttentionBlock(
                    in_channel=self.embedding_dim,
                    num_attention=self.num_attention,
                    hidden_channel=self.hidden_channel,
                )
                for _ in range(self.num_enc_layers)
            ]
        )

    def forward(self, x):
        """
        input : indexed words (batch_size, num_words)
        output : features (batch_size, embedding_dim)
        """

        out = x

        if self.use_embedding:
            out = self.embedding(x)

        for multihead_attention in self.multihead_attention_blocks:
            out = multihead_attention(out)

        return out


class SegmentEmbedding(
    nn.Embedding
):  # referenced from https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/model/embedding/segment.py
    def __init__(self, embedding_dim):
        super(SegmentEmbedding, self).__init__(3, embedding_dim)


class BERTEmbeddingBlock(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, max_seq_len=512):
        super(BERTEmbeddingBlock, self).__init__()

        self.embedding_dim = embedding_dim
        self.pos_units = [
            10000 ** (2 * i / self.embedding_dim)
            for i in range(self.embedding_dim // 2)
        ]

        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim
        )
        self.segment_embedding = SegmentEmbedding(self.embedding_dim)

        # Positional Embedding 미리 계산
        pos = torch.zeros((max_seq_len, self.embedding_dim))
        for p in range(max_seq_len):
            for i in range(0, self.embedding_dim, 2):
                pos[p, i] = torch.sin(torch.tensor(p) / self.pos_units[i // 2])
                pos[p, i + 1] = torch.cos(torch.tensor(p) / self.pos_units[i // 2])
        self.register_buffer("pos", pos)

    def forward(self, x, segment_info):
        out = self.embedding(x)

        seq_len = x.size(1)
        pos = self.pos[:seq_len, :]

        out += pos.unsqueeze(0).expand_as(out)
        out += self.segment_embedding(segment_info)

        return out


class BERT(nn.Module):

    """
    Arguments:
        num_embeddings : the number of word types
        num_transformer_block : the dimension of embedding vector
        num_enc_layers : the number of encoder stack
        embedding_dim : the dimension of embedding vector
        num_attention : the number of attention heads
        hidden_channel : the number of hidden channels in Position-wise Feed-Forward Networks

    Variables:
        out_channel : d_model
    """

    def __init__(
        self,
        num_embeddings=30000,
        num_transformer_block=6,
        num_enc_layers=1,
        embedding_dim=384,
        num_attention=12,
        hidden_channel=384,
    ):
        super(BERT, self).__init__()

        self.num_embeddings = num_embeddings
        self.num_transformer_block = num_transformer_block
        self.num_enc_layers = num_enc_layers
        self.embedding_dim = embedding_dim
        self.num_attention = num_attention
        self.hidden_channel = hidden_channel

        self.embedding = BERTEmbeddingBlock(self.num_embeddings, self.embedding_dim)
        self.lin = nn.Linear(self.embedding_dim, self.num_embeddings)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerEncoder(
                    num_embeddings=self.embedding_dim,
                    num_enc_layers=self.num_enc_layers,
                    embedding_dim=self.embedding_dim,
                    num_attention=self.num_attention,
                    hidden_channel=self.hidden_channel,
                    use_embedding=False,
                )
                for _ in range(self.num_transformer_block)
            ]
        )

    def forward(
        self, x, segment_info
    ):  # referenced from https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/model/embedding/segment.py
        x = self.embedding(x, segment_info)

        for transformer in self.transformer_blocks:
            x = transformer.forward(x)
        x = self.lin(x)
        return x


from torch.utils.data import Dataset, DataLoader


class ConversationDataset(Dataset):
    def __init__(self, patient_questions, doctor_answers, segment_embeddings):
        self.patient_questions = patient_questions
        self.doctor_answers = doctor_answers
        self.segment_embeddings = segment_embeddings

    def __len__(self):
        return len(self.patient_questions)

    def __getitem__(self, idx):
        return {
            "patient_question": self.patient_questions[idx],
            "doctor_answer": self.doctor_answers[idx],
            "segment_embedding": self.segment_embeddings[idx],
        }


def make_generation_text(inp, pred):
    outputs = ""
    for i in range(len(inp)):
        outputs += "Input | Output #{}: {} | {}\n".format(i, inp[i], pred[i])
    return outputs


import os

pat_qus = torch.tensor(pad_sequences(all_patient_ques, padding="post"))
doc_ans = torch.tensor(pad_sequences(docter_ans, padding="post"))
seg_emb = torch.tensor(pad_sequences(segmented_embedding, padding="post"))
conversation_dataset = ConversationDataset(pat_qus, doc_ans, seg_emb)

seed = 42
np.random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cuda:1" if torch.cuda.is_available() else "cpu"
print("Device: ", device)

batch_size = 32  # Define the batch size
pat_qus = torch.tensor(pad_sequences(all_patient_ques, padding="post"))
doc_ans = torch.tensor(pad_sequences(docter_ans, padding="post"))
seg_emb = torch.tensor(pad_sequences(segmented_embedding, padding="post"))
conversation_dataset = ConversationDataset(pat_qus, doc_ans, seg_emb)
conversation_dataloader = DataLoader(
    conversation_dataset, batch_size=batch_size, shuffle=True
)

# Changed
pat_qus_test = torch.tensor(pad_sequences(all_patient_ques_test, padding="post"))
doc_ans_test = torch.tensor(pad_sequences(docter_ans_test, padding="post"))
seg_emb_test = torch.tensor(pad_sequences(segmented_embedding_test, padding="post"))
conversation_dataset_test = ConversationDataset(
    pat_qus_test, doc_ans_test, seg_emb_test
)
conversation_dataloader_test = DataLoader(
    conversation_dataset_test, batch_size=batch_size, shuffle=False
)

print(len(conversation_dataloader_test))

# Replace Word Embedding by Graph Embedding -----------------------------------------------
wordembeddings = torch.from_numpy(np.load("./emb/node_embedding_384.npy"))

word2embidx = {}
with open("./rawKG/node_info_new.txt", "r") as f:
    for line in f.readlines():
        word, embidx = line.rstrip().split("\t")
        embidx = int(embidx)
        word2embidx[word] = embidx

model = BERT(num_embeddings=len(tokenizer.word_index)).to(device)

for word in word2embidx:
    if word in tokenizer.word_index.keys():
        embidx = word2embidx[word]
        tokenidx = tokenizer.word_index[word]
        model.embedding.embedding.weight.data[tokenidx] = wordembeddings[embidx].to(
            device
        )

# Done  -----------------------------------------------------------------------------------

for i, data in enumerate(conversation_dataloader):
    patient_questions = data["patient_question"].to(device)
    doctor_answers = data["doctor_answer"].to(device)
    segment_embeddings = data["segment_embedding"].to(device)
    break  # Changed

# print(patient_questions.shape)
# print(doctor_answers.shape)
# print(segment_embeddings.shape)

output = model(patient_questions, segment_embeddings)[:, 201:, :]

output.shape

import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

# Assuming pat_qus, doc_ans, seg_emb are already defined
# and that your BERT model and ConversationDataset class are properly implemented

# Create dataset and dataloader
conversation_dataset = ConversationDataset(pat_qus, doc_ans, seg_emb)
conversation_dataloader = DataLoader(
    conversation_dataset, batch_size=batch_size, shuffle=True
)

# Initialize the model
model = BERT(num_embeddings=len(tokenizer.word_index) + 10)
model.to(device)

# Define loss function and optimizer
loss_function = CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):  # num_epochs should be defined
    model.train()
    total_loss = 0
    total_loss_test = 0  # Changed

    for i, data in enumerate(conversation_dataloader):
        # Forward pass

        model.train()

        patient_questions = data["patient_question"].to(device)
        doctor_answers = data["doctor_answer"].to(device).long()
        segment_embeddings = data["segment_embedding"].to(device)

        # if(len(patient_questions)!=batch_size):
        #   print('jump')
        #   continue
        optimizer.zero_grad()
        output = model(patient_questions, segment_embeddings)

        # output = output[:, 201:, :] # Changed
        ans_seq_len = doctor_answers.shape[1]
        output = output[:, :ans_seq_len, :]  # Changed
        # output = output.softmax(dim=-1)
        # Compute loss - ensure doctor_answers is the correct target and has the right shape

        loss = loss_function(
            output.reshape(-1, output.size(-1)), doctor_answers.reshape(-1)
        )
        total_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    # Changed
    with torch.no_grad():
        for i, data in enumerate(conversation_dataloader_test):
            # Forward pass

            model.eval()

            patient_questions = data["patient_question"].to(device)
            doctor_answers = data["doctor_answer"].to(device).long()
            segment_embeddings = data["segment_embedding"].to(device)

            output = model(patient_questions, segment_embeddings)

            ans_seq_len = doctor_answers.shape[1]
            output = output[:, :ans_seq_len, :]  # Changed
            pred = torch.argmax(output, dim=-1)

            loss = loss_function(
                output.reshape(-1, output.size(-1)), doctor_answers.reshape(-1)
            )
            total_loss_test += loss.item()

            inp = [
                seq[: seq.index(0)] if seq.count(0) != 0 else seq
                for seq in patient_questions.cpu().tolist()
            ]
            inp = tokenizer.sequences_to_texts(inp)
            temp = []
            for text in inp:
                start_index = text.index("sep")
                temp.append(text[: start_index - 1])
            inp = temp
            generated = [
                seq[: seq.index(0)] if seq.count(0) != 0 else seq
                for seq in pred.cpu().tolist()
            ]
            generated = tokenizer.sequences_to_texts(generated)
            with open("./bert/pretrain_gen/gen.txt", "w") as fw:
                fw.write(make_generation_text(inp, generated))

            if i == 1:
                temp = [
                    seq[: seq.index(0)] if seq.count(0) != 0 else seq
                    for seq in patient_questions.cpu().tolist()
                ]
                print("Patient      :", tokenizer.sequences_to_texts(temp)[:1])
                temp = [
                    seq[: seq.index(0)] if seq.count(0) != 0 else seq
                    for seq in doctor_answers.cpu().tolist()
                ]
                print("Doctor(targ) :", tokenizer.sequences_to_texts(temp)[:1])
                temp = [
                    seq[: seq.index(0)] if seq.count(0) != 0 else seq
                    for seq in pred.cpu().tolist()
                ]
                print("Doctor(pred) :", tokenizer.sequences_to_texts(temp)[:1])
                print(
                    "Loss : ",
                    loss_function(
                        output[0].reshape(-1, output.size(-1)),
                        doctor_answers[0].reshape(-1),
                    ),
                )
                print("===============================================")

            if epoch % 100 == 0 and i == 1:
                with open("./bert/pred.txt", "+a") as f:
                    f.write("[Epoch] " + str(epoch + 1) + "\n")
                    temp = [
                        seq[: seq.index(0)] if seq.count(0) != 0 else seq
                        for seq in patient_questions.cpu().tolist()
                    ]
                    f.write(
                        "Patient      :"
                        + str(tokenizer.sequences_to_texts(temp)[:1])
                        + "\n"
                    )
                    temp = [
                        seq[: seq.index(0)] if seq.count(0) != 0 else seq
                        for seq in doctor_answers.cpu().tolist()
                    ]
                    f.write(
                        "Doctor(targ) :"
                        + str(tokenizer.sequences_to_texts(temp)[:1])
                        + "\n"
                    )
                    temp = [
                        seq[: seq.index(0)] if seq.count(0) != 0 else seq
                        for seq in pred.cpu().tolist()
                    ]
                    f.write(
                        "Doctor(pred) :"
                        + str(tokenizer.sequences_to_texts(temp)[:1])
                        + "\n"
                    )

    print(
        f"Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss/len(conversation_dataloader)}"
    )
    print(
        f"Epoch {epoch+1}/{num_epochs}, Test  Loss: {total_loss_test/len(conversation_dataloader_test)}"
    )

    with open("./bert/loss.txt", "+a") as f:
        f.write(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss/len(conversation_dataloader)}\n"
        )
        f.write(
            f"Epoch {epoch+1}/{num_epochs}, Test  Loss: {total_loss_test/len(conversation_dataloader_test)}\n"
        )


loss_function(output.reshape(-1, output.size(-1)), doctor_answers.reshape(-1))
