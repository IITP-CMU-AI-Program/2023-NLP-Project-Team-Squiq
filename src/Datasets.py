from torch.utils.data import Dataset

class ConversationDataset(Dataset):
    def __init__(self, patient_questions, doctor_answers, segment_embeddings):
        self.patient_questions = patient_questions
        self.doctor_answers = doctor_answers
        self.segment_embeddings = segment_embeddings

    def __len__(self):
        return len(self.patient_questions)

    def __getitem__(self, idx):
        return {
            'patient_question': self.patient_questions[idx],
            'doctor_answer': self.doctor_answers[idx],
            'segment_embedding': self.segment_embeddings[idx]
        }

class WikiDataset(Dataset):
    def __init__(self, input, label, segment_embeddings):
        self.input = input
        self.label = label
        self.segment_embeddings = segment_embeddings

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return {
            'input': self.input[idx],
            'label': self.label[idx],
            'segment_embedding': self.segment_embeddings[idx]
        }