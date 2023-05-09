
class Skill2Vec:
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased')

    def __init__(self, skill_set: list):
        self.skill_set = skill_set
        self.skill_set_emb = self.encode(self.skill_set)

    def encode(self, texts):
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input, return_dict=True)
        embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_similar_skill(self, skill: str, top_k=1):
        skill_emb = self.encode(skill)

        # Compute dot score between query and all document embeddings
        scores = torch.mm(skill_emb, self.skill_set_emb.transpose(0, 1))[0].cpu().tolist()

        # Combine docs & scores
        doc_score_pairs = list(zip(self.skill_set, scores))

        # Sort by decreasing score
        doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
        return doc_score_pairs[:top_k]


class Skill2Vec3:
    __instance = None

    def __new__(cls, skill_set: list):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
            cls.__instance.skill_set = skill_set
            cls.__instance.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            cls.__instance.model = AutoModel.from_pretrained('bert-base-uncased')
            cls.__instance.skill_set_emb = cls.__instance.encode(cls.__instance.skill_set)
        return cls.__instance

    def encode(self, texts):
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input, return_dict=True)
        embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_similar_skill(self, skill: str, top_k=1):
        skill_emb = self.encode(skill)

        # Compute dot score between query and all document embeddings
        scores = torch.mm(skill_emb, self.skill_set_emb.transpose(0, 1))[0].cpu().tolist()

        # Combine docs & scores
        doc_score_pairs = list(zip(self.skill_set, scores))

        # Sort by decreasing score
        doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
        return doc_score_pairs[:top_k]


class Skill2Vec2:
    _instance = None

    def __init__(self, skill_set: list):
        if Skill2Vec2._instance is not None:
            raise Exception("Singleton class, use Skill2Vec.get_instance() to get the instance")
        else:
            Skill2Vec2._instance = self
            self.skill_set = skill_set
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.model = AutoModel.from_pretrained('bert-base-uncased')
            self.skill_set_emb = self.encode(self.skill_set)

    @classmethod
    def get_instance(cls, skill_set: list):
        if not cls._instance:
            cls(skill_set)
        return cls._instance

    def encode(self, texts):
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input, return_dict=True)
        embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_similar_skill(self, skill: str, top_k=1):
        skill_emb = self.encode(skill)

        # Compute dot score between query and all document embeddings
        scores = torch.mm(skill_emb, self.skill_set_emb.transpose(0, 1))[0].cpu().tolist()

        # Combine docs & scores
        doc_score_pairs = list(zip(self.skill_set, scores))

        # Sort by decreasing score
        doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
        return doc_score_pairs[:top_k]

