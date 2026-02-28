import torch
import torch.nn as nn

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(MatrixFactorization, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self.user_embedding.weight.data.uniform_(0, 0.05)
        self.item_embedding.weight.data.uniform_(0, 0.05)

    def forward(self, user_ids, item_ids): #내적으로 계산.
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)
        return (user_embeds * item_embeds).sum(dim=1)
    

class FeatureAidedGMF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, genres_dim, age_dim):
        super(FeatureAidedGMF, self).__init__()
        self.user_embedding = nn.Embedding(num_users+1, embedding_dim)
        self.item_embedding = nn.Embedding(num_items+1, embedding_dim)

        self.genres_layer = nn.Linear(genres_dim,embedding_dim)
        self.age_layer = nn.Linear(age_dim,embedding_dim)

        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.normal_(self.genres_layer.weight, std=0.01)
        nn.init.normal_(self.age_layer.weight, std=0.01)

        self.age_weight = nn.Parameter(torch.tensor([0.1]))
        self.genre_weight = nn.Parameter(torch.tensor([0.1]))

        self.prediction_layer = nn.Linear(embedding_dim, 1)


    def forward(self, user_ids, item_ids,genres_features, age_features):
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)
        genres_proj = self.genres_layer(genres_features)
        age_proj = self.age_layer(age_features)

        user_embeds = user_embeds + self.age_weight * age_proj
        item_embeds = item_embeds + self.genre_weight * genres_proj

        #interaction = torch.mul(user_embeds, item_embeds)
        #output = torch.sigmoid(self.prediction_layer(interaction)) * 5.0
        return torch.sigmoid((user_embeds * item_embeds).sum(dim=1)) * 4.0 + 1.0