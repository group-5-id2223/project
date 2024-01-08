import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_words=15, num_hidden_layers=5):
        super().__init__()

        self.embedding_titles = nn.Embedding(31001, 50)
        self.spatial_dropout = nn.Dropout2d(0.2)
        self.rnn_titles = nn.LSTM(50, 128)

        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(259, 259),
                nn.ReLU(),
                nn.BatchNorm1d(259),
                nn.Dropout(0.5)
            )
            for _ in range(num_hidden_layers)
        ])

        self.output_layer = nn.Linear(259, 1)

    def forward(self, input_titles, input_domains, input_dayofweeks, input_hours):
        embedding_titles = self.embedding_titles(input_titles)
        spatial_dropout = self.spatial_dropout(embedding_titles)
        rnn_titles, _ = self.rnn_titles(spatial_dropout.permute(1, 0, 2))

        concat = torch.cat([rnn_titles[-1], input_domains, input_dayofweeks, input_hours], dim=1)
        for layer in self.hidden_layers:
            concat = layer(concat)

        output = self.output_layer(concat)
        return output