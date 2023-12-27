# Hackernews Stars Predictor
This project aims to **predict the number of stars a post will receive on Hackernews**. It is an AI project for the KTH course [Scalable Machine Learning and Deep Learning](https://id2223kth.github.io/).

## Data
The dataset contains information about posts on hackernews such as the number of comments and the number of starts the post received, the karma of the user who posted the post, the time the post was created, and the title of the post. The dataset is generated based on the [Hackernews API](https://github.com/HackerNews/API).

## Pipelines
The project contains two pipelines: **inference** and **feature** pipelines. The feature pipeline extracts everyday the data from hackernews to update the dataset. The inference pipeline trains a model and saves it to a file. The prediction pipeline loads the model and predicts the number of stars a post will receive based on the title of the post, the number of comments, and the time the post was created.