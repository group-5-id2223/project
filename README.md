# Hackernews Stars Predictor
This project aims to **predict the number of stars a post will receive on Hackernews**. It is an AI project for the KTH course [Scalable Machine Learning and Deep Learning](https://id2223kth.github.io/).

- Problem description: what are the data sources and the prediction problem that you will be building a ML System for?
- Tools: what tools you are going to use? In the course we mainly used Decision Trees and PyTorch/Tensorflow, but you are free to explore new tools and technologies.
- Data: what data will you use and how are you going to collect it?
- Methodology and algorithm: what method(s) or algorithm(s) are you proposing?

## Data
The dataset contains information about posts on hackernews such as the number of comments and the number of starts the post received, the karma of the user who posted the post, the time the post was created, and the title of the post. The dataset is generated based on the [Hackernews API](https://github.com/HackerNews/API).

## Pipelines
The project contains two pipelines: **inference** and **feature** pipelines. The feature pipeline extracts everyday the data from hackernews to update the dataset. The inference pipeline trains a model and saves it to a file. The prediction pipeline loads the model and predicts the number of stars a post will receive based on the title of the post, the number of comments, and the time the post was created.

## Methodology
The based is based on the relevance of the title of the post and the domain its posted on given the hour of a day in a week. Title is tokenized and and embedded with a Linear Tokenizer and passed through an LSTM to learn sequential features. The output of the LSTM is concatenated to the categorized domain of the post URL along with the hour and the day category.

These concatenated features are then passed through a linear layers to obtain the predicted score.
Minimizing `mse` loss as typical for regression problems will not work, as the model will realize that selecting 1 unilaterally accomplishes this task the best.

Instead, create a hybrid loss of `mae`, `msle`, and `poisson` (see Keras's docs for more info: https://github.com/keras-team/keras/blob/master/keras/losses.py) The latter two losses can account for very high values much better; perfect for the hyper-skewed data.

## Websites
Online: [link](https://huggingface.co/spaces/ID2223/hackernews-upvotes-predictor)
Monitor: [link](https://huggingface.co/spaces/ID2223/hackernews-upvotes-predictor-monitor)