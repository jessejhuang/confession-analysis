![Word Clouds of r/confession topics](https://raw.githubusercontent.com/jessejhuang/confession-analysis/3b65769ff49715ac5d068a52f2867efad3b66146/src/graphs/WordClouds.png)

### Data
1000 randomly selected r/confession comments from each month of 2017

[ Source](https://www.reddit.com/r/bigquery/comments/3cej2b/17_billion_reddit_comments_loaded_on_bigquery/) 


### Background:
Last year I did a CS class project on topic modeling 
of r/confession using Latent Dirichlet allocation. The 
project report with my initial implementation details
can be found [here](https://github.com/jessejhuang/confession-analysis/blob/master/report.pdf). The main change 
I’ve made since the class is switching from a scatter 
plot to word clouds for each topic. Each word cloud 
was created from the 25 words most strongly associated 
with a topic (not including stop words).
