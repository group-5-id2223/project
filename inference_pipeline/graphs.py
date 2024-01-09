import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import mean_squared_error, confusion_matrix
import hopsworks

project = hopsworks.login(project='id2223_enric')
fs = project.get_feature_store()
dataset_api = project.get_dataset_api() 

hackernews_fg = fs.get_feature_group("hackernews_predictions", 1)
new_df = hackernews_fg.read(read_options={"use_hive": True})

# Plot the predicted score vs the actual score
sns.set_theme(style="darkgrid")
sp = sns.scatterplot(x="score", y="predicted_score", data=new_df)
sp.xaxis.set_label_text("Actual score")
sp.yaxis.set_label_text("Predicted score")
sp.set_title("Actual vs Predicted score")
sp.set(xlim=(-10, 1000), ylim=(-10, 1000))
# add a padding of -1 to the plot


fig = sp.get_figure()
fig.savefig("./scatter_plot.png")
dataset_api.upload("./scatter_plot.png", "Resources/images", overwrite=True)
