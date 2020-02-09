from plotnine import *
import pandas as pd
df = pd.read_csv("really_biased_removed_data.csv")
# import ipdb; ipdb.set_trace()
x = (ggplot(aes(x='Points-Removed', y='Discrimination'), data=df) +\
    geom_line(size=1.5) +\
    # stat_smooth(colour='blue', span=0.2) +\
    stat_summary() +\
    # xlim(0, 85) +\
    facet_wrap(['H1Units','H2Units','Batch'], nrow=3, ncol=4,scales = 'free', labeller='label_both', shrink=False) + \
    xlab("Real Biased Points Removed") + \
    ylab("Percentage Discrimination remaining") + \
    ggtitle("Facet plot for remaining discrimination for each setting (real biased points removed)") +\
    theme(axis_text_x = element_text(size=6), dpi=251) +\
    theme_xkcd()
    )

x.save("real_biased_points_discm.png", height=12, width=12)

df1 = pd.read_csv("really_biased_removed_data.csv")
df2 = pd.read_csv("really_biased_removed_data.csv")

df1 = df1.drop("Test-Accuracy", axis=1)
df1 = df1.rename(columns={"Discrimination":"Kind"})
df1 = df1.rename(columns={"Training-Accuracy":"Accuracy"})
df1.loc[df1['Kind'] > 0, 'Kind'] = 'Training'

df2 = df2.drop("Training-Accuracy", axis=1)
df2 = df2.rename(columns={"Discrimination":"Kind"})
df2 = df2.rename(columns={"Test-Accuracy":"Accuracy"})
df2.loc[df2['Kind'] > 0, 'Kind'] = 'Testing'

df_new = pd.concat([df1, df2])
# import ipdb; ipdb.set_trace()
y = (ggplot(aes(x='Points-Removed', y='Accuracy', color='Kind'), data=df_new) +\
    geom_line(size=1.5) +\
    # geom_path(x='Points-Removed', y='Training-Accuracy', size=1.5, inherit_aes=False) +\
    # geom_path(x='Points-Removed', y='Test-Accuracy', size=1.5, inherit_aes=False) +\
    # stat_smooth(colour='blue', span=0.2) +\
    stat_summary() +\
    # xlim(0, 85) +\
    facet_wrap(['H1Units','H2Units','Batch'], nrow=3, ncol=4,scales = 'free', labeller='label_both', shrink=False) + \
    xlab("Real Biased Points Removed") + \
    ylab("Percentage Accuracy of training model") + \
    ggtitle("Facet plot for accuracy for each setting (real biased points removed)") +\
    theme(axis_text_x = element_text(size=6), dpi=251) +\
    theme_xkcd()
    )

y.save("real_biased_points_accuracy.png", height=12, width=12)