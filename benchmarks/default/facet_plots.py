from plotnine import *
import pandas as pd

# df1 = pd.read_csv("default_results_first120.csv")
# df2 = pd.read_csv("default_results_last120.csv")
# df = pd.concat([df1, df2])


# indexNames = df[df['Percentage-removal'] >= 5.0].index
# df.drop(indexNames , inplace=True)
# # import ipdb; ipdb.set_trace()
# x = (ggplot(aes(x='Percentage-removal', y='Discm_percent', color='perm'), data=df) +\
#     geom_point(size=1.5) +\
#     # geom_line(size=1.5) +\
#     # stat_smooth(colour='blue', span=0.2) +\
#     # stat_summary() +\
#     xlim(0, 5) +\
#     facet_wrap(['H1Units','H2Units','batch'], nrow=3, ncol=4, scales = 'fixed', labeller='label_both') + \
#     xlab("Biased Points Removed (percentage of training points)") + \
#     ylab("Percentage Discrimination remaining") + \
#     ggtitle("Facet plot for remaining discrimination for each setting (Default prediction)") +\
#     theme(axis_text_x = element_text(size=6), dpi=151) +\
#     theme_seaborn()
#     )

# x.save("decrease-discm_default_upto5.png", height=12, width=12)

df = pd.read_csv("default_results_pointsremoved.csv")


# indexNames = df[df['Percentage-removal'] >= 5.0].index
# df.drop(indexNames , inplace=True)
# import ipdb; ipdb.set_trace()
x = (ggplot(aes(x='Percentage-removal', y='Discm_percent', color='perm'), data=df) +\
    geom_point(size=1.5) +\
    # geom_line(size=1.5) +\
    # stat_smooth(colour='blue', span=0.2) +\
    # stat_summary() +\
    # xlim(0, 5) +\
    facet_wrap(['H1Units','H2Units','batch'], nrow=3, ncol=4, scales = 'fixed', labeller='label_both') + \
    xlab("Biased Points Removed (percentage of training points)") + \
    ylab("Percentage Discrimination remaining") + \
    ggtitle("Facet plot for remaining discrimination for each setting (Default prediction)") +\
    theme(axis_text_x = element_text(size=6), dpi=151) +\
    theme_seaborn()
    )

x.save("decrease-discm_default_pointremoved_upto5.png", height=12, width=12)

