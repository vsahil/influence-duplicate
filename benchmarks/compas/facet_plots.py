

from plotnine import *
# from plotnine import ggplot, geom_point, geom_line, aes, stat_smooth, facet_wrap
# from plotnine.data import mtcars
# from plotnine import positions
# from plotnine import xlab, ylab, ggtitle, geom_boxplot, geom_path, geom_ribbon, geom_arrow
import pandas as pd
# df1 = pd.read_csv("compas_results_first80.csv")
# df2 = pd.read_csv("compas_results_second80.csv")
# df3 = pd.read_csv("compas_results_last80.csv")
# df = pd.concat([df1, df2, df3])

# df1 = pd.read_csv("compas_two_year_results_first120.csv")
# df2 = pd.read_csv("compas_two_year_results_last120.csv")
# df = pd.concat([df1, df2])

# # import ipdb; ipdb.set_trace()
# x = (ggplot(aes(x='Percentage-removal', y='Discm_percent', color='perm'), data=df) +\
#     geom_point(size=1.5) +\
#     # stat_smooth(colour='blue', span=0.2) +\
#     # stat_summary() +\
#     xlim(0, 5) +\
#     facet_wrap(['H1Units','H2Units','batch'], nrow=3, ncol=4, scales = 'fixed', labeller='label_both', shrink=False) + \
#     xlab("Biased Points Removed (percentage of training points)") + \
#     ylab("Percentage Discrimination remaining") + \
#     ggtitle("Facet plot for remaining discrimination for each setting (Compas recidivism)") +\
#     theme(axis_text_x = element_text(size=6), dpi=151) +\
#     theme_seaborn()
#     )

# x.save("decrease-discm_compas_two_year_upto5.png", height=12, width=12)


df1 = pd.read_csv("compas_two_year_results_pointsremoved_first120.csv")
df2 = pd.read_csv("compas_two_year_results_pointsremoved_last120.csv")
df = pd.concat([df1, df2])

x = (ggplot(aes(x='Percentage-removal', y='Discm_percent', color='perm'), data=df) +\
    geom_point(size=1.5) +\
    # stat_smooth(colour='blue', span=0.2) +\
    # stat_summary() +\
    # xlim(0, 5) +\
    facet_wrap(['H1Units','H2Units','batch'], nrow=3, ncol=4, scales = 'fixed', labeller='label_both', shrink=False) + \
    xlab("Biased Points Removed (percentage of training points)") + \
    ylab("Percentage Discrimination remaining") + \
    ggtitle("Facet plot for remaining discrimination for each setting (Compas recidivism)") +\
    theme(axis_text_x = element_text(size=6), dpi=151) +\
    theme_seaborn()
    )

x.save("decrease-discm_compas_two_year_pointsremoved_upto6.png", height=12, width=12)


