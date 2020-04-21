

from plotnine import *
# from plotnine import ggplot, geom_point, geom_line, aes, stat_smooth, facet_wrap
# from plotnine.data import mtcars
# from plotnine import positions
# from plotnine import xlab, ylab, ggtitle, geom_boxplot, geom_path, geom_ribbon, geom_arrow
import pandas as pd
df1 = pd.read_csv("adult_high_removal_boom_first120.csv")
df2 = pd.read_csv("adult_high_removal_bam_last120.csv")
df = pd.concat([df1, df2])
# import ipdb; ipdb.set_trace()
x = (ggplot(aes(x='Removal-percentage', y='Discm_percent', color='DataSplit'), data=df) +\
    geom_point(size=1.5) +\
    # stat_smooth(colour='blue', span=0.2) +\
    # stat_summary() +\
    # ylim(0, 20) +\
    facet_wrap(['H1Units','H2Units','Batch'], nrow=3, ncol=4, scales = 'fixed', labeller='label_both', shrink=False) + \
    xlab("Biased Points Removed (percentage of training points)") + \
    ylab("Percentage Discrimination remaining") + \
    ggtitle("Facet plot for remaining discrimination for each setting (Adult income)") +\
    theme(axis_text_x = element_text(size=6), dpi=151) +\
    theme_seaborn()
    )

# x.save("summary.png", height=12, width=12)
x.save("discm_high_removal.png", height=12, width=12)

# (ggplot(mtcars, aes('wt', 'mpg', color='factor(gear)'))
#  + geom_point()
#  + stat_smooth(method='lm')
#  + facet_wrap('~gear'))

# (ggplot(ggplot.save()))


# import pandas as pd
# import numpy as np
# from pandas.api.types import CategoricalDtype
# from plotnine import *
# from plotnine.data import mpg
# %matplotlib inline

# x = (ggplot(mpg)         # defining what data to use
#  + aes(x='class')    # defining what variable to use
#  + geom_bar(size=20) # defining the type of plot to use
# )
