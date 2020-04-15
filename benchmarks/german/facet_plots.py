

from plotnine import *
# from plotnine import ggplot, geom_point, geom_line, aes, stat_smooth, facet_wrap
# from plotnine.data import mtcars
# from plotnine import positions
# from plotnine import xlab, ylab, ggtitle, geom_boxplot, geom_path, geom_ribbon, geom_arrow
import pandas as pd
# df = pd.read_csv("all_german_discm_data.csv")
df = pd.read_csv("german_credit_mega_results0.csv")
# import ipdb; ipdb.set_trace()
x = (ggplot(aes(x='Removed-points', y='Discm-percent', color='Data-Split'), data=df) +\
    geom_point(size=1.5) +\
    # stat_smooth(colour='blue', span=0.2) +\
    # stat_summary() +\
    # xlim(0, 85) +\
    facet_wrap(['H1Units','H2Units','Batch'], nrow=3, ncol=4,scales = 'fixed', labeller='label_both', shrink=False) + \
    xlab("Biased Points Removed") + \
    ylab("Percentage Discrimination remaining") + \
    ggtitle("Facet plot for remaining discrimination for each setting (German Credit)") +\
    theme(axis_text_x = element_text(size=6), dpi=50) +\
    theme_seaborn()
    )

# x.save("summary.png", height=12, width=12)
x.save("points_redone_upto700.png", height=12, width=12)

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
