from plotnine import *
import pandas as pd

def discrimination_plot():
    # df = pd.read_csv("really_biased_removed_without_test_points_removed.csv")
    df = pd.read_csv("really_biased_removed_biased_test_points_removed.csv")
    df['Discrimination'] = df['Discrimination'].div(1000)

    # import ipdb; ipdb.set_trace()
    x = (ggplot(aes(x='Points-Removed', y='Discrimination'), data=df) +\
        geom_line(size=1.5) +\
        stat_smooth(colour='blue', span=0.3) +\
        # stat_summary() +\
        # xlim(0, 85) +\
        facet_wrap(['H1Units','H2Units','Batch'], nrow=3, ncol=4,scales = 'fixed', labeller='label_both', shrink=False) + \
        xlab("Real Biased Points Removed") + \
        ylab("Percentage Discrimination remaining") + \
        # ggtitle("Facet plot for remaining discrimination(biased training points removed)") +\
        ggtitle("Facet plot for remaining discrimination(biased training and testing points removed)") +\
        theme(axis_text_x = element_text(size=6), dpi=51) +\
        theme_seaborn()
        )

    # x.save("real_biasedpoints_removed_plots/discm_without_test_points_removed.png", height=12, width=12)
    x.save("real_biasedpoints_removed_plots/discm_test_points_removed.png", height=12, width=12)


def accuracy_plots():
    df1 = pd.read_csv("really_biased_removed_without_test_points_removed.csv")
    df2 = pd.read_csv("really_biased_removed_without_test_points_removed.csv")

    # df1 = pd.read_csv("really_biased_removed_biased_test_points_removed.csv")
    # df2 = pd.read_csv("really_biased_removed_biased_test_points_removed.csv")

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
        # geom_ribbon(size=1.5, alpha=1, ymin=0.0, ymax = 0.90) +\
        geom_line(size=1.5, alpha=1) +\
        geom_smooth(size=1.5, alpha=1) +\
        # geom_path(x='Points-Removed', y='Training-Accuracy', size=1.5, inherit_aes=False) +\
        # geom_path(x='Points-Removed', y='Test-Accuracy', size=1.5, inherit_aes=False) +\
        # stat_smooth(colour='blue', span=0.2) +\
        # stat_summary() +\
        # xlim(0, 85) +\
        facet_wrap(['H1Units','H2Units','Batch'], nrow=3, ncol=4, labeller='label_both', shrink=False) + \
        xlab("Real Biased Points Removed") + \
        ylab("Percentage Accuracy of training model") + \
        ggtitle("Facet plot for accuracy metrics (biased training points removed)") +\
        # ggtitle("Facet plot for accuracy metrics (biased training and testing points removed)") +\
        theme(axis_text_x = element_text(size=6), dpi=51) +\
        theme_seaborn()
        )

    y.save("real_biasedpoints_removed_plots/accuracy_without_test_removed.png", height=12, width=12)
    # y.save("real_biasedpoints_removed_plots/accuracy_test_removed.png", height=12, width=12)


if __name__ == "__main__":
    discrimination_plot()
    # accuracy_plots()