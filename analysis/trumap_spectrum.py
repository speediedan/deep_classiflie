import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from adjustText import adjust_text
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple


class TrumapSpectrum(object):
    def __init__(self, ts_data: Tuple) -> None:
        (self.vals, self.labels, self.target_token_tup, self.ss_name, self.umap_bounds_tup) = ts_data
        plt.ioff()
        plt.xkcd()
        self.fig = plt.figure(figsize=(3, 2))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.build_ts()

    def build_ts(self) -> None:
        # have broken the obscenely complicated mpl configuration into a sequence of functions
        self.title_config()
        self.axis_config()
        self.plt_data()
        self.legend_config()
        self.plot_data_annotations()
        self.plot_finalize_save()

    def title_config(self) -> None:
        # config title
        ttl = self.ax.title
        ttl.set_position([.5, 1.05])
        title_props = dict(boxstyle='round, pad=0.5', facecolor='lightgrey', alpha=0.4)
        self.ax.set_title(f'Most Similar Statements/Predictions from the Training Set', fontsize=8, color='darkgreen',
                          bbox=title_props, fontweight='normal', pad=5)

    def axis_config(self) -> None:
        # config x-axis and data points
        # TODO: change to bulk set (self.ax.set(**kwargs))
        self.ax.set_xmargin(0.1)
        self.ax.set_ymargin(0.1)
        self.ax.set_zmargin(0.1)
        self.ax.autoscale()
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])
        self.ax.minorticks_off()
        self.ax.xaxis.pane.set_edgecolor('grey')
        self.ax.yaxis.pane.set_edgecolor('grey')
        self.ax.zaxis.pane.set_edgecolor('grey')
        self.ax.xaxis.pane.set_alpha(0.8)
        self.ax.yaxis.pane.set_alpha(0.8)
        self.ax.zaxis.pane.set_alpha(0.8)
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False

    def plt_data(self) -> None:
        self.ax.scatter(self.target_token_tup[2][0], self.target_token_tup[2][1], self.target_token_tup[2][2],
                        color="darkblue", s=3)
        for i, (v, l) in enumerate(zip(self.vals, self.labels)):
            self.ax.scatter(self.vals[i][0], self.vals[i][1], self.vals[i][2], color="darkgreen", s=3) if l == 0 \
                else self.ax.scatter(self.vals[i][0], self.vals[i][1], self.vals[i][2], color="red", s=3)

    def legend_config(self) -> None:
        falsehoods, no_falsehoods, tbd = (mpatches.Patch(color='red', label='WP labeled False'),
                                                      mpatches.Patch(color='darkgreen', label='No WP False label'),
                                                      mpatches.Patch(color='darkblue', label='Target: Label TBD'))
        leg = self.ax.legend(handles=[falsehoods, no_falsehoods, tbd], bbox_to_anchor=(-0.70, 0.55),
                             loc='lower left', prop={'size': 6}, labelspacing=0.2)
        for handle, text in zip(leg.legendHandles, leg.get_texts()):
            text.set_color((handle.get_facecolor()[0], handle.get_facecolor()[1], handle.get_facecolor()[2]))

    def plot_data_annotations(self) -> None:
        texts = []
        for i, (v, l) in enumerate(zip(self.vals, self.labels)):
            texts.append(
                self.ax.text(v[0], v[1], v[2], i, fontsize=7, color="darkgreen", zorder=0, rotation=0)) if l == 0 \
                else texts.append(self.ax.text(v[0], v[1], v[2], i, fontsize=7, color="red", zorder=0, rotation=0))
        texts.append(self.ax.text(self.target_token_tup[2][0], self.target_token_tup[2][1], self.target_token_tup[2][2],
                                  'Target', weight="bold", fontsize=7, color="darkblue", rotation=0))
        adjust_text(texts, autoalign='xy')

    def plot_finalize_save(self) -> None:
        self.fig.savefig(self.ss_name, dpi=150, bbox_inches='tight', pad_inches=0.15, transparent=False,
                         backend="cairo")
        plt.close(self.fig)
