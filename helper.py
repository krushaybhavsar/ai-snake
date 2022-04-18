import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from IPython import display

# plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    prop = font_manager.FontProperties(fname='/home/krushay/Desktop/Projects/ai-snake/monsterrat.tff')
    plt.xlabel('Number of Games', fontproperties=prop)
    plt.ylabel('Score', fontproperties=prop)
    plt.plot(scores, label='Score', color='#79c0ff')
    plt.plot(mean_scores, label='Mean Score', color='#84edc1')
    plt.ylim(ymin=0)
    plt.gcf().set_facecolor('#1f272b')
    plt.gca().set_facecolor('#1f272b')
    # set axis font
    for item in ([plt.gca().xaxis.label, plt.gca().yaxis.label] +
                    plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
            item.set_fontproperties(prop)
    plt.legend(loc='upper left', prop=prop)
    plt.gca().xaxis.label.set_color('#dfe7ef')
    plt.gca().yaxis.label.set_color('#dfe7ef')
    plt.gca().tick_params(axis='x', colors='#dfe7ef')
    plt.gca().tick_params(axis='y', colors='#dfe7ef')
    plt.gca().spines['bottom'].set_color('#dfe7ef')
    plt.gca().spines['left'].set_color('#dfe7ef')
    plt.gca().spines['top'].set_color('#dfe7ef')
    plt.gca().spines['right'].set_color('#dfe7ef')
#     plt.text(len(scores)-1, scores[-1], str(scores[-1]))
#     plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]), fontproperties=prop)
    plt.show(block=False)
    plt.pause(.1)
