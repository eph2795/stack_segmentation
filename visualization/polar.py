def plot_polar(
        title,
        models,
        stacks,
        metrics_th,
        data,
        stacks_name_mapping,
        models_name_mapping,
        markers,
        h, w):
    plt.style.use('seaborn-white')
    subplot_kw = {'projection': 'polar'}
    gridspec_kw = {'left': 0.1, 'right': 0.9, 'bottom': 0.05, 'top': 0.95,
                   'wspace': 0.1, 'hspace': 0.075}

    fig_kw = {'figsize': (10 * w, 10 * h)}
    title_properties = {'weight': 'semibold', 'size': 30}
    legend_properties = {'weight': 'semibold', 'size': 30}
    ylabels_properties = {'weight': 'semibold', 'size': 16}
    xlabels_properties = {'weight': 'semibold', 'size': 22}

    fig, axes = plt.subplots(h, w,  # sharey='row', #sharex='col',
                             subplot_kw=subplot_kw,
                             gridspec_kw=gridspec_kw,
                             **fig_kw)

    axes = list(chain(*axes))

    for i, (stack, axis) in enumerate(
            zip(sorted(stacks, key=lambda item: stacks_name_mapping[item]), axes)):

        results = data[data['stack'] == stack]

        x = list(np.linspace(0, 2 * np.pi, len(results) + 1))
        axis.set_xticks(x)
        labels = [models_name_mapping[item] for item in results['model']]
        #         axis.set_thetagrids(np.linspace(0, 360, len(results) + 1), frac=1.5)
        axis.tick_params(axis='x', pad=10)
        axis.set_xticklabels(labels,
                             fontdict=xlabels_properties)

        for j, metric_name in enumerate(metrics_th):
            y = list(results[metric_name])
            #             x.append(x[0])
            axis.plot(x, y + [y[0]],
                      linestyle='-',
                      marker=markers[j],
                      label=metric_name,
                      markersize=15,
                      linewidth=4)  # , label=stacks_name_mapping[k], linewidth=3)
        #             items = sorted(results.items(), key=lambda item: stacks_name_mapping[item[0]])
        #             for k, v in items:
        #                 x, y = v[metric_name]
        #                 axis.plot(x, y, label=stacks_name_mapping[k], linewidth=3)
        #             axis.set_xlim([0, 1])
        #             axis.set_ylim([0, 1.05])

        #             plt.xticks(np.arange(len(y) - 1), results['Stack'])

        #             axis.set_rticks(np.arange(0.8, 1.0, 21))

        #             axis.set_yticks([0] + list(np.arange(0.05, 0.951, 0.1)) + [1])
        #             axis.tick_params(labelsize=18)
        #             axis.set_xlabel('Threshold', fontsize=22)
        #             axis.set_ylabel(metric_name[0].upper() + metric_name[1:], fontsize=22)
        axis.set_rmin(0.7)
        axis.set_rmax(1)
        axis.set_rticks(np.linspace(0.8, 1, 9))
        axis.set_yticklabels([round(x, 3) for x in np.linspace(0.8, 1, 9)],
                             fontdict=ylabels_properties)
        if i == len(stacks) - 1:
            axis.legend(loc='lower right',
                        bbox_to_anchor=(0, 0, 1.8, 1.8),
                        prop=legend_properties)
        #         else:
        #             axis.legend(loc='center', bbox_to_anchor=(0, 0, 0.7, 0.7), fontsize=16)
        axis.set_title('{}) Models quality on {}'.format(string.ascii_letters[i],
                                                         stacks_name_mapping[stack]),
                       fontdict=legend_properties)

    if len(stacks) < len(axes):
        axes[-1].set_visible(False)

    #     fig.suptitle('Models quality on diffenent stacks', fontdict=title_properties)
    #     fig.text(0.5, 0.025, 'Threshold', ha='center', va='center', fontsize=40)
    #     fig.text(0.05, 0.5, metric_name.upper(),
    #              ha='center', va='center', rotation=90, fontsize=40)

    plt.savefig('./article_results/polar_plot_{}.jpg'.format(title))
    plt.show()