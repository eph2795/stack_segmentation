def plot_thresholded_metric(models, metrics_th, h, w):
    plt.style.use('seaborn-white')
    fig_kw = {'figsize': (10 * w, 10 * h)}
    gridspec_kw = {  # 'nrows': h, 'ncols': 2 * w,
        'left': 0.1, 'right': 0.9, 'bottom': 0.05, 'top': 0.95,
        'wspace': 0.025, 'hspace': 0.1}
    suptitle_properties = {'weight': 'semibold', 'size': 50}
    title_properties = {'weight': 'semibold', 'size': 40}
    legend_properties = {'weight': 'semibold', 'size': 30}
    labels_properties = {'weight': 'semibold', 'size': 20}
    #     gs = gridspec.GridSpec()
    #     gs.update(left=0.1, right=0.9, bottom=0.05, top=0.95)

    #     fig = plt.figure(figsize=(10 * w , 10 * h))
    fig, axes = plt.subplots(h, w, sharey='row',  # sharex='col',
                             gridspec_kw=gridspec_kw, **fig_kw)

    axes = list(chain(*axes))
    for i, (fname, axis) in enumerate(zip(models, axes)):
        results = load('../output_data/' + fname)

        axis.set_xlim([0, 1])
        axis.set_ylim([0, 1.05])
        axis.set_xticks([round(item, 2) for item in np.arange(0.05, 0.951, 0.1)])
        axis.set_xticklabels([round(item, 2) for item in np.arange(0.05, 0.951, 0.1)],
                             fontdict=labels_properties)
        axis.set_yticks([0] + [round(item, 2) for item in np.arange(0.05, 0.951, 0.1)] + [1])
        axis.set_yticklabels([0] + [round(item, 2) for item in np.arange(0.05, 0.951, 0.1)] + [1],
                             fontdict=labels_properties)
        #             axis.tick_params(fontdict=ylabels_properties)
        axis.set_title(models_name_mapping[fname.split('.')[0]], fontsize=24)

        for metric_name in metrics_th:
            items = sorted(results.items(), key=lambda item: stacks_name_mapping[item[0]])
            #             print(type(items))
            for k, v in items:
                #                 print(k, v)
                x, y = v[metric_name]
                axis.plot(x, y,
                          label=stacks_name_mapping[k],
                          linewidth=4)

        if i == len(models) - 1:
            axis.legend(loc='lower right',
                        bbox_to_anchor=(0, 0, 1.8, 1.8),
                        prop=legend_properties)
        axis.set_title('{}) {} {} from threshold'.format(string.ascii_letters[i],
                                                         models_name_mapping[fname.split('.')[0]],
                                                         metrics_th[0]),
                       fontdict=legend_properties)
    #             axis.set_xlabel('Threshold', fontsize=22)
    #             axis.set_ylabel(metric_name[0].upper() + metric_name[1:], fontsize=22)
    if len(models) < len(axes):
        axes[-1].set_visible(False)
    #     gs.tight_layout(fig)

    #     fig=axes[0,0].figure\
    #     fig = axes[0].figure

    #     fig.suptitle('IOU from threshold dependence for all models',
    #                  fontdict=suptitle_properties)
    fig.text(0.5, 0.025, 'Threshold',
             fontdict=title_properties,
             ha='center', va='center')
    fig.text(0.05, 0.5, metric_name.upper(),
             fontdict=title_properties,
             ha='center', va='center', rotation=90)

    plt.savefig('../output_data/results/metrics_plot_{}_{}.jpg'
                .format('_'.join(models), '_'.join(metrics_th)))
    plt.show()