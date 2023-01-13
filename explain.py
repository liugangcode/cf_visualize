import os
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from joblib import dump, load
from sklearn.manifold import TSNE

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

import dice_ml



def get_args():
    # settings
    parser = argparse.ArgumentParser(description='Models for polymer predictions')
    parser.add_argument('--query', type=int, default=0,
                        help='query index')
    parser.add_argument('--use_index', type=int, default=2,
                        help='use model indexfrom 5 cv')
    parser.add_argument('--use_model_test', type=bool, default=False,
                        help='use test set from the cross validation splitting')
    parser.add_argument('--num_explain', type=int, default=2,
                        help='number of counterfactual explanations')
    args = parser.parse_args()
    return args

def main(args):
    data_df = pd.read_csv('NF-BoT-IoT.csv')
    use_index = args.use_index
    query_index = args.query

    model = load('model_cache/BoT_IoT_rf_{}.joblib'.format(use_index))
    print('model', model)
    cache_data = np.load('model_cache/index_{}.npz'.format(use_index))
    train_idx, test_idx = cache_data['train'], cache_data['test']
    data_df_features = data_df.drop(['Attack','IPV4_SRC_ADDR', 'L4_SRC_PORT','IPV4_DST_ADDR', 'L4_DST_PORT'], axis=1)
    d = dice_ml.Data(dataframe=data_df_features.loc[train_idx],
                    continuous_features=['PROTOCOL','L7_PROTO', 'IN_BYTES','OUT_BYTES', 'IN_PKTS','OUT_PKTS','TCP_FLAGS','FLOW_DURATION_MILLISECONDS'],
                    outcome_name='Label')
    m = dice_ml.Model(model=model, backend="sklearn")
    exp = dice_ml.Dice(d, m, method='random')
    # exp = dice_ml.Dice(d, m, method='genetic')

    if args.use_model_test:
        query_instance = data_df_features.drop(columns="Label").loc[test_idx][query_index:query_index+1]
        query_label = data_df_features["Label"].loc[test_idx].to_numpy()[query_index:query_index+1]
    else:
        query_instance = data_df_features.drop(columns="Label")[query_index:query_index+1]
        query_label = data_df_features["Label"].to_numpy()[query_index:query_index+1]
    
    # print('real query label value', query_label)
    
    #---- starting generate counterfactuals ----#
    dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=args.num_explain, desired_class="opposite", random_seed=0)
    dice_exp.visualize_as_dataframe()
    cfe_res = dice_exp.cf_examples_list[0].final_cfs_df

    query_feature = query_instance.to_numpy()
    cfe_feature = cfe_res.drop(columns="Label").to_numpy()
    cfe_labels = cfe_res["Label"].to_numpy()
    #---- ending generate counterfactuals ----#

    ### transform the feature to visualize
    features_to_visualize = np.concatenate((query_feature, cfe_feature), axis=0)
    label_to_visulize = np.concatenate((query_label, cfe_labels), axis=0)
    features_train = data_df_features.drop(columns="Label").loc[train_idx].to_numpy()
    label_train = model.predict(data_df_features.drop(columns="Label").loc[train_idx])

    features_to_visualize_scaled = stats.zscore(features_to_visualize, axis=1, ddof=0)
    features_train_scaled = stats.zscore(features_train, axis=1, ddof=0)

    plot_only_test = False
    visual_model = TSNE(n_components=2, init='pca', random_state=0, learning_rate='auto', n_iter=500, verbose=0, perplexity=2, metric='l2', n_jobs=15)
    def select_examples(features, labels, num_examples=4000):
        # Select num_examples examples from each class
        selected_examples = []
        unique_labels= np.unique(labels)
        for i in unique_labels:
            selected_examples.extend(np.where(labels == i)[0][:num_examples])
        return features[selected_examples], labels[selected_examples]
    ### subsample for tsne visualization
    features_train_scaled, label_train = select_examples(features_train_scaled, label_train)
    if plot_only_test:
        tot_feat = features_to_visualize_scaled
    else:
        tot_feat = np.concatenate((features_train_scaled, features_to_visualize_scaled), axis=0)
    X_tot = visual_model.fit_transform(tot_feat)
    if plot_only_test:
        X_train = tot_feat
        label_train = label_to_visulize
    else:
        X_train, X_visual = X_tot[:len(features_train_scaled)], X_tot[len(features_train_scaled):]


    # define color map
    cmap = plt.cm.Set1
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # print('cmaplist', cmaplist[0], cmaplist[1])
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('mcm', cmaplist, cmap.N)
    # define figure and axis
    fig = plt.figure(1, figsize=(10, 6))
    plt.clf()
    ax = fig.add_subplot(111)

    #---- define plotting for sampled training examples ----#
    def plot_train_sampled_points(ax, X_train, label_train, cmaplist):
        unique_labels= np.unique(label_train)
        markers = ['x', '+', 'o', 'v', '^', '<', '>', 's', 'p', 'P', '*', 'h', 'H', 'X', 'D', 'd', '|', '_']
        for i in unique_labels:
            cur_idx = np.where(label_train == i)[0]
            cur_x, cur_y = X_train[cur_idx], label_train[cur_idx]
            cur_color = cmaplist[cur_y[0]]
            marker = markers[cur_y[0]]
            ax.scatter(cur_x[:, 0], cur_x[:, 1], color=cur_color, s=25, label='Predicted Label {}'.format(cur_y[0]), marker=marker, zorder=-1, alpha=0.2)

    if not plot_only_test:
        plot_train_sampled_points(ax, X_train, label_train, cmaplist)

    #---- define plotting for query and explanation ----#
    def plot_query_cf_points(ax, X_visual, label_to_visulize, cmaplist, explained_size=150):
        ax.scatter(X_visual[0, 0], X_visual[0, 1], color=cmaplist[label_to_visulize[0]], s=explained_size, marker='P',label='Query')
        cf_marker = ['<', '>', '^', '<', 'D', '*', 's', '1','2','3','4']
        for i in range(1, len(X_visual)):
            ## default marker '*'
            color = cmaplist[label_to_visulize[i]]
            marker = cf_marker[i-1]
            ax.scatter(X_visual[i, 0], X_visual[i, 1], color=color, s=explained_size, marker=marker,label='CF Exp-{}'.format(i))
    plot_query_cf_points(ax, X_visual, label_to_visulize, cmaplist, explained_size=150)

    #---- define plotting for arrow  ----#
    def draw_arrow_with_text(ax, query_instance, cfe_res, X_visual, exp_idx=None):
        feat_name_list = ['PROTOCOL','L7_PROTO', 'IN_BYTES','OUT_BYTES', 'IN_PKTS','OUT_PKTS','TCP_FLAGS','FLOW_DURATION_MILLISECONDS']
        cfe_features = cfe_res.to_numpy()[:,:-1] 
        query_feature = query_instance.to_numpy().reshape(-1)
        cf_exp_idx, cf_exp_feat  = ((cfe_features - query_feature) != 0.).nonzero()[0], ((cfe_features - query_feature) != 0.).nonzero()[1]
        for i in np.unique(cf_exp_idx):
            if exp_idx is not None and i != exp_idx:
                continue
            annotate_txt = 'CF Exp-{} w/ Feature Change \n'.format(i+1)
            for relative_idx, j in enumerate(cf_exp_feat[cf_exp_idx == i]):
                old_feature = query_feature[j]
                new_feature = cfe_features[i, j]
                feat_name = feat_name_list[j]
                annotate_txt += '{}. {}: {}->{}\n'.format(relative_idx+1, feat_name, int(old_feature), int(new_feature))
            x_query_pos = (X_visual[0, 0], X_visual[0, 1])
            x_cf_pos = (X_visual[i+1, 0], X_visual[i+1, 1])
            ax.annotate(text='', xy=x_query_pos, xytext=x_cf_pos, arrowprops=dict(arrowstyle='<-', color='k', lw=1.5, mutation_scale=15), color='k', fontsize=6, zorder=1, weight='bold')
            x_min, x_max = ax.get_xlim()[0], ax.get_xlim()[1]
            y_min, y_max = ax.get_ylim()[0], ax.get_ylim()[1]
            ax.text(.5, .99, annotate_txt, ha='center', va='top', fontsize=10, zorder=1, weight='bold', transform=ax.transAxes)

    #---- define zoom in region ----# 
    ## show explanation in separate zoom in region
    if True:
        def zoom_in_explanation(ax, X_train, label_train, X_visual, label_to_visulize, query_instance, cfe_res, exp_idx, extend_area = 10):
            def calculate_region_min_max(ax, X_visual, exp_idx):
                x_min, x_max = min(X_visual[0, 0].min(), X_visual[exp_idx+1, 0]), max(X_visual[0, 0].max(), X_visual[exp_idx+1, 0])
                y_min, y_max = min(X_visual[0, 1].min(), X_visual[exp_idx+1, 1]), max(X_visual[0, 1].max(), X_visual[exp_idx+1, 1])
                extend_x, extend_y = (x_max-x_min)/1.5, (y_max-y_min)/1.5
                extend_xy = max(extend_x, extend_y)
                axis_x_min, axis_x_max = ax.get_xlim()[0], ax.get_xlim()[1]
                axis_y_min, axis_y_max = ax.get_ylim()[0], ax.get_ylim()[1]
                return max(axis_x_min, x_min - extend_xy), min(axis_x_max, x_max + extend_xy), max(axis_y_min, y_min - extend_xy), min(axis_y_max, y_max + extend_xy)

            def calculate_core_region_ratio(X_train, x_min, x_max, y_min, y_max):
                zoom_x_len, zoom_y_len = x_max - x_min, y_max - y_min
                train_x_max, train_y_max = X_train.max(axis=0)[0], X_train.max(axis=0)[1]
                train_x_min, train_y_min = X_train.min(axis=0)[0], X_train.min(axis=0)[1]
                all_x_len, all_y_len = train_x_max - train_x_min, train_y_max - train_y_min
                core_area_ratio = all_x_len*all_y_len / ( zoom_x_len*zoom_y_len)
                return core_area_ratio
            x_min, x_max, y_min, y_max = calculate_region_min_max(ax, X_visual, exp_idx)
            core_area_ratio = calculate_core_region_ratio(X_train, x_min, x_max, y_min, y_max)
            zoom_scale, borderpad = 10**(np.log10(core_area_ratio)/2) / 2, -10
            print('ratio: ', core_area_ratio, 'zoom_scale', zoom_scale, 'ratio/zoom_scale: ', core_area_ratio/zoom_scale)
            zoom_in_are_list = ['lower left', 'upper right', 'upper left', 'lower right', 'right', 'center left', 'center right', 'upper center', 'lower center', 'center']
            bbox_list = [(0,0), (1,1), (0,1), (1,0), (1,0.5), (0,0.5), (0.5,1), (0.5,0)]
            locs = [[2,4], [2,4], [1,3],[1,3], [1,4], [2,3], [1,2],[3,4]]
            position_indexing_order = [0, 3, 1, 2, 4, 5, 6, 7]
            pos_idx = position_indexing_order[exp_idx]
            axins = zoomed_inset_axes(ax, zoom_scale, loc=zoom_in_are_list[pos_idx], bbox_to_anchor=bbox_list[pos_idx], bbox_transform=ax.transAxes, borderpad=borderpad)
            cur_locs = locs[pos_idx]
            mark_inset(ax, axins, loc1=cur_locs[0], loc2=cur_locs[1], fc="none", ec="0.5", lw=2)
            axins.set_xlim([x_min,x_max])
            axins.set_ylim([y_min,y_max])
            plot_train_sampled_points(axins, X_train, label_train, cmaplist)
            plot_query_cf_points(axins, X_visual, label_to_visulize, cmaplist, explained_size=75)
            draw_arrow_with_text(axins, query_instance, cfe_res, X_visual, exp_idx=exp_idx)
        
        for exp_idx in range(1, len(X_visual)):
            exp_idx -= 1
            zoom_in_explanation(ax, X_train, label_train, X_visual, label_to_visulize, query_instance, cfe_res, exp_idx, extend_area = 10)

    n_col = 3 + args.num_explain
    if args.num_explain > 2:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=n_col, frameon=False)
    else:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=n_col, frameon=False)

    os.makedirs('figures/', exist_ok=True)
    fig_name = 'figures/use{}_query{}'.format(args.use_index, args.query)

    ## write to text
    # query_cp = query_instance.copy()
    # query_cp['Label'] = query_label
    # query_cp = pd.concat([query_cp, cfe_res])
    # query_cp[query_cp.columns.values] = query_cp[query_cp.columns.values].astype(int)
    # query_cp.to_csv(fig_name+'.txt', sep='\t', index=False)

    plt.savefig(fig_name, bbox_inches='tight')
    return dice_exp


if __name__ == "__main__":
    args = get_args()
    os.makedirs('model_cache', exist_ok=True)
    main(args)
