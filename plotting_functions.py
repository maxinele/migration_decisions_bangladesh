import pandas as pd
import matplotlib as mp # Core matplotlib library
import matplotlib.pyplot as plt # Matplotlib plotting functions
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcol
import seaborn as sns
import numpy as np
import shap

# Get (customized) best fold or mean plot for every model 
def plot_barshap(
    dict_shaps, 
    dict_train_cols,
    name_dict,
    do_mean,
    do_best,
    modelname,
    bestfold=None,
    n_range=None, 
    selected_folds=None,
    threshold=None,
    n_max_values=None,
    figure_label=None,
    save_fig_path=None,
        
):
    # Intiate whether to plot only best fold or take the mean over all repetitions
    if do_mean:
        ls_shaps = []
        for fold in range(n_range):
            shapvals_sfold = pd.DataFrame(
                dict_shaps[modelname][fold].values,
                columns=dict_train_cols[modelname][fold].columns
            ).abs().mean()
            ls_shaps.append(shapvals_sfold)
        shapvals_s = pd.concat(ls_shaps,axis=1).mean(1).sort_values(ascending=False)

    elif do_best:
        ls_shaps = []
        for fold in selected_folds:
            shapvals_sfold = pd.DataFrame(
                dict_shaps[modelname][fold].values,
                columns=dict_train_cols[modelname][fold].columns
            ).abs().mean()
            ls_shaps.append(shapvals_sfold)
        shapvals_s = pd.concat(ls_shaps,axis=1).mean(1).sort_values(ascending=False)

    else:
        shapvals_s = pd.DataFrame(
            dict_shaps[modelname][bestfold].values,
            columns=dict_train_cols[modelname][bestfold].columns
        ).abs().mean().sort_values(ascending=False)
        
    shapvals_top = shapvals_s.copy()
    if threshold:
        shapvals_top = shapvals_s[shapvals_s >= threshold]
        shapvals_rest = shapvals_s[shapvals_s < threshold]
        shapvals_top[f'Sum of {len(shapvals_rest)} other features'] = shapvals_rest.sum()

    if n_max_values:
        shapvals_top = shapvals_top.iloc[:n_max_values]
        shapvals_rest = shapvals_s.iloc[n_max_values:]
        shapvals_top[f'Sum of {len(shapvals_rest)} other features'] = shapvals_rest.sum()

    print(shapvals_top)
    print(shapvals_top.sum())
    shapvals_plot = shapvals_top[::-1]
    shapvals_plot = shapvals_plot.rename(name_dict)
        
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.barh(range(len(shapvals_plot)), shapvals_plot,color="darkblue")

    # Axes
    plt.yticks(range(len(shapvals_plot)),shapvals_plot.index)
    plt.xlabel('mean(|SHAP value|)',size=14)
    
    # Adjust x-axis ticks to show every other value
    x_ticks = ax.get_xticks()  # Get current ticks
    ax.set_xticks(x_ticks[::2])  # Set every other tick

    # Layout
    ax.grid(axis='y',color='grey', linestyle='--', linewidth=0.3)
    ax.set_facecolor('white')
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    
    if figure_label is not None:
        plt.text(1.0, -0.1, figure_label, transform=ax.transAxes,
                 fontsize=20, verticalalignment='top', horizontalalignment='right')

    if save_fig_path != None:
        plt.savefig(save_fig_path,bbox_inches="tight", dpi=400,
                    transparent=False)
        plt.show(fig)
        plt.close()


    else:

        plt.show(fig)

def plot_barshap_sumfeatures(
    dict_shaps, 
    dict_train_cols,
    modelname,
    feature_sets,
    feature_set_names,
    do_mean=False,
    bestfold=None,
    n_range=None, 
    threshold=None,
    n_max_values=None,
    figure_label=None,
    save_fig_path=None,
):
    # --- aggregate SHAP values ---
    if do_mean:
        ls_shaps = []
        for fold in range(n_range):
            shapvals_sfold = pd.DataFrame(
                dict_shaps[modelname][fold].values,
                columns=dict_train_cols[modelname][fold].columns
            ).abs().mean()

            # sum within feature sets
            collect_sums = [shapvals_sfold.loc[feat].sum() for feat in feature_sets]
            ls_shaps.append(collect_sums)

        shapvals_top = pd.DataFrame(ls_shaps, columns=feature_set_names).mean().sort_values(ascending=False)

    elif bestfold is not None:
        shapvals_s = pd.DataFrame(
            dict_shaps[modelname][bestfold].values,
            columns=dict_train_cols[modelname][bestfold].columns
        ).abs().mean()

        collect_sums = [shapvals_s.loc[feat].sum() for feat in feature_sets]
        shapvals_top = pd.Series(collect_sums, index=feature_set_names).sort_values(ascending=False)

    else:
        raise ValueError("Specify do_mean=True or bestfold index.")

    # --- threshold / top-N on feature sets ---
    if threshold:
        keep = shapvals_top[shapvals_top >= threshold]
        rest = shapvals_top[shapvals_top < threshold]
        keep[f'Sum of {len(rest)} other feature sets'] = rest.sum()
        shapvals_top = keep

    if n_max_values:
        keep = shapvals_top.iloc[:n_max_values]
        rest = shapvals_top.iloc[n_max_values:]
        keep[f'Sum of {len(rest)} other feature sets'] = rest.sum()
        shapvals_top = keep

    # --- plot ---
    shapvals_plot = shapvals_top[::-1]  # reverse for barh

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(shapvals_plot)), shapvals_plot, color="darkblue")

    ax.set_yticks(range(len(shapvals_plot)))
    ax.set_yticklabels(shapvals_plot.index, fontsize=14)
    ax.set_xlabel('mean(|SHAP value|)', size=14)
    ax.grid(axis='y', color='grey', linestyle='--', linewidth=0.3)
    ax.set_facecolor('white')
    ax.spines[['top','right']].set_visible(False)

    if figure_label:
        ax.text(1.0, -0.1, figure_label, transform=ax.transAxes,
                fontsize=20, verticalalignment='top', horizontalalignment='right')

    if save_fig_path:
        plt.savefig(save_fig_path, bbox_inches="tight", dpi=400, transparent=False)
        plt.show(fig)
        plt.close()
    else:
        plt.show(fig)


def beeswarmplot_continuous(
    shapvals,
    model,
    varname, 
    name_dict,
    colvarname=None,
    path_figure=None,
    figure_label=None,
):
    mname = model['model_name']
    shap_exp = shapvals   # can be pooled or single fold

    # feature name to color by
    colvar = colvarname if colvarname else varname

    # Build dataframe
    df_shap = pd.DataFrame(
        shap_exp.values,
        columns=shap_exp.feature_names
    )
    
    df_actuals = pd.DataFrame(
        shap_exp.data,
        columns=shap_exp.feature_names
    )[[colvar]].add_suffix('_actuals')

    df_combined = df_shap.join(df_actuals)

    # --- plotting ---
    fig, ax = plt.subplots(figsize=(15, 10))  
    cmap = mcol.LinearSegmentedColormap.from_list("MyCmapName",["blue","r"])
    norm = matplotlib.colors.Normalize(
        vmin=df_combined[f'{colvar}_actuals'].min(), 
        vmax=df_combined[f'{colvar}_actuals'].max()
    )

    sns.swarmplot(
        data=df_combined, 
        x=varname,
        y=[""]*len(df_combined[varname]),
        hue=f"{colvar}_actuals",
        size=2.5,
        palette={val: cmap(norm(val)) for val in df_combined[f'{colvar}_actuals']}
    )
    plt.gca().legend_.remove()

    ax.set_facecolor('white')
    ax.grid(axis='y',color='grey', linestyle='--', linewidth=0.3)
    ax.grid(axis='x',color='grey', linestyle='--', linewidth=0.3)
    ax.margins(x=0,y=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)
    plt.axvline(x=0, color='black', linestyle='--')

    # colorbar
    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    fig.add_axes(ax_cb)
    cb1 = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=cmap, norm=norm, orientation='vertical')

    # Labels
    renamed_string = name_dict.get(colvar, colvar)
    renamed_string_v = name_dict.get(varname, varname)
    cb1.set_label(f'Values for {renamed_string}', size=22)
    ax.set_xlabel(f'Shap values for {renamed_string_v}', size=22) 

    if figure_label is not None:
        plt.text(0, 1.05, figure_label, transform=ax.transAxes,
                 fontsize=20, verticalalignment='top', horizontalalignment='left')

    if path_figure: 
        plt.savefig(path_figure, bbox_inches="tight", dpi=400, transparent=False)
        plt.show(fig)
    else:
        plt.show(fig)

def df_interactions_abs(
    shap_interactions_dict,
    train_dict,
    model,
    name_dict,
    inter_vars,
    selected_folds=None,
    remove_zeros=False,
    agg_effect=False,
    path_out=None,
    save_fig=False, 
    colorpalette='inferno_r',
):
    mname = model['model_name']
    
    # collect fold-wise interaction matrices
    mats = []
    for f in (selected_folds if selected_folds is not None else shap_interactions_dict[mname].keys()):
        mean_shap = np.abs(shap_interactions_dict[mname][f]).mean(0)
        mats.append(mean_shap)
    
    # average across folds
    mean_shap = np.mean(mats, axis=0)
    
    # get dataframe
    df_shap_inter = pd.DataFrame(
        mean_shap, 
        index=train_dict[mname][f].columns, 
        columns=train_dict[mname][f].columns
    )
    
    # double the diagonal
    np.fill_diagonal(df_shap_inter.values, np.diagonal(df_shap_inter) * 2)
    
    # select variables of interest
    df_plot = df_shap_inter.loc[inter_vars, inter_vars].rename(columns=name_dict, index=name_dict)
    
    if remove_zeros:
        remov_columns = df_plot.loc[:, ~(df_plot > 0.001).any(axis=0)].columns
        inter_vars = [v for v in inter_vars if v not in remov_columns]
        df_plot = df_shap_inter.loc[inter_vars, inter_vars].rename(columns=name_dict, index=name_dict)

    # plot
    mask = np.triu(np.ones_like(df_plot, dtype=bool))
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        df_plot.round(3), mask=mask, cmap=colorpalette, annot=True, cbar=True, ax=ax,
        square=True, cbar_kws={"shrink": 0.8}, linewidths=.5
    )
    plt.yticks(rotation=0, size=14)
    plt.xticks(rotation=90, size=14)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    
    # diagonal fill
    for i in range(len(df_plot)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=True, color='#E5E4E2'))
    
    ax.add_patch(plt.Rectangle((0, 0), df_plot.shape[1], df_plot.shape[0], 
                               fill=False, edgecolor='black', lw=3, zorder=1))
    ax.grid(False)
    
    if save_fig and path_out is not None:
        plt.savefig(path_out, bbox_inches="tight", dpi=400, transparent=False)
        plt.show(fig)
        plt.close()
    else:
        plt.show(fig)

    return df_plot, df_shap_inter

def pool_shap_selected(shapdict_model, selected_folds, sample_size=10000, random_state=1308):
    """
    Pool SHAP values across selected folds for one model.
    Optionally subsample rows for plotting.
    """
    # concatenate SHAP values, data, and base_values
    values = np.concatenate([shapdict_model[f].values for f in selected_folds], axis=0)
    data   = np.concatenate([shapdict_model[f].data   for f in selected_folds], axis=0)
    base   = np.concatenate([shapdict_model[f].base_values for f in selected_folds], axis=0)

    feature_names = shapdict_model[selected_folds[0]].feature_names

    # subsample if too many rows
    n = values.shape[0]
    if sample_size and n > sample_size:
        rng = np.random.default_rng(seed=random_state)
        idx = rng.choice(n, size=sample_size, replace=False)
        values, data, base = values[idx], data[idx], base[idx]

    return shap.Explanation(values=values,
                            data=data,
                            base_values=base,
                            feature_names=feature_names)