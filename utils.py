import numpy as np
import pandas as pd
from scipy.io import loadmat
import scipy.stats as st
import statsmodels.api as sm
from statsmodels.formula.api import ols
import glob

## Functions to read input
def load_psychophys(pp_fn):
    df = pd.read_csv(pp_fn, sep='\t')
    return df

def load_gaba(gaba_fn, pres_cond='occ_binoc'):
    gdf = pd.read_csv(gaba_fn, sep='\t')
    return gdf[gdf.Presentation==pres_cond] # this is the gaba measure we want to use

def load_fmri(fmri_fn):
    return pd.read_csv(fmri_fn, sep='\t')

## Functions for dealing with psychophysics data
def get_interocular_diff(g, field):
    if len(g) < 2:
        return g
    else:
        assert(len(g)==2)
        nde_mean = g[g.Eye=='Nde'][field].iloc[0]
        de_mean = g[g.Eye=='De'][field].iloc[0]
        g['ValueDiff'] = nde_mean - de_mean
        g['ValueRatio'] = nde_mean/de_mean
        return g

def big_anova(df, model_str):
    """Do the big ANOVA on psychophysical data requested by reviewer #1 of CC sub."""
    model = ols(model_str, data=df).fit()
    aov_table = sm.stats.anova_lm(model, typ=2)
    return aov_table[['F', 'PR(>F)']]

def calc_rs(df, permute=False):
    if permute:
        gaba = np.random.permutation(df['GABA'])
        measure = np.random.permutation(df['value'])
        corr = st.spearmanr(gaba, measure).correlation
    else:
        corr = st.spearmanr(df['GABA'], df['value']).correlation
    if np.isnan(corr):
        corr = 0
    return pd.Series({"correlation": corr})

def calc_diff(df, field='correlation'):
    c = df
    acode = 'PWA' if 'PWA' in c.index else 'Amblyope'
    ccode = 'NSP' if 'NSP' in c.index else 'Control'
    amb_de = c.loc[acode,'De'][field]
    amb_nde = c.loc[acode, 'Nde'][field]
    amb_diff = amb_nde-amb_de
    con_de = c.loc[ccode,'De'][field]
    con_nde = c.loc[ccode, 'Nde'][field]
    con_diff = con_nde-con_de
    pop_diff = amb_diff - con_diff
    return [amb_diff, con_diff, pop_diff]

def compare_rs(df, n_boot=2, verbose=False, resample=False):
    """Takes a data frame consisting of one Task/Orientation/Presentation only and generates bootstrapped p-vals for correlations."""
    #print(f"\n\n****{df.name}*****\n")
    if verbose:
        print(f"given df: (head)",
              df[['Population','Eye','Subject','GABA','value']].head(),
              sep="\n")
    corrs = df.groupby(['Population', 'Eye']).apply(calc_rs)
    acode = 'PWA' if 'PWA' in corrs.index else 'Amblyope'
    ccode = 'NSP' if 'NSP' in corrs.index else 'Control'

    observed_corrs_ordered = (corrs.loc[acode,'De']['correlation'],
                            corrs.loc[acode, 'Nde']['correlation'],
                            corrs.loc[ccode,'De']['correlation'],
                            corrs.loc[ccode, 'Nde']['correlation'])
    #print(corrs, sep="\n")
    # Structure to hold the bootstrap iterations of the individual eye correlations
    corrs_permute = np.empty((4, n_boot), dtype=np.float32)
    # Structure to hold the final bootstrapped p-values for each of those 4 correlations
    pvals_corrs = np.zeros(4)

    # Now calculate the NDE-DE diff for AMB and CON, and also the difference between these (p_real)
    a_real, c_real, p_real = calc_diff(corrs, field='correlation')
    #print(f"Real (observed) r_s differences:\nA\tC\tP\n{a_real:.3}\t{c_real:.3}\t{p_real:.3}")
    real_diffs = (a_real, c_real, p_real)
    # Bootstrap structure for the differences
    rs_permute = np.empty((3, n_boot), dtype=np.float32)
    # Structure to hold the final bootstrapped p-values for each of those 3 differences
    pvals_diffs = np.zeros(3)

    for i in range(n_boot):
        # sample with replacement
        if resample:
            samples = df.groupby(['Population', 'Eye'], as_index=False).apply(
                lambda x: x.sample(n=len(x), replace=True)).reset_index()
        else:
            samples = df
        permute_corrs = samples.groupby(['Population', 'Eye']).apply(calc_rs, permute=True)
        amb_diff, con_diff, pop_diff = calc_diff(permute_corrs, field='correlation')
        if verbose:
            print(f"resampled to: (iteration {i})",
              samples[['Population','Eye','Subject','GABA','value']].head(), sep="\n")
            print(corrs)
            print(f"Amb Nde-De: {amb_diff}")
            print(f"Con Nde-De: {con_diff}")
            print(f"Amb - Con: {pop_diff}")
        rs_permute[:, i] = [amb_diff, con_diff, pop_diff]
        amb_de = permute_corrs.loc[acode,'De']['correlation']
        amb_nde = permute_corrs.loc[acode, 'Nde']['correlation']
        con_de = permute_corrs.loc[ccode,'De']['correlation']
        con_nde = permute_corrs.loc[ccode, 'Nde']['correlation']
        corrs_permute[:, i] = [amb_de, amb_nde, con_de, con_nde]     
    comps = ['Amb NDE vs DE', 'Con NDE vs DE', 'Pop Amb vs Con']
    corrs_in_order = ['Amb DE', 'Amb NDE', 'Con DE', 'Con NDE']
    print("\nPercentiles for individual eye correlations:")
    for i in range(4):
        #p = np.percentile(corrs_permute[i, :], np.array([0, 0.5, 1, 1.5, 2, 2.5, 5, 25, 50, 75, 95, 97.5, 100]))
        obs_pct = (np.count_nonzero(observed_corrs_ordered[i]>corrs_permute[i, :])/n_boot)
        if obs_pct > .5:
            pval = (1-obs_pct)*2
        else:
            pval = obs_pct * 2
        pvals_corrs[i] = pval
        print(corrs_in_order[i], f"\nObserved value of {observed_corrs_ordered[i]:.3f} is greater than {obs_pct:.3f} of bootstrap distribution, corresponding to p={pval:.2f}.")
    print("\nPercentiles for permuted r_s differences:")
    for i in range(3):
        #p = np.percentile(rs_permute[i, :], np.array([0, 0.5, 1, 1.5, 2, 2.5, 5, 25, 50, 75, 95, 97.5, 100]))
        obs_pct = (np.count_nonzero(real_diffs[i]>rs_permute[i, :])/n_boot)
        if obs_pct > .5:
            pval = (1-obs_pct)*2
        else:
            pval = obs_pct * 2
        pvals_diffs[i] = pval
        print(comps[i], f"\nObserved value of {real_diffs[i]:.3f} is greater than {obs_pct:.3f} of bootstrap distribution, corresponding to p={pval:.2f}.")
    
    rs_df = pd.DataFrame({
        'amb_de':corrs_permute[0, :],
        'amb_nde':corrs_permute[1, :],
        'con_de':corrs_permute[2, :],
        'con_nde':corrs_permute[3, :],
        'amb_rdiff':rs_permute[0, :],
        'con_rdiff':rs_permute[1, :],
        'pop_rdiff':rs_permute[2, :]})

    print(f"{pvals_corrs}, {pvals_diffs}\n")
    return rs_df, pvals_corrs, pvals_diffs

def calculate_orientation_selective_suppression(df, col='value', **kwargs):
    if len(df.Orientation.unique())==2:
        v1 = df[df.Orientation=='Iso'][col].iloc[0]
        v2 = df[df.Orientation=='Cross'][col].iloc[0]
        iso_cross_oss_ratio = v1/v2
    else:
        iso_cross_oss_ratio = np.nan
    return pd.Series(iso_cross_oss_ratio, ['value'])