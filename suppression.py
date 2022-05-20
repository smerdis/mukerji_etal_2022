# suppression.py
#
# Functions to read in, deal with, and plot the psychophysical data extracted from MATLAB

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as tick

import seaborn as sns

import scipy.stats as st

import itertools as it

import utils

def annotate_facet_df(xcol, ycol, tracecol, what='rho', **kwargs):
    """Annotate each level of hue variable on each facet of graph (i.e. multiple times per facet)"""
    #print(f"Annotating\n{tracecol}\n{xcol}\n{ycol}\n{kwargs}")
    ax = plt.gca()
    if kwargs['pvals'] is not None:
        pvals = kwargs.pop("pvals") #pvals from bootstrap
    else:
        pvals = None
    data = kwargs.pop("data") #pvals from bootstrap
    trace = data[tracecol].unique()[0] # 'Persons with\nAmblyopia, DE' etc
    pal = kwargs['palette']
    presentation = kwargs.pop("presentation") #pvals from bootstrap
    n_thistrace = len(data[tracecol])
    assert(n_thistrace==len(data[xcol])==len(data[ycol]))
    if n_thistrace > 2:
        # if this graph has 8 lines (2 presentation conditions x 2 populations x 2 eyes), annotate only the correct 4, which are in kwarg 'presentation'
        # if this graph has 4 lines (2 populations x 2 eyes for one presentation condition, which is then not in data.columns), annotate it
        if ('Presentation' in data.columns and data.Presentation.iloc[0]==presentation) or ('Presentation' not in data.columns):
            if trace == "Persons with\nAmblyopia, DE" or trace=="Normally-sighted\npersons, DE":
                pos = (0.5, 0.9)
                if trace == "Persons with\nAmblyopia, DE":
                    pval = pvals[0]
                if trace=="Normally-sighted\npersons, DE":
                    pval = pvals[2]
            if trace == "Persons with\nAmblyopia, NDE" or trace=="Normally-sighted\npersons, NDE":
                pos = (0.5, 0.85)
                if trace == "Persons with\nAmblyopia, NDE":
                    pval=pvals[1]
                if trace=="Normally-sighted\npersons, NDE":
                    pval=pvals[3]
            color = pal[trace]
            if what=='rho':
                result = st.spearmanr(data[xcol], data[ycol])
                annotation = fr"N={n_thistrace}, $\rho$={result.correlation:.2f}, p={pval:.2f}"
            elif what=='slope':
                result = st.linregress(data[xcol], data[ycol])
                annotation = fr"N={n_thistrace}, slope={result.slope:.2f}, p={pval:.2f}"
            ax.text(*pos, annotation, fontsize=12, transform=ax.transAxes, fontdict={'color': color}, horizontalalignment='center')


def gaba_vs_psychophys_plot(gv, gr, legend_box = [0.89, 0.55, 0.1, 0.1], legend_img = True, log = False, ylim = None, annotate=True, boot_func=utils.compare_rs, **kwargs):
    """Plotting function for GABA vs. psychophysical measures, with annotations etc."""
    print(gv)#, gr)
    with sns.plotting_context(context="paper", font_scale=0.8):
        xvar = "GABA"
        yvar = "value"
        try:
            n_boot = kwargs['n_boot']
        except KeyError:
            n_boot = 1000
        if boot_func == utils.compare_rs:
            what='rho'
        elif boot_func == utils.compare_slopes:
            what = 'slope'
        g = sns.lmplot(data=gr, x=xvar, y=yvar, **kwargs)
        if annotate:
            anno_groups = gr.groupby(['Task','Orientation','Presentation'])
            for agv, agr in anno_groups:
                print(agv[-1])
                iterations, pvals_corrs, pvals_diffs = boot_func(agr, n_boot=n_boot, verbose=False, resample=False)
                g.map_dataframe(annotate_facet_df, 'GABA', 'value', 'Trace', what=what, pvals=pvals_corrs, palette=kwargs['palette'], presentation=agv[-1]) #runs on each level of hue in each facet

        for axi, ax in enumerate(g.axes.flat): #set various things on each facet
            if log:
                ax.set_yscale('log')
            if gv[-1] in ("ThreshPredCritical", "ThreshElev"): # assumes 'measure' is last grouping variable
                ax.axhline(1, color='grey', linestyle='dotted') # facilitation-suppression line
            if ylim is not None:
                if type(ylim[0]) is tuple: # tuple of tuples, i.e. different ylims for amb and con
                    ax.set_ylim(*ylim[axi%2])
                else:
                    ax.set_ylim(*ylim)
            ax.yaxis.set_major_locator(tick.FixedLocator([0.5, 1, 2, 3, 5, 10]))
            ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))
            #ax.yaxis.set_minor_formatter(tick.FormatStrFormatter('%.1f'))
            ax.yaxis.set_minor_formatter(tick.NullFormatter())
            #newax = g.fig.add_axes(legend_box[axi], anchor='NE')
            #newax.imshow(im)
            #newax.axis('off')
            if 'Dicho' in gv:
                ax.legend(loc='lower left', title='Annulus presented to:\n(other eye viewed surround)')
            elif 'Mono' in gv:
                ax.legend(loc='lower left', title='Annulus and surround\npresented to:')
            else: # page grouping variables don't include presentation, i.e. there are 4 subplots on one page
                ax.legend(loc='lower left', title='Annulus presented to:')

        if g._legend:
            g._legend.set_title(f"Target presented to")
            #g._legend.set_bbox_to_anchor([legend_box[0], legend_box[1]-0.16, legend_box[2], legend_box[3]])

        x_lbl = "GABA:Creatine ratio"
        y_lbl = {'BaselineThresh':'Baseline contrast discrimination threshold (C%)',
                'ThreshElev':'Relative threshold\n(multiples of baseline)',
                'ThreshPredCritical':'Relative threshold, multiples of baseline\n(>1 indicates suppression, <1 facilitation)',
                'OSSSRatio':'Orientation-selective surround suppression\n(Iso-surround:cross-surround ratio)'}
        g.set_axis_labels(x_lbl, y_lbl[gv[-1]])

    plt.close(g.fig)
    return(g)
