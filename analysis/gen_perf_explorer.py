from typing import Dict, List, Tuple, MutableMapping
from pathlib import Path
from collections import OrderedDict
import math

import pandas as pd
from bokeh.models import ColumnDataSource, CDSView, CustomJS, CustomJSFilter, TableColumn, DataTable, Div, \
    RadioButtonGroup, Legend, HoverTool, CustomJSHover, Panel, Tabs
from bokeh.settings import settings
from bokeh.plotting import Figure
from bokeh.io import curdoc
from bokeh.embed import components

import analysis.dashboard_constants as dashboard_constants
import constants as db_constants
from utils.core_utils import save_bokeh_tags


def set_radio_logic(stmttable_cds: ColumnDataSource, stmt_view: CDSView, word_import_div: Div, local_metrics_div: Div,
                    null_set_warn_div: Div,
                    tnt_grp: RadioButtonGroup, b_grp: RadioButtonGroup, cc_grp: RadioButtonGroup) -> None:
    # set additional callback on radio button change
    someargs = dict(stmttable_cds=stmttable_cds, stmt_view=stmt_view, word_import_div=word_import_div,
                    null_set_warn_div=null_set_warn_div,
                    local_metrics_div=local_metrics_div,
                    local_metrics_static_open=dashboard_constants.local_metrics_static_open,
                    tnt_grp=tnt_grp, b_grp=b_grp, cc_grp=cc_grp)
    radio_callback = CustomJS(args=someargs, code=dashboard_constants.radio_callback_code)
    tnt_grp.js_on_change('active', radio_callback)
    b_grp.js_on_change('active', radio_callback)
    cc_grp.js_on_change('active', radio_callback)


def init_radio_groups() -> Tuple[RadioButtonGroup, ...]:
    tweet_nontweet_grp = RadioButtonGroup(labels=["nontweets", "tweets"], active=0, css_classes=['t_nt_grp'],
                                          width=200, width_policy='fit')
    conf_bucket_grp = RadioButtonGroup(labels=["max", "min"], active=0, css_classes=['bucket_grp'], width=100,
                                       width_policy='fit')
    pred_exp_class_grp = RadioButtonGroup(labels=["tp", "tn", "fp", "fn"], active=0, css_classes=['cc_grp'], width=200,
                                          width_policy='fit')
    return tweet_nontweet_grp, conf_bucket_grp, pred_exp_class_grp


def init_explorer_divs(pred_stmt_dict: Dict) -> Tuple[Div, ...]:
    default_idx = min([i for i, (b, c) in enumerate(zip(pred_stmt_dict['bucket_type'], pred_stmt_dict['tp']))
                       if b == 'max_ppv_nontweets' and c == 1])
    word_import_div = Div(text=pred_stmt_dict['pred_exp_attr_tups'][default_idx][1], height_policy='max',
                          width_policy='max', width=300, min_height=20, align='start', margin=(5, 5, 5, 5),
                          css_classes=['box', 'word_import'])
    local_stats_mapping = {k: pred_stmt_dict[k][0] for k in ['conf_percentile', 'pos_pred_ratio', 'neg_pred_ratio',
                                                             'bucket_type']}
    for k in ['bucket_acc', 'pos_pred_acc', 'neg_pred_acc']:
        local_stats_mapping[k] = f"{round(pred_stmt_dict[k][0] * 100)}%"
    local_metrics_base_tag = dashboard_constants.init_local_metrics_summ.format(**local_stats_mapping)
    local_metrics_grid = Div(text=local_metrics_base_tag, width_policy="max", width=325, max_width=550,
                             margin=(0, 5, 5, 0), css_classes=['box'])
    null_set_warn_div = Div(text="", width_policy='max', height_policy='min', width=195, max_width=600, height=20,
                            align='start',
                            margin=(3, 3, 3, 3), css_classes=['box', 'null_set_warn'], visible=False)
    return word_import_div, local_metrics_grid, null_set_warn_div


def set_stmttbl_logic(stmttable_cds: ColumnDataSource, word_import_div: Div, local_metrics_div: Div,
                      stmt_view: CDSView) -> None:
    someargs = dict(stmttable_cds=stmttable_cds, word_import_div=word_import_div, local_metrics_div=local_metrics_div,
                    local_metrics_static_open=dashboard_constants.local_metrics_static_open, stmt_view=stmt_view)
    tblcallback = CustomJS(args=someargs, code=dashboard_constants.stmttbl_callback_code)
    stmttable_cds.selected.js_on_change('indices', tblcallback)


def build_stmt_table(pred_stmt_dict: Dict, tnt_grp: RadioButtonGroup, b_grp: RadioButtonGroup,
                     cc_grp: RadioButtonGroup) -> Tuple[DataTable, ColumnDataSource, CDSView]:
    stmttable_cds = ColumnDataSource(pred_stmt_dict)
    stmtcolumn = [TableColumn(field="statement_text", title="Statement", width=4000)]
    js_filter = CustomJSFilter(args=dict(stmttable_cds=stmttable_cds, tnt_grp=tnt_grp, b_grp=b_grp, cc_grp=cc_grp),
                               code=dashboard_constants.cdsview_jsfilter_code)
    stmt_view = CDSView(source=stmttable_cds, filters=[js_filter])
    stmttable = DataTable(source=stmttable_cds, columns=stmtcolumn, header_row=False, index_position=None,
                          view=stmt_view,
                          height_policy='max',
                          width_policy='max',
                          fit_columns=False,
                          scroll_to_selection=False, margin=(5, 5, 5, 5), width=300, height=300,
                          css_classes=['box', 'stmt_table'])
    return stmttable, stmttable_cds, stmt_view


def build_cmatrix_rpt_dfs(perf_exp_dict: Dict) -> OrderedDict:
    cmatrix_df_dict = OrderedDict()
    for k, v in perf_exp_dict.items():
        if k in db_constants.TEST_CMATRICES:
            temp_df = pd.DataFrame.from_records(v, columns=['yweeks', 'tp_ratio', 'tn_ratio', 'fp_ratio', 'fn_ratio'],
                                                coerce_float=True)
            temp_df['yweeks'] = temp_df['yweeks'].apply(lambda x: pd.to_datetime(x))
            cmatrix_df_dict[k] = temp_df.set_index('yweeks')
        else:
            cmatrix_df_dict[k] = pd.DataFrame.from_records(v,
                                                           columns=['confidence_bucket', 'max_conf_percentile',
                                                                    'avg_wc', 'fp_ratio', 'fn_ratio', 'tn_ratio',
                                                                    'tp_ratio', 'acc', 'confidence', 'acc_conf_delta'],
                                                           coerce_float=True).set_index('confidence_bucket')
    return cmatrix_df_dict


def gen_cmatrix_legend(p: Figure) -> Figure:
    legend = Legend(items=[tuple((n, [r])) for n, r in zip(dashboard_constants.CMATRIX_CLASS_NAMES, p.renderers)],
                    location="top_center", orientation="horizontal", click_policy='hide', border_line_alpha=0,
                    background_fill_alpha=0)
    legend.label_text_color = '#003566'
    p.add_layout(legend, 'above')
    return p


def cmatrix_plot_format(plot: Figure, format_cols: List, hover_code: str) -> Figure:
    plot.toolbar.logo = None
    plot.yaxis.axis_label_text_font_size = "16px"
    plot.yaxis.axis_label_text_color = '#003566'
    plot.xaxis.major_label_text_color = '#003566'
    plot.y_range.range_padding = 0
    plot.hover.point_policy = "follow_mouse"
    plot.hover.attachment = "above"
    cust_formatters = {}
    for f in format_cols:
        code_ref = dashboard_constants.bucket_cust_format.format(field_name=f) if f == 'max_percentile_conf' else \
            dashboard_constants.scale_format_code.format(field_name=f)
        cust_formatters[f] = CustomJSHover(args=dict(source=plot.renderers[0].data_source), code=code_ref)
    custom_hover_format = CustomJS(code=hover_code)
    format_dict = {f'@{c}': cust_formatters[c] for c in format_cols}
    plot.add_tools(HoverTool(tooltips=None, callback=custom_hover_format, formatters=format_dict))
    someargs = dict(yaxis=plot.yaxis[0], xaxis=plot.xaxis[0], legend=plot.legend[0])
    fig_resize_callback = CustomJS(args=someargs, code=dashboard_constants.fig_resize_callback_code)
    plot.js_on_change('inner_width', fig_resize_callback)
    return plot


def gen_temporal_cmatrices(plot_config: Dict, v: pd.DataFrame) -> Figure:
    cust_plot_tools = {**plot_config}
    v.index = v.index.get_level_values('yweeks').strftime("%b-%d")
    v.index.name = 'yweeks'
    p = Figure(x_range=v.index.get_level_values('yweeks').tolist(), **cust_plot_tools, css_classes=["cmatrix_fig"])
    stack_fields = ['tp_ratio', 'tn_ratio', 'fp_ratio', 'fn_ratio']
    p.vbar_stack(stackers=stack_fields, x='yweeks', color=dashboard_constants.cust_rg_palette, source=v, width=1.0)
    p = gen_cmatrix_legend(p)
    p.grid.minor_grid_line_color = '#eeeeee'
    p.xaxis.major_label_orientation = math.pi / 4
    p.x_range.range_padding = 0
    p.xaxis.major_label_text_font_size = "12px"
    p = cmatrix_plot_format(p, stack_fields, dashboard_constants.temporal_custom_hover_code)
    return p


def gen_conf_cmatrices(plot_config: Dict, v: pd.DataFrame) -> Figure:
    cust_plot_tools = {'x_axis_label': 'Confidence Bucket', **plot_config}
    p = Figure(x_range=(v.index.min(), v.index.max()), **cust_plot_tools)
    stack_fields = ['tp_ratio', 'tn_ratio', 'fp_ratio', 'fn_ratio']
    p.vbar_stack(stackers=stack_fields, x='confidence_bucket', color=dashboard_constants.cust_rg_palette,
                 source=v, width=1.0)
    p = gen_cmatrix_legend(p)
    p.grid.minor_grid_line_color = '#eeeeee'
    p.xaxis.axis_label_text_font_size = "16px"
    p.xaxis.axis_label_text_color = '#003566'
    stack_fields.extend(['max_conf_percentile', 'acc'])
    p = cmatrix_plot_format(p, stack_fields, dashboard_constants.conf_custom_hover_code)
    return p


def build_cmatrices(cmatrix_dfs: Dict) -> OrderedDict:
    cmatrix_figs = OrderedDict()

    plot_size_and_tools = {'sizing_mode': 'scale_both', 'border_fill_alpha': 0,
                           'background_fill_alpha': 0, 'min_width': 200, 'min_height': 200, 'max_width': 600,
                           'max_height': 600, 'y_axis_label': 'Cumulative Percent Samples', 'tools': ['reset']}
    for k, v in cmatrix_dfs.items():
        if k in db_constants.TEST_CMATRICES:
            p = gen_temporal_cmatrices(plot_size_and_tools, v)
            cmatrix_figs[k] = p
        else:
            p = gen_conf_cmatrices(plot_size_and_tools, v)
            cmatrix_figs[k] = p
    return cmatrix_figs


def build_perf_exp_doc(config: MutableMapping, perf_exp_dict: Dict, debug_mode: bool = False) -> None:
    # define core document objects
    curdoc().clear()
    bokeh_base = Path(config.experiment.dc_base).parent / f"{dashboard_constants.BOKEH_DEST}"
    tag_files = [dashboard_constants.TEMPORAL_TABS_FILE, dashboard_constants.CONF_TABS_FILE]
    cmatrix_rpt_dfs = build_cmatrix_rpt_dfs(perf_exp_dict)
    cmatrix_figs = build_cmatrices(cmatrix_rpt_dfs)
    if debug_mode:
        settings.py_log_level = 'debug'
        settings.log_level = 'debug'
    temporal_cmatrices, conf_cmatrices = [], []
    for (k, v), t in zip(cmatrix_figs.items(), dashboard_constants.cmatrix_tab_labels):
        if k in db_constants.TEST_CMATRICES:
            temporal_cmatrices.append(Panel(child=v, title=t))
        else:
            conf_cmatrices.append(Panel(child=v, title=t))
    tabs_config = {'min_width': 200, 'min_height': 200, 'max_width': 600, 'max_height': 600,
                   'sizing_mode': 'scale_both'}
    temporal_tabs, conf_tabs = Tabs(tabs=temporal_cmatrices, **tabs_config), Tabs(tabs=conf_cmatrices, **tabs_config)
    script, tags = components((temporal_tabs, conf_tabs))
    save_bokeh_tags(bokeh_base, script, dashboard_constants.PERF_EXPLORER_SCRIPT_FILE, tags, tag_files)
