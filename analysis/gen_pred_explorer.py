from typing import Dict, Tuple,  MutableMapping
from pathlib import Path

from bokeh.models import ColumnDataSource, CDSView, CustomJS, CustomJSFilter, TableColumn, DataTable, Div, \
    RadioButtonGroup
from bokeh.settings import settings
from bokeh.io import curdoc
from bokeh.embed import components


import analysis.dashboard_constants as dashboard_constants
from utils.core_utils import save_bokeh_tags


def set_radio_logic(stmttable_cds: ColumnDataSource, stmt_view: CDSView, word_import_div: Div, local_metrics_div: Div,
                    null_set_warn_div: Div, tnt_grp: RadioButtonGroup, b_grp: RadioButtonGroup,
                    cc_grp: RadioButtonGroup) -> None:
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
                       if b == 'max_acc_nontweets' and c == 1])
    word_import_div = Div(text=pred_stmt_dict['pred_exp_attr_tups'][default_idx][1], height_policy='max',
                          width_policy='max', width=300, min_height=20, align='start', margin=(5, 5, 5, 5),
                          css_classes=['box', 'word_import'])
    local_stats_mapping = {k: pred_stmt_dict[k][0] for k in ['conf_percentile', 'bucket_type']}
    for k in ['bucket_acc', 'pos_pred_acc', 'neg_pred_acc']:
        local_stats_mapping[k] = f"{round(pred_stmt_dict[k][0] * 100, 1)}%"
    for k in ['pos_pred_ratio', 'neg_pred_ratio']:
        local_stats_mapping[k] = f"{round(pred_stmt_dict[k][0], 2)}"
    local_metrics_base_tag = dashboard_constants.init_local_metrics_summ.format(**local_stats_mapping)
    local_metrics_grid = Div(text=local_metrics_base_tag, width_policy="max", width=325, max_width=550,
                             margin=(0, 5, 5, 0), css_classes=['box'])
    null_set_warn_div = Div(text="", width_policy='max', height_policy='min', width=195, max_width=600, height=20,
                            align='start', margin=(3, 3, 3, 3), css_classes=['box', 'null_set_warn'], visible=False)
    return word_import_div, local_metrics_grid, null_set_warn_div


def set_stmttbl_logic(stmttable_cds: ColumnDataSource, word_import_div: Div, local_metrics_div: Div,
                      stmt_view: CDSView) -> None:
    someargs = dict(stmttable_cds=stmttable_cds, word_import_div=word_import_div, local_metrics_div=local_metrics_div,
                    local_metrics_static_open=dashboard_constants.local_metrics_static_open, stmt_view=stmt_view)
    tblcallback = CustomJS(args=someargs, code=dashboard_constants.stmttbl_callback_code)
    stmttable_cds.selected.js_on_change('indices', tblcallback)


def build_global_metrics_div(global_stats: Tuple) -> Div:
    global_stats_mapping = {
        'nt_mcc': global_stats[2], 'nt_auc': global_stats[1], 'nt_acc': global_stats[0], 'nt_tp': global_stats[3],
        'nt_tn': global_stats[4], 'nt_fp': global_stats[5], 'nt_fn': global_stats[6],
        't_mcc': global_stats[9], 't_auc': global_stats[8], 't_acc': global_stats[7], 't_tp': global_stats[10],
        't_tn': global_stats[11], 't_fp': global_stats[12], 't_fn': global_stats[13],
        'all_mcc': global_stats[16], 'all_auc': global_stats[15], 'all_acc': global_stats[14],
        'all_tp': global_stats[17], 'all_tn': global_stats[18], 'all_fp': global_stats[19], 'all_fn': global_stats[20]
    }
    global_stats_div_text = dashboard_constants.global_metrics_summ.format(**global_stats_mapping)
    global_stats_summ_div = Div(text=global_stats_div_text, margin=(5, 5, 5, 5), width_policy="max", width=325,
                                max_width=600, css_classes=['box', 'metric_summ'])
    return global_stats_summ_div


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
                          scroll_to_selection=False, margin=(5, 5, 5, 5), width=300, height=300, min_height=50,
                          css_classes=['box', 'stmt_table'])
    return stmttable, stmttable_cds, stmt_view


def build_pred_exp_doc(config: MutableMapping, pred_stmt_dict: Dict, global_metric_stats: Tuple,
                       debug_mode: bool = False) -> None:
    # define core document objects
    curdoc().clear()
    global_metrics_div = build_global_metrics_div(global_metric_stats)
    t_nt_grp, bucket_grp, cc_grp = init_radio_groups()
    stmttable, stmttable_cds, stmt_view = build_stmt_table(pred_stmt_dict, t_nt_grp, bucket_grp, cc_grp)
    word_import_div, local_metrics_div, null_set_warn_div = init_explorer_divs(pred_stmt_dict)
    set_radio_logic(stmttable_cds, stmt_view, word_import_div, local_metrics_div, null_set_warn_div, t_nt_grp,
                    bucket_grp, cc_grp)
    set_stmttbl_logic(stmttable_cds, word_import_div, local_metrics_div, stmt_view)
    bokeh_base = Path(config.experiment.dc_base).parent / f"{dashboard_constants.BOKEH_DEST}"
    tag_files = [dashboard_constants.GLOBAL_METRICS_TAG_FILE, dashboard_constants.LOCAL_METRICS_TAG_FILE,
                 dashboard_constants.WORD_IMPORTANCE_TAG_FILE, dashboard_constants.STMT_TABLE_TAG_FILE,
                 dashboard_constants.NULL_STMT_TAG_FILE, dashboard_constants.RADIO_GRPS_TAG_FILE]
    if debug_mode:
        settings.py_log_level = 'debug'
        settings.log_level = 'debug'
    script, tags = components((global_metrics_div, local_metrics_div, word_import_div, stmttable, null_set_warn_div,
                               t_nt_grp, bucket_grp, cc_grp))
    save_bokeh_tags(bokeh_base, script, dashboard_constants.STMT_EXPLORER_SCRIPT_FILE, (*tags[0:-3], (tags[-3:])),
                    tag_files)
