STMT_EXPLORER_SCRIPT_FILE = "stmt_explorer_script.html"
PERF_EXPLORER_SCRIPT_FILE = "perf_explorer_script.html"
WORD_IMPORTANCE_TAG_FILE = "word_importance_tag.html"
STMT_TABLE_TAG_FILE = "stmt_table_tag.html"
GLOBAL_METRICS_TAG_FILE = "global_metrics_tag.html"
LOCAL_METRICS_TAG_FILE = "local_metrics_tag.html"
RADIO_GRPS_TAG_FILE = "radio_grps_tags.html"
NULL_STMT_TAG_FILE = "null_stmt_tags.html"
TEMPORAL_TABS_FILE = "temporal_tabs_tags.html"
CONF_TABS_FILE = "conf_tabs_tags.html"
BOKEH_DEST = "docs/_includes/bokeh_viz"
CMATRIX_CLASS_NAMES = ['True Positive', 'True Negative', 'False Positive', 'False Negative']
cmatrix_tab_labels = ['All', 'Nontweets', 'Tweets', 'All', 'Nontweets', 'Tweets']
global_metrics_summ = """
            <span class="box metric_title">Salient Global Model<sup id="a0" class="small"><a href="#f0">[0]</a></sup> Metrics</span>
            <table class="metric_summ metric_summary">
                <thead class="metric-label" >
                    <tr>
                        <th>Stmt Type</th>
                        <th>AUC</th>
                        <th>MCC</th>
                        <th>Acc.</th>
                        <th>TP</th>
                        <th>TN</th>
                        <th>FP</th>
                        <th>FN</th>
                    </tr>
                </thead>
                <tbody>
                <tr>
                    <td class="metric-label">All</td>
                    <td>{all_auc}</td>
                    <td>{all_mcc}</td>
                    <td>{all_acc}</td>
                    <td>{all_tp}</td>
                    <td>{all_tn}</td>
                    <td>{all_fp}</td>
                    <td>{all_fn}</td>
                </tr>
                <tr>
                    <td class="metric-label">Non-tweets</td>
                    <td>{nt_auc}</td>
                    <td>{nt_mcc}</td>
                    <td>{nt_acc}</td>
                    <td>{nt_tp}</td>
                    <td>{nt_tn}</td>
                    <td>{nt_fp}</td>
                    <td>{nt_fn}</td>
                </tr>
                <tr>
                    <td class="metric-label">Tweets</td>
                    <td>{t_auc}</td>
                    <td>{t_mcc}</td>
                    <td>{t_acc}</td>
                    <td>{t_tp}</td>
                    <td>{t_tn}</td>
                    <td>{t_fp}</td>
                    <td>{t_fn}</td>
                </tr>
                </tbody>
            </table>"""
local_metrics_static_open = """
    <div class="local-metric-grid-container">
        <div class="bucket-level-ttl local_title">Confidence Bucket-Level Metrics<sup id="a1" class="small"><a href="#f1">[1] </a></sup></div>
        <div class="bucket-name metric-label">Bucket</div>
        <div class="bucket-acc metric-label">Tot. Accuracy</div>
        <div class="percentile-conf metric-label">Percentile Conf</div>
        <div class="pred-acc-head metric-label">Pred. Accuracy</div>
        <div class="pred-acc-pos metric-label">Pos</div>
        <div class="pred-acc-neg metric-label">Neg</div>
        <div class="pred-ratio-head metric-label">Pred. Ratios</div>
        <div class="pred-ratio-pos metric-label">Pos</div>
        <div class="pred-ratio-neg metric-label">Neg</div>
"""
init_local_metrics_summ = local_metrics_static_open + """
        <div class="bn-val grid-data">{bucket_type}</div>
        <div class="pc-val grid-data">{conf_percentile}</div>
        <div class="ba-val grid-data">{bucket_acc}</div>
        <div class="pap-val grid-data">{pos_pred_acc}</div>
        <div class="pan-val grid-data">{neg_pred_acc}</div>
        <div class="prp-val grid-data">{pos_pred_ratio}</div>
        <div class="prn-val grid-data">{neg_pred_ratio}</div>
    </div>
    """
cust_rg_palette = tuple(['#a7d2bd', '#a0f2cb', '#f2a0a0', '#ffe2e2'])
fig_resize_callback_code = """
yaxis.axis_label_text_font_size=`${cb_obj.inner_width/30}px`;
xaxis.axis_label_text_font_size=`${cb_obj.inner_width/30}px`;
legend.label_text_font_size=`${cb_obj.inner_width/35}px`;
"""
stmttbl_callback_code = """
var selected_index = stmttable_cds.selected.indices[0];
var selected_stmt = stmttable_cds.data['statement_text'][selected_index];
console.log('Selected stmt is ' + selected_stmt);
var local_metrics_dynamic = `
        ${local_metrics_static_open}
        <div class="bn-val grid-data">${stmttable_cds.data['bucket_type'][selected_index]}</div>
        <div class="pc-val grid-data">${stmttable_cds.data['conf_percentile'][selected_index]}</div>
        <div class="ba-val grid-data">${(stmttable_cds.data['bucket_acc'][selected_index]*100).toFixed(1)}%</div>
        <div class="pap-val grid-data">${(stmttable_cds.data['pos_pred_acc'][selected_index]*100).toFixed(1)}%</div>
        <div class="pan-val grid-data">${(stmttable_cds.data['neg_pred_acc'][selected_index]*100).toFixed(1)}%</div>
        <div class="prp-val grid-data">${stmttable_cds.data['pos_pred_ratio'][selected_index].toFixed(2)}</div>
        <div class="prn-val grid-data">${stmttable_cds.data['neg_pred_ratio'][selected_index].toFixed(2)}</div>
    </div>`;
word_import_div.text = stmttable_cds.data['pred_exp_attr_tups'][selected_index][1];
local_metrics_div.text = local_metrics_dynamic;
"""
cdsview_jsfilter_code = """
const indices = []
var tnt_active_index = tnt_grp.active;
var b_active_index = b_grp.active;
var cc_active_index = cc_grp.active;
var tnt_val = tnt_grp.labels[tnt_active_index];
var b_val = b_grp.labels[b_active_index];
var cc_val = cc_grp.labels[cc_active_index];
var curr_bucket = '';
if (tnt_val== 'nontweets') {
    if (b_val == 'max') {
        curr_bucket = 'max_ppv_nontweets';
    } else {
        curr_bucket = 'min_ppv_nontweets';
    }
} else {
    if (b_val == 'max') {
        curr_bucket = 'max_ppv_tweets';
    } else {
        curr_bucket = 'min_ppv_tweets';
    }
}
for (var i = 0; i <= stmttable_cds.data['bucket_type'].length; i++) {
    if (stmttable_cds.data['bucket_type'][i] == curr_bucket && stmttable_cds.data[cc_val][i] == 1) {
        indices.push(i)
    }
}
return indices
"""
radio_callback_code = """
var tnt_active_index = tnt_grp.active;
var b_active_index = b_grp.active;
var cc_active_index = cc_grp.active;
var tnt_val = tnt_grp.labels[tnt_active_index];
var b_val = b_grp.labels[b_active_index];
var cc_val = cc_grp.labels[cc_active_index];
console.log('Currently filtering:' + tnt_val + ', ' + b_val + ', ' + cc_val);
stmttable_cds.change.emit();
stmttable_cds.selected.change.emit();
if (stmt_view.indices.length == 0) {
    word_import_div.text = "";
    null_set_warn_div.text = `No ${cc_val} statements in ${b_val} ${tnt_val} bucket. Please select another class.`;
    null_set_warn_div.visible = true;
    var local_metrics_dynamic = `
        ${local_metrics_static_open}
        <div class="bn-val grid-data"></div>
        <div class="pc-val grid-data"></div>
        <div class="ba-val grid-data"></div>
        <div class="pap-val grid-data"></div>
        <div class="pan-val grid-data"></div>
        <div class="prp-val grid-data"></div>
        <div class="prn-val grid-data"></div>
        </div>`
} else {
    null_set_warn_div.visible = false;
    var selected_index = stmt_view.indices[0];
    var selected_stmt = stmttable_cds.data['statement_text'][selected_index];
    word_import_div.text = stmttable_cds.data['pred_exp_attr_tups'][selected_index][1];
    var local_metrics_dynamic = `
            ${local_metrics_static_open}
            <div class="bn-val grid-data">${stmttable_cds.data['bucket_type'][selected_index]}</div>
            <div class="pc-val grid-data">${stmttable_cds.data['conf_percentile'][selected_index]}</div>
            <div class="ba-val grid-data">${(stmttable_cds.data['bucket_acc'][selected_index]*100).toFixed(1)}%</div>
            <div class="pap-val grid-data">${(stmttable_cds.data['pos_pred_acc'][selected_index]*100).toFixed(1)}%</div>
            <div class="pan-val grid-data">${(stmttable_cds.data['neg_pred_acc'][selected_index]*100).toFixed(1)}%</div>
            <div class="prp-val grid-data">${stmttable_cds.data['pos_pred_ratio'][selected_index].toFixed(2)}</div>
            <div class="prn-val grid-data">${stmttable_cds.data['neg_pred_ratio'][selected_index].toFixed(2)}</div>
        </div>`;
    local_metrics_div.text = local_metrics_dynamic;
}
"""
scale_format_code = """
    var curr_idx = special_vars.index
    return (source.data['{field_name}'][curr_idx]*100).toFixed(1) + "%"
    """
bucket_cust_format = """
    var curr_idx = special_vars.index
    return Math.round(source.data['{field_name}'][curr_idx]*100) + "%"
    """
temporal_custom_hover_code = """
var cust_div = `<div>
    <div>
        <span class="tooltip_header">@yweeks</span>
    </div>
    <div>
        <span>TP: @tp_ratio{custom}</span>
    </div>
    <div>
        <span>TN: @tn_ratio{custom}</span>
    </div>
    <div>
        <span>FP: @fp_ratio{custom}</span>
    </div>
    <div>
        <span>FN: @fn_ratio{custom}</span>
    </div>
</div>`;
cb_obj.tooltips = cust_div;
"""
conf_custom_hover_code = """
var cust_div = `<div>
    <div>
        <span class="tooltip_header">Conf Bucket: @max_conf_percentile{custom}</span>
    </div>
    <div>
        <span>Bucket Acc: @acc{custom}</span>
    </div>
    <div>
        <span>TP: @tp_ratio{custom}</span>
    </div>
    <div>
        <span>TN: @tn_ratio{custom}</span>
    </div>
    <div>
        <span>FP: @fp_ratio{custom}</span>
    </div>
    <div>
        <span>FN: @fn_ratio{custom}</span>
    </div>
</div>`;
cb_obj.tooltips = cust_div;
"""
