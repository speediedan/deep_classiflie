import re
from typing import List, Tuple, Optional

import numpy as np
from captum.attr._utils.visualization import VisualizationDataRecord, _get_color, format_special_tokens
from weasyprint import HTML

import utils.constants as constants
import analysis.dashboard_constants as dashboard_constants


def fmt_word_import(words: List[str], importances: np.ndarray, token_mask: List[int] = None,
                    invert_colors: bool = False, pred_exp_mode: bool = False) -> Optional[str]:
    if importances is None or len(importances) == 0:
        if pred_exp_mode:
            return None
        else:
            return """<div class="stats2"><div class="box word_import"></div></div>"""
    assert len(words) <= len(importances)
    tags = ["""<div class="stats2"><div class="box word_import">"""] if not pred_exp_mode else []
    regex_dict = {'punc_bef_fix': re.compile(r"(\s(?=[,’'!:.\-?]))"),
                  'punc_aft_fix': re.compile(r"([$:\-](?=(</font>)))"),
                  'is_deci_comm': re.compile(r"(?<=[.,](?=(</font>)))"),
                  'in_num': re.compile(r"\d(?=(</font>))"),
                  'sp_clean': re.compile(r"▁")
                  }
    include_space = True
    in_num = False
    for word, importance in zip(words, importances[: len(words)]):
        if word not in token_mask:
            word = format_special_tokens(word)
            color = _get_color(importance) if not invert_colors else _get_color((importance * -1))
            word_formatted = f" {word}" if include_space else f"{word}"
            unwrapped_tag = f'<mark style="background-color: {color};">{word_formatted}</mark>'
            unwrapped_tag = unwrapped_tag.translate(str.maketrans(r'“”', r'""'))
            for regex in ['punc_bef_fix', 'sp_clean']:
                unwrapped_tag = regex_dict[regex].sub('', unwrapped_tag)
            include_space = False if ((in_num and regex_dict['is_deci_comm'].search(unwrapped_tag))
                                      or regex_dict['punc_aft_fix'].search(unwrapped_tag)) else True
            in_num = True if regex_dict['in_num'].search(unwrapped_tag) else False
            tags.append(unwrapped_tag)
    if not pred_exp_mode:
        tags.append("</div></div>")
    return "".join(tags)


def fmt_target_pred(datarecord: VisualizationDataRecord) -> Tuple[str, str, str, str]:
    if datarecord.true_class is not None:
        if datarecord.pred_class == 1:
            target_color = "red"
            target_text = "Labeled False"
        else:
            target_color = "green"
            target_text = "No Falsehood Label"
    else:
        target_color = "white"
        target_text = "TBD"
    if datarecord.pred_class == 1:
        pred_color = "red"
        pred_text = "Labeled False"
    else:
        pred_color = "green"
        pred_text = "No Falsehood Label"
    return pred_color, pred_text, target_color, target_text


def calc_confidence(ext_rec: Tuple) -> Tuple[str, str]:
    ppv = ext_rec[5][5]
    if round(ppv, 2) >= 0.80:
        confidence_level = "High"
        confidence_color = "green"
    elif round(ppv, 2) >= 0.60:
        confidence_level = "Moderate"
        confidence_color = "white"
    else:
        confidence_level = "Low"
        confidence_color = "red"
    confidence = f"{round(ppv * 100)} percent" if confidence_level != "Low" else "Low"
    return confidence, confidence_color


def fmt_header_footer(rel_style_path: str) -> Tuple[str, str]:
    html_header = f"""
    <!DOCTYPE html>
    <html lang = "en-US">
        <head>
        <meta charset="UTF-8">
        <title>Detailed Prediction Report</title>
        <link rel="stylesheet" type="text/css" href="{rel_style_path}" />
        </head>
        <body>
        """
    html_footer = """</body></html>"""
    return html_header, html_footer


def fmt_smry_row(pred_color: str, pred_text: str, target_color: str, target_text: str, confidence_color: str,
                 ext_rec: Tuple) -> str:
    bucket_acc, ppv, npv = ext_rec[5][1], ext_rec[5][5], ext_rec[5][6]
    # tweet_warn = """<br/><span class="small">(see footnote<sup>[7]</sup> regarding tweet inference)</span>""" \
    #    if not ext_rec[9] else ""
    tweet_warn = ""  # disabling tweet warn for this release. May remove next release.
    summary_row = f"""
    <div class="statswrapper">
    <div class="stats1">
    <div class="box prediction">Prediction<sup class="small">[1]</sup>:<mark class="{pred_color}"> {pred_text}</mark>
    <br/> Actual<sup class="small">[2]</sup>:<mark class="{target_color}"> {target_text}</mark></div>
    <div class="box local_acc"><span class="und_dec">Prediction accuracy (local)<sup class="small">[3]</sup>:</span><br/>
    <mark class="green">Acc:</mark> {round(bucket_acc*100, 1)}%, <mark class="green">PPV:</mark><mark class="{confidence_color}"> {round(ppv*100, 1)}%</mark>, <mark class="green">NPV:</mark> {round(npv*100, 1)}%  {tweet_warn} </div>
    <div class="box global_acc"><span class="und_dec">Salient model<sup class="small">[4]</sup> metrics (global) (TP-TN-FP-FN):</span><br/>
    Nontweets: <mark class="mid white"><mark class="green"> AUC:</mark> {ext_rec[8][1]} <mark class="green"> MCC:</mark> {ext_rec[8][2]} <mark class="green"> ACC:</mark> {ext_rec[8][0]} <mark class="cm_small white">({ext_rec[8][3]}-{ext_rec[8][4]}-{ext_rec[8][5]}-{ext_rec[8][6]})</mark></mark><br/>
    Tweets: <mark class="mid white"><mark class="green"> AUC:</mark> {ext_rec[8][8]} <mark class="green"> MCC:</mark> {ext_rec[8][9]} <mark class="green"> ACC:</mark> {ext_rec[8][7]} <mark class="cm_small white">({ext_rec[8][10]}-{ext_rec[8][11]}-{ext_rec[8][12]}-{ext_rec[8][13]})</mark></mark><br/>
      </div>
    </div>
        """
    return summary_row


def fmt_smry_row_tmp(pred_color: str, pred_text: str, target_color: str, target_text: str, confidence_color: str,
                     confidence: str, ext_rec: Tuple) -> str:
    global_stats = ext_rec[8]
    global_stats_mapping = {
        'nt_mcc': global_stats[2], 'nt_auc': global_stats[1], 'nt_acc': global_stats[0], 'nt_tp': global_stats[3],
        'nt_tn': global_stats[4], 'nt_fp': global_stats[5], 'nt_fn': global_stats[6],
        't_mcc': global_stats[9], 't_auc': global_stats[8], 't_acc': global_stats[7], 't_tp': global_stats[10],
        't_tn': global_stats[11], 't_fp': global_stats[12], 't_fn': global_stats[13]
    }
    global_stats_div_html = dashboard_constants.global_metrics_summ.format(**global_stats_mapping)
    summary_row = f"""
    <div class="statswrapper">
        <div class="stats1">
        <div class="box prediction">Prediction<sup class="small">[1]</sup>:<mark class="{pred_color}"> {pred_text}</mark>
        <br/> Actual<sup class="small">[2]</sup>:<mark class="{target_color}"> {target_text}</mark></div>
        <div class="box local_acc">Prediction accuracy (local)<sup class="small">[3]</sup>:<br/>
<mark class="{confidence_color} emph"> {confidence} </mark></div> 
        {global_stats_div_html}
        </div>
    """
    return summary_row


def fmt_pred_exp_attr(ext_rec: Tuple, pred_color: str) -> Tuple[float, str]:
    stmt_attr = round(ext_rec[2], 2)
    max_word = f"""<mark class={pred_color}> {ext_rec[3]}: {round(ext_rec[0], 2)} </mark>"""
    return stmt_attr, max_word


def fmt_attr_row(ext_rec: Tuple, pred_color: str) -> str:
    attr_row = f"""
    <div class="stats3">
    <div class="box legend">Attribution Key: Word increased...<br/>
    <mark style="background-color: hsl(0, 75%, 50%); "> prediction of "falsehood" label</mark>&nbsp;&nbsp;<br/>
    <mark style="background-color: hsl(120, 75%, 50%); "> prediction of "falsehood" label absence</mark>
    </div>
    <div class="box max_word">Attribution Score: <mark class="white">{round(ext_rec[2], 2)}</mark>
    <br/>Max Word<sup class="small">[5]</sup>:
    <mark class={pred_color}> {ext_rec[3]}: {round(ext_rec[0], 2)} </mark></div>
    </div>
        """
    return attr_row


def fmt_smlr_stmts(ext_rec: Tuple) -> List[str]:
    stmts_trunc = []
    max_stmt_len = constants.MAX_STMT_LEN
    for erec in ext_rec[4]:
        stmt_color = "green" if erec[1] == 0 else "red"
        stmt_tokens = erec[0].split(" ")
        stmt_text = stmt_tokens[0:min(len(erec[0]), max_stmt_len)]
        stmt_filt = " ".join(stmt_text)
        if len(stmt_tokens) > max_stmt_len:
            stmt_filt = stmt_filt + "..."
        stmts_trunc.append((stmt_filt, stmt_color))
    stmt_lis = [f"""<mark class="{s[1]}"><li>{s[0]}</li></mark>""" for s in stmts_trunc]
    return stmt_lis


def fmt_plot_box(logo_path: str, ss_image: str, stmt_lis: List[str]) -> str:
    plot_box = f"""
    <div class="statsplot1">
    <img class="logo" src="{logo_path}" alt="deep_classiflie_logo" />
    <img class="ss" src="{ss_image}" alt="detailed_report" />
    <div class="box stmt_summs">
    <div class="corner_note">Most Similar Statements/Predictions<sup class="white small">[6]</sup>:</div>
    <ol start="0">{"".join(stmt_lis)}</ol>
    </div>
    </div>
    """
    return plot_box


def fmt_notes_box(ext_rec: Tuple) -> str:
    notes_box = f"""
    <div class="footnotes">
    <ol>
    <li>prediction of whether Washington Post's Fact Checker will add this claim to its "Trump False Claims" DB</li>
    <li>if claim was included in WP's Fact Checker false claims DB at time of original model training</li>
    <li>accuracy estimated by sorting & bucketing the test set sigmoid outputs, averaging performance in each bucket
    <li>global metrics relate to the current model's performance on a test set comprised of ~13K statements made between 
    {ext_rec[5][13].strftime('%Y-%m-%d')} and {ext_rec[5][14].strftime('%Y-%m-%d')}. Training, validation and test sets 
    are chronologically disjoint. </li>
    <li>subject to interpretability filter, some subword tokens have been omitted to facilitate interpretability</li>
    <li>similarity calculated by minimizing a linear function of the L2 pairwise distance between sentence embeddings 
    and target/train sentence sigmoid output delta</li>
    </ol>
    </div>
    </div>
        """
    return notes_box


def gen_detailed_report(datarecord: VisualizationDataRecord, ext_rec: Tuple, ss_image: str, rel_style_path: str,
                        token_mask: List[int], invert_colors: bool = False, logo_path: str = None) -> str:
    html_header, html_footer = fmt_header_footer(rel_style_path)
    pred_color, pred_text, target_color, target_text = fmt_target_pred(datarecord)
    confidence, confidence_color = calc_confidence(ext_rec)
    summary_row = fmt_smry_row(pred_color, pred_text, target_color, target_text, confidence_color, ext_rec)
    word_import_box = fmt_word_import(datarecord.raw_input, datarecord.word_attributions, token_mask, invert_colors)
    attr_row = fmt_attr_row(ext_rec, pred_color)
    stmt_lis = fmt_smlr_stmts(ext_rec)
    plot_box = fmt_plot_box(logo_path, ss_image, stmt_lis)
    notes_box = fmt_notes_box(ext_rec)
    dom = [html_header, summary_row, word_import_box, attr_row, plot_box, notes_box, html_footer]
    htmlout = "".join(dom)
    return htmlout


def gen_per_pred_reports(datarecords: List[VisualizationDataRecord], ext_recs: List[Tuple] = None,
                         ss_images: List = None, paths_tup: Tuple = None, token_mask: List = None,
                         invert_colors: bool = False, pub_thresholds: Tuple = (1, 1)) -> List[Tuple]:
    pred_report_path, style_path, logo_path, tweet_rpt_path = paths_tup
    unpublished_reports = []
    for i, (datarecord, ext_rec, ss_image) in enumerate(zip(datarecords, ext_recs, ss_images)):
        is_stmt = ext_rec[9]
        htmlout = gen_detailed_report(datarecord, ext_rec, ss_image, style_path, token_mask, invert_colors, logo_path)
        accuracy_conf = round(ext_rec[5][5], 3)
        if is_stmt:
            conf_b = True if accuracy_conf > pub_thresholds[1] else False
        else:
            conf_b = True if accuracy_conf > pub_thresholds[0] else False
        pred_name = f'{ext_rec[6]}_{ext_rec[7]}' if not (ext_rec[6] is None or ext_rec[7] is None) else f'pred_{i}'
        pred_report = f'{pred_report_path}/{pred_name}.html'
        with open(pred_report, 'w') as writer:
            writer.write(htmlout)
        report_image = f'{tweet_rpt_path}/{pred_name}.png'
        HTML(pred_report).write_png(target=report_image, presentational_hints=True)
        unpublished_reports.append((report_image, ext_rec[6], ext_rec[7], conf_b, datarecord.pred_class, ss_image,
                                    pred_report))
    return unpublished_reports


def gen_pred_exp_attr_tup(datarecord: VisualizationDataRecord, ext_rec: Tuple, token_mask: List[int],
                          invert_colors: bool = False) -> Tuple[float, str, str]:
    pred_color, _, _, _ = fmt_target_pred(datarecord)
    word_import_html = fmt_word_import(datarecord.raw_input, datarecord.word_attributions, token_mask, invert_colors,
                                       pred_exp_mode=True)
    stmt_attr, max_word_html = fmt_pred_exp_attr(ext_rec, pred_color)
    return stmt_attr, word_import_html, max_word_html


def pred_exp_attr(datarecords: List[VisualizationDataRecord], ext_recs: List[Tuple] = None, token_mask: List = None,
                  invert_colors: bool = False, **_) -> Tuple[List, Tuple]:
    global_metrics_summ = ext_recs[0][8]
    pred_exp_tups = []
    for i, (datarecord, ext_rec) in enumerate(zip(datarecords, ext_recs)):
        pred_exp_tup = gen_pred_exp_attr_tup(datarecord, ext_rec, token_mask, invert_colors)
        pred_exp_tups.append(pred_exp_tup)
    return pred_exp_tups, global_metrics_summ