"""Create a concise, evidence-based project summary PDF."""

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "output" / "pdf" / "banking_crisis_ews_summary.pdf"


def fmt(value: float) -> str:
    return f"{value:.3f}"


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D1D5DB"))
    canvas.line(doc.leftMargin, 1.3 * cm, A4[0] - doc.rightMargin, 1.3 * cm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#4B5563"))
    canvas.drawString(doc.leftMargin, 0.85 * cm, "Banking Crisis Early Warning System")
    canvas.drawRightString(A4[0] - doc.rightMargin, 0.85 * cm, f"Page {doc.page}")
    canvas.restoreState()


def styled_table(rows, widths):
    cell_style = ParagraphStyle(
        "TableCell", fontName="Helvetica", fontSize=8.5, leading=10.5, textColor=colors.black
    )
    header_style = ParagraphStyle(
        "TableHeader", fontName="Helvetica-Bold", fontSize=8.5, leading=10.5, textColor=colors.white
    )
    formatted_rows = [
        [Paragraph(str(value), header_style if row_index == 0 else cell_style) for value in row]
        for row_index, row in enumerate(rows)
    ]
    table = Table(formatted_rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F4C5C")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("LEADING", (0, 0), (-1, -1), 11),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F7F7")]),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def main():
    panel = pd.read_csv(ROOT / "data" / "processed" / "banking_crisis_model_ready.csv")
    tabular = pd.read_csv(ROOT / "results" / "nested_cv_results.csv", index_col=0)
    lstm = pd.read_csv(ROOT / "results" / "lstm_cv_results.csv", index_col=0)
    thresholds = pd.read_csv(ROOT / "results" / "lightgbm_threshold_diagnostics.csv")
    shap = pd.read_csv(ROOT / "results" / "lightgbm_shap_importance.csv")
    coverage = pd.read_csv(ROOT / "results" / "candidate_feature_coverage.csv")
    extended = pd.read_csv(ROOT / "results" / "extended_feature_cv_results.csv", index_col=0)
    comparison = pd.concat([tabular, lstm]).sort_values("PRAUC_mean", ascending=False)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCustom", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=22, leading=27, textColor=colors.HexColor("#0F4C5C"), alignment=TA_CENTER, spaceAfter=12))
    styles.add(ParagraphStyle(name="Subhead", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=13, leading=16, textColor=colors.HexColor("#0F4C5C"), spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle(name="BodyCustom", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.5, leading=14, textColor=colors.HexColor("#1F2937"), spaceAfter=6))
    styles.add(ParagraphStyle(name="Callout", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=10, leading=14, textColor=colors.HexColor("#0F4C5C"), backColor=colors.HexColor("#E6F2F2"), borderColor=colors.HexColor("#8ABDBB"), borderWidth=0.5, borderPadding=9, spaceBefore=8, spaceAfter=9))

    doc = SimpleDocTemplate(str(OUT_PATH), pagesize=A4, leftMargin=1.65 * cm, rightMargin=1.65 * cm, topMargin=1.5 * cm, bottomMargin=1.75 * cm)
    story = []

    story.append(Paragraph("Banking Crisis Early Warning System", styles["TitleCustom"]))
    story.append(Paragraph("Project summary and validation audit", styles["BodyCustom"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Bottom line: the pipeline runs end-to-end from the supplied processed panel, with leakage-safe country-grouped validation. The LSTM is the strongest evaluated model, but the small event count and incomplete raw World Bank coverage mean this is a research screening tool, not a deployment-ready policy engine.", styles["Callout"]))

    story.append(Paragraph("Project Scope", styles["Subhead"]))
    story.append(Paragraph(
        f"The system predicts the first year of a systemic banking crisis across {panel['country'].nunique()} emerging-market countries from {panel['prediction_year'].min()} to {panel['prediction_year'].max()}. It contains {len(panel):,} country-years and {int(panel['crisis_t'].sum())} crisis onsets ({panel['crisis_t'].mean() * 100:.1f}% prevalence). Features are lagged one year, so the target-year crisis label is never paired with same-year macroeconomic information.",
        styles["BodyCustom"],
    ))
    scope_rows = [
        ["Component", "Verified design"],
        ["Target", "First year of a systemic banking crisis episode"],
        ["Predictors", "Six lagged macro-financial indicators, including FX depreciation"],
        ["Validation", "Six crisis-balanced, country-grouped outer folds with inner validation"],
        ["Models", "Logistic regression, random forest, XGBoost, LightGBM, and LSTM"],
    ]
    story.append(styled_table(scope_rows, [4.0 * cm, 12.1 * cm]))

    story.append(Paragraph("Data Readiness", styles["Subhead"]))
    story.append(Paragraph("The supplied model-ready panel passed shape, target-count, fold-assignment, and leakage tests. Feature missingness remains material: 10% for FX depreciation and roughly 36-55% for the World Bank indicators. The raw World Bank workbook covers only 16 of the 24 selected countries; the code now stops rather than silently rebuilding an incomplete panel. A complete 24-country World Bank re-export is required for fully raw-data reproducibility. A separate official-API enrichment extract now supplies coverage-audited banking and external-vulnerability candidates.", styles["BodyCustom"]))

    story.append(PageBreak())
    story.append(Paragraph("Validation and Modeling Audit", styles["TitleCustom"]))
    story.append(Paragraph("Controls added and verified in the final workflow", styles["BodyCustom"]))
    audit_rows = [
        ["Control", "Why it matters", "Status"],
        ["One-year feature lag", "Prevents target-year information from entering an early-warning score", "Verified"],
        ["Country-grouped folds", "Keeps all years of each country in exactly one outer partition", "Verified"],
        ["Nested validation", "Keeps model selection and threshold selection away from outer test countries", "Verified"],
        ["Past-only filling", "Uses country history, then train-only same-year and expanding fallbacks", "Verified"],
        ["Earliest-year gaps", "Uses a zero-information placeholder only when no historical value exists", "Verified"],
        ["Raw coverage guard", "Prevents incomplete source files from overwriting the model-ready panel", "Added"],
    ]
    story.append(styled_table(audit_rows, [3.4 * cm, 9.0 * cm, 3.7 * cm]))

    story.append(Paragraph("Evaluation Interpretation", styles["Subhead"]))
    story.append(Paragraph("PR-AUC is the primary ranking metric because crisis onsets are rare. A random score would yield a PR-AUC near 0.035. AUC remains useful for broad rank ordering but can look optimistic in an imbalanced problem. Reported uncertainty is the standard deviation across the six held-out country folds, not a confidence interval.", styles["BodyCustom"]))
    story.append(Paragraph("Model Results", styles["Subhead"]))
    result_rows = [["Model", "AUC", "PR-AUC", "Assessment"]]
    for model, row in comparison.iterrows():
        assessment = "Best observed" if model == "LSTM" else ("Useful benchmark" if model == "Basel_benchmark" else "Below LSTM")
        result_rows.append([model.replace("_", " "), f"{fmt(row.AUC_mean)} +/- {fmt(row.AUC_std)}", f"{fmt(row.PRAUC_mean)} +/- {fmt(row.PRAUC_std)}", assessment])
    story.append(styled_table(result_rows, [4.2 * cm, 4.0 * cm, 4.0 * cm, 3.9 * cm]))
    story.append(Paragraph("The LSTM achieved the highest held-out PR-AUC (0.112). Logistic regression was competitive (0.103), which is valuable evidence that simple, transparent relationships remain strong in this small sample. The fixed LightGBM reference configuration did not exceed the credit-to-GDP benchmark on PR-AUC and should not be selected for deployment without additional tuning and calibration.", styles["Callout"]))

    story.append(PageBreak())
    story.append(Paragraph("Diagnostics and Recommendation", styles["TitleCustom"]))
    threshold_summary = thresholds[["test_precision", "test_recall", "test_f2", "test_alert_rate"]].mean()
    story.append(Paragraph("LightGBM Alert Threshold Diagnostics", styles["Subhead"]))
    story.append(Paragraph(f"Thresholds were chosen within each outer fold using inner-validation F2 optimization, then applied once to held-out countries. Mean held-out recall was {threshold_summary.test_recall * 100:.1f}% at precision of {threshold_summary.test_precision * 100:.1f}%, with a {threshold_summary.test_alert_rate * 100:.1f}% alert rate. This is recall-oriented screening behavior, not a sufficiently precise autonomous alerting rule.", styles["BodyCustom"]))
    threshold_rows = [["Metric", "Average across six held-out folds"], ["Recall", f"{threshold_summary.test_recall * 100:.1f}%"], ["Precision", f"{threshold_summary.test_precision * 100:.1f}%"], ["F2 score", fmt(threshold_summary.test_f2)], ["Alert rate", f"{threshold_summary.test_alert_rate * 100:.1f}%"]]
    story.append(styled_table(threshold_rows, [6.0 * cm, 10.1 * cm]))

    story.append(Paragraph("Reference LightGBM Drivers", styles["Subhead"]))
    driver_rows = [["Rank", "Feature", "Mean absolute SHAP value"]]
    for rank, row in enumerate(shap.head(4).itertuples(index=False), start=1):
        driver_rows.append([str(rank), row.feature.replace("_t_minus_1", " (t-1)").replace("_", " "), fmt(row.mean_abs_shap)])
    story.append(styled_table(driver_rows, [2.0 * cm, 9.6 * cm, 4.5 * cm]))

    story.append(Paragraph("External-Vulnerability Feature Screen", styles["Subhead"]))
    debt_coverage = coverage.loc[coverage["feature"] == "external_debt_service_exports_t_minus_1", "crisis_onset_coverage_pct"].item()
    reserves_coverage = coverage.loc[coverage["feature"] == "reserves_to_short_term_debt_t_minus_1", "crisis_onset_coverage_pct"].item()
    base_pr = extended.loc["LightGBM_baseline", "PRAUC_mean"]
    extended_pr = extended.loc["LightGBM_external_vulnerability", "PRAUC_mean"]
    extended_auc = extended.loc["LightGBM_external_vulnerability", "AUC_mean"]
    story.append(Paragraph(f"The new official World Bank extract supports two testable external indicators: debt service to exports ({debt_coverage:.1f}% crisis-onset coverage) and reserves to short-term debt ({reserves_coverage:.1f}%). Adding both to LightGBM increased AUC to {extended_auc:.3f} and PR-AUC from {base_pr:.3f} to {extended_pr:.3f}. The PR-AUC gain is modest relative to variation across folds, so this remains an experimental extension rather than a replacement for the LSTM.", styles["BodyCustom"]))

    story.append(Paragraph("Recommended Next Steps", styles["Subhead"]))
    story.append(Paragraph("1. Re-export World Bank indicators for all 24 countries and rerun the complete raw-data pipeline. 2. Treat the LSTM and logistic regression as candidate research models, then add calibration, probability stability checks, and decision-cost analysis before choosing an operating threshold. 3. Add an external holdout period or new-country holdout before deployment claims. 4. Investigate twin-currency-crisis features only after they are available without leakage. 5. Maintain a human review queue for every model alert; current precision is too low for automated intervention.", styles["BodyCustom"]))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Created {OUT_PATH}")


if __name__ == "__main__":
    main()
