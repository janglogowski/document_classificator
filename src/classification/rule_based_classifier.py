import os
import yaml
import json
from sklearn.metrics import classification_report

#--- load config ---#
cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
ROOT = os.path.abspath(cfg["paths"]["project_root"])

#--- select ocr engine and scan level  ---#
engine = "easy_ocr"      # easy_ocr or "tesseract_ocr"
level  = "level1"        # level1/level2/level3

#--- paths to extracted texts ---#
DOCS_FOLDER = os.path.join(ROOT,cfg["paths"]["data"]["extracted"][engine][level])
os.makedirs(DOCS_FOLDER, exist_ok=True)

#--- load docs and labels ---#
docs = []
true = []
for root, _, files in os.walk(DOCS_FOLDER):
    for fn in files:
        if not fn.lower().endswith(".txt"):
            continue
        text = open(os.path.join(root, fn), encoding="utf-8").read().strip()
        if not text:
            continue
        docs.append(text)
        name = fn.lower()
        if "bom" in name:
            true.append("bill_of_materials")
        elif "product" in name:
            true.append("product_data_sheet")
        elif "quality" in name:
            true.append("quality_checklist")
        elif "daily" in name:
            true.append("daily_report")
        elif "inspection" in name:
            true.append("inspection_report")
        elif "maintenance" in name:
            true.append("maintenance_log")
        else:
            true.append("other")

#--- rule-based classifier ---#
def rule_classifier(text):
    t = text.lower()
    if any(k in t for k in ["unit price","line total","bill of material"]):
        return "bill_of_materials"
    if any(k in t for k in ["flow rate","motor power","voltage"]):
        return "product_data_sheet"
    if "quality inspection checklist" in t:
        return "quality_checklist"
    if any(k in t for k in ["shift","operation","defect qty"]):
        return "daily_report"
    if any(k in t for k in ["measured (mm)","deviation","pass/fail"]):
        return "inspection_report"
    if any(k in t for k in ["maintenance type","performed by","downtime"]):
        return "maintenance_log"
    return "other"

pred = [rule_classifier(d) for d in docs]

#--- evaluate ---#
total     = len(true)
correct   = sum(t == p for t,p in zip(true, pred))
incorrect = total - correct
accuracy  = correct / total if total else 0.0

class_report = classification_report(true, pred, digits=4, output_dict=True, zero_division=0)

print("=== Rule-based Classifier ===")
print(f"Total docs:     {total}")
print(f"Correct:        {correct}")
print(f"Incorrect:      {incorrect}")
print(f"Accuracy:       {accuracy:.4%}\n")
print(classification_report(true, pred, digits=4,zero_division=0))

#--- save JSON ---#
metrics = {
    "engine": engine,
    "level": level,
    "total": total,
    "correct": correct,
    "incorrect": incorrect,
    "accuracy": accuracy,
    "classification_report": class_report}

out_dir = os.path.join(ROOT,cfg["paths"]["data"]["models"]["rule_based_classifier"][engine],f"rule_based_metrics_{level}.json")

with open(out_dir, "w", encoding="utf-8") as jf:
    json.dump(metrics, jf, indent=2, ensure_ascii=False)

print(f"\nSaved metrics >>>> {out_dir}")
