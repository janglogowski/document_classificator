import os
import joblib
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import mobilenet_v2
from PIL import Image
from pdf2image import convert_from_path
from collections import OrderedDict


def load_vsd_model(classifier, level, cfg):
    ocr_engine = cfg["defaults"]["ocr_engine"]
    if classifier == "tfidf_lr":
        path = os.path.join(cfg["paths"]["project_root"], cfg["paths"]["data"]["models"]["lr_doc_vs_drw"][ocr_engine])
        return joblib.load(os.path.join(path, f"lr_doc_vs_drw_{level}.pkl")), joblib.load(os.path.join(path, f"doc_vs_drw_vect_{level}.pkl"))
    else:
        state_path = os.path.join(cfg["paths"]["project_root"], cfg["paths"]["data"]["models"]["cnn_doc_vs_drw"], f"cnn_doc_vs_drw_{level}.pth")
        model = mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, 2)
        state = torch.load(state_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        return model


def load_doc_type_model(classifier, engine, level, cfg):
    if classifier == "tfidf_lr":
        path = os.path.join(cfg["paths"]["project_root"], cfg["paths"]["data"]["models"]["lr_doc_type_classifier"][engine])
        model = joblib.load(os.path.join(path, f"doc_type_classifier_{level}.pkl"))
        vectorizer = joblib.load(os.path.join(path, f"doc_type_vect_{level}.pkl"))
        return (model, vectorizer)
    else:
        state_path = os.path.join(cfg["paths"]["project_root"], cfg["paths"]["data"]["models"]["cnn_doc_type_classifier"], f"cnn_doc_type_classifier_{level}.pth")
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 6)
        state = torch.load(state_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        names = ["BOM", "DAILY_REPORT", "INSPECTION_REPORT", "MAINTENANCE_LOG", "PRODUCT_DATA_SHEET", "QUALITY_CHECKLIST"]
        return (model, names)



def predict_doc_vs_drw(path, classifier, model, text, cfg):
    ocr_engine = cfg["defaults"]["ocr_engine"]
    if classifier == "tfidf_lr":
        model, vect = model
        return model.predict(vect.transform([text]))[0] if text else "document"
    else:
        ext = os.path.splitext(path)[1].lower()
        if ext in cfg[ocr_engine]["image_extensions"]:
            pil = Image.open(path).convert("RGB")
        else:
            pg = convert_from_path(path, dpi=cfg[ocr_engine]["pdf_dpi"], poppler_path=cfg["paths"]["poppler"]["bin_path"])
            pil = pg[0].convert("RGB") if pg else None
        if pil is None:
            return "undefined"
        tf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        idx = model(tf(pil).unsqueeze(0)).argmax(dim=1).item()
        return ["document", "tech_drw"][idx]


def predict_doc_type(path, classifier, model_bundle, ocr_func, text, cfg):
    ocr_engine = cfg["defaults"]["ocr_engine"]
    if classifier == "tfidf_lr":
        model, vectorizer = model_bundle
        return model.predict(vectorizer.transform([text]))[0] if text else "undefined"
    else:
        model, names = model_bundle
        ext = os.path.splitext(path)[1].lower()
        if ext in cfg[ocr_engine]["image_extensions"]:
            pil = Image.open(path).convert("RGB")
        else:
            pg = convert_from_path(path, dpi=cfg[ocr_engine]["pdf_dpi"], poppler_path=cfg["paths"]["poppler"]["bin_path"])
            pil = pg[0].convert("RGB") if pg else None
        if pil is None:
            return "undefined"
        tf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        idx = model(tf(pil).unsqueeze(0)).argmax(dim=1).item()
        return names[idx]

