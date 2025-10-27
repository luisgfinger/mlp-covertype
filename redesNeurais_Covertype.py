
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Atividade 3 – Redes Neurais Artificiais (Covertype + MLP)
Autor: Luis Gustavo Grando Finger
Curso: Engenharia de Software | Disciplina: Inteligência Artificial

Como usar (exemplos):
  python atividade3_mlp_covtype.py --mode quick
  python atividade3_mlp_covtype.py --mode full --max-iter 200 --cv 3
  python atividade3_mlp_covtype.py --export-report resultados/relatorio_auto.md

Requisitos:
  - Python 3.9+ recomendado
  - scikit-learn, pandas, numpy, matplotlib, seaborn (opcional)
"""
import argparse
import os
import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

RANDOM_STATE = 42

def load_covtype(as_frame=True):
    """
    Tenta carregar a base Covertype via scikit-learn.
    Caso esteja sem internet, o scikit-learn utiliza o cache local (~/.scikit_learn).
    Se falhar, mostra instruções de download manual.
    """
    try:
        data = fetch_covtype(as_frame=as_frame)
        X = data.data
        y = data.target
        # Covertype tem rótulos [1..7]; não há ajuste necessário, mas vamos forçar int
        y = y.astype(int)
        return X, y
    except Exception as e:
        print("[ERRO] Não foi possível baixar/carregar a base Covertype automaticamente.\n"
              "Opções:\n"
              "  1) Conectar à internet e rodar novamente.\n"
              "  2) Baixar manualmente a base do UCI (https://archive.ics.uci.edu/dataset/31/covertype)\n"
              "     e converter para CSV, depois usar --csv /caminho/dados.csv e --target nome_da_coluna_classe.\n"
              f"Detalhes do erro: {e}")
        sys.exit(1)

def load_from_csv(csv_path, target_col):
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Coluna alvo '{target_col}' não encontrada no CSV. Colunas: {list(df.columns)}")
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])
    return X, y

def get_param_grid(mode="quick"):
    """
    Grade de hiperparâmetros:
      - quick: pequena (execução rápida p/ validar pipeline)
      - full: maior (execução mais demorada p/ melhores resultados)
    Você pode ajustar via argumentos CLI também.
    """
    if mode == "quick":
        return {
            "mlp__hidden_layer_sizes": [(64,), (128,), (64, 64)],
            "mlp__activation": ["relu", "tanh"],
            "mlp__alpha": [1e-4, 1e-3],
            "mlp__learning_rate_init": [1e-3],
            "mlp__max_iter": [100],
        }
    else:  # full
        return {
            "mlp__hidden_layer_sizes": [(128,), (256,), (128, 64), (256,128), (128,128,64)],
            "mlp__activation": ["relu", "tanh", "logistic"],
            "mlp__alpha": [1e-5, 1e-4, 1e-3],
            "mlp__learning_rate_init": [5e-4, 1e-3, 2e-3],
            "mlp__max_iter": [150, 200],
        }

def build_pipeline(max_iter=100, early_stopping=True):
    mlp = MLPClassifier(
        random_state=RANDOM_STATE,
        max_iter=max_iter,
        early_stopping=early_stopping,
        n_iter_no_change=10,
        validation_fraction=0.1
    )
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),  # dados esparsos em Covertype (one-hot), evitar centrar
        ("mlp", mlp)
    ])
    return pipe

def plot_and_save_confusion_matrix(y_true, y_pred, outdir, title):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(6,6))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fname = os.path.join(outdir, f"{title.lower().replace(' ', '_')}_cm.png")
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    return fname

def main():
    parser = argparse.ArgumentParser(description="Atividade 3 – MLP na base Covertype")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick", help="Tamanho da busca em grade (grid).")
    parser.add_argument("--csv", type=str, default=None, help="Caminho para CSV da base (opcional).")
    parser.add_argument("--target", type=str, default=None, help="Nome da coluna alvo no CSV (se usar --csv).")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proporção de teste.")
    parser.add_argument("--cv", type=int, default=3, help="N de folds para StratifiedKFold.")
    parser.add_argument("--max-iter", type=int, default=None, help="Sobrescreve max_iter do MLP.")
    parser.add_argument("--export-report", type=str, default=None, help="Exporta um relatório-resumo em Markdown.")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    outdir = os.path.join("resultados", timestamp)
    os.makedirs(outdir, exist_ok=True)

    # Carregar dados
    if args.csv:
        if not args.target:
            print("Ao usar --csv, informe --target com o nome da coluna de classe.")
            sys.exit(1)
        X, y = load_from_csv(args.csv, args.target)
        origem = f"CSV: {args.csv} (target={args.target})"
    else:
        X, y = load_covtype(as_frame=True)
        origem = "fetch_covtype (scikit-learn)"

    # Split treino/teste estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=RANDOM_STATE, stratify=y
    )

    # Pipeline + Grid
    pipe = build_pipeline(max_iter=args.max_iter or (150 if args.mode=="full" else 100))
    param_grid = get_param_grid(args.mode)

    # Se usuário alterou max_iter, injeta isso no grid também (para consistência)
    if args.max_iter is not None:
        param_grid["mlp__max_iter"] = [args.max_iter]

    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=RANDOM_STATE)

    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        scoring=["accuracy", "f1_macro"],
        refit="f1_macro",
        n_jobs=-1,
        verbose=1,
        return_train_score=False
    )

    print("[INFO] Iniciando GridSearchCV...")
    gs.fit(X_train, y_train)

    # Resultados de validação
    cv_results = pd.DataFrame(gs.cv_results_)
    cv_results.to_csv(os.path.join(outdir, "gridsearch_results.csv"), index=False)
    print("[OK] gridsearch_results.csv salvo.")

    # Melhor modelo
    best = gs.best_estimator_
    print("[MELHOR] params:", gs.best_params_)
    print("[MELHOR] f1_macro (val):", gs.best_score_)

    # Avaliação em teste
    y_pred = best.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")

    print(f"[TESTE] accuracy={acc:.4f} | f1_macro={f1m:.4f}")
    cls_rep = classification_report(y_test, y_pred, digits=4)
    with open(os.path.join(outdir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(cls_rep)
    print("[OK] classification_report.txt salvo.")

    # Matriz de confusão
    cm_path = plot_and_save_confusion_matrix(y_test, y_pred, outdir, "MLP Covertype")
    print(f"[OK] Matriz de confusão salva em: {cm_path}")

    # Salvar metadados do experimento
    meta = {
        "origem_dados": origem,
        "amostras_total": int(X.shape[0]),
        "features_total": int(X.shape[1]),
        "split": {"test_size": args.test_size},
        "cv": args.cv,
        "best_params": gs.best_params_,
        "f1_macro_val": float(gs.best_score_),
        "accuracy_test": float(acc),
        "f1_macro_test": float(f1m),
        "timestamp": timestamp,
        "script": "atividade3_mlp_covtype.py",
        "observacoes": "Preencha observações adicionais no relatório."
    }
    with open(os.path.join(outdir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("[OK] metadata.json salvo.")

    # Report auto (opcional)
    if args.export_report:
        md = []
        md.append(f"# Atividade 3 – MLP (Covertype)\n")
        md.append(f"**Origem dos dados:** {origem}\n")
        md.append(f"**Tamanho:** {X.shape[0]} amostras, {X.shape[1]} atributos\n")
        md.append(f"**Divisão treino/teste:** {1-args.test_size:.0%}/{args.test_size:.0%}\n")
        md.append("## Melhor configuração (validação)\n")
        md.append("```json\n" + json.dumps(gs.best_params_, indent=2) + "\n```\n")
        md.append(f"**F1 macro (val):** {gs.best_score_:.4f}\n")
        md.append("## Desempenho no conjunto de teste\n")
        md.append(f"- Accuracy: {acc:.4f}\n- F1 macro: {f1m:.4f}\n")
        md.append("### Classification report\n")
        md.append("```\n" + cls_rep + "\n```\n")
        md.append(f"![Matriz de Confusão]({os.path.basename(cm_path)})\n")
        Path(args.export_report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.export_report, "w", encoding="utf-8") as f:
            f.write("\n".join(md))
        print(f"[OK] Relatório automático salvo em: {args.export_report}")

if __name__ == "__main__":
    main()
