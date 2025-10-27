# 🧠 Atividade 3 – Redes Neurais Artificiais (MLP na Covertype)

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Academic-green.svg)](LICENSE)
![Status](https://img.shields.io/badge/status-Completed-success.svg)
![Language](https://img.shields.io/badge/language-Python-yellow.svg)

Este projeto implementa uma **Rede Neural Artificial do tipo MLP (Multi-Layer Perceptron)** aplicada à base de dados **Covertype**, utilizando a biblioteca `scikit-learn`.  
O objetivo é **classificar tipos de cobertura florestal** com base em atributos ambientais e topográficos, avaliando o desempenho do modelo e explorando o ajuste de hiperparâmetros via **Grid Search**.

---

## 📘 **Descrição do Projeto**

O script `redesNeurais_Covertype.py` realiza o **treinamento, validação e avaliação** de um classificador MLP (Multi-Layer Perceptron) sobre a base Covertype (UCI Machine Learning Repository).  

Ele permite testar tanto uma execução rápida (modo `quick`) quanto uma busca completa de parâmetros (modo `full`), além de exportar relatórios automáticos com métricas e a matriz de confusão.

---

## 🎯 **Objetivos**

- Aplicar conceitos de **Redes Neurais Artificiais** em um problema real de **classificação supervisionada**.  
- Analisar o impacto de diferentes **hiperparâmetros** no desempenho do MLP.  
- Avaliar o modelo com **validação cruzada** e métricas como *accuracy* e *F1-macro*.  
- Gerar relatórios e gráficos de desempenho automaticamente.

---

## ⚙️ **Funcionalidades**

✅ Carregamento automático da base **Covertype** via `scikit-learn`  
✅ Pipeline completo com `StandardScaler` e `MLPClassifier`  
✅ Busca em grade (GridSearchCV) com validação cruzada  
✅ Avaliação com métricas e relatório detalhado  
✅ Geração automática de:
- `classification_report.txt`  
- `gridsearch_results.csv`  
- `metadata.json`  
- `mlp_covertype_cm.png` (matriz de confusão)  
✅ Exportação opcional de relatório Markdown (`--export-report`)

---

## 🧩 **Requisitos**

Antes de rodar o projeto, instale os pacotes necessários:

```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

**Requisitos mínimos:**
- Python **3.9+**
- Internet (para baixar a base Covertype, caso não esteja em cache)

---

## 🚀 **Como executar**

Clone o repositório e entre na pasta do projeto:

```bash
git clone https://github.com/luisgfinger/mlp-covertype.git
cd mlp-covertype
```

---

### 🧪 **Execução rápida (modo quick)**

Ideal para validar o funcionamento do pipeline:

```bash
python redesNeurais_Covertype.py --mode quick
```

- Testa poucas combinações de parâmetros.  
- Executa rapidamente (5–10 minutos).  
- Gera resultados na pasta `resultados/YYYYmmdd-HHMMSS/`.

---

### 🔍 **Execução completa (modo full)**

Realiza uma busca detalhada de hiperparâmetros e maior número de iterações:

```bash
python redesNeurais_Covertype.py --mode full --max-iter 200 --cv 3
```

- Explora várias combinações de camadas, ativações e taxas de aprendizado.  
- Pode levar **20–40 minutos**, dependendo do hardware.  
- Resultados mais precisos e melhor acurácia (~90%).

---

### 🧾 **Gerar relatório automático**

Cria um relatório Markdown com resumo, métricas e imagem da matriz de confusão:

```bash
python redesNeurais_Covertype --mode full --export-report resultados/relatorio_auto.md
```

O arquivo conterá:
- Parâmetros ideais encontrados
- Métricas de validação e teste
- Relatório de classificação completo
- Link da matriz de confusão

---

## 📊 **Saída esperada**

Pasta `resultados/YYYYmmdd-HHMMSS/` conterá:

| Arquivo | Descrição |
|----------|------------|
| `gridsearch_results.csv` | Resultados da validação cruzada |
| `classification_report.txt` | Métricas detalhadas (precision, recall, F1) |
| `mlp_covertype_cm.png` | Matriz de confusão do modelo |
| `metadata.json` | Resumo do experimento |
| `relatorio_auto.md` *(opcional)* | Relatório automático |

---

## 📈 **Resultados obtidos (exemplo)**

- **Acurácia (accuracy):** 0.9000  
- **F1-macro:** 0.8650  
- **Principais acertos:** classes 0, 1 e 2  
- **Principais confusões:** entre classes 0 e 1 (florestas semelhantes)  

A matriz de confusão mostra forte concentração na diagonal principal, indicando **alta taxa de acerto** e **boa generalização** do modelo.

---

## 🧠 **Estrutura do Projeto**

```
├── redesNeurais_Covertype.py      # Script principal
├── resultados/                     # Saídas dos experimentos
│   ├── 20251027-171200/            # Execução com timestamp
│   │   ├── gridsearch_results.csv
│   │   ├── classification_report.txt
│   │   ├── mlp_covertype_cm.png
│   │   ├── metadata.json
│   │   └── relatorio_auto.md
└── README.md
```

---

## 🧾 **Licença**

Este projeto é de uso acadêmico e foi desenvolvido como parte da disciplina **Inteligência Artificial** do curso de **Engenharia de Software** na **Fundação Assis Gurgacz (FAG)**.

---

## 👨‍💻 **Autor**

**Luis Gustavo Grando Finger**  
Engenharia de Software – Fundação Assis Gurgacz (FAG)  
📧 [luisgfinger@gmail.com](mailto:luisgfinger@gmail.com)
