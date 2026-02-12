# 🚀 Multi Language Sentiment Engine

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.10-0194E2.svg)](https://mlflow.org/)
[![Prometheus](https://img.shields.io/badge/Prometheus-2.48-E6522C.svg)](https://prometheus.io/)
[![Redis](https://img.shields.io/badge/Redis-7-DC382D.svg)](https://redis.io/)
[![scikit-learn](https://img.shields.io/badge/scikit-learn-1.4-F7931E.svg)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[English](#english) | [Português](#português)

---

## English

### 🎯 Overview

**Multi Language Sentiment Engine** — Advanced data science project: multi-language-sentiment-engine

Total source lines: **2,331** across **16** files in **2** languages.

### ✨ Key Features

- **Production-Ready Architecture**: Modular, well-documented, and following best practices
- **Comprehensive Implementation**: Complete solution with all core functionality
- **Clean Code**: Type-safe, well-tested, and maintainable codebase
- **Easy Deployment**: Docker support for quick setup and deployment

### 🚀 Quick Start

#### Prerequisites
- Python 3.12+
- Docker and Docker Compose (optional)

#### Installation

1. **Clone the repository**
```bash
git clone https://github.com/galafis/multi-language-sentiment-engine.git
cd multi-language-sentiment-engine
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```




## 🐳 Docker

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov --cov-report=html

# Run with verbose output
pytest -v
```

### 📁 Project Structure

```
multi-language-sentiment-engine/
├── config/
│   ├── README.md
│   ├── __init__.py
│   ├── db_config.yaml
│   ├── kafka_config.yaml
│   ├── logging_config.yaml
│   └── model_config.yaml
├── data/
│   ├── __init__.py
│   └── tweets_sample.json
├── deployment/
│   ├── README.md
│   ├── docker-compose.yml
│   └── k8s_deployment.yaml
├── docs/
│   └── README.md
├── scripts/
│   └── download_models.py
├── src/
│   ├── api/
│   │   └── rest_api.py
│   ├── data/
│   │   └── __init__.py
│   ├── evaluation/
│   │   └── __init__.py
│   ├── models/
│   │   └── transformer_model.py
│   ├── preprocessing/
│   │   ├── README.md
│   │   └── __init__.py
│   ├── streaming/
│   │   ├── aggregation_service.py
│   │   ├── preprocessing_service.py
│   │   └── sentiment_analysis_service.py
│   └── visualization/
│       └── __init__.py
├── tests/
│   ├── README.md
│   ├── __init__.py
│   ├── test_api.py
│   └── test_transformer_model.py
├── CONTRIBUTING.md
├── README.md
└── requirements.txt
```

### 🛠️ Tech Stack

| Technology | Usage |
|------------|-------|
| Python | 15 files |
| HTML | 1 files |

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### 👤 Author

**Gabriel Demetrios Lafis**

- GitHub: [@galafis](https://github.com/galafis)
- LinkedIn: [Gabriel Demetrios Lafis](https://linkedin.com/in/gabriel-demetrios-lafis)

---

## Português

### 🎯 Visão Geral

**Multi Language Sentiment Engine** — Advanced data science project: multi-language-sentiment-engine

Total de linhas de código: **2,331** em **16** arquivos em **2** linguagens.

### ✨ Funcionalidades Principais

- **Arquitetura Pronta para Produção**: Modular, bem documentada e seguindo boas práticas
- **Implementação Completa**: Solução completa com todas as funcionalidades principais
- **Código Limpo**: Type-safe, bem testado e manutenível
- **Fácil Implantação**: Suporte Docker para configuração e implantação rápidas

### 🚀 Início Rápido

#### Pré-requisitos
- Python 3.12+
- Docker e Docker Compose (opcional)

#### Instalação

1. **Clone the repository**
```bash
git clone https://github.com/galafis/multi-language-sentiment-engine.git
cd multi-language-sentiment-engine
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```




### 🧪 Testes

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov --cov-report=html

# Run with verbose output
pytest -v
```

### 📁 Estrutura do Projeto

```
multi-language-sentiment-engine/
├── config/
│   ├── README.md
│   ├── __init__.py
│   ├── db_config.yaml
│   ├── kafka_config.yaml
│   ├── logging_config.yaml
│   └── model_config.yaml
├── data/
│   ├── __init__.py
│   └── tweets_sample.json
├── deployment/
│   ├── README.md
│   ├── docker-compose.yml
│   └── k8s_deployment.yaml
├── docs/
│   └── README.md
├── scripts/
│   └── download_models.py
├── src/
│   ├── api/
│   │   └── rest_api.py
│   ├── data/
│   │   └── __init__.py
│   ├── evaluation/
│   │   └── __init__.py
│   ├── models/
│   │   └── transformer_model.py
│   ├── preprocessing/
│   │   ├── README.md
│   │   └── __init__.py
│   ├── streaming/
│   │   ├── aggregation_service.py
│   │   ├── preprocessing_service.py
│   │   └── sentiment_analysis_service.py
│   └── visualization/
│       └── __init__.py
├── tests/
│   ├── README.md
│   ├── __init__.py
│   ├── test_api.py
│   └── test_transformer_model.py
├── CONTRIBUTING.md
├── README.md
└── requirements.txt
```

### 🛠️ Stack Tecnológica

| Tecnologia | Uso |
|------------|-----|
| Python | 15 files |
| HTML | 1 files |

### 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

### 👤 Autor

**Gabriel Demetrios Lafis**

- GitHub: [@galafis](https://github.com/galafis)
- LinkedIn: [Gabriel Demetrios Lafis](https://linkedin.com/in/gabriel-demetrios-lafis)
