# Multi-Language Sentiment Engine

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-005571?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Hugging Face Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow?logo=huggingface)](https://huggingface.co/docs/transformers/index)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Redis](https://img.shields.io/badge/Redis-7.0-DC382D?logo=redis)](https://redis.io/)
[![Prometheus](https://img.shields.io/badge/Prometheus-2.47.0-E6522C?logo=prometheus)](https://prometheus.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🇧🇷 Português

### Visão Geral

Este repositório apresenta um **Mecanismo de Análise de Sentimento Multi-idioma** robusto e escalável, desenvolvido para processar e analisar o sentimento de textos em diversas línguas. Utilizando modelos de *transformers* de última geração, o sistema oferece capacidades de análise de sentimento, detecção de emoções e análise baseada em aspectos, tudo através de uma API RESTful de alta performance.

### Funcionalidades Principais

*   **Análise de Sentimento Multi-idioma**: Suporte para múltiplos idiomas com modelos otimizados para cada língua.
*   **Detecção de Emoções**: Identificação de emoções como alegria, tristeza, raiva, medo, surpresa, desgosto e neutralidade.
*   **Análise de Sentimento Baseada em Aspectos**: Avaliação do sentimento em relação a aspectos específicos mencionados no texto.
*   **Detecção de Sarcasmo**: Capacidade de identificar sarcasmo em textos.
*   **API RESTful com FastAPI**: Interface de programação de aplicações (API) moderna e eficiente.
*   **Métricas e Monitoramento**: Integração com Prometheus para monitoramento.
*   **Cache com Redis**: Otimização de performance através de cache de resultados.
*   **Arquitetura Modular**: Código bem organizado e modular.

### Arquitetura do Sistema

![Diagrama de Arquitetura](docs/architecture_diagram.png)

### Tecnologias Utilizadas

*   **Python**: Linguagem de programação principal.
*   **FastAPI**: Framework web para construção da API RESTful.
*   **Hugging Face Transformers**: Biblioteca para modelos de linguagem baseados em transformers.
*   **PyTorch**: Biblioteca de aprendizado de máquina.
*   **Redis**: Banco de dados em memória para cache.
*   **Prometheus**: Sistema de monitoramento e alerta.

### Instalação e Uso

1.  **Clone o repositório**:
    ```bash
    git clone https://github.com/GabrielDemetriosLafis/multi-language-sentiment-engine.git
    cd multi-language-sentiment-engine
    ```

2.  **Instale as dependências**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Execute a API**:
    ```bash
    uvicorn src.api.rest_api:app --host 0.0.0.0 --port 8000 --reload
    ```

### Documentação da API

Acesse a documentação interativa (Swagger UI) em `http://localhost:8000/docs`.

---

## 🇬🇧 English

### Overview

This repository presents a robust and scalable **Multi-Language Sentiment Analysis Engine**, designed to process and analyze the sentiment of texts in various languages. Utilizing state-of-the-art *transformer* models, the system offers sentiment analysis, emotion detection, and aspect-based analysis capabilities, all through a high-performance RESTful API.

### Key Features

*   **Multi-Language Sentiment Analysis**: Support for multiple languages with optimized models for each language.
*   **Emotion Detection**: Identification of emotions such as joy, sadness, anger, fear, surprise, disgust, and neutrality.
*   **Aspect-Based Sentiment Analysis**: Evaluation of sentiment regarding specific aspects mentioned in the text.
*   **Sarcasm Detection**: Ability to identify sarcasm in texts.
*   **RESTful API with FastAPI**: Modern and efficient Application Programming Interface (API).
*   **Metrics and Monitoring**: Integration with Prometheus for monitoring.
*   **Caching with Redis**: Performance optimization through result caching.
*   **Modular Architecture**: Well-organized and modular code.

### System Architecture

![Architecture Diagram](docs/architecture_diagram.png)

### Technologies Used

*   **Python**: Main programming language.
*   **FastAPI**: Web framework for building the RESTful API.
*   **Hugging Face Transformers**: Library for transformer-based language models.
*   **PyTorch**: Machine learning library.
*   **Redis**: In-memory database for caching.
*   **Prometheus**: Monitoring and alerting system.

### Installation and Usage

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/GabrielDemetriosLafis/multi-language-sentiment-engine.git
    cd multi-language-sentiment-engine
    ```

2.  **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the API**:
    ```bash
    uvicorn src.api.rest_api:app --host 0.0.0.0 --port 8000 --reload
    ```

### API Documentation

Access the interactive documentation (Swagger UI) at `http://localhost:8000/docs`.

---

**Author**: Gabriel Demetrios Lafis

