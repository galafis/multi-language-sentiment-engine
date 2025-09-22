# Deployment Guide

## 📋 Visão Geral

Este diretório contém todos os arquivos necessários para implantar o **Multi-Language Sentiment Analysis Engine** em diferentes ambientes (local, cloud, Kubernetes). Os arquivos foram organizados para facilitar tanto deployments de desenvolvimento quanto produção.

## 📁 Estrutura dos Arquivos

```
deployment/
├── Dockerfile                 # Container de produção otimizado
├── docker-compose.yml         # Orquestração local dos serviços
├── k8s_deployment.yaml        # Manifesto Kubernetes para produção
└── README.md                  # Este arquivo
```

## 🐳 Docker

### Dockerfile

O `Dockerfile` é otimizado para produção e inclui:

- **Base Image**: Python 3.9 slim para melhor performance
- **Segurança**: Usuário não-root para execução
- **Otimizações**: Cache de layers, dependências pré-instaladas
- **Health Check**: Monitoramento automático da saúde do container
- **Gunicorn**: Servidor WSGI de produção com workers otimizados

#### Build da Imagem

```bash
# Construir a imagem do sentiment engine
docker build -t sentiment-engine:latest -f deployment/Dockerfile .

# Build com argumentos personalizados
docker build \
  --build-arg PYTHON_VERSION=3.9 \
  --build-arg APP_ENV=production \
  -t sentiment-engine:v1.0.0 \
  -f deployment/Dockerfile .
```

#### Executar Container Individual

```bash
# Executar o serviço principal
docker run -d \
  --name sentiment-api \
  -p 8000:8000 \
  -e KAFKA_BOOTSTRAP_SERVERS=localhost:9092 \
  -e REDIS_URL=redis://localhost:6379 \
  sentiment-engine:latest

# Com variáveis de ambiente customizadas
docker run -d \
  --name sentiment-api \
  -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  sentiment-engine:latest
```

## 🏗️ Docker Compose

### Serviços Disponíveis

O `docker-compose.yml` inclui toda a infraestrutura necessária:

- **sentiment-api**: API principal do engine
- **kafka**: Broker de mensagens Apache Kafka
- **zookeeper**: Coordenação para Kafka
- **redis**: Cache e armazenamento de sessões
- **elasticsearch**: Busca e analytics (opcional)
- **kibana**: Visualização de logs (opcional)
- **prometheus**: Métricas de monitoramento
- **grafana**: Dashboards de monitoramento

### Comandos Docker Compose

```bash
# Subir todos os serviços
docker-compose -f deployment/docker-compose.yml up -d

# Subir apenas serviços essenciais
docker-compose -f deployment/docker-compose.yml up -d sentiment-api kafka redis

# Ver logs dos serviços
docker-compose -f deployment/docker-compose.yml logs -f sentiment-api

# Parar e remover todos os containers
docker-compose -f deployment/docker-compose.yml down

# Rebuild e restart dos serviços
docker-compose -f deployment/docker-compose.yml up --build -d
```

### Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto:

```bash
# Configurações do Kafka
KAFKA_BOOTSTRAP_SERVERS=kafka:9092
KAFKA_AUTO_OFFSET_RESET=latest

# Configurações do Redis
REDIS_URL=redis://redis:6379/0

# Configurações da API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Configurações dos Modelos
MODEL_CACHE_DIR=/app/models
DEFAULT_MODEL=xlm-roberta-base

# Configurações de Log
LOG_LEVEL=INFO
LOG_FORMAT=json
```

## ☸️ Kubernetes

### k8s_deployment.yaml

O manifesto Kubernetes inclui:

- **Deployment**: Especificação da aplicação principal
- **Service**: Exposição da API para outros serviços
- **ConfigMap**: Configurações da aplicação
- **Persistent Volume**: Armazenamento para modelos e logs
- **Horizontal Pod Autoscaler**: Escalabilidade automática

### Deploy no Kubernetes

```bash
# Aplicar o manifesto
kubectl apply -f deployment/k8s_deployment.yaml

# Verificar o status do deployment
kubectl get deployments
kubectl get pods -l app=sentiment-engine

# Ver logs da aplicação
kubectl logs -l app=sentiment-engine --follow

# Port-forward para teste local
kubectl port-forward svc/sentiment-engine 8000:8000
```

### Configurações Específicas do K8s

```bash
# Criar namespace dedicado
kubectl create namespace sentiment-analysis
kubectl apply -f deployment/k8s_deployment.yaml -n sentiment-analysis

# Configurar recursos e limites
kubectl patch deployment sentiment-engine -p '{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "sentiment-engine",
          "resources": {
            "requests": {"cpu": "500m", "memory": "1Gi"},
            "limits": {"cpu": "2", "memory": "4Gi"}
          }
        }]
      }
    }
  }
}'
```

## 🌥️ Deploy em Cloud

### AWS ECS

```bash
# Build e push para ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com
docker tag sentiment-engine:latest 123456789012.dkr.ecr.us-east-1.amazonaws.com/sentiment-engine:latest
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/sentiment-engine:latest

# Deploy usando ECS CLI
ecs-cli compose --file deployment/docker-compose.yml service up --cluster sentiment-cluster
```

### Google Cloud Run

```bash
# Build e push para GCR
docker tag sentiment-engine:latest gcr.io/PROJECT_ID/sentiment-engine:latest
docker push gcr.io/PROJECT_ID/sentiment-engine:latest

# Deploy no Cloud Run
gcloud run deploy sentiment-engine \
  --image gcr.io/PROJECT_ID/sentiment-engine:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Azure Container Instances

```bash
# Push para ACR
az acr build --registry myregistry --image sentiment-engine:latest -f deployment/Dockerfile .

# Deploy no ACI
az container create \
  --resource-group myResourceGroup \
  --name sentiment-engine \
  --image myregistry.azurecr.io/sentiment-engine:latest \
  --cpu 2 --memory 4 \
  --ports 8000
```

## 🔧 Configurações de Produção

### Health Checks

```bash
# Testar health check da API
curl http://localhost:8000/health

# Response esperado:
{
  "status": "healthy",
  "timestamp": "2025-09-22T19:30:00Z",
  "version": "1.0.0",
  "services": {
    "kafka": "connected",
    "redis": "connected",
    "model_loading": "ready"
  }
}
```

### Monitoramento

```bash
# Métricas do Prometheus
http://localhost:9090/targets

# Dashboards do Grafana
http://localhost:3000
# Login: admin/admin

# Logs no Kibana
http://localhost:5601
```

### Backup e Persistência

```bash
# Backup dos modelos ML
docker run --rm \
  -v sentiment_models:/models \
  -v $(pwd)/backup:/backup \
  alpine tar czf /backup/models_$(date +%Y%m%d).tar.gz -C /models .

# Backup de dados do Redis
docker exec redis redis-cli --rdb /data/dump_$(date +%Y%m%d).rdb
```

## 🚨 Troubleshooting

### Problemas Comuns

1. **Container não inicia**:
   ```bash
   docker logs sentiment-api
   # Verificar variáveis de ambiente e dependências
   ```

2. **Erro de conexão com Kafka**:
   ```bash
   # Testar conectividade
   docker exec -it kafka kafka-topics.sh --bootstrap-server localhost:9092 --list
   ```

3. **Modelos não carregam**:
   ```bash
   # Verificar espaço em disco e permissões
   docker exec sentiment-api ls -la /app/models
   ```

4. **Performance baixa**:
   ```bash
   # Monitorar recursos
   docker stats sentiment-api
   # Ajustar workers do Gunicorn no Dockerfile
   ```

### Logs e Debugging

```bash
# Logs detalhados da aplicação
docker-compose logs -f --tail=100 sentiment-api

# Debug mode (não use em produção)
docker run -e LOG_LEVEL=DEBUG sentiment-engine:latest

# Executar shell no container
docker exec -it sentiment-api /bin/bash
```

## 📊 Performance e Scaling

### Otimizações Recomendadas

- **CPU**: Mínimo 2 cores para produção
- **RAM**: 4GB para carregar modelos transformer
- **Disco**: SSD para cache de modelos (>10GB)
- **Rede**: Baixa latência para Kafka e Redis

### Scaling Horizontal

```bash
# Docker Compose
docker-compose up --scale sentiment-api=3

# Kubernetes
kubectl scale deployment sentiment-engine --replicas=5

# Auto-scaling baseado em CPU
kubectl autoscale deployment sentiment-engine --cpu-percent=70 --min=2 --max=10
```

---

## 🤝 Contribuição

Para contribuir com melhorias nos arquivos de deployment:

1. Teste as mudanças em ambiente local
2. Valide com diferentes configurações
3. Atualize esta documentação
4. Submeta um pull request

## 📞 Suporte

Para problemas relacionados ao deployment:
- 🐛 Issues: [GitHub Issues](https://github.com/galafis/multi-language-sentiment-engine/issues)
- 📧 Email: gabriel@galafis.dev
- 📖 Docs: [Documentação Completa](../docs/)
