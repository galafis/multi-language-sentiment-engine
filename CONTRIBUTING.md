# Diretrizes de Contribuição

Bem-vindo ao projeto Multi-Language Sentiment Engine! Agradecemos o seu interesse em contribuir.

Para garantir um processo de colaboração eficiente e produtivo, por favor, siga estas diretrizes:

## Como Contribuir

1.  **Fork o Repositório**: Comece fazendo um fork do repositório para a sua conta GitHub.
2.  **Clone o Repositório**: Clone o seu fork para a sua máquina local:
    ```bash
    git clone https://github.com/SEU_USUARIO/multi-language-sentiment-engine.git
    cd multi-language-sentiment-engine
    ```
3.  **Crie uma Branch**: Crie uma nova branch para a sua feature ou correção de bug:
    ```bash
    git checkout -b feature/sua-feature-name
    # ou
    git checkout -b bugfix/sua-bugfix-name
    ```
4.  **Faça Suas Alterações**: Implemente suas alterações, adicione novos recursos ou corrija bugs.
5.  **Escreva Testes**: Certifique-se de que suas alterações são cobertas por testes unitários e de integração. Adicione novos testes se necessário.
6.  **Execute os Testes**: Antes de submeter, execute todos os testes para garantir que nada foi quebrado:
    ```bash
    pytest
    ```
7.  **Formate o Código**: Use um formatador de código como `black` ou `flake8` para manter a consistência do estilo de código.
8.  **Commit Suas Alterações**: Escreva mensagens de commit claras e concisas:
    ```bash
    git commit -m "feat: Adiciona nova funcionalidade X" # para novas funcionalidades
    # ou
    git commit -m "fix: Corrige bug Y" # para correção de bugs
    ```
9.  **Envie para o GitHub**: Envie suas alterações para o seu fork no GitHub:
    ```bash
    git push origin feature/sua-feature-name
    ```
10. **Abra um Pull Request (PR)**: Abra um Pull Request do seu fork para a branch `main` do repositório original. Descreva suas alterações em detalhes e referencie quaisquer issues relevantes.

## Padrões de Código

*   Siga o estilo de código PEP 8 para Python.
*   Use docstrings para documentar funções, classes e módulos.
*   Mantenha as funções pequenas e focadas em uma única responsabilidade.

## Relato de Bugs

Se você encontrar um bug, por favor, abra uma issue no GitHub e forneça o máximo de detalhes possível, incluindo:

*   Passos para reproduzir o bug.
*   Comportamento esperado.
*   Comportamento atual.
*   Sua configuração de ambiente (OS, versão do Python, dependências).

## Sugestões de Recursos

Se você tiver uma ideia para um novo recurso, sinta-se à vontade para abrir uma issue para discutir a proposta antes de começar a trabalhar nela.

Obrigado por sua contribuição!
