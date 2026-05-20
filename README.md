# clinical-risk-weighted-pooling

Este projeto demonstra uma técnica intermediária entre **embeddings simples** e **mecanismos completos de atenção (Transformers)**.

O objetivo é entender como um modelo pode **aprender automaticamente a importância de cada palavra em uma frase**, sem usar estruturas complexas como self-attention.

## Problema

Dado um texto:

```
"paciente com dor intensa"
```

Após embedding, temos:

$$
X = [x_1, x_2, x_3, x_4]
$$

onde:

$$
x_i \in \mathbb{R}^d
$$

Para tarefas como classificação, precisamos converter isso em um vetor fixo.

## Abordagens Tradicionais

### 1. Mean Pooling

$$
v = \frac{1}{n} \sum_{i=1}^{n} x_i
$$

**Problema**
- Todas as palavras têm o mesmo peso
- Ignora relevância semântica

### 2. Max Pooling

Seleciona o maior valor por dimensão.

**Problema**
- Perde estrutura global
- Sensível a ruído

## Solução: Weighted Pooling

### Ideia central

Aprender um peso para cada palavra:

$$
v = \sum_{i=1}^{n} \alpha_i x_i
$$

Onde:

$$
\alpha_i = \text{softmax}(\text{score}(x_i))
$$

### Intuição

O modelo aprende coisas como:
- `"crítico"` → peso alto
- `"leve"` → peso baixo
- `"com"` → quase irrelevante

### Arquitetura do Modelo

```
Input (tokens)
        ↓
Embedding Layer
        ↓
Weighted Pooling
        ↓
Camada Linear
        ↓
Classificação
```

## O Componente-chave: Weighted Pooling

### Definição

```python
class WeightedPooling(nn.Module):
```

### 1. Score de cada token

```python
scores = self.scorer(embeddings)
```

$$
\text{score}(x_i) = W x_i + b
$$

### 2. Normalização com Softmax

```python
weights = torch.softmax(scores, dim=1)
```

$$
\alpha_i =
\frac{
e^{\text{score}(x_i)}
}{
\sum_{j=1}^{n} e^{\text{score}(x_j)}
}
$$

### 3. Soma ponderada

```python
pooled = torch.sum(embeddings * weights, dim=1)
```

$$
v = \sum_{i=1}^{n} \alpha_i x_i
$$

## Tratamento de Padding

Padding é mascarado:

```python
scores = scores.masked_fill(mask == 0, -1e9)
```

Isso garante que:

- tokens `<pad>` não influenciem o resultado
- o softmax ignore posições inválidas

## Interpretabilidade

O modelo retorna:

```python
return pooled, weights
```

Você pode inspecionar os pesos aprendidos:

```
Frase: sintomas graves e dor intensa

sintomas -> 0.15
graves   -> 0.28
e        -> 0.02
dor      -> 0.20
intensa  -> 0.35
```

O modelo aprendeu a priorizar termos relevantes.

## Dataset do Projeto

Domínio: **classificação de risco clínico**

| Texto                 | Label |
| --------------------- | ----- |
| paciente com dor leve | 0     |
| sintomas graves       | 1     |

## Treinamento

```python
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters())
```

Treinamento:

```python
loss.backward()
optimizer.step()
```

## O que está sendo aprendido?

O modelo aprende simultaneamente:

1. Representação das palavras (embeddings)
2. Importância contextual de cada token
3. Regra de decisão final

## Trade-offs

### Vantagens
- Simples
- Interpretável
- Baixo custo computacional
- Melhor que mean/max pooling

### Limitações
- Não captura interação entre palavras
  - Exemplo: `"não grave"` vs `"grave"`
- Não modela dependências longas
- Não usa contexto global

## Conexão com Attention

| Modelo           | Interação entre tokens | Query/Key |
| ---------------- | ---------------------- | --------- |
| Weighted Pooling | ❌                      | ❌         |
| Self-Attention   | ✅                      | ✅         |

## Insight-chave

Weighted Pooling é equivalente a:

> "attention sem contexto global"

Ou seja:

$$
\text{score}(x_i) \neq \text{score}(x_i, x_j)
$$

O peso depende apenas do token atual.

## Extensões

### 1. Melhorar o scorer

Trocar:

```python
nn.Linear(d, 1)
```

por:

```python
nn.Sequential(
    nn.Linear(d, d),
    nn.ReLU(),
    nn.Linear(d, 1)
)
```

### 2. Vetor de atenção global

$$
\text{score}(x_i) = v^T x_i
$$

### 3. Multi-head Pooling

Criar múltiplos "olhares" sobre a frase.

## Conclusão

Weighted Pooling é um **degrau conceitual essencial**:

```text
Mean Pooling
    ↓
Weighted Pooling
    ↓
Self-Attention
    ↓
Transformer
```

Ele introduz:

- aprendizagem de relevância
- normalização via softmax
- explicabilidade

sem a complexidade total dos Transformers.

## Próximo Passo

Implementar:

**Self-Attention completa**

Com:

$$
\text{score}(q, k) = q^T k
$$

## Perguntas para reflexão

1. Por que o modelo funciona mesmo sem considerar relações entre palavras?
2. Em que casos isso quebra?
3. Como isso se comporta em textos longos?
4. Isso escala para linguagem natural real?

## Observação importante

Este projeto usa:

- PyTorch
- Tokenização simplificada

Em produção, usaríamos:

- Tokenizers (BPE, WordPiece)
- Embeddings pré-treinados
- Transformers

## Boas práticas

- Separar dados, modelo e treinamento
- Versionar experimentos
- Analisar os pesos aprendidos

## TL;DR

Weighted Pooling é uma:

> média ponderada aprendida via softmax

É a forma mais simples de "atenção" funcional.
