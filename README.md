# Regressao-Logistica-Classificador
### Descrição
Treinando um modelo de classificação utilizando a técnica de regressão logistica, aplicando em uma base de dados para classificar registros e compartilhar os resultados(Censo de 1994 - EUA).

O objetivo é prever se uma pessoa possui renda anual <= ou > 50 mil dólares por ano.

**Percentual mínimo a ser batido** -> Base Line Classifier = 0.7559 (ZeroR).

### Resultados - Validação Cruzada - StratifiedKFold
**Precisão** | **Pré-Processamentos** | **Desvio Padrão**
| :------: | :------: | :------: |
0.7914 | LabelEncoder | 0.0117
0.7974 | OneHotEncoder | 0.0080
0.8247 | LabelEncoder + StandardScaler | 0.0056
**0.8517** | **OneHotEncoder + StandardScaler** | **0.0070**
0.8517 | LabelEnconder + OneHotEncoder + StandardScaler | 0.0070 

### Matriz de Confusão(Média de todas as 10 execuções):
### Matriz de Confusão (Média):
**x** | 0 | 1
| :------: | :------: | :------: |
0 | **2353.5** | 118.5
1 | 560.4 | **223.7**

A Matriz na tabela acima é formada pela média de todas as matrizes geradas ao longo de 10 execuções usando pré-processamentos OneHotEncoder + StandardScaler.

A diagonal principal (em negrito) destaca os registros classificados corretamente.

### Bibliotecas usadas:
- Pandas
- Sklearn
- Numpy

### Ferramentas Usadas:
- Anaconda
- Spyder

### Fonte da Base de Dados: 
- Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

### Como usar:
1. Faça o download do classificador já treinado dispoível neste mesmo repositório [aqui](https://github.com/juliomrodrigues/Regressao-Logistica-Classificador/blob/main/classificador_regressao_logistica.sav).
2. Abra o arquivo.py que deseja usar o classificador ou então crie um novo.
3. Execute o código abaixo para fazer a importação:
~~~~python
import pickle
classificador = pickle.load(open('classificador_regressao_logistica.sav', 'rb'))

~~~~~
4. Pronto, agora o classficador está pronto para ser usado.

#### Outros Classificadores:
- [Naive Bayes](https://github.com/juliomrodrigues/Classificador-Naive-Bayes)
- [Árvore de Decisão](https://github.com/juliomrodrigues/Arvore-de-Decisao)
- [Random Forest](https://github.com/juliomrodrigues/Random-Forest-Classificador)
- [Regras](https://github.com/juliomrodrigues/Classificador-Regras)
- [KNN](https://github.com/juliomrodrigues/Classificador-KNN)
- [SVM](https://github.com/juliomrodrigues/Classificador-SVM)
- [Rede Neural](https://github.com/juliomrodrigues/Classificador-Rede-Neural)
