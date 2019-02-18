# SmartParking

**_Projeto de um Sistema de Gestão de estacionamentos inteligente._**

Passos para conclusão:

- [x] Detecção da placa na imagem

- [x] Indentificação dos Caracteres

- [ ] Reconhecimento dos Caracteres

- [ ] Obtenção da imagem via camera IP

- [ ] Estrutura do banco de dados

- [ ] Desenvolvimento Banco de dados

- [ ] Código de reconhecimento com requisição no banco de dados

- [ ] Template do WebServer



# Ambiente de desenvolvimento

Desenvolvimento na instalação mínima do SO **Ubuntu 18.04 LTS Bionic Beaver** utilizando **virtualenvwrapper 4.8.2** como ferramenta para manipulação de ambientes virtuais do **virtualenv**

Links:
* Ubuntu 18.04 LTS - Download: http://releases.ubuntu.com/18.04/  (Desktop Image)
* VirtualEnvWrapper - Documentação: https://virtualenvwrapper.readthedocs.io/en/latest/
* VirtualEnv - Documentação: https://virtualenv.pypa.io/en/latest/

Algumas opções de ferramentas para desenvolver o código são o **PyCharm** e o **Visual Studio Code**

Links:
* PyCharm - Download: https://www.jetbrains.com/pycharm/download/
* Visual Studio Code - Download: https://code.visualstudio.com/download

# Relatório de atualização

**11/01/2019** :arrow_right:
Código de detecção de placas funcionando corretamente, é necessário definir como será posicionada a câmera que irá capturar a imagem ou melhorar a detecção para evitar alguns erros

Os mínimo de módulos necessários são:
```
> imutils==0.5.2

> numpy==1.15.4

> Pillow==5.3.0

> pytesseract==0.2.5

> OpenCV == 3.4.1

> scikit-image==0.14.1

> scikit-learn==0.20.2
```
Para vizualizar os pacotes instalados no ambiente virtual, basta fazer:
```
pip freeze
```

**16/02/2019**  :arrow_right:
Foi feito um downgrade na biblioteca scikit-learn para a versão 0.17. A versão mais atual (0.20.2) tem problemas para abrir os treinamentos e falhas na convergência dos cálculos.

Para fazer downgrade foram utilizados os comandos:
```
$ pip uninstall scikit-learn
$ pip install scikit-learn==0.17
``` 
Depois de instalada, para confirmar a instalação
```
$ python
```

```Python
>>> import sklearn
>>> sklearn.__version__
0.17
```