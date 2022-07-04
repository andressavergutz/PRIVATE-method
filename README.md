**IoTReGuard Method [old PRIVATE Method]**
============================

FeatureExtraction requer Python v3 e Tshark instalados, e deve ser testado em um ambiente Linux (pode ser uma VM).

1. Instalação das dependências (no terminal execute):
  
  Execute: ./downloadLibraries.sh

Obs.: Pressione Yes para as perguntas de confirmação de instalação dos pacotes.

2. Cole os arquivos .pcap na pasta 'pcaps'
  Nesse link você encontra os pcaps por dia. São de dispositivos IoT. Faça o download de todos.
  https://iotanalytics.unsw.edu.au/iottraces

Obs.: a ferramenta já vem com dois pcaps na pasta pcap (verifique), caso não tiver, faça o download de algum pcap no link acima.

3. Verifique os caminhos no arquivo P2P_CONSTANTS.py 
  
  P2P_CONSTANTS.PY = CONSTANTES

Obs.: Siga a mesma ordem de diretórios do repositório original.

4. Execute: python3 FilterPackets.py
  
  FilterPackets = RECEBE COMO ENTRADA OS ARQUIVOS PCAPS E EXECUTA O TSHARK PARA EXTRAIR OS PACOTES E 
  FLUXOS, BEM COMO AS PRINCIPAIS CARACT. DO TRÁFEGO DA REDE
    
    ENTRADA: *.pcap da pasta './pcaps'
    SAÍDA: *.csv com as características extraídas para cada dia e salva nas pastas './flowdata' e './packetdata'
    
    ( *.pcap/pcap > *.csv/packetdata ) 
  
5. Execute: python3 ProcessCsvFile.py
  
  ProcessCsvFile = EXTRAI AS CARACTERISTICAS ESTATÍSTICAS DO TRÁFEGO DA REDE (EX. NÚMERO DE PACOTES, MÉDIA DO NUM DE BYTES).
  EXTRAI POR NÍVEL DE FLUXO E PACOTE. 
    
    ENTRADA: *.csv da pasta './packetdata'
    SAÍDA: 1) *.csv com as características estatísticas para cada dia, salvos nas pastas './flowplots' e './packetplots'
           1) também salva dois arquivos .csvs (all_flow_samples.csv AND all_packet_samples.csv) na pasta './all_final_samples'   (concatenação de todas as características de todos os dias em um arquivo -- útil para plots e algoritmos ML)
   
  ( *.csv/packetdata > '*.csv/flowplots' AND '*.csv/packetplots' AND '*.csv/all_final_samples')  

6. Execute: python3 Classification_ML_Holdout.py
  
  Classification_ML_Holdout.py = PARA A CLASSIFICAÇÃO DOS DISPOSITIVOS IOT POR MEIO DE ALGORITMOS DE APRENDIZAGEM DE MÁQUINA SEGUINTE TRAIN TEST SPLIT
  
  Obs.: Altere o algoritmo utilizado (linhas 46 a 57 > deixe apenas descomentado o algoritmo que irá usar) e número de execuções (linha 93 > range(2))

    ENTRADA: '/all_final_samples/packet OR flow_samples.csv'
    SAÍDA: '/graphics/packet-level-MLresults.csv' OR '/graphics/flow-level-MLresults.csv'


6. Execute: python3 Classification_ML_kfold.py
  
  Classification_ML_Holdout.py = PARA A CLASSIFICAÇÃO DOS DISPOSITIVOS IOT POR MEIO DE ALGORITMOS DE APRENDIZAGEM DE MÁQUINA SEGUINDO KFOLD
  
  Obs.: Altere o algoritmo utilizado (linhas 46 a 57 > deixe apenas descomentado o algoritmo que irá usar) e número de execuções (linha 93 > range(2))

    ENTRADA: '/all_final_samples/packet OR flow_samples.csv'
    SAÍDA: '/graphics/packet-level-MLresults.csv' OR '/graphics/flow-level-MLresults.csv'


####GRÁFICOS:

7. Execute: python3 PlotGraphFeatures.py
  
  PlotGraphFeatures.py = PARA GERAÇÃO DE GRÁFICOS DE COMPORTAMENTO DO TRÁFEGO DA REDE. 
   
    ENTRADA: '*.csv/flowplots' OR '*.csv/packetplots'
    SAÍDA: os gráficos são gerados nas pastas './graphics/packet-level/' OR './graphics/flow-level/'

8. Execute: python3 PlotGraphML.py
  
  PlotGraphML.py = PARA GERAÇÃO DE GRÁFICOS DOS RESULTADOS DOS ALGORITMOS DE ML.
   
    ENTRADA: '*.csv/flowplots' OR '*.csv/packetplots'
    SAÍDA: os gráficos são gerados nas pastas './graphics/packet-level/' OR './graphics/flow-level/'




##############################################################################


