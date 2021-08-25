FeatureExtraction
============================

###ESTADO ATUAL BY ANDRESSA

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
    
    ENTRADA: todos*.pcap da pasta './pcaps'
    SAÍDA: todos*.csv com as características extraídas para cada dia e salva nas pastas './flowdata' e './packetdata'
    
    ( todos*.pcap/pcap > todos*.csv/flowdata AND todos*.csv/packetdata ) 
  
5. Execute: python3 ProcessCsvFile.py
  
  ProcessCsvFile = EXTRAI AS CARACTERISTICAS ESTATÍSTICAS DO TRÁFEGO DA REDE (EX. NÚMERO DE PACOTES, MÉDIA DO NUM DE BYTES).
  EXTRAI POR NÍVEL DE FLUXO E PACOTE. 
    
    ENTRADA: todos*.csv das pastas './flowdata' e './packetdata'
    SAÍDA: 1) todos*.csv com as características estatísticas para cada dia e salva nas pastas './flowplots' e './packetplots'
           2) também salva dois arquivos .csvs (all_flow_samples.csv AND all_packet_samples.csv) na pasta './all_final_samples'   (concatenação de todas as características de todos os dias em um arquivo -- útil para plots e algoritmos ML)
   
  ( todos*.csv/flowdata AND todos*.csv/packetdata > 'todos*.csv/flowplots' AND 'todos*.csv/packetplots' AND 'todos*.csv/all_final_samples')  

6. Execute: python3 classific_ML.py
  
  classific_ML.py = PARA A CLASSIFICAÇÃO DOS DISPOSITIVOS IOT POR MEIO DE ALGORITMOS DE APRENDIZAGEM DE MÁQUINA
  
  Obs.: Altere o algoritmo utilizado e número de execuções (linha 93 > range(2))

    ENTRADA: packet-OR-flow_samples.csv/all_final_samples
    SAÍDA: 'packet-level-MLresults.csv/graphics' OR 'flow-level-MLresults.csv/graphics'


####GRÁFICOS:

7. Execute: python3 PlotGraphFeatures.py
  
  PlotGraphFeatures.py = PARA GERAÇÃO DE GRÁFICOS DE COMPORTAMENTO DO TRÁFEGO DA REDE. 
    ENTRADA: 'todos*.csv/flowplots' OR 'todos*.csv/packetplots'
    SAÍDA: os gráficos são gerados nas pastas './graphics/packet-level/' OR './graphics/flow-level/'

8. Execute: python3 PlotGraphML.py
  
  PlotGraphML.py = PARA GERAÇÃO DE GRÁFICOS DOS RESULTADOS DOS ALGORITMOS DE ML.
    ENTRADA: 'todos*.csv/flowplots' OR 'todos*.csv/packetplots'
    SAÍDA: os gráficos são gerados nas pastas './graphics/packet-level/' OR './graphics/flow-level/'




##############################################################################


