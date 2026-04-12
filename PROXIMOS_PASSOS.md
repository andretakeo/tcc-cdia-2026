# Próximos

- ~~Criar base igualitária para evitar viés~~ **FEITO**
  - Implementado class weights em todos os 7 modelos do Stage 7:
    - PyTorch (BiLSTM, Transformer, TCN): peso por amostra na BCELoss
    - XGBoost: scale_pos_weight
    - Logistic Regression / Random Forest: class_weight="balanced"
