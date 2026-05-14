#TFE: Financial Anomalies Detection 📈🤖
This repository contains the source code and experimental pipeline for the Master's Thesis: "Intelligent System for Early Detection of Financial Anomalies using Unsupervised Learning".

Financial markets are highly dynamic environments characterized by volatility clustering and extreme events (fat tails), such as Flash Crashes. Traditional statistical risk models, which often assume normal data distributions, fail to accurately predict or identify these complex structural anomalies.

To address this challenge, this project develops an automated early-warning system utilizing Unsupervised Artificial Intelligence. By analyzing large volumes of historical financial time series (ingested via the yfinance API), the system autonomously learns the "normal" behavior of portfolios and flags significant risk spikes.

Following the CRISP-DM methodology, the technical architecture presents a rigorous comparative analysis across three stages:

1. Statistical Baseline: Traditional parametric methods (e.g., moving Z-Score) to establish a performance threshold.

2. Classical Machine Learning: Isolation Forest and Local Outlier Factor (LOF) for spatial isolation and local density-based anomaly detection.

3. Deep Learning: LSTM-Autoencoders designed to capture temporal dependencies and detect market crashes through reconstruction error thresholds.

The ultimate goal of this project is to provide a robust, explainable tool to mitigate financial risks and optimize decision-making.

Authors: Jose Manuel Ruiz-Mateos & Daniel Rueda
