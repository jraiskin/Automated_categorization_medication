## Previous experiments (prior to 13/4)  
Learning rate of 10^-1 was established to under perform.  
Lower dimensional hidden states consistently under perform.  
No clear relation regarding:  
- Character embeddings of dimension 4 / one hot vector  
- Keep probability  


## Experiment: 13/4  
Experiemnt run on 4 different batches, (LSTM, GRU) X (feed-forward, bidirectional)  
- Learning rate: 10^-2 / 10^-3  
- Character embeddings (4 dimensions) vs one hot vector representation  
- Hidden state size: 64 / 128  
- L2 norm regularization constant: 10^-3 / 10^-4  
- Target replication regularization constant: 0.3 / 0.1  


- SIMPLE GRU (best test accuracy 93.5%, at spike ~96.5%)  
    - learning rate of 10^-3 is too slow for practical purposes (within the thesis time constraints), so the chosen learning rate is 10^-2  
    - it seems that the one hot vector representation out performs the character embeddings, by a small margin but consistently  
    - hidden_state_size: not clear between 64,128 (128 had a spike in accuracy towards the end)  
    - L2 wieghts regularization constant of 10^-4 seems to out performs by a small and consistent margin  
    - Target replication weight of 0.3 leads to faster convergence with slightly better performance (although the "spike" was in the 0.1 group)  

- Bidirectional GRU (best test accuracy 93.5%, at spike ~96.0%)  
    - SAME learning rate  
    - SAME one hot  
    - hidden_state_size: similar, 128 could be slightly better but not clear  
    - L2 wieghts regularization: 10^-3 slightly out performs, especially after a "dip" in one model accuracy over shuts, motivation for noisy activation  
    - SAME Target replication  

- SIMPLE LSTM (best test accuracy ~91.5%)  
    - SAME learning rate  
    - SAME one hot (one hot even better here)  
    - hidden_state_size: similar, 128 could be slightly better but not clear  
    - L2 wieghts regularization: not clear whether there is any significant effect  
    - SAME Target replication  

- Bidirectional LSTM (best test accuracy ~90.5%)  
- SAME learning rate  
- SAME one hot (one hot even better here)  
- hidden_state_size: similar, 128 could be slightly better but not clear  
- L2 wieghts regularization: 10^-3 slightly out performs  
- SAME Target replication  


