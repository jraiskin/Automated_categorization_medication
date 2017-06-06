
## Document outline
- Cover  

- Abstract  
    presenting the research problem and its importance, the main results, conclusion and how the thesis advances the field  

- Acknowledgements, table of content  

- Introduction  
    - The research area  
    - Previous work in this area  
    - Research problem and why it's worthwhile studying  
    - Thesis objective: (how) advance knowledge in the field  
    - Personal motivation: why this topic, why it is important  
    - Research method (brief)  
    - Structure: paragraph about each chapter, contribution of the chapter  
    - Expectations of the reader's knowledge and familiarity with the topics (is it ok to "hide" all opt related here?)  

- Preliminaries: NN and RNN  
    - NN background and history, often reaches state-of-the-art performance in various tasks (vision, speech, text)  
    - RNN and its different types (SRN, GRU, LSTM)  
    - Data feed direction  

- Related work  
    - Information retrieval and classification in medical data domain  

- Method  
    - Similarity-based label-suggestion procedure
        - Data scarsity, some ATC codes have no perscription variance (only one x,y pair)  
        - Relatively few labeled data with lots of ublabeled data  
        - Adopting labels of unlabeled text found similar to labeled observation, define similarity measure  
        - Tried enhancing the data with and without unsing external data sources (wiki.de, drugbank, compendium didnt work)  
        - Sampling procedure, exploring suggestions @ different similarities thresholds, getting feedback from Patrick  
        - Analyzing Patrick's feedback, setting a similarity cut-off, analyzing the enhanced data-set  
        - Classifiers trained on both original data and enhanced data  
    - Models  
        - RNN models
            - LSTM, GRU RNN architectures  
            - Possible hyper-parameters (learning rate, embeddings/one_hot, hidden state size)  
            - Data feed directions (forward and bidirectional)  
            - Regularization, importance of regularization in NN and RNNs (L2 norm, dropout, target replication)  
            - Activation functions and noisy activation  
        - Within model, choose by cost / between models, choose by MRR (reporting accuracy as well)  

- Experiments  
    - Data structure  
        - Review all given variables, focus on prescriptions and ATC code  
        - ATC code hierarchical structure  
        - Targets can be non-unique (i.e. a drug may have multiple valid ATC codes)  
        - ATC codes manually anotated to a unique set of ATC codes (i.e. 1 ATC code per prescription)  
        - Mention that we did not use Mimic data for training (GSN codes, text too clean w/o variation)  
    - Setup  
        - Comparing trained classifiers with and without similarity-based label-suggestion  
        - Data preperatoion and preprocessing (deifference for linear and RNN models, filtering vs "near-no-filtering")  
        - Descriptive stats (inculde visualizations, # of labels, observations etc)  
    - Evaluation:  
        - (Shortly) binary classification: accuracy, mean-reciprocal-rank, recall, precision, f-score  
        - Cost function requirements (average over observations, only depends on output values [not on activations])  
        - Cross-entropy in depth, necessary adjustments for regularization techniques  
        - Explain why negative log likelihood and perplexity don't make sense (not a language model)  
    - Results  
        - Baseline linear classifier (SVM)  
        - RNN results  
            - Compare between different architectures and hyper-parameter settings  
            - Evaluation metrics, training time (epochs and wall-time)  
        - Evaulation at varying string lengths  
    - Discussion  
        - Qunatitive and qualitively, where does our classifier underperforms, low-hanging fruits etc.  
        - Future research  

- Conclusion  

- Bibliography, Declaration of originality  

