## Explain

Data strucrute
- ATC codes, hierarchical structure
- Targets can be non-unique, hand anotated and mapped to a unique set defined by those annotations.
    This "non-uniqueness" is solved by mapping, disambiguating labels, according to hand-annotated labels (by physicians)
    
## Rough outline
- Cover  

- Abstract  
    presenting the research problem, the main results, conclusion and how the thesis advances the field  
    
- Acknowledgements, table of content  

- Introduction  
    - The research area  
    - Previous work in this area  
    - Research problem and why it's worthwhile studying  
    - Thesis objective: (how) advance knowledge in the field  
    - Personal motivation: why this topic, why it is important  
    - Research method (brief)  
    - Structure: paragraph about each chapter, contribution of the chapter  
    - Expectations of the reader's knowledge and familiarity with the topics  
    
- Related work  
    - NN:  
        neural networks background and history, function approximators that learn intermediate representations (theorems?)  
        NN increased success in various tasks over the years (vision, speech, text)  
        background on network architectures: 
        "neural unit" (piecemeal linear with non linear "activations"), popular activation functions,  
        fully conected 1 layer NN  
        introducing depth  
        convolutional NN ?  
        GAN  
        RNN, (bi-directional?)  
        LSTM (including previous iterations)  
        short discussion on models considered but not used (char2char, seq2seq, word embeddings e.g. GloVe and word2vec)  
        discussion on importance or regulatization in general, especially in NN (# of params etc)  
        discussion on gradient-based learning, SGD, batch grad descent, optimization, local minima  
    
    - Hierarchical classification:  
        TBC, according to papers collected  
    
    - Assessment metrics:  
        SHORT: binary classification: accuracy, precision, f-score  
        Cost function requirements, 
        loss function for classification  
        SHORT: Quadratic, Exponentional, Hellinger distance, Kullback–Leibler divergence, Itakura–Saito distance  
        Explain why negative log likelihood and perplexity don't make sense (not a language model)  
        Cross-entropy in depth, adjustments for target replication  
    
- Method  
    How we plan to plan to approach the problem, what possible solutions we want to try.  
    Possible solutions may include (not final):  
    - Explore LSTM depth (literature found up to 3 LSTM "layers")  
    - Experinemt (each?) LSTM with different feeding methods: forward, backward, bi-directional  
    - Experinemt with drop-out, target replication  
    - GAN (? not sure) (need to choose network architecture for G and A networks, w&w/o drop-out)  
    - Encoder-decoder (classification as translation)  
    - For each (???) experiment on "flat" vs. hierarchical class structure  
    
- Experiments
    - Data  
        descriptive (inculde visualizations, # of labels, observations etc)  
        structure, preperation and preprocessing (disambiguation of possible non-unique labels)  
        
    - Task evaluation
        
    - Baseline  
        TBC (linear classifier, SVM / log-reg / plain-vanilla LSTM)  
        
    - Results  
        include evaluation metrics, training time (epochs and wall-time)  
        compare between different structures, hyper-params and baseline
        
    - Discussion  
    
- Conclusion  

- Bibliography, Declaration of originality  
