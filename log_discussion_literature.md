## Explain

Data strucrute
- ATC codes, hierarchical structure
- Targets can be non-unique, hand anotated and mapped to a unique set defined by those annotations.
    This "non-uniqueness" is solved by mapping, disambiguating labels, according to hand-annotated labels (by physicians)
    
## Rough outline
- Cover, abstract, acknowledgements, table of content
- Introduction
    present the problem, current available solutions, motivation to solve
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
        short discussion on models considered but not used (char2char, seq2seq, word embeddings such as GloVe and word2vec)  
        discussion on importance or regulatization in general, especially in NN (# of params etc)  
        discussion on SGD, batch grad descent, optimization, local minima  
    
    - Hierarchical classification:  
        TBC, according to papers collected  
    
    - Assessment metrics:  
        binary classification: accuracy, precision, f-score  
        Cost function requirements, 
        loss function for classification 
        (Quadratic, Cross-entropy, Exponentional, Hellinger distance, Kullback–Leibler divergence, Itakura–Saito distance)  
        cross entropy adjustments for target replication  
    
- Experiments
    - Data  
        descriptive (inculde visualizations, # of labels, observations etc)  
        structure, preperation and preprocessing (disambiguation of possible non-unique labels)  
        
    - Task evaluation
        
    - Baseline  
    
    - Approach
        
    - Results
        include evaluation metrics, training time (epochs and wall-time)  
        compare between different structures, hyper-params and baseline
        
    - Discussion
    
    - Bibliography, Declaration of originality
    
- Conclusion
    
## Baseline
- Train a linear classifier, SVM / log-reg / plain-vanilla LSTM (TBC)

## Literature review - general topics
- RNN classification
    
    - [Finding Structure in Time]
    (http://onlinelibrary.wiley.com/doi/10.1207/s15516709cog1402_1/abstract)
    
        Introduced RNN (simple RNN, later referred to as SRN)
    
    - [Long Short-Term Memory]
    (http://dl.acm.org/citation.cfm?id=1246450)
    
        Introduced LSTM
    
    - [Learning to Diagnose with LSTM Recurrent Neural Networks]
    (https://arxiv.org/abs/1511.03677)
    
        TBC
    
    - [Sequential Short-Text Classification with Recurrent and Convolutional Neural Networks]
    (https://arxiv.org/abs/1603.03827)
    
        TBC
    
    - [Character-Aware Neural Language Models]
    (https://arxiv.org/abs/1508.06615)
    
        Combining CNN and LSTM for charachter-level classification
    
    - [Recurrent Neural Networks for Robust Real-World Text Classification]
    (http://dl.acm.org/citation.cfm?id=1331869)
    
        xRNN model, 2 hidden layer, controlled by a hysteresis function
    
    - [A C-LSTM Neural Network for Text Classification]
    (https://arxiv.org/abs/1511.08630)
    
        Propose C-LSTM model, utilizes components from CNN and LSTM to extract features and represent latent vectors.
        Gives a descent overview of related work, N-gram feature extraction through convolution, LSTM,
        padding and word vector initialization.
        
    - [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation]
    (https://arxiv.org/abs/1406.1078)
    
        TBC
    
    - [Recurrent Neural Network for Text Classification with Multi-Task Learning]
    (https://arxiv.org/abs/1605.05101)
    
        TBC
    
- Bidirectional RNN classification
    
    - [Neural Machine Translation by Jointly Learning to Align and Translate]
    (https://arxiv.org/abs/1409.0473)
    
        RNN encoder-decoder, bi-directional RNN encoder
    
- char2char, seq2seq
    
    - [Sequence to Sequence Learning with Neural Networks]
    (https://arxiv.org/abs/1409.3215)
    
        TBC
        
- Generative Adversarial Network (GAN)
    
    - [Generative Adversarial Networks]
    (https://arxiv.org/abs/1406.2661)
    
        Introducing GAN
    
    - [Improved Techniques for Training GANs]
    (https://arxiv.org/abs/1606.03498)
    
        TBC
    
    - [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks]
    (https://arxiv.org/abs/1511.06434)
    
        TBC
    
- Hierarchical structures classification (no mention of NN)
    
    - [A survey of hierarchical classification across different application domains] 
    (http://link.springer.com/article/10.1007/s10618-010-0175-9)
    
        “IS-A” relationship, tree and DAG structure, (non) mandatory leaf node prediction and the blocking problem, 
        big-bang (or global) classifiers, flat classification approach, local classifier per node (LCN), 
        local classifier per parent node (LCPN) and local classifier per level (LCL)
    - [Large Margin Methods for Structured and Interdependent Output Variables]
    (http://www.jmlr.org/papers/volume6/tsochantaridis05a/tsochantaridis05a.pdf)
        
        (not sure if relevant)
        
    - [Large Margin Methods for Label Sequence Learning]
    (http://www.eecs.yorku.ca/course_archive/2005-06/F/6002B/Readings/EuroSpeech2003.pdf)
        
        Risk functions (Hamming risk), Conditional random fields (CRFs)
    
    - [Development of a Hierarchical Classification System with 
    Artificial Neural Networks and FT-IR Spectra for the Identification of Bacteria]
    (http://journals.sagepub.com/doi/abs/10.1366/0003702001948619)
    
        Hierarchical classification, different classifer for each level and for each node
    
    - [Hierarchical text classification and evaluation] 
    (http://ieeexplore.ieee.org/abstract/document/989560/?reload=true)
    
        TBC
    
    - [Large margin hierarchical classification]
    (http://dl.acm.org/citation.cfm?id=1015374)
    
        TBC
    
- NN robust classification methods wrt misspelling

- NN introduction, derivation, optimization, overfitting etc
    
    - L2 regularization
    
    - early stopping
        
        - [On Early Stopping in Gradient Descent Learning]
        (http://link.springer.com/article/10.1007/s00365-006-0663-2)
        
            TBC
        
        - [Boosting with early stopping: Convergence and consistency]
        (https://arxiv.org/abs/math/0508276v1)
        
            TBC
        
    - drop-out
    
        - [Improving neural networks by preventing co-adaptation of feature detectors]
        (https://arxiv.org/abs/1207.0580)
        
            Introducing drop-out
        
        - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting]
        (http://jmlr.org/papers/v15/srivastava14a.html)
        
            Show that dropout improves the performance of neural networks on supervised learning tasks in vision,
            speech recognition, document classification and computational biology.
        
        - [Semi-supervised Sequence Learning]
        (https://arxiv.org/abs/1511.01432)
        
            TBC
        
        - [Recurrent Neural Network Regularization]
        (https://arxiv.org/abs/1409.2329)
        
            TBC
        
        - [Dropout improves Recurrent Neural Networks for Handwriting Recognition]
        (https://arxiv.org/abs/1312.4569)
        
            TBC
        
    - auxilary labels
    
        - [Using the Future to "Sort Out" the Present: Rankprop and Multitask Learning for Medical Risk Evaluation]
        (https://papers.nips.cc/paper/1081-using-the-future-to-sort-out-the-present-rankprop-and-multitask-learning-for-medical-risk-evaluation)
        
            Learning from the future with multitask learning
        
        - [Learning Many Related Tasks at the Same Time With Backpropagation]
        (http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.54.6346)
        
            Multitask backprop
        
    - target replication
        
        - [Deeply-Supervised Nets]
        (https://arxiv.org/abs/1409.5185)
        
            Companion objective (different to layer-wise pre-training), 
            introducing a classifier (e.g. SVM) to each layer
        
        - [Beyond Short Snippets: Deep Networks for Video Classification]
        (https://arxiv.org/abs/1503.08909)
        
            Target replication after each video frame
        
        - [Semi-supervised Sequence Learning]
        (https://arxiv.org/abs/1511.01432)
        
            TBC
        
    - optimization
        
        - [Adam: A Method for Stochastic Optimization]
        (https://arxiv.org/abs/1412.6980v8)
        
            Adam, an algorithm for first-order gradient-based optimization of stochastic objective functions,
            based on adaptive estimates of lower-order moments.
        

#### formatting examples
- [x] Finish my changes
- [ ] Push my commits to GitHub
- [ ] Open a pull request


    - [name]
    (url)
    
        TBC
    
