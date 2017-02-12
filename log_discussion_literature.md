## Explain:
Deep learning:
- Learn mappings (NN) and intermediate representations (layers depth)
- Theorems from deep learning, approximating functions

Data strucrute
- ATC codes, hierarchical structure
- Targets can be non-unique, hand anotated and mapped to a unique set defined by those annotations
    

## Baseline
- Train a linear classifier, SVM / log-reg (TBC)

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
- char2char, seq2seq
    
    - [Sequence to Sequence Learning with Neural Networks]
    (https://arxiv.org/abs/1409.3215)
    
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
- NN overfitting
    - L2 regularization
    - early stopping
    - drop-out
    
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
        
- Target replication in squence data context

#### formatting examples
- [x] Finish my changes
- [ ] Push my commits to GitHub
- [ ] Open a pull request


    - [name]
    (url)
    
        TBC
    
