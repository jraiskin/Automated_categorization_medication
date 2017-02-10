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

#### text sequence -> class (hierarchical structure?)

- RNN classification
- Bidirectional RNN classification
- char2char, seq2seq
- NN classification of hierarchical structures
    - [A survey of hierarchical classification across different application domains] 
    (http://link.springer.com/article/10.1007/s10618-010-0175-9)
    
        “IS-A” relationship, tree and DAG structure, (non) mandatory leaf node prediction and the blocking problem, 
        big-bang (or global) classifiers, flat classification approach, local classifier per node (LCN), 
        local classifier per parent node (LCPN) and local classifier per level (LCL)
    - [Large Margin Methods for Structured and Interdependent Output Variables]
    (http://www.jmlr.org/papers/volume6/tsochantaridis05a/tsochantaridis05a.pdf)
        
        
    - [Large Margin Methods for Label Sequence Learning]
    (http://www.eecs.yorku.ca/course_archive/2005-06/F/6002B/Readings/EuroSpeech2003.pdf)
        
        Risk functions (Hamming risk), Conditional random fields (CRFs)

- NN robust classification methods wrt misspelling
- NN overfitting
    - L2 regularization
    - early stopping
    - drop-out
    - auxilary labels
    - target replication
- Target replication in squence data context

#### formatting examples
- [x] Finish my changes
- [ ] Push my commits to GitHub
- [ ] Open a pull request
