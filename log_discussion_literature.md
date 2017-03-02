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
    
## Baseline
- Train a baseline before trying out more fancy algos.  

## Literature review - general topics
- RNN classification
    
    - [ ] [Finding Structure in Time]
    (http://onlinelibrary.wiley.com/doi/10.1207/s15516709cog1402_1/abstract)
    
        Introduced RNN (simple RNN, later referred to as SRN)
    
    - [ ] [Long Short-Term Memory]
    (http://dl.acm.org/citation.cfm?id=1246450)
    
        Introduced LSTM
    
    - [ ] [Learning to Diagnose with LSTM Recurrent Neural Networks]
    (https://arxiv.org/abs/1511.03677)
    
        TBC
    
    - [ ] [Sequential Short-Text Classification with Recurrent and Convolutional Neural Networks]
    (https://arxiv.org/abs/1603.03827)
    
        TBC
    
    - [ ] [Character-Aware Neural Language Models]
    (https://arxiv.org/abs/1508.06615)
    
        CNN on n-grams, fed into an LSTM, for charachter-level classification.  
        Does not utilize word embeddings at all, thus significantly less parameters to train.  
        Character embeddings of dimension 4.  
        Touches on recurrent neural network language model (RNN-LM)  
    
    - [ ] [Recurrent Neural Networks for Robust Real-World Text Classification]
    (http://dl.acm.org/citation.cfm?id=1331869)
    
        xRNN model, 2 hidden layer, controlled by a hysteresis function
    
    - [ ] [A C-LSTM Neural Network for Text Classification]
    (https://arxiv.org/abs/1511.08630)
    
        Propose C-LSTM model, utilizes components from CNN and LSTM to extract features and represent latent vectors.
        Gives a descent overview of related work, N-gram feature extraction through convolution, LSTM,
        padding and word vector initialization.
        
    - [ ] [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation]
    (https://arxiv.org/abs/1406.1078)
    
        RNN-encoder-decoder setup for machine translation.  
        Present a modified unit with "update" and "reset" gates.  
        Encoding to a latent vector-space by the hidden state at time t == "Summary".  
        Decoding as a function of (Summary, last predicted symbol, last hidden state {of the decoder}).  
    
    - [ ] [Recurrent Neural Network for Text Classification with Multi-Task Learning]
    (https://arxiv.org/abs/1605.05101)
    
        Using RNN architecture to simultaneously train on several text classification tasks.
        Has reasonably nice visuallizations (probably not so relevant otherwise).  
    
    - [x] [ImageNet Classification with Deep Convolutional Neural Networks]
    (https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
    
        Introducing AlexNet, CNN image classification  
    
    - [x] [Going Deeper with Convolutions]
    (https://arxiv.org/abs/1409.4842)
    
        Introducing the Inception module in a deep CNN  
    
    - [x] [Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation]
    (https://arxiv.org/abs/1609.08144v2)
    
        LSTM-based translation NN   
        
- Bidirectional RNN classification
    
    - [ ] [Neural Machine Translation by Jointly Learning to Align and Translate]
    (https://arxiv.org/abs/1409.0473)
    
        RNN encoder-decoder, bi-directional RNN encoder (probably not so relevant otherwise).  
    
- char2char, seq2seq
    
    - [ ] [Sequence to Sequence Learning with Neural Networks]
    (https://arxiv.org/abs/1409.3215)
    
        TBC
        
- Generative Adversarial Network (GAN)
    
    - [ ] [Generative Adversarial Networks]
    (https://arxiv.org/abs/1406.2661)
    
        Introducing GAN
    
    - [ ] [Improved Techniques for Training GANs]
    (https://arxiv.org/abs/1606.03498)
    
        TBC
    
    - [ ] [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks]
    (https://arxiv.org/abs/1511.06434)
    
        TBC
    
- Hierarchical structures classification (no mention of NN)
    
    - [ ] [A survey of hierarchical classification across different application domains] 
    (http://link.springer.com/article/10.1007/s10618-010-0175-9)
    
        “IS-A” relationship, tree and DAG structure, (non) mandatory leaf node prediction and the blocking problem, 
        big-bang (or global) classifiers, flat classification approach, local classifier per node (LCN), 
        local classifier per parent node (LCPN) and local classifier per level (LCL)
    - [ ] [Large Margin Methods for Structured and Interdependent Output Variables]
    (http://www.jmlr.org/papers/volume6/tsochantaridis05a/tsochantaridis05a.pdf)
        
        (not sure if relevant)
        
    - [ ] [Large Margin Methods for Label Sequence Learning]
    (http://www.eecs.yorku.ca/course_archive/2005-06/F/6002B/Readings/EuroSpeech2003.pdf)
        
        Risk functions (Hamming risk), Conditional random fields (CRFs)
    
    - [ ] [Development of a Hierarchical Classification System with 
    Artificial Neural Networks and FT-IR Spectra for the Identification of Bacteria]
    (http://journals.sagepub.com/doi/abs/10.1366/0003702001948619)
    
        Hierarchical classification, different classifer for each level and for each node
    
    - [ ] [Hierarchical text classification and evaluation] 
    (http://ieeexplore.ieee.org/abstract/document/989560/?reload=true)
    
        TBC
    
    - [ ] [Large margin hierarchical classification]
    (http://dl.acm.org/citation.cfm?id=1015374)
    
        TBC
    
- NN robust classification methods wrt misspelling

- NN introduction, derivation, optimization, overfitting etc
    
    - Introduction and historical context
    
        - [ ] [Neural Networks and Deep Learning]
        (http://neuralnetworksanddeeplearning.com)
        
            Gives a nice overview and introduction to NN
            
        - [x] [A logical calculus of the ideas immanent in nervous activity]
        (https://link.springer.com/article/10.1007/BF02478259)
        
            First paper on NN (1943)
                
        - [x] [Principles of neurodynamics: perceptrons and the theory of brain mechanisms]
        (https://books.google.ca/books/about/Principles_of_neurodynamics.html?id=7FhRAAAAMAAJ&hl=en)
        
            Introducing perceptrons
                                
    - L2 regularization
    
    - early stopping
        
        - [ ] [On Early Stopping in Gradient Descent Learning]
        (http://link.springer.com/article/10.1007/s00365-006-0663-2)
        
            TBC
        
        - [ ] [Boosting with early stopping: Convergence and consistency]
        (https://arxiv.org/abs/math/0508276v1)
        
            TBC
        
    - drop-out
    
        - [ ] [Improving neural networks by preventing co-adaptation of feature detectors]
        (https://arxiv.org/abs/1207.0580)
        
            Introducing drop-out
        
        - [ ] [Dropout: A Simple Way to Prevent Neural Networks from Overfitting]
        (http://jmlr.org/papers/v15/srivastava14a.html)
        
            Show that dropout improves the performance of neural networks on supervised learning tasks in vision,
            speech recognition, document classification and computational biology.
        
        - [ ] [Semi-supervised Sequence Learning]
        (https://arxiv.org/abs/1511.01432)
        
            TBC
        
        - [ ] [Recurrent Neural Network Regularization]
        (https://arxiv.org/abs/1409.2329)
        
            TBC
        
        - [ ] [Dropout improves Recurrent Neural Networks for Handwriting Recognition]
        (https://arxiv.org/abs/1312.4569)
        
            TBC
        
        - [ ] [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks]
        (https://arxiv.org/abs/1512.05287)
        
            TBC
        
    - auxilary labels
    
        - [ ] [Using the Future to "Sort Out" the Present: Rankprop and Multitask Learning for Medical Risk Evaluation]
        (https://papers.nips.cc/paper/1081-using-the-future-to-sort-out-the-present-rankprop-and-multitask-learning-for-medical-risk-evaluation)
        
            Learning from the future with multitask learning
        
        - [ ] [Learning Many Related Tasks at the Same Time With Backpropagation]
        (http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.54.6346)
        
            Multitask backprop
        
    - target replication
        
        - [ ] [Deeply-Supervised Nets]
        (https://arxiv.org/abs/1409.5185)
        
            Companion objective (different to layer-wise pre-training), 
            introducing a classifier (e.g. SVM) to each layer
        
        - [ ] [Beyond Short Snippets: Deep Networks for Video Classification]
        (https://arxiv.org/abs/1503.08909)
        
            Target replication after each video frame
        
        - [ ] [Semi-supervised Sequence Learning]
        (https://arxiv.org/abs/1511.01432)
        
            TBC
        
    - optimization
        
        - [ ] [Adam: A Method for Stochastic Optimization]
        (https://arxiv.org/abs/1412.6980v8)
        
            Adam, an algorithm for first-order gradient-based optimization of stochastic objective functions,
            based on adaptive estimates of lower-order moments.
        
        - [ ] [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift]
        (https://arxiv.org/abs/1502.03167)
        
            Normalizing each batch to speed up learning (can be less careful with learning rates).  
            Can also be considered as a mean to regularize.  
        
        - [ ] [Highway Networks]
        (https://arxiv.org/abs/1505.00387)
        
            Assisting gradient-based learning of deep networks by allowing information flow downstream.    
        


## progress log  
**Commands**:  
comp-pdf-bib-tex Yarden_master_thesis  
ssh pytharski.math.ethz.ch  
ethmathSFTP  
- **1/3:** 
Fixed a lot of small latex and bibtex related bugs. 
To re-compile bibtex, use "comp-pdf-bib-tex Yarden_master_thesis" (compiles pdf, then bibtex *correctly*, then pdf x 2).
Started on NN preliminaries (in Related work chapter). Wrote on perceptron.
SVG file format - print to PDF.  
Todo:  
    - Work on github md file, add a check-box to quoted sources.  
    - Continue on activation functions, add different kinds of activations + figures.  
    (https://en.wikipedia.org/wiki/Activation_function,
    https://en.wikipedia.org/wiki/Heaviside_step_function, 
    https://en.wikipedia.org/wiki/Sigmoid_function, 
    https://en.wikipedia.org/wiki/Rectifier_(neural_networks),
    leaky ReLU,
    https://en.wikipedia.org/wiki/Hyperbolic_function#Hyperbolic_tangent _)  
    - Add background (motivation) on recent surge of NN success in vision, etc.  
    - Add background (motivation) BRIEF on NN as function approximators.  
    - Write on multi-layer perceptron, fully connected neural network.  
    - Gradient-based learning methods, backprop.   
    - **Next step**: RNN (wait with CNN), LSTM  
- **2/3:** 
Try and secure a work-station in CAB E81.  
Focus on state-of-the-art, no so much on a review of methods.  
Set up connection to math deprtment cluster, have TF1.0! (see _commands_).  
Finished writing about feed-forward NN, backprop (shallow).  
[Convert ris to bib file] (https://www.bruot.org/ris2bib/)  
Todo:  
    - Wite on RNN.  
    (http://colah.github.io/posts/2015-08-Understanding-LSTMs/)  
    - Look for state-of-the-art in the field (or in analogous, if doesn't apply).  
    
