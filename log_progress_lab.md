
## progress log  
**Commands**:  
comp-pdf-bib-tex Yarden_master_thesis  
ssh pytharski.math.ethz.ch  
ethmathSFTP  
**Guides / Useful Links:**  
https://wiki.math.ethz.ch/SfSInfo/RemoteAccessIntern  
https://blogs.ethz.ch/isgdmath/tensorflow/  
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
    - Write on RNN.  
    (http://colah.github.io/posts/2015-08-Understanding-LSTMs/)  
    - Look for state-of-the-art in the field (or in analogous, if doesn't apply).  
- **3/3:** 
Wrote on RNN and LSTM, equations figures and all.  
Fixed minor issues in latex equations (reference and aligning).  
Todo:  
    - Hierarchical classification.  
    - Look for state-of-the-art in the field (or in analogous, if doesn't apply).  
    
- **6/3:** 
Got data (on the end of Friday 3/3).  
Initial inspection shows that there are typically a handful of (distinct) examples for each ATC code. 
This is confirmed by histograms and filtering.  
Could try and enrich the data by looking at less frequent variants of FREETXT, 
to see if we could easily label them (e.g. by using the ATC convection table).  
Reorganized the literature review file on Github.  
Todo:  
    - Read through papers
    (https://arxiv.org/abs/1502.01710,  
    https://arxiv.org/abs/1607.01759,  
    https://www.cs.colorado.edu/~jbg/docs/2015_acl_dan.pdf,  
    http://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification)  
    - Consult Carsten and Patrick regarding possibility to expand the data-set.  
    - Think of topics to talk with Carsten (print structure?).  
    - Finish organizing literature review (if not already).  

- **7/3:** 
Migrated functions to utils directory.  
Wrote functions for n-grams, {char:count}, {n-gram:count}.  
Created sparse features and labels matricies, to be used by SVM (for benchmark).  
Fit SVM, linear multiclass produces perfect fit (needs further investigation).  
Todo:  
    - Read through papers (still, haven't gotten around to it)  
    (https://arxiv.org/abs/1502.01710,  
    https://arxiv.org/abs/1607.01759,  
    https://www.cs.colorado.edu/~jbg/docs/2015_acl_dan.pdf,  
    http://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification)  
    - Finish working on multiclass SVM.  
    - Character embeddings?  


