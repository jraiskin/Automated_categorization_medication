
## progress log  
**Commands**:  
comp-pdf-bib-tex Yarden_master_thesis  
ssh pytharski.math.ethz.ch  
ethmathSFTP  
**Guides / Useful Links:**  
https://wiki.math.ethz.ch/SfSInfo/RemoteAccessIntern  
https://blogs.ethz.ch/isgdmath/tensorflow/  
**Long-Term Goals / Guidelines:**  
Explain why we did not use Mimic data for training (GSN codes, text too clean w/o variation).  
Potentially validate LSTM architecture on Mimic data. Perform prediction on text (+timestamp), the class would be the perscription given shortly after (given timestamp). Look for literature if something similar has been done before.  

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

- **9/3:** 
Automated the feature creation and filtering process for the SVM linear classifier.  
Automated SVM hyper-parameters setup, set defaults as kwargs.  
Added utils (seed, save, load etc.).  
Initial commit for a simple LSTM model (including Tensorboard, checkpoints).  
Todo:  
    - Work on char-level embeddings for the LSTM architecture.  

- **10/3:** 
Tensorflow provides built-in embeddings support with tf.nn.embedding_lookup. Can use example here (https://github.com/dhwajraj/deep-siamese-text-similarity/blob/master/siamese_network.py).  
Worked on Tensorflow embeddings visualizer and Tensorboard. Still not perfect but not high priority now.  
Todo:  
    - Create (filtered) character lists as inputs.  
    - Create embeddings (lookup) for these inputs.  
    - Create 'one-hot' representation for the labels.  

- **11/3:** 
Spliting data into train-test set using train_test_split (sklearn).  
Transformed observations into filtered character lists.  
Created variables needed for RNN (character set size, number of unique labels etc.).  
Started working on LSTM simple model (links in file to tutorials / examples of archtectures, embeddings lookups).  
Todo:  
    - Finish LSTM architecture module.  
    - Create embeddings lookup for characters inputs.  

- **13/3:** 
Meeting with Carsten and Patrick.  
Suggested to output a ranked list (top k) of classes, as a suggestion.  
Possibly extending the data-set by trying to match patterns from unlabled data with labeled data. Hand over results to Patrick, so that he could sample the output to assess reliability.  
Padded inputs to have same length (introduced a padding symbol).  
Made some initial changes to SVM in order to output "top-k" ranked list, alongside probabilities.  
Todo (this is for more than one day):  
    - Check if Wikipedia pages on ATC codes would yield any benefit.  
    - Finish LSTM architecture module.  
    - Create embeddings lookup for characters inputs. If using "one-hot" representation, change to *untrainable* Identity matrix.  
    - Extract feasible ATC labels from unlabeled observations (n-grams, character BOW features, nearest neighbor?).  
    - "Un-functionize": make a simple LSTM / RNN **working** architecture.  
        - **X:** Padded list of symbols. Create mapping to index, initialize "embeddings matrix".  
        - **Y:** Create mapping to index.  
        - **Don't regard for now (rather, later):** functions, embedding-visualizer.  

- **14/3:** 
Worked on LSTM model with some progress but nothing completed yet (despite quite some effort).  

- **15/3:** 
A successful day!  
A simple LSTM model is working on a *toy example*.  
Able to adapt it to the *real data*. I've launched a small experiment with 1000 epoches. 
It seems to work, cost going down.  
The model has a dropout wrapper and calculated cross entropy loss.  
Todo:  
    - Look for potential flaws / mistakes in the model.  
    - Clean up code.  
    - "Functionize" the code, create name scopes, Tensorboard summaries, saver object.  
    - Check if Wikipedia pages on ATC codes would yield any benefit.  
    - Extract feasible ATC labels from unlabeled observations (n-grams, character BOW features, nearest neighbor?).  

- **16/3:** 
Fixed one-hot initialization (as an identity matrix constant).  
"Functionized" the code with name scopes (might need some minor additional work).  
Cleaned out the code (massively).  
Tensorboad summaries are in place, graph is visible (might also need some minor work).  
Todo:  
    - Fix (check) minor name scoping issues in Tensorboard.  
    - Figure out how saving and recovery work in Tensorboard.  
    - Check if Wikipedia pages on ATC codes would yield any benefit.  
    - Extract feasible ATC labels from unlabeled observations (n-grams, character BOW features, nearest neighbor?).  


- **17/3:** 
Cleaned up code, created functions to automate and migrated utils to utils_nn.  
Set up paths to data files over Euler in utils.  
Code submission to Euler works as expected! returns output into the folder from which the job was launched.  
Started woking on embeddings visualizer, visualizing character embeddings (with character labels).  
Todo:  
    - Finish character embeddings visualizer (https://www.tensorflow.org/get_started/embedding_viz).  
    - Figure out checkpoint saving and loading (https://www.tensorflow.org/programmers_guide/variables).  
    - Check if Wikipedia pages on ATC codes would yield any benefit.  
    - Extract feasible ATC labels from unlabeled observations (n-grams, character BOW features, nearest neighbor?).  

- **18/3:** 
Changed the implementation from a (series of) functions to a class!  
This would enable calling a 'train' method or to load a model from a checkpoint.  
Limited success on the character embeddings visualizer.  
Found nice resources [from a blog post](http://andrewmatteson.name/index.php/2017/02/19/using-tensorboard-projector-external-data/) and [an example on github](https://github.com/normanheckscher/mnist-tensorboard-embeddings/blob/master/mnist_t-sne.py).  
Will try to work on it more today.  
Todo (basically unchaged):  
    - Finish character embeddings visualizer (https://www.tensorflow.org/get_started/embedding_viz).  
    - Figure out checkpoint saving and loading (https://www.tensorflow.org/programmers_guide/variables).  
    - Check if Wikipedia pages on ATC codes would yield any benefit.  
    - Extract feasible ATC labels from unlabeled observations (n-grams, character BOW features, nearest neighbor?).  

- **20/3:** 
Meeting with Carsten:  
Expand the data-set and create a validation set (maybe even test set), utilizing the unlabeled data.
Map the unlabeled text and labeled text to features (say, n-grams and character 'bag-of-words') and find Jaccard similarity.
Keep a heap of top 5 suggestion and their similarities. Output top suggestions if sim >= threshold.  
Once we have validation set, hyper-parameters would be optimized on it.  
Possible model variations: sampling proportional to f(frequency), with f as log.  
Hierarchical classification - currently takes the back sit (due to lack of data, perhaps would be feasible with patient data).  
**DONE:** 
    - Jaccard sim function, with BOW dictionaries as input.  
    - Compare labeled and unlabeled data with Jaccard sim, keep top 5 suggestions and their similarities. Output top suggestions if sim >= threshold.  
    - Save a CSV file with text, label, Jaccard_sim.  
Todo:  
    - Follow up on suggested labels: send to Patrick.  
    - Finish character embeddings visualizer (https://www.tensorflow.org/get_started/embedding_viz).  
    - Figure out checkpoint saving and loading (https://www.tensorflow.org/programmers_guide/variables).  

