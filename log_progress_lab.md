
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

- **21/3:** 
Worked on restoring *session* and *checkpoint*, as well as on reproducible results (setting a seed).  
It seems like restoring a checkpoint (and session) is working, but can't tell for sure, since results are irreproducible. 
This [stackoverflow post](http://stackoverflow.com/questions/34500052/tensorflow-saving-and-restoring-session) deals with restoring.  
Other than that, tried to work on setting a seed. My understanding is that it is a graph-wide seed, 
although [TF documentation}(https://www.tensorflow.org/api_docs/python/tf/Graph) is lacking, as usual.  
[This code snippent](https://groups.google.com/a/tensorflow.org/forum/#!topic/discuss/Yu3OFF3_GLc) is supposed to work on CPU.  
Also check out [this post](http://stackoverflow.com/questions/36288235/how-to-get-stable-results-with-tensorflow-setting-random-seed) on setting a graph-wide seed.  
Todo:  
    - Work on setting a seed, to get reproducible results.  
    - Figure out checkpoint saving and loading.  
    - Finish character embeddings visualizer (https://www.tensorflow.org/get_started/embedding_viz).  

- **23/3:** 
Worked on saving and restoring session chechpoints.  
Saving and restoring works as expected!  
Fixed minor issues with saving and printing steps.  
Still **can't make reproducible code with tf seed**.  
Patrick gave the go-ahead to use the suggested data as if it was labeled data.  
Todo:  
    - Encorporate the suggested data as normal data. Make sure there are no duplicates. Make sure to have (Text, ATC, frequency) information on all data points.  
    - Modify LSTM class to have a training and validation feed dict. Report accuracy on validation data.  
    - Generate a "top-k" prediction method and a "top-k" accuracy measure.  
    - Sample / duplicate data-points proportional to f(frequncy), with f, say, log.  
    - Finish character embeddings visualizer (https://www.tensorflow.org/get_started/embedding_viz).  

- **26/3:** 
Corrected path to unlabeled suggestion CSV (for both 'yarden' and 'raiskiny' cluster users).  
While working on merging suggested file and original file (with syncing frequencies), found out that there were character encoding issues (not able to match text strings).  
Fixing this issue would likely resolve the "duplicates" issue as well (as they were duplicates due to incorrect encoding).  
Ran unlabeled data suggestion procedure again on Euler.  
Todo:  
    - Encorporate the suggested data as normal data. Make sure there are no duplicates. Make sure to have (Text, ATC, frequency) information on all data points.  
    - Sample / duplicate data-points proportional to f(frequncy), with f, say, log, or f=1 to unchange the data.  
    - Clean-up merging data files (migrate to different file, make data generation consistent across different files, including linear model).  
    - Explore the newly merged data-set (same way initial data was explored). Gain insights as to how to split to train / validation sets.  
    - Modify LSTM class to have a training and validation feed dict. Report accuracy on validation data.  
    - Generate a "top-k" prediction method and a "top-k" accuracy measure.  
    - Finish character embeddings visualizer (https://www.tensorflow.org/get_started/embedding_viz).  

- **27/3:** 
Results of second run of unlabeled data suggestion procedure were successful. Yielded results can now be matched with original data.  
Created same data representation from suggested data, along with labels and frequency data.  
Data is merged and fed into transformations that the LSTM model accepts (array of character indecies).  
Still left: replicate "rows" according to a given function / rule.  
Todo:  
    - Sample / duplicate data-points proportional to f(frequncy), with f, say, log, or f=1 to unchange the data.  
    - Clean-up merging data files (migrate to different file, make data generation consistent across different files, including linear model).  
    - Explore the newly merged data-set (same way initial data was explored). Gain insights as to how to split to train / validation sets.  
    - Modify LSTM class to have a training and validation feed dict. Report accuracy on validation data.  
    - Generate a "top-k" prediction method and a "top-k" accuracy measure.  
    - Finish character embeddings visualizer (https://www.tensorflow.org/get_started/embedding_viz).  

- **28/3:** 
Sampling proportional to frequency is working ("scaling"). Can you "unscaled" version to retain original proportions.
Code has been automated and cleaned, utils migrated (and slightly modified).  
Updated the exploratory notebook (with analyses and histograms) to accomodate merged data, made code more concide, flexible and consistent.  
Need to start thinking of ways to split to training-validation sets (maintaining that all labels are present in training set).  
Split should occur after merging the data sources, but possibly before scaling (after would make sense as well).  
Todo:  
    - Split data into training-validation sets. Possibly, sample for validation set in a multiple step process:  
        - Restrict to a subset of labels to draw from (either uniformly or inversly proportional to frequency)  
        - Sample from each drawn label a certain number of observations (up to, say, half of available observations)  
        - Make sure sample size is in a certain range  
        - Split (x,y,freq) into two sets seperate sets  
        - Shuffle boths sets (with same permutation)  
        - Look for possible literature on the subject  
    - Modify LSTM class to have a training and validation feed dict. Report accuracy on validation data.  
    - Generate a "top-k" prediction method and a "top-k" accuracy measure.  
    - Finish character embeddings visualizer (https://www.tensorflow.org/get_started/embedding_viz).  

- **31/3:** 
Split data to validation and training sets. Both sets can be scaled and fed into the LSTM model.  
Validation and training data are accepted as feeding dicts to the LSTM model.  
Still need to make TF infer the "batch size", i.e. not specifying it 
(rather passing it as None). Doesn't work for some reason.  
Todo:  
    - Fix shaping problem in TF (accept placeholders' size as None).  
    - Report accuracy on validation data.  
    - Generate a "top-k" prediction method and a "top-k" accuracy measure.  
    - Finish character embeddings visualizer (https://www.tensorflow.org/get_started/embedding_viz).  
   
- **1/4:** 
TF can now infer "batch-size". Changed the way data is read into the LSTM function (reshaping etc.).  
Tried different ways to track both training and validation accuracy (including constructing a string using tf.cond).  
Ended up with a solution of having 2 write function for each model, 
seperating written summaries into 2 different directories.  
Tracking training and validation accuracy seems to work, 
but the *validation* accuracy seems to be consistently higher (suspicious).  
There's a tf.nn.in_top_k function, should use it as well.  
Todo:  
    - Make sure accuracy on validation data is correct (no bugs).  
    - Generate a "top-k" prediction method and a "top-k" accuracy measure 
        (look at [tf.nn.in_top_k function](https://www.tensorflow.org/api_docs/python/tf/nn/in_top_k)).  
    - Finish character embeddings visualizer (https://www.tensorflow.org/get_started/embedding_viz).  

- **3/4:** 
Successful day overall!  
Fixed TF summaries, now training and test error tracking works fine.  
When evaluating test set, dropout keep probability is set to 1.  
A new metric "is in top k" is introduced (and working). k is set to 5 (parameter).  
SOLVED Embeddings visualizer! Tensorboard should be called from the git repo directory.  
Ran a small-scale experiment locally. Seems to work fine.  
A big experiment is currently running on Euler.  
Not clear: how to relatively easily apply l2 normalization to weights.  
Todo:  
    - Clean up code.  
    - Go over linear classifier code:  
        - Data initialization;  
        - Test-training split;  
        - Evaluation (including top_k);  
        - Hyperparameters optimization.  
    - Fix tf.seed issue

- **4/4:** 
Meeting with Carsten:  
Try discarding labels that appear less than k (10) times from both training and test sets.  
Consider applying other IR metrics, such as *Reciprocal Rank*.  
think of ways to incorporate external knowledge into the model.  
e.g. If we had a comprehensive list of all legal medicine (say, from swissmedic / wikipedia), 
we could add that data as a perscription (frequency of 1) as additional labeled data, in the "data enhancement" stage.  
Fixed dropout keep probability, set to 1.0 whenever evaluating (dropout applied only while training).  
The procedure used to split the data to training-test sets is called stratified sampling.  
Todo:  
    - Track Reciprocal Rank.  
    - Discard labels that appear less than k times from both data sets.  
    - Apply L2 weight normalization [using compute and apply gradients]
        (https://www.tensorflow.org/api_docs/python/tf/train/Optimizer).  
    - Look at 3/4 experiment. Why strange behavior occurs?:  
        - seems to be correlated with a large hidden state size (of dimension 32).  
        - what other hyper-parameters affect the strange behavior of these graphs.  
        - what's common for all the under-performers (learning rate of 1.0E-2).  
    - Possible other directions to explore:  
        - Target replication.  
        - Dynamic learning rate schedule.  
        - Backward / bi-directional data feeding.  
        - LSTM up to 3 layers deep.  
        - CNN-LSTM combined models.  
        - GRU RNN.  
        - Encoder-decoder(?).  
    - Go over linear classifier code:  
        - Data initialization;  
        - Test-training split;  
        - Evaluation (including top_k);  
        - Hyperparameters optimization.  
    - Fix tf.seed issue  

- **5/4:** 
Reproducibility problem solve! The main culprit was iterating over a sets and dictionaries (non deterministic ordering).  
Applied L2 norm regularization for all trainable variables (with the same coefficient).  
Discarding "rare" labels from both data sets is now an option, as part of the train-test data split.  
Implementing reciprocal rank proves to be challenging (applying functions on tensors with unknown ranks).  
Some ideas regarding how to implement target replication.  
Todo:  
    - Target replication:  
        - create a bool placeholder  
        - conditional duplicate each row in Y by seq_len in all feed dictionaries  
        - conditional take all LSTM outputs (not just last) , reshape (?) so that each is like an observation  
    - Dynamic learning rate schedule.  
    - Backward / bi-directional data feeding.
    - Look at 3/4 experiment. Why strange behavior occurs?:  
        - seems to be correlated with a large hidden state size (of dimension 32).  
        - what other hyper-parameters affect the strange behavior of these graphs.  
        - what's common for all the under-performers (learning rate of 1.0E-2).  
    - Possible other directions to explore:  
        - LSTM up to 3 layers deep.  
        - CNN-LSTM combined models.  
        - GRU RNN.  
        - Encoder-decoder(?).  
    - Go over linear classifier code:  
        - Data initialization;  
        - Test-training split;  
        - Evaluation (including top_k);  
        - Hyperparameters optimization.  
    - Track Reciprocal Rank.  

- **6/4:** 
Implemented target replication!  
Implemented bidirectional LSTM!  
Implemented dynamic learning rate.  
Modified hyper-parameter string generator to accommodate new functionality.  
Fixed some side effects that happened when keep_rare_labels is False.  
Setting keep_rare_labels to False, amends the abnormalities that arose between training-test (loss(train) < loss(test)).  
Todo:  
    - Look at 3/4 experiment. Why strange behavior occurs?:  
        - seems to be correlated with a large hidden state size (of dimension 32).  
        - what other hyper-parameters affect the strange behavior of these graphs.  
        - what's common for all the under-performers (learning rate of 1.0E-2).  
    - Possible other directions to explore:  
        - LSTM up to 3 layers deep.  
        - CNN-LSTM combined models.  
        - GRU RNN.  
        - Encoder-decoder(?).  
    - Go over linear classifier code:  
        - Data initialization;  
        - Test-training split;  
        - Evaluation (including top_k);  
        - Hyperparameters optimization.  
    - Track Reciprocal Rank.  

- **7/4:** 
Fixed minor encoding issues (german characters).  
Cost procedure is more transparent now. Now also reporting the different components of the cost function 
(with name scoping).  
Linear classifier: data initialization and merging runs as expected. Need still to split into training-validation sets.  
Running a big experiment on Euler.  
Todo:  
    - Linear classifier code:  
        - Test-training split;  
        - Evaluation (including top_k and Reciprocal Rank);  
        - Hyperparameters optimization.  
    - Possible other directions to explore:  
        - LSTM up to 3 layers deep.  
        - CNN-LSTM combined models.  
        - GRU RNN.  
        - Encoder-decoder(?).  
    - Track Reciprocal Rank.  

- **8/4:** 
LSTM model can now track less summaries (such as histograms) and not save models, to save log space.  
Running an Euler experiment again (output too big).  
Linear SVM: implemented Mean Reciprocal Rank, train-test split and performed hyper parameter search.  
SVM is performing surprisingly well, after discarding "rare" labels (same procedure as in the LSTM input).  
Todo:  
    - Look at experiment results.  
    - Think of SVM results meaning.  
    - Possible other directions to explore:  
        - LSTM up to 3 layers deep.  
        - CNN-LSTM combined models.  
        - GRU RNN.  
        - Encoder-decoder(?).  
    - Track Reciprocal Rank.  

- **10/4:** 
Meeting with Carsten:  
Look closer at the SVM, compare to logistic regression. Specifically into the data preprocessing step 
(no unfair advantage to the SVM).  
Further investigate the parameters, see what affects the model the most and increase search variation there 
(learning rate, hidden state size, possibly regularization constants).  
Directions for later investigation:  
    - Introduce noise sources, such as noisy activation functions.  
    - Context where LSTM outperforms, such as shorter key strokes, random unknown characters. 
        e.g. plot accuracy vs how many characters the classifier "sees".  
    - Explor ensemble methods, e.g. a joint prediction based on weighted LSTM and linear classifiers.  
Next meeting on Wed (19/4) 10:00.  
Todo:  
    - Fix saving step to be the last step.  
    - Check SVM data pre-processing stage, look into Logistic regression as well ('logistic' kernel).  
    - Look at experiment results, what hyper parameters affects the model the most.  
    - Look into noisy activation functions.  
    - Possible other directions to explore:  
        - Introduce noise sources, such as noisy activation functions.  
        - Context where LSTM outperforms, such as shorter key strokes, random unknown characters. 
            e.g. plot accuracy vs how many characters the classifier "sees".  
        - Explor ensemble methods, e.g. a joint prediction based on weighted LSTM and linear classifiers.  
    - Track Reciprocal Rank.  

