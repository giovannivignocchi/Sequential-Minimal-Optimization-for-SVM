# Sequential minimal optimization

The main purpose of the project is to analyze the optimization methods, currently emplyed to perform 
the training of the support vector machines (SVM).

## The scope of the project

The first part of the project consisted in studing the three following papers:
1. Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines,   [Platt 1998](http://www.microsoft.com/en-us/research/publication/sequential-minimal-optimization-a-fast-algorithm-for-training-support-vector-machines/);
2. Making Large-Scale SVM Learning Practical,   [Joachims 1998](http://www.cs.cornell.edu/people/tj/publications/joachims_99a.pdf);
3. Working Set Selection Using Second Order Information for Training Support Vector Machines,   [Fan, Chem, Lin 2005](www.jmlr.org/papers/volume6/fan05a/fan05a.pdf);

The first article analyses the Sequential Minimal Optimization method (SMO) proposed by Platt. <br />Whereas in the second and the third, the authors proposed two different improvements of the SMO that, respectively, employ a 1<sup>st</sup> order and 2<sup>nd</sup> order method to select the Lagrange multipliers (LMs) that composed the working set.

## Implementation
In the second stage of the project, the algorithms have been developed by using Matlab. 
<br />For the sake of completeness, in addition to the Sequential Minimal Optimization methods proposed in the papers, two more versions of the SMO have been implemented, In particular:

1. "Improvements to Platt's SMO algorithm for SVM Classifier Design", [Keerthi 2001](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.115.5266&rep=rep1&type=pdf);

2. The original version proposed by Platt without using an auxiliary cache to store Prediction error;

The implementation proposed by Keerthi, employed a 1<sup>st</sup> order method to select the LMs of a fixed size working set (size equals 2). So it does not rely on an external quadratic solver to solve the optimization step as done by the version proposed by Joachims.

While the implementation of the Platt's version that does not use any cache to temporarly store the prediction error is due  to the fact that could be intresting to analyze the effects of the cache on the training time.

Although it might be interesting to further analyze both:
- The impact of the cache in the SMO version proposed by Platt;
- The performance of the Keerthi's SMO with respect to th Joachims' SMO  that use a working set with size equals 2;
These considerations will not be taken into account in the continuation of the project;

The folder [Algorithms](https://github.com/giovannivignocchi/Sequential-Minimal-Optimization-for-SVM/tree/master/Algorithm) collects all the implementations cited below:

- [Platt's version without Error Cache](https://github.com/giovannivignocchi/Sequential-Minimal-Optimization-for-SVM/blob/master/Algorithm/Platt/smo.m);
- [Platt's version with Error Cache](https://github.com/giovannivignocchi/Sequential-Minimal-Optimization-for-SVM/blob/master/Algorithm/Platt/smoErrorCache.m);
- [Keerthi](https://github.com/giovannivignocchi/Sequential-Minimal-Optimization-for-SVM/blob/master/Algorithm/Keerthi/KeerthiSmo.m) (1<sup>st</sup> order method);
- [Joachims](https://github.com/giovannivignocchi/Sequential-Minimal-Optimization-for-SVM/blob/master/Algorithm/Joachims/Jsmo.m) (1<sup>st</sup> order method);
- [Fan Chen and Lin](https://github.com/giovannivignocchi/Sequential-Minimal-Optimization-for-SVM/blob/master/Algorithm/Fan%20Chen%20and%20Lin/FCLsmo.m) (2<sup>nd</sup> order method);

## Testing and analysis
The third part of the project is testing and analysis stage. 
This phase is clearly divided in two sub-steps:

1. A first set of tests is focused on three different (bi-dimensional) datasets that were artificially generated.
The main goal of this first test phase is to assess the correctness of the implementations.
Indeed, it was easy to detect errors and imprecisions in the implementations, testing the algorithms against this small-size and known datasets.

2. A second series of tests is focused on four datasets of more substantial size and with a number of features greater than two.
The aim of this second set of tests is to compare how a 1<sup>st</sup> order SMO and a 2<sup>nd</sup> order SMO perform in the training phase.

To do so, the training phase is repeated using 3 different Sequental minimal Optimization implementations:

- Joachims'version with Working Set Size (WSS) equals 4
- Joachims'version with WSS equals 6
- Fan Chen and Lin

The statistics collected in the second series of test were used to gain a better understanding on how these different training techniques behave.

## First series of tests
In this phase the algorithms were tested both on perfectly separable and non-perfcetly separable data. 
<br />A sample of the artificial dataset used is shown in the images below:

Perfectly separable data:
![ls](https://user-images.githubusercontent.com/32396630/51924433-501fe680-23ed-11e9-8198-e81da13399dd.jpg)
Non-perfcetly separable data:
![nls](https://user-images.githubusercontent.com/32396630/51924099-b3f5df80-23ec-11e9-8a44-17641528ce0c.jpg)

The results obtained, training the algorithms on these artificial datasets, confirm the correctness of their implementation.
<br /> To check the results, another model was trained using the Matlab function _fitcsvm_.
To be coherent all the relevant parameters (training algorithm, tolerance, maxIter, BoxConstraint, sigma) were set to be consistent with those used by the implementations under testing. The model generated is then used as baseline to check if the other models are consistent with it.

Furthermore during the training phase different statistics are collected. Among these:
- The number of iteration performed
- Training and prediction time
- Average number of kernel evaluation per iteration
- Number of support vector generated

Since these tests are more focused on testing the overall correctness of the algorithms the results obtained are not show here.
<br />Anyhow, the interested reader can find all the results and statistics concerning this serie of tests in the folder [TEST RESULT](https://github.com/giovannivignocchi/Sequential-Minimal-Optimization-for-SVM/tree/master/TEST/on%20artificial%20dataset/TEST%20RESULTS).

## Second series of tests

The following statistics briefly describe the datasets used during this phase:

- diabetes (8 features, 576 instances)
- ringnorm (20 features, 6500 instances)
- magic (10 features, 17118 instances)

The complete datasets used, are available in the [Dataset](https://github.com/giovannivignocchi/Sequential-Minimal-Optimization-for-SVM/tree/master/Dataset/Data) folder. 

To analyze the performances of the 1<sup>st</sup> and 2<sup>nd</sup> order Sequential Minimal Optimization techniques, I thought that should be usefull to test the methods under the greates number of training settings (different dataset as well as different training parameters). To do so I repeated a **Grid search** over the parameters _C_ (Box constraint) and _sigma_ (gaussian kernel variation) for each implementation and each dataset.

As for the previous series of test, I collected several statistics, that are available in the folder [TEST RESULT](https://github.com/giovannivignocchi/Sequential-Minimal-Optimization-for-SVM/tree/master/TEST/on%20real%20dataset/TEST%20RESULTS), some of these are:

* Total number of iteration
* Training time
* Total number of kernel evaluation
* Number of support vector generated
* Several metrics about the performance of the models generated 
    * Accuracy
    * Sensitivity
    * Specificity
    * Precision
    * Recall
    * Fmeasure
    * Gmean
    
These statistics are then analyzed to show how the optimization methods respond to changes in the specified parameters.

<br />Before showing the results of the analysis, I want to stress 2 consideration about how these tests were carried out, in order to better understand the results presented in the graphs below:

1. Based on the size of the dataset under consideration, a different number of combinations of parameters is analyzed during grid search.

2. To obtained statistics concerning the performances of the models generated. I tested these on the relative tests set. 
What I want to underline is that, however validating models using test set, should be considered as a bad practice; the main porpouse of this project is not to generate the best possible model for a given set of data, but to analyze the performance of the different implementations under analysis (note how the latters are not affected by the way we validate the models).
<br />On the other hand, this choice, gave me an unbiased way to evaluate the models generated, without the burden to repeat the training procedure to perform cross-validation. Note how this choice allowed me to significantly shorten the overall training procedure time. 


The following pictures collect the results obtained during this series of tests.
<br />For each dataset two different kind of graphs are presented:
- A 3D plot that individually summarize for each method, the main performance measures obtained during the grid search phase.
- A set of plots for each value of _C_ (used during the grid search) that compare how the differenet implementations perform as _sigma_ changes.


### DIABETES 
**Fan Chen and Lin**
![flccomplessivo](https://user-images.githubusercontent.com/32396630/51791861-05625c80-21aa-11e9-80e3-4301430de884.jpg)

**Joachims with working set size equal 4**
![iterj4](https://user-images.githubusercontent.com/32396630/51791963-5e7ec000-21ab-11e9-9a11-97593fcff93e.jpg)

**Joachims with working set size equal 6**
![j6complessivo](https://user-images.githubusercontent.com/32396630/51792037-6db23d80-21ac-11e9-93d2-5bd00cc19ae1.jpg)

The following graphs compare how the differenet implementations perform as _sigma_ changes

![legnd](https://user-images.githubusercontent.com/32396630/51909976-2dcaa080-23ce-11e9-8b45-e3c2f1f44a74.jpg)

<pre>                                                    C = 2<sup>-5</sup></pre>
![diab1](https://user-images.githubusercontent.com/32396630/51970337-f82dc200-2476-11e9-9260-73fc85d5d197.jpg)
<pre>                                                    C = 2<sup>-3</sup> </pre>
![diab2](https://user-images.githubusercontent.com/32396630/51970339-f82dc200-2476-11e9-936c-6864a448ae67.jpg)
<pre>                                                    C = 2  </pre>
![diab3](https://user-images.githubusercontent.com/32396630/51970340-f8c65880-2476-11e9-9b8a-3e011a81dbb7.jpg)
<pre>                                                    C = 2<sup>7</sup>  </pre>                                                          
![diab4](https://user-images.githubusercontent.com/32396630/51970341-f8c65880-2476-11e9-8eff-ee7082781974.jpg)
<pre>                                                    C = 2<sup>9</sup>  </pre>                                                          
![diab5](https://user-images.githubusercontent.com/32396630/51970343-f8c65880-2476-11e9-98b4-ce5699505d65.jpg)
<pre>                                                    C = 2<sup>15</sup>  </pre>                                                          
![diab6](https://user-images.githubusercontent.com/32396630/51970344-f8c65880-2476-11e9-967e-fd254a4d092b.jpg)

###                                                      RINGNORM
**Fan Chen and Lin**
![fclcomplete](https://user-images.githubusercontent.com/32396630/51843395-26928c80-2313-11e9-8396-32a6712e0219.jpg)

**Joachims with working set size equal 4**
![j4complessivo](https://user-images.githubusercontent.com/32396630/51843397-285c5000-2313-11e9-874f-99940b03f7cf.jpg)

**Joachims with working set size equal 6**
![j6complessivo](https://user-images.githubusercontent.com/32396630/51843398-285c5000-2313-11e9-9e89-e371c6ea4c04.jpg)

The following graphs compare how the differenet implementations perform as _sigma_ changes
![legnd](https://user-images.githubusercontent.com/32396630/51909976-2dcaa080-23ce-11e9-8b45-e3c2f1f44a74.jpg)

<pre>                                                    C = 2<sup>-5</sup></pre>
![ring1](https://user-images.githubusercontent.com/32396630/51970349-f95eef00-2476-11e9-8eb2-4660f3762c9e.jpg)
<pre>                                                    C = 2<sup>-3</sup></pre>
![ring2](https://user-images.githubusercontent.com/32396630/51970350-f9f78580-2476-11e9-9bbd-287a5892f9a1.jpg)
<pre>                                                    C = 2  </pre>
![ring3](https://user-images.githubusercontent.com/32396630/51970352-f9f78580-2476-11e9-9a67-bd7d1d60b9ce.jpg)
<pre>                                                    C = 2<sup>7</sup>  </pre>           
![ring4](https://user-images.githubusercontent.com/32396630/51970353-fa901c00-2476-11e9-8524-a693d17c6522.jpg)
<pre>                                                    C = 2<sup>9</sup>  </pre>                          
![ring5](https://user-images.githubusercontent.com/32396630/51970354-fa901c00-2476-11e9-9175-e406a92902b4.jpg)

###                                                         MAGIC
**Fan Chen and Lin**
![complessivofcl](https://user-images.githubusercontent.com/32396630/51862186-54d99180-233e-11e9-85eb-5d94a1068e9d.jpg)

**Joachims with working set size equal 4**
![complessivoj4](https://user-images.githubusercontent.com/32396630/51862187-55722800-233e-11e9-9c6e-2f1db1adb9d4.jpg)

**Joachims with working set size equal 6**
![complessivoj6](https://user-images.githubusercontent.com/32396630/51862189-55722800-233e-11e9-9747-bf5e5681a439.jpg)

The following graphs compare how the differenet implementations perform as _sigma_ changes
![legnd](https://user-images.githubusercontent.com/32396630/51909976-2dcaa080-23ce-11e9-8b45-e3c2f1f44a74.jpg)

<pre>                                                    C = 2<sup>-3</sup> </pre>
![mag1](https://user-images.githubusercontent.com/32396630/51970345-f95eef00-2476-11e9-8a47-97309707d96d.jpg)
<pre>                                                    C = 2  </pre>
![mag2](https://user-images.githubusercontent.com/32396630/51970347-f95eef00-2476-11e9-9030-f825e7c3961d.jpg)
<pre>                                                    C = 2<sup>9</sup>  </pre>                       
![mag3](https://user-images.githubusercontent.com/32396630/51970348-f95eef00-2476-11e9-9aa5-1a6e40806074.jpg)


## Bibliography

1. Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines,   [Platt 1998](http://www.microsoft.com/en-us/research/publication/sequential-minimal-optimization-a-fast-algorithm-for-training-support-vector-machines/)
2. Making Large-Scale SVM Learning Practical,   [Joachims 1998](http://www.cs.cornell.edu/people/tj/publications/joachims_99a.pdf)
3. Working Set Selection Using Second Order Information for Training Support Vector Machines,   [Fan, Chem, Lin 2005](www.jmlr.org/papers/volume6/fan05a/fan05a.pdf)
4. Improvements to Platt's SMO algorithm for SVM Classifier Design [Keerthi 2001](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.115.5266&rep=rep1&type=pdf)
5. On the Convergence of the Decomposition Method for Support Vector Machines [Chih-Jen Lin](https://www.csie.ntu.edu.tw/~cjlin/papers/conv.pdf)
6. [_SVMlight_ library documentation](http://svmlight.joachims.org/)
7. Data Mining and Analysis: Fundamental Concepts and Algorithms (Wagner Meira, Mohammed J. Zaki)
8. A Practical Guide to Support Vector Classification [Chih-Wei Hsu, Chih-Chung Chang, and Chih-Jen Lin](https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf)

## Author

    Giovanni Vignocchi

This project has been carried out in the context of the course of Nonlinear optimization I have attended in Politecnico di Milano.
