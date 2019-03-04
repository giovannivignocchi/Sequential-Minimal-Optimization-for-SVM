# Sequential Minimal Optimization

The main purpose of the project is to analyse the optimization methods currently employed to perform the training of the support vector machines (SVM).

## The scope of the project

The first part of the project consisted in studing the three following papers:
1. Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines,   [Platt 1998](http://www.microsoft.com/en-us/research/publication/sequential-minimal-optimization-a-fast-algorithm-for-training-support-vector-machines/);
2. Making Large-Scale SVM Learning Practical,   [Joachims 1998](http://www.cs.cornell.edu/people/tj/publications/joachims_99a.pdf);
3. Working Set Selection Using Second Order Information for Training Support Vector Machines, [Fan, Chem, Lin 2005](http://www.jmlr.org/papers/volume6/fan05a/fan05a.pdf);

The first article analyses the Sequential Minimal Optimization method (SMO) proposed by Platt. <br />Whereas in the second and the third, the authors proposed two different improvements of the SMO that, respectively, employ a 1<sup>st</sup> order and 2<sup>nd</sup> order method to select the Lagrange multipliers (LMs) that composed the working set.

## Implementation
In the second stage of the project, the algorithms have been developed by using Matlab. 
<br />For the sake of completeness, in addition to the Sequential Minimal Optimization methods proposed in the papers, two more versions of the SMO have been implemented, in particular:

1. "Improvements to Platt's SMO algorithm for SVM Classifier Design", [Keerthi 2001](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.115.5266&rep=rep1&type=pdf);

2. The original version proposed by Platt without using an auxiliary cache to store prediction error;

As in the implementation proposed by Joachims, Keerthi employed a 1<sup>st</sup> order method to select the LMs.
However the latter avoids to use an external quadratic solver to solve the optimization steps.

Although it might be interesting to further analyze both:
- The impact of the cache in the SMO version proposed by Platt;
- The performance of the Keerthi's SMO with respect to the Joachims' SMO  that uses a working set with size equals 2;

These considerations will not be taken into account in the continuation of the project

The folder [Algorithms](https://github.com/giovannivignocchi/Sequential-Minimal-Optimization-for-SVM/tree/master/Algorithm) collects all the implementations cited below:

- [Platt's version without Error Cache](https://github.com/giovannivignocchi/Sequential-Minimal-Optimization-for-SVM/blob/master/Algorithm/Platt/smo.m);
- [Platt's version with Error Cache](https://github.com/giovannivignocchi/Sequential-Minimal-Optimization-for-SVM/blob/master/Algorithm/Platt/smoErrorCache.m);
- [Keerthi](https://github.com/giovannivignocchi/Sequential-Minimal-Optimization-for-SVM/blob/master/Algorithm/Keerthi/KeerthiSmo.m) (1<sup>st</sup> order method);
- [Joachims](https://github.com/giovannivignocchi/Sequential-Minimal-Optimization-for-SVM/blob/master/Algorithm/Joachims/Jsmo.m) (1<sup>st</sup> order method);
- [Fan Chen and Lin](https://github.com/giovannivignocchi/Sequential-Minimal-Optimization-for-SVM/blob/master/Algorithm/Fan%20Chen%20and%20Lin/FCLsmo.m) (2<sup>nd</sup> order method);

## Testing and analysis
The third part of the project is the test and analysis stage. 
This phase is clearly divided in two sub-steps:

1. A first set of tests is focused on three different (bi-dimensional) datasets that were artificially generated.
The main goal of this first test phase is to assess the correctness of the implementations.
Indeed, it was easy to detect errors and imprecisions in the implementations, testing the algorithms against this small-size and known datasets.

2. A second series of tests is focused on six datasets of more substantial size and with a number of features greater than two.
The aim of this second set of tests is to compare how a 1<sup>st</sup> order SMO and a 2<sup>nd</sup> order SMO perform in the training phase.

To develop the 2<sup>nd</sup> point, the training phase is repeated using 3 different Sequental Minimal Optimization implementations:
- Joachims'version with Working Set Size (WSS) equals 4;
- Joachims'version with WSS equals 6;
- Fan Chen and Lin' implementation;

The statistics collected in the second series of test were used to gain a better understanding how these different training techniques behave.

## First series of tests
In this phase the algorithms were tested both on perfectly separable and non-perfcetly separable data. 
<br />A sample of the artificial dataset used is shown in the images below:

Perfectly separable data:
![ls](https://user-images.githubusercontent.com/32396630/51924433-501fe680-23ed-11e9-8198-e81da13399dd.jpg)
Non-perfcetly separable data:
![nls](https://user-images.githubusercontent.com/32396630/51924099-b3f5df80-23ec-11e9-8a44-17641528ce0c.jpg)

The results obtained, training the algorithms on these artificial datasets, confirm the correctness of their implementation.
<br /> To check the results, another model has been trained using the Matlab function _fitcsvm_.
To be coherent, all the relevant parameters (training algorithm, tolerance, maxIter, BoxConstraint, sigma) were set to be consistent with those used by the implementations under testing. Then the model generated is used as baseline to check if its results are consistent with other models.

Furthermore during the training phase different statistics are collected. Among these:
- Number of iterations performed;
- Training and prediction time;
- Average number of kernel evaluations per iteration;
- Number of support vectors generated;

Since these tests are more focused on testing the overall correctness of the algorithms the results obtained are not shown here.
Anyhow, the interested reader can find all the results and statistics concerning this collection of tests in the folder [TEST RESULT](https://github.com/giovannivignocchi/Sequential-Minimal-Optimization-for-SVM/tree/master/TEST/1%20series%20of%20tests/TEST%20RESULTS).

## Second series of tests

The following statistics briefly describe the datasets used during this phase:

- diabetes (8 features, 576 instances)
- ringnorm (20 features, 6500 instances)
- magic (10 features, 17118 instances)
- a9a (124 features, 12682 instances)
- codrna (8 features, 59535 instances)

The complete datasets used are available in the [Dataset](https://github.com/giovannivignocchi/Sequential-Minimal-Optimization-for-SVM/tree/master/Dataset/Data) folder. 

To analyse the performances of the 1<sup>st</sup> and 2<sup>nd</sup> order Sequential Minimal Optimization techniques,
the methods under investigation have been tested under a large number of training settings (different dataset as well as different training parameters) in order to gain a better overview of their behavior.
To do this, I repeated a **Grid search** over the parameters _C_ and _sigma_ (gaussian kernel variation) for each implementation and dataset.

As for the previous serie of test, I collected several statistics that are available in the folder [TEST RESULT](https://github.com/giovannivignocchi/Sequential-Minimal-Optimization-for-SVM/tree/master/TEST/2%20series%20of%20tests/TESTS%20RESULTS/GRID%20SERACH). Some of these are:

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
    * F_measure
    * G_mean
    
Thereafter these statistics are analyzed to show how the optimization methods respond to changes  of the specified parameters.

Before showing the results of the analysis, two considerations have to be taken into account:

1. Based on the size of the dataset under consideration, a different number of combinations of parameters is analysed during grid search. Notice how for the last two dataset (given their high number of instances and features) only the _C_ parameter is investigated during grid search.

2. To obtained statistics concerning the performances of the models generated, these have been tested against the test set. 
However, to validate the model using the test set should be considered as a bad practice.
The main porpouse of this project is not to generate the best possible model for a given set of data, but to analyse the performance of the different implementations. It is relevant to note that these are not affected by the way we validate the models.
<br />On the other hand, this choice gave an unbiased way to evaluate the models generated without the burden to repeat the training procedure to perform cross-validation. It is important to underline how this choice allowed to significantly shorten the overall training procedure time. 


<br />The following pictures collect the results obtained during this series of tests.
<br />For each dataset two different kinds of graphs are presented:
- A 3D plot that for each method shows:
   - Number of iterations;
   - Accuracy;
   - Number of support vector generated;
   
- A set of plots that compare how the implementations perform as  _C_ and  _sigma_ change.


### DIABETES 
**Fan Chen and Lin**
![flccomplessivo](https://user-images.githubusercontent.com/32396630/51791861-05625c80-21aa-11e9-80e3-4301430de884.jpg)

**Joachims with working set size equals 4**
![iterj4](https://user-images.githubusercontent.com/32396630/51791963-5e7ec000-21ab-11e9-9a11-97593fcff93e.jpg)

**Joachims with working set size equals 6**
![j6complessivo](https://user-images.githubusercontent.com/32396630/51792037-6db23d80-21ac-11e9-93d2-5bd00cc19ae1.jpg)

![line](https://user-images.githubusercontent.com/32396630/52021205-6cf31180-24f4-11e9-8e35-451ac5592d7b.jpg)

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

**Joachims with working set size equals 4**
![j4complessivo](https://user-images.githubusercontent.com/32396630/51843397-285c5000-2313-11e9-874f-99940b03f7cf.jpg)

**Joachims with working set size equals 6**
![j6complessivo](https://user-images.githubusercontent.com/32396630/51843398-285c5000-2313-11e9-9e89-e371c6ea4c04.jpg)

![line](https://user-images.githubusercontent.com/32396630/52021205-6cf31180-24f4-11e9-8e35-451ac5592d7b.jpg)

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

**Joachims with working set size equals 4**
![complessivoj4](https://user-images.githubusercontent.com/32396630/51862187-55722800-233e-11e9-9c6e-2f1db1adb9d4.jpg)

**Joachims with working set size equals 6**
![complessivoj6](https://user-images.githubusercontent.com/32396630/51862189-55722800-233e-11e9-9747-bf5e5681a439.jpg)

![line](https://user-images.githubusercontent.com/32396630/52021205-6cf31180-24f4-11e9-8e35-451ac5592d7b.jpg)

![legnd](https://user-images.githubusercontent.com/32396630/51909976-2dcaa080-23ce-11e9-8b45-e3c2f1f44a74.jpg)

<pre>                                                    C = 2<sup>-3</sup> </pre>
![mag1](https://user-images.githubusercontent.com/32396630/51970345-f95eef00-2476-11e9-8a47-97309707d96d.jpg)
<pre>                                                    C = 2  </pre>
![mag2](https://user-images.githubusercontent.com/32396630/51970347-f95eef00-2476-11e9-9030-f825e7c3961d.jpg)
<pre>                                                    C = 2<sup>9</sup>  </pre>                       
![mag3](https://user-images.githubusercontent.com/32396630/51970348-f95eef00-2476-11e9-9aa5-1a6e40806074.jpg)

###                                                          A9A

###                                                         CODRNA

Due to the high number of instances that compose this dataset, would be to expensive to perform a complete gid search (over both _C_ and _sigma_) as done in the previous dataset. 
As a consequence a fixed _sigma_ is heuristically selected (using the [Jaakkola](http://image.diku.dk/shark/sphinx_pages/build/html/rest_sources/tutorials/algorithms/svmModelSelection.html) heuristic) and a grid search is performed only over the _C_ parameter.
<br/>The Sigma parameter selected is equal to 3.2497


![line](https://user-images.githubusercontent.com/32396630/52021205-6cf31180-24f4-11e9-8e35-451ac5592d7b.jpg)
![legnd](https://user-images.githubusercontent.com/32396630/51909976-2dcaa080-23ce-11e9-8b45-e3c2f1f44a74.jpg)

<pre>                                                      sigma = 3.2497 </pre>
<img width="957" alt="codrna" src="https://user-images.githubusercontent.com/32396630/53735306-c4cfc000-3e86-11e9-97ab-da583993d79b.png">



## Bibliography

1. Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines,   [Platt 1998](http://www.microsoft.com/en-us/research/publication/sequential-minimal-optimization-a-fast-algorithm-for-training-support-vector-machines/)
2. Making Large-Scale SVM Learning Practical,   [Joachims 1998](http://www.cs.cornell.edu/people/tj/publications/joachims_99a.pdf)
3. Working Set Selection Using Second Order Information for Training Support Vector Machines,  [Fan, Chem, Lin 2005](http://www.jmlr.org/papers/volume6/fan05a/fan05a.pdf)
4. Improvements to Platt's SMO algorithm for SVM Classifier Design [Keerthi 2001](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.115.5266&rep=rep1&type=pdf)
5. On the Convergence of the Decomposition Method for Support Vector Machines [Chih-Jen Lin](https://www.csie.ntu.edu.tw/~cjlin/papers/conv.pdf)
6. [_SVMlight_ library documentation](http://svmlight.joachims.org/)
7. Data Mining and Analysis: Fundamental Concepts and Algorithms (Wagner Meira, Mohammed J. Zaki)
8. A Practical Guide to Support Vector Classification [Chih-Wei Hsu, Chih-Chung Chang, and Chih-Jen Lin](https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf)

## Author

    Giovanni Vignocchi

This project has been carried out in the context of the course of Nonlinear optimization I have attended in Politecnico di Milano.
