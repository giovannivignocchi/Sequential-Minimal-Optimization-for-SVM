# Sequential minimal optimization

The main purpose of the project is to analyze the optimization methods, currently emplyed to perform 
the training of the support vector machines (SVM).

## The scope of the project

The first part of the project consisted in studing the three following paper :
1. Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines,   [Platt 1998](http://www.microsoft.com/en-us/research/publication/sequential-minimal-optimization-a-fast-algorithm-for-training-support-vector-machines/)
2. Making Large-Scale SVM Learning Practical,   [Joachims 1998](http://www.cs.cornell.edu/people/tj/publications/joachims_99a.pdf)
3. Working Set Selection Using Second Order Information for Training Support Vector Machines,   [Fan, Chem, Lin 2005](www.jmlr.org/papers/volume6/fan05a/fan05a.pdf)

The first arictle discusses the Sequential Minimal Optimization method (SMO) in its first version proposed by Platt, whereas in the second and the third, the authors proposed two different modifications of the SMO that employ, respectively, a 1st order and 2nd order method to select the Lagrange multipliers that composed the working set

## Implementation
The second stage of the project consisted in the implementation, using Matlab, of the algorithms proposed in the papers.
For the sake of completeness, in addition to the Sequential Minimal Optimization methods proposed in the papers, two more version of the SMO were implemented, In particular:

1. "Improvements to Platt's SMO algorithm for SVM Classifier Design", [Keerthi 2001](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.115.5266&rep=rep1&type=pdf)

2. The original version proposed by Platt without using an auxiliary cache to store Prediction error.

The implementation proposed by Keerthi, employed a 1st order method to select the LMs of a fixed size working set (size equals 2). So it does not rely on an external quadratic solver to solve the optimization step as done by the version proposed by Joachims.

While the implemetion of the Platt's version that does not used any cache to temporarly store the prediction error is due  to the fact that could be intresting to analyze the effects of the cache on the training time.


Although it might be interesting to further analyze both:
- The impact of the cache in the SMO version proposed by Platt.
- The performance of the Keerthi's SMO with respect to th Joachims' SMO  that use a working set with size equals 2.
These considerations will not be taken into account in the continuation of the project. 

The folder [Algorithms](https://github.com/giovannivignocchi/Sequential-Minimal-Optimization-for-SVM/tree/master/Algorithm) collects all the implementations cited above.

- Platt's version without Error Cache
- Platt's version with Error Cache
- Keerthi's version (1st order method)
- [Joachims](https://github.com/giovannivignocchi/Sequential-Minimal-Optimization-for-SVM/blob/master/Algorithm/Joachims/Jsmo.m) (1st order method)
- [Fan Chen and Lin](https://github.com/giovannivignocchi/Sequential-Minimal-Optimization-for-SVM/blob/master/Algorithm/Fan%20Chen%20and%20Lin/FCLsmo.m) (2nd order method) 

## Testing and analysis
The last part of the project consisted in comparing the training of the various implementations on different dataset.

## First series of tests
A first set of tests was carried out on three different (bidimensional) dataset that were specially generated to test how the
implementations approximate well known functions.

A sample of the artificial dataset used, is shown in the images below:


   CORNERS DATASET
<p align="left">
<img width="308" alt="corners" src="https://user-images.githubusercontent.com/32396630/50488895-ff3f5b80-0a04-11e9-8b38-b88c7de53a4a.PNG">

 LINEARLY AND NON LINERLY SEPARABLE DATASET

<img width="687" alt="mosaico" src="https://user-images.githubusercontent.com/32396630/50489264-fa7ba700-0a06-11e9-8595-23353999a8de.png">


The results of these tests are collected in the folder [TEST RESULT](https://github.com/giovannivignocchi/Sequential-Minimal-Optimization-for-SVM/tree/master/TEST/on%20artificial%20dataset/TEST%20RESULTS). For each implementations different statistics are collected in the file stat. Among these:

- The number of iteration performed
- Training and prediction time
- Avg number of kernel evaluation for iteration
- Number of support vector

To check the results obtained, another model is trained using the matlab function _fitcsvm_ and it is used as a baseline to check if the other models are consistent with it.

This series of test is performed using a slightly different version of the algorithms implemented in the final version of the project.
The way in which they were implemented led me to analyze how the Lagrange multipliers changes during the iterations of the algorithms.
This approach was no more feasible as soon as the size of the dataset grows, hence this feature was deleted in the implementations used in the following test.

## Second series of tests

A second series of tests is focused on four datasets of more substantial size and with a number of featues greater than two.
This tests are executed only for the SMO implentation proposed by [Joachims](https://github.com/giovannivignocchi/Sequential-Minimal-Optimization-for-SVM/tree/master/Algorithm/Joachims) (both with a working set of size 4 and 6) and  [Fan, Chem, Lin](https://github.com/giovannivignocchi/Sequential-Minimal-Optimization-for-SVM/tree/master/Algorithm/Fan%20Chen%20and%20Lin).

The datasets used for this series of test are available in the [Dataset](https://github.com/giovannivignocchi/Sequential-Minimal-Optimization-for-SVM/tree/master/Dataset/Data) folder. 

- diabetes (8 features, 576 instances)
- ringnorm (20 features, 6500 instances)
- magic (10 features, 17118 instances)
- cod-rna (8 features, 59535 instances)

In order to test how the different implementations perform in response to changes in parameters, a grid search over the parameter _C_ and _sigma_ is carried out, for each of the data set. 
Based on the size of the dataset under observation, a different number of change in parameters are applied during grid search.
The models generated are tested on the relative tests set. However this should be considered as a bad practice (validating model using test set), the main porpouse of this project is not to generate good models, but to analyze the performance of the different implementations under analysis. 

As for the previous series of test several statistics are collected and the results stored in the folder [TEST RESULT](https://github.com/giovannivignocchi/Sequential-Minimal-Optimization-for-SVM/tree/master/TEST/on%20real%20dataset/TEST%20RESULTS). These staistics are then analyzed to show how the optimization methods respond to changes in the specified parameters.

Statistics collected:
- Total number of iteration
- Training time
- Total number of kernel evaluation
- Number of support vector generated
- Several metrics about the performance of the model generated (accuracy, sensitivity, specificity, precision, recall, f_measure, gmean).

The following pictures collect the results obtained during this series of tests.<\n>
For each dataset two different kind of graph:

- A 3D plot that individually summarize, the performace of each implementation during the grid search phase 
- A set of plots for each value of _C_ (used in the grid search) that compare how the differenet implementations perform as _sigma_ changes

###                                                      DIABETES 
Fan Chen and Lin
![flccomplessivo](https://user-images.githubusercontent.com/32396630/51791861-05625c80-21aa-11e9-80e3-4301430de884.jpg)

Joachims with working set size equal 4
![iterj4](https://user-images.githubusercontent.com/32396630/51791963-5e7ec000-21ab-11e9-9a11-97593fcff93e.jpg)

Joachims with working set size equal 6
![j6complessivo](https://user-images.githubusercontent.com/32396630/51792037-6db23d80-21ac-11e9-93d2-5bd00cc19ae1.jpg)

The following graphs compare how the differenet implementations perform as _sigma_ changes

![legnd](https://user-images.githubusercontent.com/32396630/51909976-2dcaa080-23ce-11e9-8b45-e3c2f1f44a74.jpg)

                                                         C = 2^-5
![diab1](https://user-images.githubusercontent.com/32396630/51906444-6796a980-23c4-11e9-9f6b-391661763e64.jpg)
<pre>                                                    C = 2^-3 </pre>
![diab2](https://user-images.githubusercontent.com/32396630/51906447-6796a980-23c4-11e9-9aa7-ea162a46c727.jpg)   
<pre>                                                    C = 2  </pre>
![diab3](https://user-images.githubusercontent.com/32396630/51906449-682f4000-23c4-11e9-8649-ca94b19c0518.jpg)
<pre>                                                    C = 2^7  </pre>                                                          
![diab4](https://user-images.githubusercontent.com/32396630/51906450-682f4000-23c4-11e9-80ef-61ee3f7db269.jpg)
<pre>                                                    C = 2^9  </pre>                                                          
![diab5](https://user-images.githubusercontent.com/32396630/51906452-682f4000-23c4-11e9-95ec-cbd4834e529d.jpg)
<pre>                                                    C = 2^15  </pre>                                                          
![diab6](https://user-images.githubusercontent.com/32396630/51906455-682f4000-23c4-11e9-917e-06f2d81eacef.jpg)

###                                                      RINGNORM
Fan Chen and Lin
![fclcomplete](https://user-images.githubusercontent.com/32396630/51843395-26928c80-2313-11e9-8396-32a6712e0219.jpg)

Joachims with working set size equal 4
![j4complessivo](https://user-images.githubusercontent.com/32396630/51843397-285c5000-2313-11e9-874f-99940b03f7cf.jpg)

Joachims with working set size equal 6
![j6complessivo](https://user-images.githubusercontent.com/32396630/51843398-285c5000-2313-11e9-9e89-e371c6ea4c04.jpg)

                                                         C = 2^-5
![ring1](https://user-images.githubusercontent.com/32396630/51906430-66fe1300-23c4-11e9-8438-7332fbc118c2.jpg)
<pre>                                                    C = 2^-3 </pre>
![ring2](https://user-images.githubusercontent.com/32396630/51906434-66fe1300-23c4-11e9-8799-9e9ff7e422f3.jpg)
<pre>                                                    C = 2  </pre>
![ring3](https://user-images.githubusercontent.com/32396630/51906437-66fe1300-23c4-11e9-9712-37ee92c3440d.jpg)
<pre>                                                    C = 2^7  </pre>           
![ring4](https://user-images.githubusercontent.com/32396630/51906439-6796a980-23c4-11e9-90cf-feb56df5cfac.jpg)
<pre>                                                    C = 2^9  </pre>                          
![ring5](https://user-images.githubusercontent.com/32396630/51906441-6796a980-23c4-11e9-95d8-42da8b43fa87.jpg)

###                                                         MAGIC
Fan Chen and Lin
![complessivofcl](https://user-images.githubusercontent.com/32396630/51862186-54d99180-233e-11e9-85eb-5d94a1068e9d.jpg)

Joachims with working set size equal 4
![complessivoj4](https://user-images.githubusercontent.com/32396630/51862187-55722800-233e-11e9-9c6e-2f1db1adb9d4.jpg)

Joachims with working set size equal 6
![complessivoj6](https://user-images.githubusercontent.com/32396630/51862189-55722800-233e-11e9-9747-bf5e5681a439.jpg)

<pre>                                                    C = 2^-3 </pre>
![mag1](https://user-images.githubusercontent.com/32396630/51906425-66657c80-23c4-11e9-95c8-5a03d9eb0d30.jpg)
<pre>                                                    C = 2  </pre>
![magic2](https://user-images.githubusercontent.com/32396630/51906428-66657c80-23c4-11e9-8e3b-42a73b7711d6.jpg)
<pre>                                                    C = 2^9  </pre>                       
![mag3](https://user-images.githubusercontent.com/32396630/51906427-66657c80-23c4-11e9-9d43-155f6507ad23.jpg)


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
