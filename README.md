# Sequential minimal optimization

The main purpose of the project is to analyze the optimization methods, currently emplyed to perform 
the training of the support vector machines (SVM).

### The scope of the project

The first part of the project consisted in studing the three following paper :
1. Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines,   [Platt 1998](http://www.microsoft.com/en-us/research/publication/sequential-minimal-optimization-a-fast-algorithm-for-training-support-vector-machines/)
2. Making Large-Scale SVM Learning Practical,   [Joachims 1998](http://www.cs.cornell.edu/people/tj/publications/joachims_99a.pdf)
3. Working Set Selection Using Second Order Information for Training Support Vector Machines,   [Fan, Chem, Lin 2005](www.jmlr.org/papers/volume6/fan05a/fan05a.pdf)

The first arictle discusses the sequential minimal optimization method (SMO) in its first version proposed by Platt, whereas in the second 
and the third, the authors proposed two different modifications of the SMO that employ, respectively, a 1st order and 2nd order method 
to select the Lagrange multipliers that composed the working set

### Implementation
The second stage of the project consisted in the implementation using Matlab of the algorithm proposed in the papers.
For the sake of completeness two more version of the SMO were implemented
1. "Improvements to Platt's SMO algorithm for SVM Classifier Design",[Keerthi 2001](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.115.5266&rep=rep1&type=pdf)
2. The original version proposed by Platt without using an auxiliary cache to store Prediction error.

This choice is due to the fact that the implementation proposed by keerthi employed a 1st order method to select the LMs that composed the working set as proposed by Joachims (article nr. 2), but since he used a working set that is always composed by only 2 LMs, it does not rely on an external quadratic solver to solve the optimization step as done by the version proposed by Joachims.

While the choice to implemet the Platt's implementations that does not used any cache to temporarly store the prediction error is due  to the fact that could be intresting to analyze the effects of the cache on the training time.

The folder [Algorithms](https://github.com/giovannivignocchi/Sequential-Minimal-Optimization-for-SVM/tree/master/Algorithm) collects all the implementations cited above.

### Testing and analysis
The last part of the project consisted in comparing the training of the various implementations on different dataset.
A first set of tests was carried out on three different (bidimensional) dataset that were specially generated to test how the
implementations approximate well known functions. A sample of the artificial dataset used, is shown in the images below:


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

To validate the results, another model is trained using the matlab function _fitcsvm_ and it is used as a baseline to check if the other models are consistent with it.

This series of test is performed using a slightly different version of the algorithms implemented in the final version of the project.
The way in which they were implemented led me to analyze how the Lagrange multipliers changes during the iterations of the algorithms.
This approach was no more feasible as soon as the size of the dataset grows, since a big matrix needed to be stored in memory.

A second series of tests is focused on four datasets of more substantial size and with a number of featues greater than two.
This tests are executed only for the SMO implentation proposed by [Joachims](https://github.com/giovannivignocchi/Sequential-Minimal-Optimization-for-SVM/tree/master/Algorithm/Joachims) (both with a working set of size 4 and 6) and  [Fan, Chem, Lin](https://github.com/giovannivignocchi/Sequential-Minimal-Optimization-for-SVM/tree/master/Algorithm/Fan%20Chen%20and%20Lin).

The datasets used for this series of test are available in the [Dataset](https://github.com/giovannivignocchi/Sequential-Minimal-Optimization-for-SVM/tree/master/Dataset/Data) folder. 

- cod-rna (8 features, 59535 instances)
- diabetes (8 features, 576 instances)
- ringnorm (20 features, 6500 instances)
- magic (10 features, 17118 instances)

The training procedure is repeated, varying both the boxConstraint _C_ and the _sigma_ that control the gaussian kernel.
Based on the size of the dataset under observation, a different number of iteration is performed.
As for the previous series of test several statistics are collected and the results stored in the folder [TEST RESULT](https://github.com/giovannivignocchi/Sequential-Minimal-Optimization-for-SVM/tree/master/TEST/on%20real%20dataset/TEST%20RESULTS).


Note that, as the main scope of this project is to evaluate and compare the performances of this optimization methods, I do not investigate how the models generated work on the testing set. Even if the performance of the models are not investigated, I checked each model to be consistent with the one generated with the matlab function _fitcsvm_ (using the same parameter).


## Bibliography

1. Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines,   [Platt 1998](http://www.microsoft.com/en-us/research/publication/sequential-minimal-optimization-a-fast-algorithm-for-training-support-vector-machines/)
2. Making Large-Scale SVM Learning Practical,   [Joachims 1998](http://www.cs.cornell.edu/people/tj/publications/joachims_99a.pdf)
3. Working Set Selection Using Second Order Information for Training Support Vector Machines,   [Fan, Chem, Lin 2005](www.jmlr.org/papers/volume6/fan05a/fan05a.pdf)
4. Improvements to Platt's SMO algorithm for SVM Classifier Design [Keerthi 2001](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.115.5266&rep=rep1&type=pdf)
5. On the Convergence of the Decomposition Method for Support Vector Machines [Chih-Jen Lin](https://www.csie.ntu.edu.tw/~cjlin/papers/conv.pdf)
6. [_SVMlight_ library documentation](http://svmlight.joachims.org/)
7. Data Mining and Analysis: Fundamental Concepts and Algorithms (Wagner Meira, Mohammed J. Zaki)

## Authors

    Giovanni Vignocchi

This project has been carried out in the context of the course of Nonlinear optimization I have attended in Politecnico di Milano.
