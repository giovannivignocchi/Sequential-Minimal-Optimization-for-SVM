## Sequential minimal optimization

The main purpose of the project is to analyze the optimization methods, currently emplyed to perform 
the training of the support vector machines (SVM).

The first part of the project consisted in studing the three following paper :

1. Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines,   [Platt 1998](http://www.microsoft.com/en-us/research/publication/sequential-minimal-optimization-a-fast-algorithm-for-training-support-vector-machines/)
2. Making Large-Scale SVM Learning Practical,   [Joachims 1998](http://www.cs.cornell.edu/people/tj/publications/joachims_99a.pdf)
3. Working Set Selection Using Second Order Information for Training Support Vector Machines,   [Fan, Chem, Lin 2005](www.jmlr.org/papers/volume6/fan05a/fan05a.pdf)

The first arictle discusses the sequential minimal optimization method (SMO) in its first version proposed by Platt, whereas in the second 
and the third, the authors proposed two different modifications of the SMO that employ, respectively, a 1st order and 2nd order method 
to select the Lagrange multipliers that composed the working set

The second stage of the project consisted in the implementation using Matlab of the algorithm proposed in the papers.

For the sake of completeness two more version of the SMO were implemented
1. A modified SMO proposed by Keerthi in "Improvements to Platt's SMO algorithm for SVM Classifier Design" 
2. The original version proposed by Platt without using an auxiliary cache to store Prediction error.

This choice is due to the fact that the implementation proposed by keerthi employed a 1st order method to select the LMs that composed the
working set as proposed by Joachims (article nr. 2), but since he used a working set that is always composed by only 2 LMs, it does not 
rely on an external quadratic solver to solve the optimization step as done by the version proposed by Joachims.

The folder Algorithms collects all the implementations cited above.

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

This series f test is performed using a slightly different version of the algorithms implemented 



## Contributing


## Authors

    Giovanni Vignocchi

This project has been carried out in the context of the course of Nonlinear optimization I have attended in Politecnico di Milano.
