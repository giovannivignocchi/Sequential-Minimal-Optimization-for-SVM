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
implementations approximate well known functions.




corners
<p align="left">
  <img width="308" alt="corners" src="https://user-images.githubusercontent.com/32396630/50488895-ff3f5b80-0a04-11e9-8b38-b88c7de53a4a.PNG">
</p>
Linerly separable polynomial function
<p align="right">
<img width="311" alt="polyls" src="https://user-images.githubusercontent.com/32396630/50488896-ffd7f200-0a04-11e9-8177-e1e80341089a.PNG">
</p>




Non linerly separable polynomial function
<img width="313" alt="polynls" src="https://user-images.githubusercontent.com/32396630/50488897-ffd7f200-0a04-11e9-84f8-542ede64caf6.PNG">

Linerly separable circle
<img width="313" alt="circlels" src="https://user-images.githubusercontent.com/32396630/50488899-ffd7f200-0a04-11e9-8b7b-4cb8acb55370.PNG">

Non linerly separable circle
<img width="313" alt="circlenls" src="https://user-images.githubusercontent.com/32396630/50488900-ffd7f200-0a04-11e9-9a00-613d76eff642.PNG">




## Contributing


## Authors

    Giovanni Vignocchi

This project has been carried out in the context of the course of Nonlinear optimization I have attended in Politecnico di Milano.
