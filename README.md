# Sequential minimal optimization

The main purpose of the project is to analyze the optimization methods, currently emplyed to perform 
the training of the support vector machines (SVM).

## The scope of the project

The first part of the project consisted in studing the three following paper :
1. Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines,   [Platt 1998](http://www.microsoft.com/en-us/research/publication/sequential-minimal-optimization-a-fast-algorithm-for-training-support-vector-machines/)
2. Making Large-Scale SVM Learning Practical,   [Joachims 1998](http://www.cs.cornell.edu/people/tj/publications/joachims_99a.pdf)
3. Working Set Selection Using Second Order Information for Training Support Vector Machines,   [Fan, Chem, Lin 2005](www.jmlr.org/papers/volume6/fan05a/fan05a.pdf)

The first arictle discusses the sequential minimal optimization method (SMO) in its first version proposed by Platt, whereas in the second 
and the third, the authors proposed two different modifications of the SMO that employ, respectively, a 1st order and 2nd order method 
to select the Lagrange multipliers that composed the working set

## Implementation
The second stage of the project consisted in the implementation using Matlab of the algorithm proposed in the papers.
For the sake of completeness two more version of the SMO were implemented
1. "Improvements to Platt's SMO algorithm for SVM Classifier Design",[Keerthi 2001](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.115.5266&rep=rep1&type=pdf)
2. The original version proposed by Platt without using an auxiliary cache to store Prediction error.

This choice is due to the fact that the implementation proposed by keerthi employed a 1st order method to select the LMs that composed the working set as proposed by Joachims (article nr. 2), but since he used a working set that is always composed by only 2 LMs, it does not rely on an external quadratic solver to solve the optimization step as done by the version proposed by Joachims.

While the choice to implemet the Platt's implementations that does not used any cache to temporarly store the prediction error is due  to the fact that could be intresting to analyze the effects of the cache on the training time.

The folder [Algorithms](https://github.com/giovannivignocchi/Sequential-Minimal-Optimization-for-SVM/tree/master/Algorithm) collects all the implementations cited above.

## Testing and analysis
The last part of the project consisted in comparing the training of the various implementations on different dataset.
<pre>
----------------------------------------------- FIRST SET OF TESTS -----------------------------------------------
</pre>
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

<pre>
---------------------------------------------- SECOND SET OF TESTS ----------------------------------------------
</pre>

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

The following pictures collect the results obtained during this series of tests:

### DIABETES 
Fan Chen and Lin
![flccomplessivo](https://user-images.githubusercontent.com/32396630/51791861-05625c80-21aa-11e9-80e3-4301430de884.jpg)

Joachims with working set size equal 4
![iterj4](https://user-images.githubusercontent.com/32396630/51791963-5e7ec000-21ab-11e9-9a11-97593fcff93e.jpg)

Joachims with working set size equal 6
![j6complessivo](https://user-images.githubusercontent.com/32396630/51792037-6db23d80-21ac-11e9-93d2-5bd00cc19ae1.jpg)

### RINGNORM
Fan Chen and Lin
![fclcomplete](https://user-images.githubusercontent.com/32396630/51843395-26928c80-2313-11e9-8396-32a6712e0219.jpg)

Joachims with working set size equal 4
![j4complessivo](https://user-images.githubusercontent.com/32396630/51843397-285c5000-2313-11e9-874f-99940b03f7cf.jpg)

Joachims with working set size equal 6
![j6complessivo](https://user-images.githubusercontent.com/32396630/51843398-285c5000-2313-11e9-9e89-e371c6ea4c04.jpg)

### MAGIC
Fan Chen and Lin
![complessivofcl](https://user-images.githubusercontent.com/32396630/51862186-54d99180-233e-11e9-85eb-5d94a1068e9d.jpg)

Joachims with working set size equal 4
![complessivoj4](https://user-images.githubusercontent.com/32396630/51862187-55722800-233e-11e9-9c6e-2f1db1adb9d4.jpg)

Joachims with working set size equal 6
![complessivoj6](https://user-images.githubusercontent.com/32396630/51862189-55722800-233e-11e9-9747-bf5e5681a439.jpg)

During the grid search, I detected the presence of some parameter combinations (_C_ and _sigma_) that severely slow down the Joachims' implementations, to such an extent that they did not even reach convergence.

To further analyze the reason behind such a behaivour, I repeated the training phase using a different seed to shuffle the dataset.
What I noticed, was that the way in which the data were shuffled could have a strong impact on the speed of convergence for the Joachims implementation. On the other hand, the Fan Chen and Lin implementation does not suffer of the same issue.

*it is noteworthy that this issue is hidden in the graphs presented above, because in the points where this effect was visible I repeated the training with different shuffle of the data and plotted the statistics for the configuration where I obtained convergence.  

The following graphs show how for a lot of parameters combinations was possible to identify a _seed_ (use to shuffle the data), that causes the Joachims' implementation to do not converge.




## Bibliography

1. Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines,   [Platt 1998](http://www.microsoft.com/en-us/research/publication/sequential-minimal-optimization-a-fast-algorithm-for-training-support-vector-machines/)
2. Making Large-Scale SVM Learning Practical,   [Joachims 1998](http://www.cs.cornell.edu/people/tj/publications/joachims_99a.pdf)
3. Working Set Selection Using Second Order Information for Training Support Vector Machines,   [Fan, Chem, Lin 2005](www.jmlr.org/papers/volume6/fan05a/fan05a.pdf)
4. Improvements to Platt's SMO algorithm for SVM Classifier Design [Keerthi 2001](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.115.5266&rep=rep1&type=pdf)
5. On the Convergence of the Decomposition Method for Support Vector Machines [Chih-Jen Lin](https://www.csie.ntu.edu.tw/~cjlin/papers/conv.pdf)
6. [_SVMlight_ library documentation](http://svmlight.joachims.org/)
7. Data Mining and Analysis: Fundamental Concepts and Algorithms (Wagner Meira, Mohammed J. Zaki)
8. A Practical Guide to Support Vector Classification [Chih-Wei Hsu, Chih-Chung Chang, and Chih-Jen Lin](https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf)

## Authors

    Giovanni Vignocchi

This project has been carried out in the context of the course of Nonlinear optimization I have attended in Politecnico di Milano.
