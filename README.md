# Intro

This is an example Spark application to demonstrate using Support Vector Machine (SVM)
to classify data. The particular technique used to find the maximum-margin hyper-plane is
subgradient descent (SGD) using the Pegasos algorithm (see references below).

You should most likely use the Apache Spark ML library for SVM techniques to classify data.
However, this project's purpose is to show simply, in a self-contained file/project, how
one may easily use SGD to learn a maximum-margin hyper-plane.

# Data Format

You must use a delimited file (e.g. comma, tab, space, etc...) as input. The delimiter can
be set when you run the program. Also, the first column/field must be represent the class
with the only either 1 or -1. An trivial example is shown below.

```cvs
-1, 0, 0
1, 10, 10
```

# HOWTO use

To use this library, use spark-submit as follows.

```text
/path/to/spark-submit \
 --class com.github.vangj.spark.svm.Sgd \
 --master <master-url> \
 --deploy-mode <deploy-mode> \
 /path/to/spark-svm-assembly-0.0.1-SNAPSHOT.jar \
 --T <number-of-iterations> \
 --k <number-of-samples> \
 --lambda <regularization-parameter> \
 --seed <seed-for-randomization> \
 --delim <delimiter-for-input-file> \
 --input <path-of-input-file> \
 --output <path-of-output-file
```

Notes on parameters.

* T is the number of iterations. Specify something like 400. You may get unacceptable classification results if T is too small.
* k is the number of samples taken at each iteration. Specify something less than or equal to your sample size.
* lambda is the regularization parameter (learning rate). Specify something between [0, 1].
* seed is used for the random number generator when sampling.
* delim is used to parse your input file.

# Building
This project depends on the following.

* Java v1.8
* Scala v2.11.8

You may use the following tools to build the project.

* SBT v0.13.13 
* Maven v3.3.9 

For SBT, type in the following.

```
sbt assembly
```

For Maven, type in the following.

```
mvn package
```

# References

* [Pegasos: Primal Estimated sub-GrAdient SOlver for SVM](http://www.ee.oulu.fi/research/imag/courses/Vedaldi/ShalevSiSr07.pdf)
* [Pegasos: Primal Estimated sub-GrAdient SOlver for SVM](http://ttic.uchicago.edu/~nati/Publications/PegasosMPB.pdf)
* [The Stochastic Gradient Descent for the Primal L1-SVM Optimization Revisited](http://www.ecmlpkdd2013.org/wp-content/uploads/2013/07/255.pdf)
* [Large-Scale Support Vector Machines: Algorithms and Theory](https://cseweb.ucsd.edu/~akmenon/ResearchExam.pdf)
* [scikit-learn](http://scikit-learn.org/stable/modules/sgd.html)
* [Support vector machine](https://en.wikipedia.org/wiki/Support_vector_machine)
* [Subgradient method](https://en.wikipedia.org/wiki/Subgradient_method)
* [Spark ML Linear Methods](https://spark.apache.org/docs/2.1.0/mllib-linear-methods.html)
