## Welcome!

K-means clustering is a well-known technique for classifying a data set into "clusters". There are k clusters (a parameter of the algorithm), hence the name. Each point in the final result is closest to the center of its own cluster than the center of any other cluster. There are lots of materials on the internet which discuss it. You don't need to understand the purpose or the mathematics of it to complete the assignment. It is a good topic to be familiar with, nonetheless, so I suggest additional reading. Starting with the wikipedia articleLinks to an external site. is good.

The naive algorithmLinks to an external site. is sufficient for many data sets and that is what we'll be using in this assignment. It goes like this:

Choose k initial points (seeds); these are the initial cluster centers.
Ask each element of the set to choose the cluster center it is closest to and become a member of that cluster.
Recompute the cluster centers based on the new membership from step #2.
If step #3 did not change any clusters, then we've converged, otherwise go back to step #2 for another "generation".
In addition, we usually put an iteration limit on the number of generations, say 300, and get out if we don't converge by then.

As an example, for the graphic above, we ran the 10-means clustering algorithm on the X11 color set, picked 10 of the colors at random as seeds, then proceeded for 13 generations until it converged to the clusters displayed in the graphic.

The code for the example is included in the provided files and will be the basis for your submission in this assignment. If you download it (guaranteed to work on CS1) and run: make run_sequential.

it will build the sequential version of the k-means algorithm, run it, and produce an html file that will look similar to the example if you pull it up in a web browser. Note that different seeds can result in a different result--there is not typically a unique clustering convergence. So, in the example's case, the results will usually be a little different each time depending on the random seeds chosen by the algorithm.

Project and Requirements
Your assignment is to take the example program and modify it so that the K-means clustering algorithm is run in parallel using MPI. Please see the Provided Files section below to explain the details of what is provided and what is required. Your MPI implementation must use at least some collective routines (all Send/Recv would be unacceptable).

You must provide an abstract class mimicking the behavior of the example's KMeans<k,d> class. The class should work for any data set that is a collection of elements that are a vector of bytes of length d (just like the example). The example requires a rebuild to change k (since it is a template variable), but that's ok. In particular, it must work for the provided hw5.cpp. We will also test it with other color data sets and other subclasses and it must run as expected. Your program must work for any number of MPI processes from 1 to 32 (or more, if we had a run time environment with more nodes).

The code must be well-written C++ with appropriate style and embedded documentation similar to the example program.

There is an extra credit opportunity. You may write another client class and program that uses k-means clustering to categorize the MNIST handwritten digits data setLinks to an external site.. This must use your KMeansMPI class without modification from what is submitted for the main homework. You must get 80% or greater on the main homework to qualify for this extra credit. It is worth up to 25 extra homework points.

### Steps to run the Kmeans cluster program aganist mnist dataset

- The train.csv has the mnist data set
- I added the commands in the Makefile to run the Kmeans cluster with mnist data set
- Execute the following commands in order
1. make mnistclean
2. make mnist
3. make run_mnist
- Check the console output to see the clusters
