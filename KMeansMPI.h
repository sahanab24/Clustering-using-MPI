#pragma once

#include <mpi.h>
#include <sys/types.h>

#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

template<int k, int d>
class KMeansMPI {
public:
    // some type definitions to make things easier
    typedef std::array <u_char, d> Element;
    static constexpr int ROOT = 0;

    // debugging
    const bool VERBOSE = false;  // set to true for debugging output
#define V(stuff)         \
  if (VERBOSE) {         \
    using namespace std; \
    stuff                \
  }

    /**
     * The algorithm constructs k clusters and attempts to populate them with like
     * neighbors. This inner class, Cluster, holds each cluster's centroid (mean)
     * and the index of the objects belonging to this cluster.
     */
    struct Cluster {
        Element
                centroid;  // the current center (mean) of the elements in the cluster
        std::vector<int> elements;

        // equality is just the centroids, regarless of elements
        friend bool operator==(const Cluster &left, const Cluster &right) {
            return left.centroid ==
                   right.centroid;  // equality means the same centroid, regardless of
            // elements
        }
    };

    typedef std::array <Cluster, k> Clusters;
    const int MAX_FIT_STEPS = 300;

    void fit(Element *data, int data_n) {
        elements = data;
        n = data_n;
        fitWork(ROOT);
    }

    void fitWork(int rank) {

        std::cout << "Rank is at " << rank << std::endl;
        int numprocs;
        MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
        std::cout << "Numprocs is at " << numprocs << std::endl;

        int num_elements = n;
        MPI_Bcast(&num_elements, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
        MPI_Bcast(&num_elements, 1, MPI_INT, rank, MPI_COMM_WORLD);
        /* root process
          - initialize clusters
          - send clusters to all other processes
          - send chunks of data to all other processes
        */
        int chunk_size = num_elements / numprocs;
        int *sendcounts = new int[numprocs];
        int *displs = new int[numprocs];

        for (int i = 0; i < numprocs; i++) {
            displs[i] = i * (d * chunk_size);
            int last_chunk_size = chunk_size;
            if (i == numprocs - 1) {
                last_chunk_size = num_elements - (numprocs - 1) * chunk_size;
            }
            sendcounts[i] = d * last_chunk_size;
        }

        int recv_count = (rank == numprocs - 1)
                         ? num_elements - (numprocs - 1) * chunk_size
                         : chunk_size;

        // Divide the elements array to multiple chunks for non-root processes
        Element *local_elements = new Element[recv_count];
        MPI_Scatterv(elements, sendcounts, displs, MPI_UNSIGNED_CHAR,
                     local_elements, d * recv_count, MPI_UNSIGNED_CHAR,
                     ROOT, MPI_COMM_WORLD);

        Element *centroids = new Element[k];
        Element *oldCentroids = new Element[k];
        if (rank == ROOT) {
            reseedClusters();
            // collect centroids into an element array
            for (int i = 0; i < k; i++) {
                centroids[i] = clusters[i].centroid;
            }
        }

        // The Updated clusters from the root process
        Clusters local_clusters_root;
        // The Updated clusters from the non-root process
        Clusters local_clusters_child;

        // Boolean is to check whether centroids has been converged or not
        bool converged = false;
        int i = 0;
        while (i++ < MAX_FIT_STEPS && !converged) {
            if (rank == ROOT) {
                for (int i = 1; i < numprocs; i++) {
                    // Send centroids to all non-root processes
                    MPI_Ssend(centroids, k * d, MPI_UNSIGNED_CHAR,
                              i, 0, MPI_COMM_WORLD);
                }
                local_clusters_root = updateClusters(centroids, local_elements, recv_count, displs[rank]);
            } else {
                // Receive centroids from root process
                MPI_Recv(centroids, k * d, MPI_UNSIGNED_CHAR,
                         ROOT, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                local_clusters_child = updateClusters(centroids, local_elements, recv_count, displs[rank]);

                // Here we are sending each cluster and elements indexes information
                for (int i = 0; i < k; i++) {
                    int *elements_size = new int[1];
                    elements_size[0] = local_clusters_child[i].elements.size();
                    MPI_Ssend(elements_size, 1, MPI_INT,
                              ROOT, 0, MPI_COMM_WORLD);
                    int *elements_index = new int[elements_size[0]];
                    for (int j = 0; j < elements_size[0]; j++) {
                        elements_index[j] = local_clusters_child[i].elements[j];
                    }
                    MPI_Ssend(elements_index, elements_size[0], MPI_INT,
                              ROOT, 0, MPI_COMM_WORLD);
                    delete[] elements_size;
                    delete[] elements_index;
                }
            }

            // Calculate the new centroids
            if (rank == ROOT) {
                double new_sum[k][d];
                double new_sum_count[k][d];
                for (int i = 0; i < k; i++) {
                    for (int j = 0; j < d; j++) {
                        new_sum[i][j] = 0;
                        new_sum_count[i][j] = 0;
                    }
                }
                for (int i = 0; i < k; i++) {
                    for (std::vector<int>::size_type j = 0; j < local_clusters_root[i].elements.size(); j++) {
                        for (int jk = 0; jk < d; jk++) {
                            new_sum[i][jk] += static_cast<double>(elements[local_clusters_root[i].elements[j]][jk]);
                            new_sum_count[i][jk]++;
                        }
                    }
                    for (int cp = 1; cp < numprocs; cp++) {
                        int *elements_size = new int[1];
                        MPI_Recv(elements_size, 1, MPI_INT,
                                 cp, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        int *elements_index = new int[elements_size[0]];
                        MPI_Recv(elements_index, elements_size[0], MPI_INT,
                                 cp, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        for (int j = 0; j < elements_size[0]; j++) {
                            for (int jk = 0; jk < d; jk++) {
                                new_sum[i][jk] += static_cast<double>(elements[elements_index[j]][jk]);
                                new_sum_count[i][jk]++;
                            }
                        }
                        delete[] elements_size;
                        delete[] elements_index;
                    }
                }
                for (int j = 0; j < k; j++) {
                    for (int jk = 0; jk < d; jk++) {
                        oldCentroids[j][jk] = centroids[j][jk];
                    }
                }
                for (int j = 0; j < k; j++) {
                    for (int jk = 0; jk < d; jk++) {
                        centroids[j][jk] = static_cast<u_char>(new_sum[j][jk] / new_sum_count[j][jk]);
                    }
                }
                converged = true;
                for (int j = 0; j < k; j++) {
                    for (int jk = 0; jk < d; jk++) {
                        if (oldCentroids[j][jk] != centroids[j][jk]) {
                            converged = false;
                            break;
                        }
                    }
                }

                int *converged_status = new int[1];
                converged_status[0] = converged ? 1 : 0;
                for (int ik = 1; ik < numprocs; ik++) {
                    MPI_Ssend(converged_status, 1, MPI_INT,
                              ik, 0, MPI_COMM_WORLD);
                }
                delete[] converged_status;
            }

            if (rank != ROOT) {
                int *converged_status_child = new int[1];
                MPI_Recv(converged_status_child, 100, MPI_INT,
                         ROOT, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (converged_status_child[0] == 1) {
                    converged = true;
                }
                delete[] converged_status_child;
            }
        }

        if (rank == ROOT) {
            // assign elements to clusters
            // update clusters centroids
            for (int i = 0; i < k; i++) {
                clusters[i].centroid[0] = centroids[i][0];
                clusters[i].centroid[1] = centroids[i][1];
                clusters[i].centroid[2] = centroids[i][2];
                clusters[i].elements.clear();
            }
            std::vector <std::array<double, k>> dist =
                    updateDistances(centroids, elements, n);
            for (int i = 0; i < n; i++) {
                int min = 0;
                for (int j = 1; j < k; j++) {
                    if (dist[i][j] < dist[i][min]) {
                        min = j;
                    }
                }
                clusters[min].elements.push_back(i);
            }
        }

        delete[] sendcounts;
        delete[] displs;
        delete[] local_elements;
        delete[] centroids;
        delete[] oldCentroids;
    }

    /**
     * Recalculate the current clusters based on the new distances shown in
     * this->dist.
     */
    Clusters updateClusters(Element *centroids, Element *local_elements,
                            int local_elements_size, int buffer) {
        Clusters local_clusters;
        for (int j = 0; j < k; j++) {
            local_clusters[j].centroid[0] = centroids[j][0];
            local_clusters[j].centroid[1] = centroids[j][1];
            local_clusters[j].centroid[2] = centroids[j][2];
            local_clusters[j].elements.clear();
        }
        std::vector <std::array<double, k>> dist =
                updateDistances(centroids, local_elements, local_elements_size);
        for (int i = 0; i < local_elements_size; i++) {
            int min = 0;
            for (int j = 1; j < k; j++) {
                if (dist[i][j] < dist[i][min]) {
                    min = j;
                }
            }
            local_clusters[min].elements.push_back(i + (buffer / d));
        }
        return local_clusters;
    }

    /**
     * Calculate the distance from each element to each centroid.
     * Place into this->dist which is a k-vector of distances from each element to
     * the kth centroid.
     */
    virtual std::vector <std::array<double, k>> updateDistances(
            const Element *centroids, const Element *local_elements,
            int local_elements_size) const {
        int local_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &local_rank);
        std::vector <std::array<double, k>> dist;
        dist.resize(local_elements_size);

        for (int i = 0; i < local_elements_size; i++) {
            for (int j = 0; j < k; j++) {
                dist[i][j] = distance(centroids[j], local_elements[i]);
            }
        }
        return dist;
    }

    /**
     * Expose the clusters to the client readonly.
     * @return clusters from latest call to fit()
     */
    Clusters getClusters() { return clusters; }

protected:
    const Element *elements{
            nullptr};  // set of elements to classify into k categories
    // (supplied to latest call to fit())
    int n = 0;     // number of elements in this->elements

    Clusters clusters;  // k clusters resulting from latest call to fit()

    std::vector <std::array<double, k>>
            dist;  // dist[i][j] is the distance from
    // elements[i] to clusters[j].centroid

    virtual double distance(const Element &a, const Element &b) const = 0;

    /**
     * Get the initial cluster centroids.
     * Default implementation here is to just pick k elements at random from the
     * element set
     * @return list of clusters made by using k random elements as the initial
     * centroids
     */
    virtual void reseedClusters() {
        std::vector<int> seeds;
        std::vector<int> candidates(n);
        std::iota(candidates.begin(), candidates.end(), 0);
        auto random = std::mt19937{std::random_device{}()};
        // Note that we need C++20 for std::sample
        std::sample(candidates.begin(), candidates.end(), back_inserter(seeds), k,
                    random);
        for (int i = 0; i < k; i++) {
            clusters[i].centroid = elements[seeds[i]];
            clusters[i].elements.clear();
        }
    }
};
