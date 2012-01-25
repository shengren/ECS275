#ifndef NEW_KNN_H_
#define NEW_KNN_H_

void new_knn_api(unsigned num_data, float *h_data_x, float *h_data_y, float *h_data_z,
                 unsigned num_queries, float *h_query_x, float *h_query_y, float *h_query_z,
                 int K, unsigned *& result);

#endif  // NEW_KNN_H_
