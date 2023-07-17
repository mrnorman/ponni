
#pragma once

namespace ponni {


  inline yakl::Array<int,1,yakl::memDevice> get_shuffled_indices(int size, size_t rand_seed = 0) {
    std::random_device  rd;
    std::mt19937        gen {rd()};
    if (rand_seed > 0) gen.seed(rand_seed);
    yakl::Array<int,1,yakl::memHost> shuffled_indices_host("shuffled_indices_host",size);
    for (int i=0; i < size; i++) { shuffled_indices_host(i) = i; }
    std::shuffle( shuffled_indices_host.begin() , shuffled_indices_host.end() , gen );
    auto shuffled_indices = shuffled_indices_host.createDeviceObject();
    shuffled_indices_host.deep_copy_to(shuffled_indices);
    return shuffled_indices;
  }


  template <class real, int N>
  inline void shuffle_data( yakl::Array<real,N,yakl::memDevice> arr_in          ,
                            int shuffle_dim                                     ,
                            yakl::Array<int,1,yakl::memDevice> shuffled_indices ) {
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;
    // Reshape into 3-D array with shuffle_dim as dim2
    int dim1 = 1;   for (int i=0; i < shuffle_dim; i++) { dim1 *= arr_in.extent(i); }
    int dim2 = arr_in.extent(shuffle_dim);
    int dim3 = 1;   for (int i=shuffle_dim+1; i < N; i++) { dim3 *= arr_in.extent(i); }
    auto arr = arr_in.reshape(dim1,dim2,dim3);
    if (! shuffled_indices.initialized()) yakl::yakl_throw("ERROR: provided shuffled indices not allocated");
    if (shuffled_indices.size() != dim2) yakl::yakl_throw("ERROR: provided shuffled indices are the wrong size");
    // Perform shuffle into temporary array
    yakl::Array<real,3,yakl::memDevice> arr_tmp("arr_tmp",dim1,dim2,dim3);
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(dim1,dim2,dim3) , YAKL_LAMBDA (int i1, int i2, int i3) {
      arr_tmp(i1,i2,i3) = arr(i1,shuffled_indices(i2),i3);
    });
    // Copy shuffled array to original array
    arr_tmp.deep_copy_to(arr);
  }


  template <class real, int N>
  inline yakl::Array<int,1,yakl::memDevice> shuffle_data( yakl::Array<real,N,yakl::memDevice> arr_in ,
                                                          int    shuffle_dim                         ,
                                                          size_t rand_seed = 0                       ) {
    auto shuffled_indices = get_shuffled_indices(arr_in.extent(shuffle_dim),rand_seed);
    shuffle_data( arr_in , shuffle_dim , shuffled_indices );
    return shuffled_indices;
  }


  template <class real>
  inline void shuffle_losses( yakl::Array<real,2,yakl::memDevice,yakl::styleC> loss  ,
                              size_t rand_seed = 0                                   ) {
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;
    typedef yakl::Array<int ,1,yakl::memHost  ,yakl::styleC> intHost1d;
    typedef yakl::Array<real,2,yakl::memDevice,yakl::styleC> real2d;
    int num_ensembles = loss.extent(0);
    int num_samples   = loss.extent(1);
    // Create a shuffled set of indices on the CPU
    std::random_device  rd;
    std::mt19937        gen {rd()};
    if (rand_seed > 0) gen.seed(rand_seed);
    intHost1d shuffled_indices_host("shuffled_indices_host",num_samples);
    for (int i=0; i < num_samples; i++) { shuffled_indices_host(i) = i; }
    std::shuffle( shuffled_indices_host.begin() , shuffled_indices_host.end() , gen );
    // Copy shuffled indices to device
    auto shuffled_indices = shuffled_indices_host.createDeviceObject();
    shuffled_indices_host.deep_copy_to(shuffled_indices);
    // Perform shuffle into temporary arrays
    real2d tmp_loss("tmp_loss",num_ensembles,num_samples);
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(num_ensembles,num_samples) , YAKL_LAMBDA (int ival, int isamp) {
      int isamp_sh = shuffled_indices(isamp);
      if (ival < num_ensembles ) tmp_loss(ival,isamp) = loss(ival,isamp_sh);
    });
    // Copy shuffled arrays to original arrays
    tmp_loss.deep_copy_to(loss);
  }

}

