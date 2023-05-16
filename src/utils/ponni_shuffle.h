
#pragma once

namespace ponni {

  template <class real>
  inline void shuffle_training_data( yakl::Array<real,2,yakl::memDevice,yakl::styleC> in  ,
                                     yakl::Array<real,2,yakl::memDevice,yakl::styleC> out ,
                                     size_t rand_seed = 0                                 ) {
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;
    typedef yakl::Array<int ,1,yakl::memHost  ,yakl::styleC> intHost1d;
    typedef yakl::Array<real,2,yakl::memDevice,yakl::styleC> real2d;
    int num_inputs  = in .extent(0);
    int num_outputs = out.extent(0);
    int num_samples = in .extent(1);
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
    real2d tmp_in ("tmp_in ",num_inputs ,num_samples);
    real2d tmp_out("tmp_out",num_outputs,num_samples);
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>( std::max(num_inputs,num_outputs) , num_samples ) ,
                                      YAKL_LAMBDA (int ival, int isamp) {
      int isamp_sh = shuffled_indices(isamp);
      if (ival < num_inputs ) tmp_in (ival,isamp) = in (ival,isamp_sh);
      if (ival < num_outputs) tmp_out(ival,isamp) = out(ival,isamp_sh);
    });
    // Copy shuffled arrays to original arrays
    tmp_in .deep_copy_to(in );
    tmp_out.deep_copy_to(out);
  }

}

