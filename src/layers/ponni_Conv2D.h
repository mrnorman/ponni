
#pragma once
// Included by ponni.h

namespace ponni {

  class Conv2D {
  public:
    
    bool static constexpr overwrite_input = true;
    bool static constexpr binop           = false; // Use two inputs?
    bool static constexpr save            = false;

    struct Params {
      int    num_inputs, num_outputs, in_rows, in_cols, in_filters, out_filters, out_rows, out_cols;
      int    kernel_size_row, kernel_size_col, stride_row, stride_col;
      bool   use_zero_padding_row, use_zero_padding_col;
      real3d weights;
    };

    Params params;

    Conv2D() {}
    Conv2D( real3d const &weights , int in_cols , int in_rows , int in_filters,
               int stride_row=1 , int stride_col=1 , bool use_zero_padding_row=false , bool use_zero_padding_col=false ) {
      init(weights,in_cols,in_rows,in_filters,strice_row,strice_col,use_zero_padding_row,use_zero_padding_col);
    }


    void init( real3d const &weights , int in_cols , int in_rows , int in_filters,
               int stride_row=1 , int stride_col=1 , bool use_zero_padding_row=false , bool use_zero_padding_col=false ) {
      if ( ! weights.initialized() ) yakl::yakl_throw("ERROR: Conv2D weights vector not initialized");
      params.weights              = weights;
      params.in_rows              = in_rows;
      params.in_cols              = in_cols;
      params.in_filters           = in_filters;
      params.kernel_size_row      = weights.dimension[0];
      params.kernel_size_col      = weights.dimension[1];
      params.out_filters          = weights.dimension[2];
      params.strice_row           = stride_row;
      params.strice_col           = stride_col;
      params.use_zero_padding_row = use_zero_padding_row;
      params.use_zero_padding_col = use_zero_padding_col;
      if (use_zero_padding_row) {
        params.out_rows = (int) std::ceil( (real) in_rows / (real) stride_row );
      } else {
        params.out_rows = (int) std::ceil( (real) (in_rows-kernel_size_row+1) / (real) stride_row );
      }
      if (use_zero_padding_col) {
        params.out_cols = (int) std::ceil( (real) in_cols / (real) stride_col );
      } else {
        params.out_cols = (int) std::ceil( (real) (in_cols-kernel_size_col+1) / (real) stride_col );
      }
      params.num_inputs  = params.in_rows  * params.in_cols  * params.in_filters ;
      params.num_outputs = params.out_rows * params.out_cols * params.out_filters;
    }


    char const * get_label         () const { return "Conv2D"; }
    YAKL_INLINE static int get_num_inputs (Params const &params_in) { return params_in.num_inputs ; }
    YAKL_INLINE static int get_num_outputs(Params const &params_in) { return params_in.num_outputs; }


    YAKL_INLINE void compute_all_outputs(real2d const &input_in, real2d const &output_out, int ibatch) const {
      int num_batches = input_in.dimension[1];
      real4d input  = input_in  .reshape<4>(in_rows ,in_cols ,in_filters ,num_batches);
      real4d output = output_out.reshape<4>(out_rows,out_cols,out_filters,num_batches);
      int in_row_beg, in_row_end, out_row_beg, out_row_end;
      if (use_zero_padding_row) { in_row_beg = 0; in_row_end = in_rows-1              ; }
      else                      { in_row_beg = 0; in_row_end = in_rows-kernel_size_row; }
      if (use_zero_padding_col) { in_col_beg = 0; in_col_end = in_cols-1              ; }
      else                      { in_col_beg = 0; in_col_end = in_cols-kernel_size_col; }
      for (int ifilt_out = 0; ifilt_out < out_filters; ifilt_out++) {
        for (int irow_in = 0; irow_in < in_rows; irow_in += stride_row) {
          for (int icol_in = 0; icol_in < in_cols; icol_in += stride_col) {
            real tmp = 0;
            for (int k_row = 0; k_row < kernel_size_row; k_row++) {
              for (int k_col = 0; k_col < kernel_size_col; k_col++) {
                for (int ifilt_in = 0; ifilt_in < in_filters; ifilt_in++) {
                  tmp += weights(k_row,k_col,ifilt_out) * input(irow_in+k_row,icol_in+k_col,ifilt_in,ibatch);
                }
              }
            }
            int irow_out  = ???;
            int icol_out  = ???;
            output_out(irow_out,icol_out,ifilt_out,ibatch) = tmp;
          }
        }
      }
    }


    void print_verbose() const {
      // TODO: Prit verbose stuff here
    }


    void validate() const {
      if (! params.weights.initialized()) yakl::yakl_throw("ERROR: weights not initialized");
    }

  };

}


