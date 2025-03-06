/** A hybrid Gauss-Seidel or Jacobi iteration on a part of matrix.
\param row_indices Indices to first element of matrix's each row.
\param column_indices Column index for each element. It makes 1-1 pair with \a values.
\param values Value for each element. It makes 1-1 pair with \a column_indices.
\param row First number of row processed (it terms of rows in matrix and two vectors).
\param row_to Last, excluded, number of row processed (it terms of rows in matrix and two vectors).
\param b Values of rhs dense vector elements.
\param[in,out] x Values of lhs dense vector elements. Initially has the initial guess. On return, it
has the one-step-converged vector.
\param jacobi true: Jacobi iteration, false: hybrid Gauss-Seidel iteration. */
void jacobi_iteration_with_CSR_partial(
	__global const uint *row_indices,
	__global const uint *column_indices,
	__global const double *values,
	uint row,
	uint row_to,
	__global const double *b,
	__global double *x,
	__global const bool jacobi)
{
	for (; row < row_to; ++row)
	{
		double diag = 0;
		double res = b[row];

		index    = row_indices[row];
		index_to = row_indices[row + 1];

		for (; index < index_to; ++index)
			if (column_indices[index] == row) diag = values[index];
			else res -= values[index] * x[column_indices[index]];
		
//TODO: THIS IS WRONG!!!		if (jacobi) barrier(CLK_GLOBAL_MEM_FENCE);
		x[row] = res / diag;	// race condition but without problem (old or new value, doesn't matter)
	} 
}

/** A hybrid Gauss-Seidel or Jacobi iteration.
It must called with:
- global(0) discretization as the number of rows of matrix. It is not required to be the same. More
than rows of matrix are waste of work-items.
\param row_indices Indices to first element of matrix's each row. There are \a num_rows + 1 elements.
\param column_indices Column index for each element. It makes 1-1 pair with \a values.
\param values Value for each element. It makes 1-1 pair with \a column_indices.
\param num_rows Number of rows in matrix and two vectors.
\param b Values of rhs dense vector elements. There are \a num_rows elements.
\param[in,out] x Values of lhs dense vector elements. Initially has the initial guess. On return, it
has the one-step-converged vector. There are \a num_rows elements.
\param jacobi true: Jacobi iteration, false: hybrid Gauss-Seidel iteration. */
__kernel void jacobi_iteration_with_CSR(
	__global const uint *row_indices,
	__global const uint *column_indices,
	__global const double *values,
	__global const uint num_rows,
	__global const double *b,
	__global double *x,
	__global const bool jacobi)
{
	uint global_id   = get_global_id(0);
	uint global_size = get_global_size(0);
	uint row    =  global_id      * num_rows / global_size;
	uint row_to = (global_id + 1) * num_rows / global_size;
	
	jacobi_iteration_with_CSR_partial(row_indices, column_indices, values, row, row_to, b, x, jacobi);
}


/** A hybrid Gauss-Seidel or Jacobi iteration on a part of matrix.
\param row_indices Indices to first element of matrix's each row.
\param column_indices Column index for each element. It makes 1-1 pair with \a values.
\param values Value for each element. It makes 1-1 pair with \a column_indices.
\param row First number of row processed (it terms of rows in matrix and two vectors).
\param row_to Last, excluded, number of row processed (it terms of rows in matrix and two vectors).
\param b Values of rhs dense vector elements.
\param[in,out] x Values of lhs dense vector elements. Initially has the initial guess. On return, it
has the iterations-step-converged vector.
\param jacobi true: Jacobi iteration, false: hybrid Gauss-Seidel iteration.
\param iterations How many Jacobi or Gauss-Seidel iterations will be applied. */
void jacobi_iterations_with_CSR_partial(
	__global const uint *row_indices,
	__global const uint *column_indices,
	__global const double *values,
	uint row,
	uint row_to,
	__global const double *b,
	__global double *x,
	__global const bool jacobi,
	__global const uchar iterations)
{
	for (uchar i = 0; i < iterations; ++i)
	{
		jacobi_iteration_with_CSR_partial(row_indices, column_indices, values, row, row_to, b, x, jacobi);
		barrier(CLK_GLOBAL_MEM_FENCE);
	} 
}

/** A hybrid Gauss-Seidel or Jacobi iteration.
It must called with:
- global(0) discretization as the number of rows of matrix. It is not required to be the same. More
than rows of matrix are waste of work-items.
\param row_indices Indices to first element of matrix's each row. There are \a num_rows + 1 elements.
\param column_indices Column index for each element. It makes 1-1 pair with \a values.
\param values Value for each element. It makes 1-1 pair with \a column_indices.
\param num_rows Number of rows in matrix and two vectors.
\param b Values of rhs dense vector elements. There are \a num_rows elements.
\param[in,out] x Values of lhs dense vector elements. Initially has the initial guess. On return, it
has the iterations-step-converged vector. There are \a num_rows elements.
\param jacobi true: Jacobi iteration, false: hybrid Gauss-Seidel iteration.
\param iterations How many Jacobi or Gauss-Seidel iterations will be applied. */
__kernel void jacobi_iterations_with_CSR(
	__global const uint *row_indices,
	__global const uint *column_indices,
	__global const double *values,
	__global const uint num_rows,
	__global const double *b,
	__global double *x,
	__global const bool jacobi,
	__global const uchar iterations)
{
	uint global_id   = get_global_id(0);
	uint global_size = get_global_size(0);
	uint row    =  global_id      * num_rows / global_size;
	uint row_to = (global_id + 1) * num_rows / global_size;
	
	jacobi_iterations_with_CSR_partial(row_indices, column_indices, values, row, row_to, b, x, jacobi, iterations);
}


/** Matrix - vector multiplication on a part of a matrix.
\param row_indices Indices to first element of matrix's each row.
\param column_indices Column index for each element. It makes 1-1 pair with \a values.
\param values Value for each element. It makes 1-1 pair with \a column_indices.
\param row First number of row processed (it terms of rows in matrix and vector \a y).
\param row_to Last, excluded, number of row processed (it terms of rows in matrix and vector \a y).
\param x Values of rhs dense vector elements.
\param[out] y Values of lhs dense vector elements.
\f$ \vec y = A\cdot\vec x\f$ or \f$ \vec y += A\cdot\vec x\f$ - it depends on \a clear.
\param lr Array with get_local_size(0) uninitialized elements. It exists only for intermediate
operations.
\param clear true: The operation is \f$ \vec y = A\cdot\vec x\f$, false: the operation is
\f$ \vec y += A\cdot\vec x\f$. */
void matrix_CSR_vector_product_partial(
	__global const uint *row_indices,
	__global const uint *column_indices,
	__global const double *values,
	uint row,
	uint row_to,
	__global const double *x,
	__global double *y,
	bool clear)
{
	for (; row < row_to; ++row)
	{
		if (clear) y[row] = 0;

		index    = row_indices[row];
		index_to = row_indices[row + 1];

		for (; index < index_to; ++index)
			y[row] += values[index] * x[column_indices[index]];
	}
}

/** Matrix - vector multiplication.
It must called with:
- global(0) discretization as the number of rows of matrix. It is not required to be the same. More
than rows of matrix are waste of work-items.
\param row_indices Indices to first element of matrix's each row. There are \a num_rows + 1 elements.
\param column_indices Column index for each element. It makes 1-1 pair with \a values.
\param values Value for each element. It makes 1-1 pair with \a column_indices.
\param num_rows Number of rows in matrix and vector \a y. Vector \a x can have different number of
rows.
\param x Values of rhs dense vector elements.
\param[out] y Values of lhs dense vector elements. There are \a num_rows elements.
\f$ \vec y = A\cdot\vec x\f$ or \f$ \vec y += A\cdot\vec x\f$ - it depends on \a clear.
\param lr Array with get_local_size(0) uninitialized elements. It exists only for intermediate
operations.
\param clear true: The operation is \f$ \vec y = A\cdot\vec x\f$, false: the operation is
\f$ \vec y += A\cdot\vec x\f$. */
__kernel void matrix_CSR_vector_product(
	__global const uint *row_indices,
	__global const uint *column_indices,
	__global const double *values,
	__global const uint num_rows,
	__global const double *x,
	__global double *y,
	__global const bool clear)
{
	uint global_id   = get_global_id(0);
	uint global_size = get_global_size(0);
	uint row    =  global_id      * num_rows / global_size;
	uint row_to = (global_id + 1) * num_rows / global_size;

	matrix_vector_product_partial(row_indices, column_indices, values, row, row_to, x, y, clear);
}


/** Set vector \a x to zero.
\param row First number of row processed.
\param row_to Last, excluded, number of row processed.
\param x[out] Values of source dense vector elements (which is the result). */
void vector_zero_partial(
	uint row,
	uint row_to,
	__global double *x)
{
	for (; row < row_to; ++row)
		x[row] = 0;
}

/** Vector set \f$ \vec y = \vec x\f$ on a part of vectors.
\param row First number of row processed.
\param row_to Last, excluded, number of row processed.
\param x Values of source dense vector elements.
\param[out] y Values of target dense vector elements (which is the result). */
void vector_set_partial(
	uint row,
	uint row_to,
	__global const double *x,
	__global double *y)
{
	for (; row < row_to; ++row)
		y[row] = x[row];
}

/** Vector subtraction \f$ \vec y = \vec x - \vec y\f$ on a part of vectors.
\param row First number of row processed.
\param row_to Last, excluded, number of row processed.
\param x Values of dense vector elements of the minuend.
\param[in,out] y Values of dense vector elements of the subtrahend which is also the result. */
void vector_subtraction_partial(
	uint row,
	uint row_to,
	__global const double *x,
	__global double *y)
{
	for (; row < row_to; ++row)
		y[row] = x[row] - y[row];
}


/** Vector subtraction \f$ \vec y = \vec x - \vec y\f$ on a part of vectors.
\param row First number of row processed (it terms of rows in matrix and vector \a y).
\param row_to Last, excluded, number of row processed.
\param x Values of dense vector elements to check if it is almost zero.
\param tolerance The tolerance for zero can be a positive number or zero.
\param[in,out] is_zero Value of this pointer, initially must be non-zero. It becomes zero if vector
is not almost zero. */
void vector_is_zero_partial(
	uint row,
	uint row_to,
	__global const double *x,
	__global const double tolerance,
	__global int *is_zero)
{
	for (; row < row_to; ++row)
	{
		if (fabs(x[row]) > tolerance)
		{
			atomic_store(is_zero, 0);
			break;
		}
		if (!atomic_load(is_zero)) break;
	}
}




// ============================================================= HIGH LEVEL KERNELS & FUNCTIONS ====


// ----------------------------------------------------------------- RESIDUAL RELATED FUNCTIONS ----

/** Calculate the residual of \f$ \vec r = \vec b - A\cdot\vec x\f$.
\param row_indices Indices to first element of matrix's each row.
\param column_indices Column index for each element. It makes 1-1 pair with \a values.
\param values Value for each element. It makes 1-1 pair with \a column_indices.
\param row First number of row processed (it terms of rows in matrix and three vectors).
\param row_to Last, excluded, number of row processed (it terms of rows in matrix and three vectors).
\param b,x Values of dense vector elements.
\param[out] r Values of the residual dense vector elements which is the result. */
void residual(
	__global const uint *row_indices,
	__global const uint *column_indices,
	__global const double *values,
	uint row,
	uint row_to,
	__global const double *b,
	__global const double *x,
	__global double *r)
{
	// r_n = A_n * x_n
	matrix_CSR_vector_product_partial(row_indices, column_indices, values, row, row_to, x, r, true);
	// r_n = b_n - r_n
	vector_subtraction_partial(row, row_to, b, r);
}

/** Calculate the residual of \f$ \vec r = \vec b - A\cdot\vec x\f$ and check if it is almost zero.
\param row_indices Indices to first element of matrix's each row.
\param column_indices Column index for each element. It makes 1-1 pair with \a values.
\param values Value for each element. It makes 1-1 pair with \a column_indices.
\param row First number of row processed (it terms of rows in matrix and three vectors).
\param row_to Last, excluded, number of row processed (it terms of rows in matrix and three vectors).
\param b,x Values of dense vector elements.
\param[out] r Values of the residual dense vector elements which is the result.
\param tolerance The tolerance for zero can be a positive number or zero.
\param[in,out] is_zero Value of this pointer, initially must be non-zero. It becomes zero if vector
is not almost zero. */
bool residual_with_check(
	__global const uint *row_indices,
	__global const uint *column_indices,
	__global const double *values,
	uint row,
	uint row_to,
	__global const double *b,
	__global const double *x,
	__global double *r,
	__global const double tolerance,
	__global int *is_zero)
{
	// r_0 = b_0 - A_0  * x_0
	residual(row_indices, column_indices, values, row, row_to, b, x, r);
	// if r_0 == 0 end
	vector_is_zero_partial(row, row_to, r, tolerance, is_zero);
}


// ------------------------------------------------------------------------- FIRST STEP KERNELS ----


/** First step of geometric multigrid.
This can be only the first step of geometric multigrid, when:
- Geometric multigrid starts from different level than level 0.
- Initial guess is not zero vector for \a x.
<p>It calculates the residual and checks if it is almost zero.
<p>It is implied that next step is in higher level of multigrid (lower detail).
\param row_indices Indices to first element of matrix's each row.
\param column_indices Column index for each element. It makes 1-1 pair with \a values.
\param values Value for each element. It makes 1-1 pair with \a column_indices.
\param num_rows Number of rows in matrix and three vectors.
\param b Values of dense vector elements.
\param x Values of lhs dense vector elements. Initially has the initial guess. On return, it
has the iterations-step-converged vector.
\param[out] r Values of the residual dense vector elements which is the result. */
__kernel void multigrid_first_down_without_jacobi(
	__global const uint *row_indices,
	__global const uint *column_indices,
	__global const double *values,
	__global const uint num_rows,
	__global const double *b,
	__global const double *x,
	__global double *r)
{
	uint global_id   = get_global_id(0);
	uint global_size = get_global_size(0);
	uint row    =  global_id      * num_rows / global_size;
	uint row_to = (global_id + 1) * num_rows / global_size;

	// r_0 = b_0 - A_0  * x_0
	// if r_0 == 0 end
	residual(row_indices, column_indices, values, row, row_to, b, x, r);
}

/** First step of geometric multigrid.
This can be only the first step of geometric multigrid.
<p>Initially does \a iterations Gauss-Seidel or Jacobi iterations.
<p>Then calculates the residual and checks if it is almost zero.
<p>It is implied that next step is in higher level of multigrid (lower detail).
\param row_indices Indices to first element of matrix's each row.
\param column_indices Column index for each element. It makes 1-1 pair with \a values.
\param values Value for each element. It makes 1-1 pair with \a column_indices.
\param num_rows Number of rows in matrix and three vectors.
\param b Values of dense vector elements.
\param[in,out] x Values of lhs dense vector elements. Initially has the initial guess. On return, it
has the iterations-step-converged vector.
\param[out] r Values of the residual dense vector elements which is the result.
\param jacobi true: Jacobi iteration, false: hybrid Gauss-Seidel iteration.
\param iterations How many Jacobi or Gauss-Seidel iterations will be applied.
\param tolerance The tolerance for zero can be a positive number or zero.
\param[in,out] is_zero Value of this pointer, initially must be non-zero. It becomes zero if vector
is not almost zero. */
__kernel void multigrid_first_down_with_jacobi(
	__global const uint *row_indices,
	__global const uint *column_indices,
	__global const double *values,
	__global const uint num_rows,
	__global const double *b,
	__global double *x,
	__global double *r,
	__global const bool jacobi,
	__global const uchar iterations,
	__global const double tolerance,
	__global int *is_zero)
{
	uint global_id   = get_global_id(0);
	uint global_size = get_global_size(0);
	uint row    =  global_id      * num_rows / global_size;
	uint row_to = (global_id + 1) * num_rows / global_size;

	// x_0 = Jacobi(A_0, b_0, x_0, iterations)
	// barrier
	jacobi_iterations_with_CSR_partial(row_indices, column_indices, values, row, row_to, b, x, jacobi, iterations);
	// r_0 = b_0 - A_0  * x_0
	// if r_0 == 0 end
	residual_with_check(row_indices, column_indices, values, row, row_to, b, x, r, tolerance, is_zero);
}


// ------------------------------------------------------------------------- GOING DOWN KERNELS ----


/** A going down-down step of geometric multigrid.
Initially restricts previous calculated residual to lower detail (higher level).
<p>Then it does \a iterations Gauss-Seidel or Jacobi iterations.
<p>Then calculates the residual.
<p>It is implied that next step is in higher level of multigrid (lower detail).
\param row_indicesR Indices to first element of restriction matrix's each row.
\param column_indicesR Column index for each element. It makes 1-1 pair with \a valuesR.
\param valuesR Value for each element. It makes 1-1 pair with \a column_indicesR.
\param r_prev Values of the previous step residual dense vector elements.
\param row_indicesA Indices to first element of matrix A each row.
\param column_indicesA Column index for each element. It makes 1-1 pair with \a valuesA.
\param valuesA Value for each element. It makes 1-1 pair with \a column_indicesA.
\param num_rows Number of rows in matrices and vectors \a b, \a x, \a r. Not in vector \a r_prev.
\param b Values of dense vector elements.
\param[out] x Values of lhs dense vector elements. Initially is zero. On return, it has the
iterations-step-converged vector.
\param[out] r Values of the residual dense vector elements which is the result.
\param jacobi true: Jacobi iteration, false: hybrid Gauss-Seidel iteration.
\param iterations How many Jacobi or Gauss-Seidel iterations will be applied. */
__kernel void multigrid_down_down(
	__global const uint *row_indicesR,
	__global const uint *column_indicesR,
	__global const double *valuesR,
	__global const double *r_prev,
	__global const uint *row_indicesA,
	__global const uint *column_indicesA,
	__global const double *valuesA,
	__global const uint num_rows,
	__global double *b,
	__global double *x,
	__global double *r,
	__global const bool jacobi,
	__global const uchar iterations)
{
	uint global_id   = get_global_id(0);
	uint global_size = get_global_size(0);
	uint row    =  global_id      * num_rows / global_size;
	uint row_to = (global_id + 1) * num_rows / global_size;

	// b_n = R_{n-1} * r_{n-1}
	// no requirement for barrier for b_n
	matrix_CSR_vector_product_partial(row_indicesR, column_indicesR, valuesR, row, row_to, r_prev, b, true);
	// x_n = 0
	vector_zero_partial(row, row_to, x);
	barrier(CLK_GLOBAL_MEM_FENCE); // for x, because jacobi iterations need whole vector
	// x_n = Jacobi(A_n, b_n, x_n, iterations)
	// barrier
	jacobi_iterations_with_CSR_partial(row_indices, column_indices, values, row, row_to, b, x, jacobi, iterations);
	// r_n = b_n - A_n * x_n
	residual(row_indices, column_indices, values, row, row_to, b, x, r);
}


/** A step of geometric multigrid which is the local higher level (lower detail) but not the global highest level.
Initially restricts previous calculated residual to lower detail (higher level).
<p>Then it does \a iterations Gauss-Seidel or Jacobi iterations.
\param row_indicesR Indices to first element of restriction matrix's each row.
\param column_indicesR Column index for each element. It makes 1-1 pair with \a valuesR.
\param valuesR Value for each element. It makes 1-1 pair with \a column_indicesR.
\param r_prev Values of the previous step residual dense vector elements.
\param row_indicesA Indices to first element of matrix A each row.
\param column_indicesA Column index for each element. It makes 1-1 pair with \a valuesA.
\param valuesA Value for each element. It makes 1-1 pair with \a column_indicesA.
\param num_rows Number of rows in matrices and vectors \a b, \a x, \a r. Not in vector \a r_prev.
\param b Values of dense vector elements.
\param[out] x Values of lhs dense vector elements. Initially is zero. On return, it has the
iterations-step-converged vector.
\param[out] r Values of the residual dense vector elements which is the result.
\param jacobi true: Jacobi iteration, false: hybrid Gauss-Seidel iteration.
\param iterations How many Jacobi or Gauss-Seidel iterations will be applied. */
__kernel void multigrid_local_maxima(
	__global const uint *row_indicesR,
	__global const uint *column_indicesR,
	__global const double *valuesR,
	__global const double *r,
	__global const uint *row_indicesA,
	__global const uint *column_indicesA,
	__global const double *valuesA,
	__global const uint num_rows,
	__global double *b,
	__global double *x,
	__global const bool jacobi,
	__global const uchar iterations)
{
	uint global_id   = get_global_id(0);
	uint global_size = get_global_size(0);
	uint row    =  global_id      * num_rows / global_size;
	uint row_to = (global_id + 1) * num_rows / global_size;

	// b_n = R_{n-1} * r_{n-1}
	// no requirement for barrier for b_n
	matrix_CSR_vector_product_partial(row_indicesR, column_indicesR, valuesR, row, row_to, r, b, true);
	// x_n = 0
	vector_zero_partial(row, row_to, x);
	barrier(CLK_GLOBAL_MEM_FENCE); // for x, because jacobi iterations need whole vector
	// x_n = Jacobi(A_n, b_n, x_n, iterations)
	// barrier
	jacobi_iterations_with_CSR_partial(row_indicesA, column_indicesA, valuesA, row, row_to, b, x, jacobi, iterations);
}


/** The step on highest level (lowest detail) of geometric multigrid.
Restricts previous calculated residual to lower detail (higher level).
<p>It is implied that next step is in higher level of multigrid (lower detail).
\param row_indices Indices to first element of restriction matrix's each row.
\param column_indices Column index for each element. It makes 1-1 pair with \a valuesR.
\param values Value for each element. It makes 1-1 pair with \a column_indicesR.
\param r Values of the previous step residual dense vector elements.
\param num_rows Number of rows in matrices and vectors \a b, \a x, \a r. Not in vector \a r.
\param[out] b Values of dense vector elements, which is the result. */
__kernel void multigrid_global_maxima(
	__global const uint *row_indices,
	__global const uint *column_indices,
	__global const double *values,
	__global const double *r,
	__global const uint num_rows,
	__global double *b)
{
	uint global_id   = get_global_id(0);
	uint global_size = get_global_size(0);
	uint row    =  global_id      * num_rows / global_size;
	uint row_to = (global_id + 1) * num_rows / global_size;

	// b_n = R_{n-1} * r_{n-1}
	matrix_CSR_vector_product_partial(row_indices, column_indices, values, row, row_to, r, b, true);



// --------------------------------------------------------------------------- GOING UP KERNELS ----


/** A going up-up step of geometric multigrid.
Initially interpolated previous calculated error to higher detail (lower level).
<p>Then it does \a iterations Gauss-Seidel or Jacobi iterations.
\param row_indicesI Indices to first element of interpolation matrix's each row.
\param column_indicesI Column index for each element. It makes 1-1 pair with \a valuesI.
\param valuesI Value for each element. It makes 1-1 pair with \a column_indicesI.
\param x_prev Values of the previous step error dense vector elements.
\param row_indicesA Indices to first element of matrix A each row.
\param column_indicesA Column index for each element. It makes 1-1 pair with \a valuesA.
\param valuesA Value for each element. It makes 1-1 pair with \a column_indicesA.
\param num_rows Number of rows in matrices and vectors \a b, \a x. Not in vector \a x_prev.
\param b Values of dense vector elements.
\param[in, out] x Values of lhs dense vector elements. On return, it has the
iterations-step-converged vector.
\param jacobi true: Jacobi iteration, false: hybrid Gauss-Seidel iteration.
\param iterations How many Jacobi or Gauss-Seidel iterations will be applied. */
__kernel void multigrid_up_up(
	__global const uint *row_indicesI,
	__global const uint *column_indicesI,
	__global const double *valuesI,
	__global const double *x_prev,
	__global const uint *row_indicesA,
	__global const uint *column_indicesA,
	__global const double *valuesA,
	__global const uint num_rows,
	__global const double *b,
	__global double *x,
	__global const bool jacobi,
	__global const uchar iterations)
{
	uint global_id   = get_global_id(0);
	uint global_size = get_global_size(0);
	uint row    =  global_id      * num_rows / global_size;
	uint row_to = (global_id + 1) * num_rows / global_size;

	// x_n += I_n * x_{n+1}
	matrix_CSR_vector_product_partial(row_indicesI, column_indicesI, valuesI, row, row_to, x_prev, x, false);
	barrier(CLK_GLOBAL_MEM_FENCE); // for x, because jacobi iterations need whole vector
	// x_n = Jacobi(A_n, b_n, x_n, iterations)
	// barrier
	jacobi_iterations_with_CSR_partial(row_indicesA, column_indicesA, valuesA, row, row_to, b, x, jacobi, iterations);
}


/** The step on lowest level (higher detail) of geometric multigrid.
This cannot be the first step of geometric multigrid. It appears when algorithm comes from a higher
step, then this is the lowest step then it goes to a higher step again.
<p>Initially interpolates previous calculated residual to lower level (higher detail).
<p>Then it does \a iterations Gauss-Seidel or Jacobi iterations.
<p>Then calculates the residual.
<p>It is implied that next step is in higher level of multigrid (lower detail).
\param row_indicesI Indices to first element of interpolation matrix's each row.
\param column_indicesI Column index for each element. It makes 1-1 pair with \a valuesI.
\param valuesI Value for each element. It makes 1-1 pair with \a column_indicesI.
\param x_prev Values of the previous step error dense vector elements.
\param row_indicesA Indices to first element of matrix A each row.
\param column_indicesA Column index for each element. It makes 1-1 pair with \a valuesA.
\param valuesA Value for each element. It makes 1-1 pair with \a column_indicesA.
\param num_rows Number of rows in matrices and vectors \a b, \a x, \a r. Not in vector \a r_prev.
\param b Values of dense vector elements.
\param[out] x Values of lhs dense vector elements. Initially is zero. On return, it has the
iterations-step-converged vector.
\param[out] r Values of the residual dense vector elements which is the result.
\param jacobi true: Jacobi iteration, false: hybrid Gauss-Seidel iteration.
\param iterations How many Jacobi or Gauss-Seidel iterations will be applied.
\param tolerance The tolerance for zero can be a positive number or zero.
\param[in,out] is_zero Value of this pointer, initially must be non-zero. It becomes zero if vector
is not almost zero. */
__kernel void multigrid_global_minima(
	__global const uint *row_indicesI,
	__global const uint *column_indicesI,
	__global const double *valuesI,
	__global const double *x_prev,
	__global const uint *row_indicesA,
	__global const uint *column_indicesA,
	__global const double *valuesA,
	__global const uint num_rows,
	__global const double *b,
	__global double *x,
	__global double *r,
	__global const bool jacobi,
	__global const uchar iterations,
	__global const double tolerance,
	__global int *is_zero)
{
	uint global_id   = get_global_id(0);
	uint global_size = get_global_size(0);
	uint row    =  global_id      * num_rows / global_size;
	uint row_to = (global_id + 1) * num_rows / global_size;

	// x_0 += I_0 * x_1
	matrix_CSR_vector_product_partial(row_indicesI, column_indicesI, valuesI, row, row_to, x_prev, x, false);
	barrier(CLK_GLOBAL_MEM_FENCE); // for x, because jacobi iterations need whole vector
	// x_0 = Jacobi(A_0, b_0, x_0, iterations)
	// barrier
	jacobi_iterations_with_CSR_partial(row_indicesA, column_indicesA, valuesA, row, row_to, b, x, jacobi, iterations);
	// r_0 = b_0 - A_0 * x_0
	// if r_0 == 0 end
	residual_with_check(row_indices, column_indices, values, row, row_to, b, x, r, tolerance, is_zero);
}


/** The step on a lower level (higher detail) of geometric multigrid.
Initially interpolates previous calculated residual to lower level (higher detail).
<p>Then it does \a iterations Gauss-Seidel or Jacobi iterations.
<p>Then calculates the residual.
<p>It is implied that next step is in higher level of multigrid (lower detail).
\param row_indicesI Indices to first element of interpolation matrix's each row.
\param column_indicesI Column index for each element. It makes 1-1 pair with \a valuesI.
\param valuesI Value for each element. It makes 1-1 pair with \a column_indicesI.
\param x_prev Values of the previous step error dense vector elements.
\param row_indicesA Indices to first element of matrix A each row.
\param column_indicesA Column index for each element. It makes 1-1 pair with \a valuesA.
\param valuesA Value for each element. It makes 1-1 pair with \a column_indicesA.
\param num_rows Number of rows in matrices and vectors \a b, \a x, \a r. Not in vector \a r_prev.
\param b Values of dense vector elements.
\param[out] x Values of lhs dense vector elements. Initially is zero. On return, it has the
iterations-step-converged vector.
\param[out] r Values of the residual dense vector elements which is the result.
\param jacobi true: Jacobi iteration, false: hybrid Gauss-Seidel iteration.
\param iterations How many Jacobi or Gauss-Seidel iterations will be applied. */
__kernel void multigrid_local_minima(
	__global const uint *row_indicesR,
	__global const uint *column_indicesR,
	__global const double *valuesR,
	__global const double *x_prev,
	__global const uint *row_indicesA,
	__global const uint *column_indicesA,
	__global const double *valuesA,
	__global const uint num_rows,
	__global double *b,
	__global double *x,
	__global double *r,
	__global const bool jacobi,
	__global const uchar iterations)
{
	uint global_id   = get_global_id(0);
	uint global_size = get_global_size(0);
	uint row    =  global_id      * num_rows / global_size;
	uint row_to = (global_id + 1) * num_rows / global_size;

	// x_n += I_n * x_{n+1}
	matrix_CSR_vector_product_partial(row_indicesI, column_indicesI, valuesI, row, row_to, x_prev, x, false);
	barrier(CLK_GLOBAL_MEM_FENCE); // for x, because jacobi iterations need whole vector
	// x_n = Jacobi(A_n, b_n, x_n, iterations)
	// barrier
	jacobi_iterations_with_CSR_partial(row_indicesA, column_indicesA, valuesA, row, row_to, b, x, jacobi, iterations);
	// r_n = b_n - A_n * x_n
	residual(row_indices, column_indices, values, row, row_to, b, x, r);
}
