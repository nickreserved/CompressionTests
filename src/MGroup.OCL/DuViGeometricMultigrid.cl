#ifdef USE_LOCAL_MEMORY	// faster compilation
/** A Jacobi iteration.
\param row_indices_on_column_indices Indices to first block of matrix's each row.
\param row_indices_on_delta_distance_indices Distance array index of each row's block distances in
block (elements - 1 in block).
\param column_indices Column index for the first element of each row's block.
\param delta_distances Distance between block's elements (0 is 1) - First distance of block is
actually the number of distances in block (elements - 1 in block).
\param value_indices The index to a distinct value. They make pairs with \a delta_distances.
\param global_values Distinct values of matrix elements.
\param b Values of rhs dense vector elements.
\param x Values of lhs dense vector elements.
\param[out] y Values of result dense vector elements.
\param rows Number of rows in matrix.
\param elements Number of distinct elements in matrix.
\param values Distinct values of matrix elements. */
void copy_values(
	__global const double *global_values,
	__local double *values,
	uint elements)
{
	uint id = get_local_id(0);
    uint size = get_local_size(0);

    for (; id < elements; id += size)
        values[id] = global_values[id];

    barrier(CLK_LOCAL_MEM_FENCE);
}
#endif


//-------------------------------------------------------------- MATRIX - VECTOR MULTIPLICATION ----


/** Matrix - vector multiplication on a row of a matrix.
\param row_indices_on_column_indices Indices to first block of matrix's each row.
\param row_indices_on_delta_distance_indices Distance array index of each row's block distances in
block (elements - 1 in block).
\param column_indices Column index for the first element of each row's block.
\param delta_distances Distance between block's elements (0 is 1) - First distance of block is
actually the number of distances in block (elements - 1 in block).
\param value_indices The index to a distinct value. They make pairs with \a delta_distances.
\param values Distinct values of matrix elements.
\param row Ρow processed (it terms of rows in matrix and vector \a y).
\param x Values of rhs dense vector elements.
\param[out] y Values of lhs dense vector elements.
\f$ \vec y = A\cdot\vec x\f$ or \f$ \vec y += A\cdot\vec x\f$ - it depends on \a clear.
\param lr Array with get_local_size(0) uninitialized elements. It exists only for intermediate
operations.
\param clear true: The operation is \f$ \vec y = A\cdot\vec x\f$, false: the operation is
\f$ \vec y += A\cdot\vec x\f$. */
void matrix_vector_product_partial(
	__global const uint *row_indices_on_column_indices,
	__global const uint *row_indices_on_delta_distance_indices,
	__global const uint *column_indices,
	__global const uchar *delta_distances,
	__global const ushort *value_indices,
#ifdef USE_LOCAL_MEMORY
	__local const double *values,
#else
	__global const double *values,
#endif
	uint row,
	__global const double *x,
	__global double *y,
	uchar clear)
{
	if (clear) y[row] = 0;

	uint col_index    = row_indices_on_column_indices[row];
	uint col_index_to = row_indices_on_column_indices[row + 1];
	
	uint dist_index = row_indices_on_delta_distance_indices[row];
	
	for (; col_index < col_index_to; ++col_index)
	{
		uint column = column_indices[col_index];
		
		uint dist_index_to = dist_index + delta_distances[dist_index] + 1;
				
		for (;;)
		{
			y[row] += values[value_indices[dist_index]] * x[column];
			// break check cannot be in for(;;)
			++dist_index;
			if (dist_index == dist_index_to) break;
			column += delta_distances[dist_index] + 1;	// because can access out of bounds
		}
	}
}

/** Matrix - vector multiplication.
It must called with:
- global(0) discretization as the number of rows of matrix.
\param row_indices_on_column_indices Indices to first block of matrix's each row.
\param row_indices_on_delta_distance_indices Distance array index of each row's block distances in
block (elements - 1 in block).
\param column_indices Column index for the first element of each row's block.
\param delta_distances Distance between block's elements (0 is 1) - First distance of block is
actually the number of distances in block (elements - 1 in block).
\param value_indices The index to a distinct value. They make pairs with \a delta_distances.
\param global_values Distinct values of matrix elements.
\param x Values of rhs dense vector elements.
\param[out] y Values of lhs dense vector elements.
\f$ \vec y = A\cdot\vec x\f$ or \f$ \vec y += A\cdot\vec x\f$ - it depends on \a clear.
\param lr Array with get_local_size(0) uninitialized elements. It exists only for intermediate
operations.
\param clear true: The operation is \f$ \vec y = A\cdot\vec x\f$, false: the operation is
\f$ \vec y += A\cdot\vec x\f$.
\param rows Number of rows in matrix.
\param elements Number of distinct elements in matrix.
\param values Distinct values of matrix elements. */
__kernel void matrix_vector_product(
	__global const uint *row_indices_on_column_indices,
	__global const uint *row_indices_on_delta_distance_indices,
	__global const uint *column_indices,
	__global const uchar *delta_distances,
	__global const ushort *value_indices,
#ifdef USE_LOCAL_MEMORY
	__global const double *global_values,
#else
	__global const double *values,
#endif
	__global const double *x,
	__global double *y,
	uchar clear,
#ifdef USE_LOCAL_MEMORY
	uint rows,
	uint elements,
	__local double *values
	)
{
	copy_values(global_values, values, elements);
#else
	uint rows)
{
#endif

	uint row = get_global_id(0);
	if (row >= rows) return;

	matrix_vector_product_partial(	row_indices_on_column_indices,
									row_indices_on_delta_distance_indices,
									column_indices, delta_distances,
									value_indices, values, row, x, y, clear);
}


// ----------------------------------------------------------------- RESIDUAL RELATED FUNCTIONS ----


/** Calculate the residual of \f$ \vec r = \vec b - A\cdot\vec x\f$.
\param row_indices_on_column_indices Indices to first block of matrix's each row.
\param row_indices_on_delta_distance_indices Distance array index of each row's block distances in
block (elements - 1 in block).
\param column_indices Column index for the first element of each row's block.
\param delta_distances Distance between block's elements (0 is 1) - First distance of block is
actually the number of distances in block (elements - 1 in block).
\param value_indices The index to a distinct value. They make pairs with \a delta_distances.
\param values Distinct values of matrix elements.
\param row Row processed (it terms of rows in matrix and three vectors).
\param b,x Values of dense vector elements.
\param[out] r Values of the residual dense vector elements which is the result. */
void residual_partial(
	__global const uint *row_indices_on_column_indices,
	__global const uint *row_indices_on_delta_distance_indices,
	__global const uint *column_indices,
	__global const uchar *delta_distances,
	__global const ushort *value_indices,
#ifdef USE_LOCAL_MEMORY
	__local const double *values,
#else
	__global const double *values,
#endif
	uint row,
	__global const double *b,
	__global const double *x,
	__global double *r)
{
	// r_n = A_n * x_n
	matrix_vector_product_partial(	row_indices_on_column_indices,
									row_indices_on_delta_distance_indices,
									column_indices, delta_distances,
									value_indices, values, row, x, r, true);
	// r_n = b_n - r_n
	r[row] = b[row] - r[row];
}

/** Calculate the residual of \f$ \vec r = \vec b - A\cdot\vec x\f$.
\param row_indices_on_column_indices Indices to first block of matrix's each row.
\param row_indices_on_delta_distance_indices Distance array index of each row's block distances in
block (elements - 1 in block).
\param column_indices Column index for the first element of each row's block.
\param delta_distances Distance between block's elements (0 is 1) - First distance of block is
actually the number of distances in block (elements - 1 in block).
\param value_indices The index to a distinct value. They make pairs with \a delta_distances.
\param global_values Distinct values of matrix elements.
\param b,x Values of dense vector elements.
\param[out] r Values of the residual dense vector elements which is the result.
\param rows Number of rows in matrix.
\param elements Number of distinct elements in matrix.
\param values Distinct values of matrix elements. */
__kernel void residual(
	__global const uint *row_indices_on_column_indices,
	__global const uint *row_indices_on_delta_distance_indices,
	__global const uint *column_indices,
	__global const uchar *delta_distances,
	__global const ushort *value_indices,
#ifdef USE_LOCAL_MEMORY
	__global const double *global_values,
#else
	__global const double *values,
#endif
	__global const double *b,
	__global const double *x,
	__global double *r,
#ifdef USE_LOCAL_MEMORY
	uint rows,
	uint elements,
	__local double *values
	)
{
	copy_values(global_values, values, elements);
#else
	uint rows)
{
#endif

	uint row = get_global_id(0);
	if (row >= rows) return;

	// r = b - A  * x
	residual_partial(row_indices_on_column_indices, row_indices_on_delta_distance_indices,
						column_indices, delta_distances, value_indices, values, row, b, x, r);
}

/** Calculate the residual of \f$ \vec r = \vec b - A\cdot\vec x\f$ and check if it is almost zero.
\param row_indices_on_column_indices Indices to first block of matrix's each row.
\param row_indices_on_delta_distance_indices Distance array index of each row's block distances in
block (elements - 1 in block).
\param column_indices Column index for the first element of each row's block.
\param delta_distances Distance between block's elements (0 is 1) - First distance of block is
actually the number of distances in block (elements - 1 in block).
\param value_indices The index to a distinct value. They make pairs with \a delta_distances.
\param global_values Distinct values of matrix elements.
\param b,x Values of dense vector elements.
\param[out] r Values of the residual dense vector elements which is the result.
\param tolerance The tolerance for zero can be a positive number or zero.
\param[in,out] is_zero Value of this pointer, initially must be non-zero. It becomes zero if vector
is not almost zero.
\param rows Number of rows in matrix.
\param elements Number of distinct elements in matrix.
\param values Distinct values of matrix elements. */
__kernel void residual_with_check(
	__global const uint *row_indices_on_column_indices,
	__global const uint *row_indices_on_delta_distance_indices,
	__global const uint *column_indices,
	__global const uchar *delta_distances,
	__global const ushort *value_indices,
#ifdef USE_LOCAL_MEMORY
	__global const double *global_values,
#else
	__global const double *values,
#endif
	__global const double *b,
	__global const double *x,
	__global double *r,
	double tolerance,
	__global int *is_zero,//__global atomic_int *is_zero,
#ifdef USE_LOCAL_MEMORY
	uint rows,
	uint elements,
	__local double *values
	)
{
	copy_values(global_values, values, elements);
#else
	uint rows)
{
#endif

	uint row = get_global_id(0);
	if (row >= rows) return;

	// r = b_0 - A_0  * x_0
	residual_partial(row_indices_on_column_indices, row_indices_on_delta_distance_indices,
						column_indices, delta_distances, value_indices, values, row, b, x, r);
	// if r_0 == 0 end
	double c = fabs(r[row]);
	if (isnan(r[row]) || c > 1e50) atomic_or (is_zero, 2);
	else if (c > tolerance)        atomic_and(is_zero, 2);	//atomic_store(is_zero, 0);
}


// ---------------------------------------------- ITERATIONS OF GAUSS - SEIDEL & JACOBI METHODS ----


#ifndef USE_LOCAL_MEMORY 	// jacobi_initial_iteration() is the same. Use the version without USE_LOCAL_MEMORY
/** An initial Jacobi iteration with initial guess vector zero.
\param preconditioner The Jacobi preconditioner vector of matrix multiplied with w = 2 / lmax where
lmax is an upper bound of eigenvalues of matrix.
\param b Values of rhs dense vector elements.
\param[out] x Values of result dense vector elements.
\param rows Number of rows in matrix. */
__kernel void jacobi_initial_iteration(
	__global const double *preconditioner,
	__global const double *b,
	__global double *x,
	uint rows)
{
	uint row = get_global_id(0);
	if (row >= rows) return;

	// x = x + w * D^-1 * (b - A * x) where x = 0, so x = w * D^-1 * b
	x[row] = preconditioner[row] * b[row];
}
#endif


/** A Jacobi iteration.
\param preconditioner The Jacobi preconditioner vector of matrix multiplied with w = 2 / lmax where
lmax is an upper bound of eigenvalues of matrix.
\param row_indices_on_column_indices Indices to first block of matrix's each row.
\param row_indices_on_delta_distance_indices Distance array index of each row's block distances in
block (elements - 1 in block).
\param column_indices Column index for the first element of each row's block.
\param delta_distances Distance between block's elements (0 is 1) - First distance of block is
actually the number of distances in block (elements - 1 in block).
\param value_indices The index to a distinct value. They make pairs with \a delta_distances.
\param global_values Distinct values of matrix elements.
\param b Values of rhs dense vector elements.
\param x Values of lhs dense vector elements.
\param[out] y Values of result dense vector elements.
\param rows Number of rows in matrix.
\param elements Number of distinct elements in matrix.
\param values Distinct values of matrix elements. */
__kernel void jacobi_iteration(
	__global const double *preconditioner,
	__global const uint *row_indices_on_column_indices,
	__global const uint *row_indices_on_delta_distance_indices,
	__global const uint *column_indices,
	__global const uchar *delta_distances,
	__global const ushort *value_indices,
#ifdef USE_LOCAL_MEMORY
	__global const double *global_values,
#else
	__global const double *values,
#endif
	__global const double *b,
	__global const double *x,
	__global double *y,
#ifdef USE_LOCAL_MEMORY
	uint rows,
	uint elements,
	__local double *values
	)
{
	copy_values(global_values, values, elements);
#else
	uint rows)
{
#endif

	uint row = get_global_id(0);
	if (row >= rows) return;
	
	// final result: y = x + w * D^-1 * (b - A * x)
	// y = b - A * x
	residual_partial(row_indices_on_column_indices, row_indices_on_delta_distance_indices,
						column_indices, delta_distances, value_indices, values, row, b, x, y);
	// y = x + w * D^-1 * y
	y[row] = x[row] + preconditioner[row] * y[row];
}


/** A Gauss-Seidel iteration.
\param preconditioner The Jacobi preconditioner vector of matrix multiplied with w = 2 / lmax where
lmax is an upper bound of eigenvalues of matrix.
\param row_indices_on_column_indices Indices to first block of matrix's each row.
\param row_indices_on_delta_distance_indices Distance array index of each row's block distances in
block (elements - 1 in block).
\param column_indices Column index for the first element of each row's block.
\param delta_distances Distance between block's elements (0 is 1) - First distance of block is
actually the number of distances in block (elements - 1 in block).
\param value_indices The index to a distinct value. They make pairs with \a delta_distances.
\param global_values Distinct values of matrix elements.
\param b Values of rhs dense vector elements.
\param[out] x Values of result dense vector elements.
\param y Intermediate buffer.
\param rows Number of rows in matrix.
\param elements Number of distinct elements in matrix.
\param values Distinct values of matrix elements. */
__kernel void gauss_seidel_iteration(
	__global const double *preconditioner,
	__global const uint *row_indices_on_column_indices,
	__global const uint *row_indices_on_delta_distance_indices,
	__global const uint *column_indices,
	__global const uchar *delta_distances,
	__global const ushort *value_indices,
#ifdef USE_LOCAL_MEMORY
	__global const double *global_values,
#else
	__global const double *values,
#endif
	__global const double *b,
	__global double *x,
	__global double *y,
#ifdef USE_LOCAL_MEMORY
	uint rows,
	uint elements,
	__local double *values
	)
{
	copy_values(global_values, values, elements);
#else
	uint rows)
{
#endif

	uint row = get_global_id(0);
	if (row >= rows) return;
	
	// final result: x += w * D^-1 * (b - A * x)
	// y = b - A * x
	residual_partial(row_indices_on_column_indices, row_indices_on_delta_distance_indices,
						column_indices, delta_distances, value_indices, values, row, b, x, y);
	// x += w * D^-1 * y
	x[row] += preconditioner[row] * y[row];
}