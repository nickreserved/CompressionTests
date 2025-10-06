//-------------------------------------------------------------- MATRIX - VECTOR MULTIPLICATION ----


/** Matrix - vector multiplication on a row of a matrix.
\param row_indices Indices to first element of matrix's each row.
\param column_indices Column index for each element. It makes 1-1 pair with \a values.
\param values Value for each element. It makes 1-1 pair with \a column_indices.
\param row Ρow processed (it terms of rows in matrix and vector \a y).
\\param x Values of rhs dense vector elements.
\param[out] y Values of lhs dense vector elements.
\f$ \vec y = A\cdot\vec x\f$ or \f$ \vec y += A\cdot\vec x\f$ - it depends on \a clear.
\param lr Array with get_local_size(0) uninitialized elements. It exists only for intermediate
operations.
\param clear true: The operation is \f$ \vec y = A\cdot\vec x\f$, false: the operation is
\f$ \vec y += A\cdot\vec x\f$. */
void matrix_vector_product_partial(
	__global const uint *row_indices,
	__global const uint *column_indices,
	__global const double *values,
	uint row,
	__global const double *x,
	__global double *y,
	uchar clear)
{
	if (clear) y[row] = 0;

	uint index    = row_indices[row];
	uint index_to = row_indices[row + 1];

	for (; index < index_to; ++index)
		y[row] += values[index] * x[column_indices[index]];
}

/** Matrix - vector multiplication.
It must called with:
- global(0) discretization as the number of rows of matrix. It is not required to be the same. More
than rows of matrix are waste of work-items.
\param row_indices Indices to first element of matrix's each row. There are \a num_rows + 1 elements.
\param column_indices Column index for each element. It makes 1-1 pair with \a values.
\param values Value for each element. It makes 1-1 pair with \a column_indices.
\param x Values of rhs dense vector elements.
\param[out] y Values of lhs dense vector elements. There are \a num_rows elements.
\f$ \vec y = A\cdot\vec x\f$ or \f$ \vec y += A\cdot\vec x\f$ - it depends on \a clear.
\param lr Array with get_local_size(0) uninitialized elements. It exists only for intermediate
operations.
\param clear true: The operation is \f$ \vec y = A\cdot\vec x\f$, false: the operation is
\f$ \vec y += A\cdot\vec x\f$.
\param rows Number of rows in matrix. */
__kernel void matrix_vector_product(
	__global const uint *row_indices,
	__global const uint *column_indices,
	__global const double *values,
	__global const double *x,
	__global double *y,
	uchar clear,
	uint rows)
{
	uint row = get_global_id(0);
	if (row >= rows) return;

	matrix_vector_product_partial(row_indices, column_indices, values, row, x, y, clear);
}


// ----------------------------------------------------------------- RESIDUAL RELATED FUNCTIONS ----


/** Calculate the residual of \f$ \vec r = \vec b - A\cdot\vec x\f$.
\param row_indices Indices to first element of matrix's each row.
\param column_indices Column index for each element. It makes 1-1 pair with \a values.
\param values Value for each element. It makes 1-1 pair with \a column_indices.
\param row Row processed (it terms of rows in matrix and three vectors).
\param b,x Values of dense vector elements.
\param[out] r Values of the residual dense vector elements which is the result. */
void residual_partial(
	__global const uint *row_indices,
	__global const uint *column_indices,
	__global const double *values,
	uint row,
	__global const double *b,
	__global const double *x,
	__global double *r)
{
	// r = A * x
	matrix_vector_product_partial(row_indices, column_indices, values, row, x, r, true);
	// r = b - r
	r[row] = b[row] - r[row];
}

/** Calculate the residual of \f$ \vec r = \vec b - A\cdot\vec x\f$.
\param row_indices Indices to first element of matrix's each row.
\param column_indices Column index for each element. It makes 1-1 pair with \a values.
\param values Value for each element. It makes 1-1 pair with \a column_indices.
\param b,x Values of dense vector elements.
\param[out] r Values of the residual dense vector elements which is the result. */
__kernel void residual(
	__global const uint *row_indices,
	__global const uint *column_indices,
	__global const double *values,
	__global const double *b,
	__global const double *x,
	__global double *r,
	uint rows)
{
	uint row = get_global_id(0);
	if (row >= rows) return;
	
	// r = b - A * x
	residual_partial(row_indices, column_indices, values, row, b, x, r);
}

/** Calculate the residual of \f$ \vec r = \vec b - A\cdot\vec x\f$ and check if it is almost zero.
\param row_indices Indices to first element of matrix's each row.
\param column_indices Column index for each element. It makes 1-1 pair with \a values.
\param values Value for each element. It makes 1-1 pair with \a column_indices.
\param b,x Values of dense vector elements.
\param[out] r Values of the residual dense vector elements which is the result.
\param tolerance The tolerance for zero can be a positive number or zero.
\param[in,out] is_zero Value of this pointer, initially must be non-zero. It becomes zero if vector
is not almost zero. */
__kernel void residual_with_check(
	__global const uint *row_indices,
	__global const uint *column_indices,
	__global const double *values,
	__global const double *b,
	__global const double *x,
	__global double *r,
	double tolerance,
	__global int *is_zero,//__global atomic_int *is_zero
	uint rows)
{
	uint row = get_global_id(0);
	if (row >= rows) return;
	
	// r_0 = b_0 - A_0  * x_0
	residual_partial(row_indices, column_indices, values, row, b, x, r);
	// if r_0 == 0 end
	double c = fabs(r[row]);
	if (isnan(r[row]) || c > 1e50) atomic_or (is_zero, 2);
	else if (c > tolerance)        atomic_and(is_zero, 2);	//atomic_store(is_zero, 0);
}


// -------------------------------------------------------------------- GAUSS - SEIDEL & JACOBI ----


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

/** A Jacobi iteration.
\param preconditioner The Jacobi preconditioner vector of matrix multiplied with w = 2 / lmax where
lmax is an upper bound of eigenvalues of matrix.
\param row_indices Indices to first element of matrix's each row.
\param column_indices Column index for each element. It makes 1-1 pair with \a values.
\param values Value for each element. It makes 1-1 pair with \a column_indices.
\param b Values of rhs dense vector elements.
\param x Values of lhs dense vector elements.
\param[out] y Values of result dense vector elements.
\param rows Number of rows in matrix. */
__kernel void jacobi_iteration(
	__global const double *preconditioner,
	__global const uint *row_indices,
	__global const uint *column_indices,
	__global const double *values,
	__global const double *b,
	__global const double *x,
	__global double *y,
	uint rows)
{
	uint row = get_global_id(0);
	if (row >= rows) return;
	
	// final result: y = x + w * D^-1 * (b - A * x)
	// y = b - A * x
	residual_partial(row_indices, column_indices, values, row, b, x, y);
	// y = x + w * D^-1 * y
	y[row] = x[row] + preconditioner[row] * y[row];
}

/** A hybrid Gauss-Seidel iteration.
\param preconditioner The Jacobi preconditioner vector of matrix multiplied with w = 2 / lmax where
lmax is an upper bound of eigenvalues of matrix.
\param row_indices Indices to first element of matrix's each row.
\param column_indices Column index for each element. It makes 1-1 pair with \a values.
\param values Value for each element. It makes 1-1 pair with \a column_indices.
\param b Values of rhs dense vector elements.
\param[in,out] x Values of lhs dense vector elements. Initially has the initial guess. On return, it
has the one-step-converged vector.
\param y Intermediate buffer.
\param rows Number of rows in matrix. */
__kernel void gauss_seidel_iteration(
	__global const double *preconditioner,
	__global const uint *row_indices,
	__global const uint *column_indices,
	__global const double *values,
	__global const double *b,
	__global double *x,
	__global double *y,
	uint rows)
{
	uint row = get_global_id(0);
	if (row >= rows) return;
	
	// final result: x += w * D^-1 * (b - A * x)
	// y = b - A * x
	residual_partial(row_indices, column_indices, values, row, b, x, y);
	// x += w * D^-1 * y
	x[row] += preconditioner[row] * y[row];
}
