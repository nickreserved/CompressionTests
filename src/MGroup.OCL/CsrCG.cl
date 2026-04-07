//--------------------------------------------------------------------------------- DOT PRODUCT ----


/** Dot product common code.
\param partial Output vector with <tt>size == num_of_workgroups</tt>.
\param local_mem Accumulation local memory with <tt>size == local_size</tt> -- without useful data
before and after execution.
\param sum Sum of workitem. */
void dot_common(
	__global double* partial,
	__local double* local_mem,
	double sum)
{
	uint lid = get_local_id(0);
	uint group_id = get_group_id(0);
	uint local_size = get_local_size(0);
	
	// Store to local memory
	local_mem[lid] = sum;
	barrier(CLK_LOCAL_MEM_FENCE);

	// Reduction inside work-group
	for (uint stride = local_size >> 1; stride > 0; stride >>= 1)
	{
		if (lid < stride)
			local_mem[lid] += local_mem[lid + stride];
		barrier(CLK_LOCAL_MEM_FENCE);
	}

    // On result per work-group
	if (lid == 0)
		partial[group_id] = local_mem[0];
}


/** Dot product first pass.
If vector has size of \a n then it is prefered <tt>local_size < global_size << n</tt>.
\param v1,v2 The vectors. They have size of \a n.
\param partial Output vector with <tt>size == num_of_workgroups</tt>. It is used on next stage.
\param local_mem Accumulation local memory with <tt>size == local_size</tt> -- without useful data
before and after execution.
\param n The size of vectors \a v1, \a v2. */
__kernel void dot_partial(
	__global const double* v1,
	__global const double* v2,
	__global double* partial,
	__local double* local_mem,
	uint n)
{
	uint gid = get_global_id(0);
	uint global_size = get_global_size(0);

	// Grid-stride loop
	double sum = 0.0;
	for (uint i = gid; i < n; i += global_size)
		sum += v1[i] * v2[i];
	
	dot_common(partial, local_mem, sum);
}


/** Dot product second and last pass.
Must be <tt>local_size == global_size</tt> which means only 1 workgroup.
\param partial Input vector from previous stage with <tt>size == n</tt>.
\param result The result dot product. It has <tt>size >= 1</tt>.
\param local_mem Accumulation local memory with <tt>size == local_size</tt> -- without useful data
before and after execution.
\param n Number of workgroups of previous stage. */
__kernel void dot_finalize(
	__global const double* partial,
	__global double* result,
	__local double* local_mem,
	uint n)
{
	uint lid = get_local_id(0);
	uint local_size = get_local_size(0);

	// Workgroup-stride loop
	// n can be lower than local_size
	double sum = 0.0;
	for (uint i = lid; i < n; i += local_size)
		sum += partial[i];

	dot_common(result, local_mem, sum);
}


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


// ---------------------------------------------------------------------- CONJUGATE GRADIENT RELATED



/** Calculate the the direction \a p and the direction \a z on algorithm initialization.
This happens once, before the main algorithm's loop.
The difference from \c initialize() is that this function implies initial vector \a x guess as zero.
\param M The Jacobi preconditioner of matrix \a A, inverted. It is the main diagonal of matrix \a A
with its elements inverted.
\param r The vector \a b.
\param[out] p,z The direction vectors.
\param rows Number of rows in matrix, vectors and preconditioner. */
__kernel void initialize0(
	__global const double *M,
	__global const double *r,
	__global double *p,
	__global double *z,
	uint rows)
{
	uint row = get_global_id(0);
	if (row >= rows) return;
	
	p[row] = z[row] = M[row] * r[row]; // p = z = M^-1 * r
}


/** Calculate the residual \a r, the direction \a p and the direction \a z on algorithm initialization.
This happens once, before the main algorithm's loop.
\param row_indices Indices to first element of matrix's each row.
\param column_indices Column index for each element. It makes 1-1 pair with \a values.
\param values Value for each element. It makes 1-1 pair with \a column_indices.
\param M The Jacobi preconditioner of matrix \a A, inverted. It is the main diagonal of matrix \a A
with its elements inverted.
\param x Initial guess for the unknown vector \a x.
\param[in,out] r The vector \a b before the call and the residual vector \f$ \vec r = \vec b - A\cdot\vec x\f$
after the call.
\param[out] p,z The direction vectors.
\param rows Number of rows in matrix, vectors and preconditioner. */
__kernel void initialize(
	__global const uint *row_indices,
	__global const uint *column_indices,
	__global const double *values,
	__global const double *M,
	__global const double *x,
	__global double *r,
	__global double *p,
	__global double *z,
	uint rows)
{
	uint row = get_global_id(0);
	if (row >= rows) return;
	
	// r = b - A * x
	r[row] = -r[row];	// r = -b
	matrix_vector_product_partial(row_indices, column_indices, values, row, x, r, false);// r += A * x
	r[row] = -r[row]; // r = -r
	// p = z = M^-1 * r
	p[row] = z[row] = M[row] * r[row];
}


/** Dot product second and last pass and calculation of coefficient \a a.
First vector of dot is the \a p and second is the \f$ A\cdot p\f$.
On this kernel must be <tt>local_size == global_size</tt>.
\param partial Input vector from previous stage with <tt>size == n</tt>.
\param a It has <tt>size == 2</tt>. <tt>a[0]</tt> must be \f$ r\cdot z\f$ while <tt>a[1]</tt>
becomes the coefficient \a a.
\param local_mem Accumulation local memory with <tt>size == local_size</tt> -- without useful data
before and after execution.
\param n Number of workgroups of previous stage. */
__kernel void dot_finalize_and_calc_a(
	__global const double *partial,
	__global double *a,
	__local double *local_mem,
	uint n)
{
	uint lid = get_local_id(0);

	if (lid == 0) a[1] = a[0];	// it will be overwritten

	dot_finalize(partial, a, local_mem, n);

	if (lid == 0) a[1] /= a[0];
}


/** Update vectors \a x, \a r, \a z and check residual r for convergence.
\param M The inverted Jacobi matrix of matrix \a A. Size is n.
\param Ap The \f$ A\cdot\vec p\f$ vector. Size is n.
\param p The p vector. Size is n.
\param a It has <tt>size == 2</tt>. <tt>a[1]</tt> must be the coefficient \a a.
\param x The direction \a x vector. Size is \a n.
\param r The residual \a r vector. Size is \a n.
\param z The \a z vector. Size is \a n.
\param n The size of all vectors.
\param is_zero Must be 1. After function, bit 1 means convergence failure (finish) and bit 0 means
convergence success (finish). 0 means no convergence yet (continue).
\param tolerance If even one element of residual has larger value than this, algorithm is not
converged. It must be positive. */
__kernel void update_x_r_z_check_r(
	__global const double *M,
	__global const double *Ap,
	__global const double *p,
	__global const double *a,
	__global double *x,
	__global double *r,
	__global double *z,
	uint n,
	__global atomic_uint *is_zero,
	double tolerance)
{
	uint gid = get_global_id(0);
	if (gid >= n) return;
	x[gid] += a[1] * p[gid];
	r[gid] -= a[1] * Ap[gid];
	z[gid] = M[gid] * r[gid];
	double c = fabs(r[gid]);
	if (isnan(r[gid]) || c > 1e50) atomic_fetch_or (is_zero, 2);	// flag 2: convergence error -- stop now
	else if (c > tolerance)        atomic_fetch_and(is_zero, 2); // remove flag 1: not converged -- continue
}


/** Dot product second and last pass and calculation of coefficient \a b.
First vector of dot is the r and second is the \a z.
On this kernel must be <tt>local_size == global_size</tt>.
\param partial Input vector from previous stage with <tt>size == n</tt>.
\param b The result dot product. It has <tt>size == 2</tt>. <tt>a[0]</tt> is the old
\f$ \vec r\cdot\vec z\f$ and becomes the new, while <tt>a[1]</tt> becomes the coefficient \a b.
\param local_mem Accumulation local memory with <tt>size == local_size</tt> -- without useful data
before and after execution.
\param n Number of workgroups of previous stage. */
__kernel void dot_finalize_and_calc_b(
	__global const double *partial,
	__global double *b,
	__local double *local_mem,
	uint n)
{
	uint lid = get_local_id(0);
	if (lid == 0) b[1] = b[0];	// it will be overwritten

	dot_finalize(partial, b, local_mem, n);	// b[0] = r * z (new)
	
	if (lid == 0) b[1] = b[0] / b[1];
}


/** Update vector \a p.
\param z The \a z vector. Size is \a n.
\param p The direction vector. Size is \a n.
\param a It has <tt>size == 2</tt>. <tt>a[1]</tt> must be the coefficient \a b.
\param x The direction x vector. Size is n.
\param r The residual r vector. Size is n.
\param n The size of all vectors. */
__kernel void update_p(
	__global const double *z,
	__global double *p,
	__global const double *b,
	uint n)
{
	uint gid = get_global_id(0);
	if (gid >= n) return;
	p[gid] = z[gid] + b[1] * p[gid];
}
