/** A hybrid Gauss-Seidel or Jacobi iteration.
It must called with:
- global(0) discretization as the number of rows of matrix.
- local(0) discretization as a number 1/20 to 1/10 than average elements per row.
\param row_indices Indices to first element of matrix's each row.
\param column_indices Column index for each element. It makes 1-1 pair with \a values.
\param values Value for each element. It makes 1-1 pair with \a column_indices.
\param b Values of rhs dense vector elements.
\param[in,out] x Values of lhs dense vector elements. Initially has the initial guess. On return, it
has the one-step-converged vector.
\param lr Array with get_local_size(0) uninitialized elements. It exists only for intermediate
operations.
\param jacobi true: Jacobi iteration, false: hybrid Gauss-Seidel iteration. */
__kernel void iteration_step_with_CSR(
	__global const uint *row_indices,
	__global const uint *column_indices,
	__global const double *values,
	__global const double *b,
	__global double *x,
	__local double *lr,
	bool jacobi)
{
	uint row = get_global_id(0);
	uint block = get_local_id(0);
	uint index_from = row_indices[row];
	uint index_to   = row_indices[row + 1];
	uint total_blocks = get_local_size(0);
	uint size = index_to - index_from;
	index_from =  block      * size / total_blocks;
	index_to   = (block + 1) * size / total_blocks;

	__local double diag; // initialization is prohibited
	if (!block)
	{
		diag = 0;
		lr[block] = b[row];
	}
	else lr[block] = 0;

	if (column_indices[index_from] <= row && column_indices[index_to] >= row)
		for (; index_from < index_to; ++index_from)
			if (column_indices[index_from] == row) diag = values[index_from];
			else lr[block] -= values[index_from] * x[column_indices[index_from]];
	else
		for (; index_from < index_to; ++index_from)
			lr[block] -= values[index_from] * x[column_indices[index_from]];

	barrier(CLK_LOCAL_MEM_FENCE);


	// Binary reduction (one of two - then - one of four - etc work-items)
	for (uint dist = 1; dist < total_blocks; dist <<= 1)
	{
		if (!(block & dist))
		{
			uint distant = block + dist;
			if (distant < total_blocks) lr[block] += lr[distant];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	if (jacobi) barrier(CLK_GLOBAL_MEM_FENCE);
	if (!block) x[row] = lr[0] / diag;
}


/** Matrix - vector multiplication.
It must called with:
- global(0) discretization as the number of rows of matrix.
- local(0) discretization as a number 1/20 to 1/10 than average elements per row.
\param row_indices Indices to first element of matrix's each row.
\param column_indices Column index for each element. It makes 1-1 pair with \a values.
\param values Value for each element. It makes 1-1 pair with \a column_indices.
\param x Values of rhs dense vector elements.
\param[out] y Values of lhs dense vector elements.
\f$ \vec y = A\cdot\vec x\f$ or \f$ \vec y += A\cdot\vec x\f$ - it depends on \a clear.
\param lr Array with get_local_size(0) uninitialized elements. It exists only for intermediate
operations.
\param clear true: The operation is \f$ \vec y = A\cdot\vec x\f$, false: the operation is
\f$ \vec y += A\cdot\vec x\f$. */
__kernel void matrix_vector_product(
	__global const uint *row_indices,
	__global const uint *column_indices,
	__global const double *values,
	__global const double *x,
	__global double *y,
	__local double *lr,
	bool clear)
{
	uint row = get_global_id(0);
	uint block = get_local_id(0);
	uint index_from = row_indices[row];
	uint index_to   = row_indices[row + 1];
	uint total_blocks = get_local_size(0);
	uint size = index_to - index_from;
	index_from =  block      * size / total_blocks;
	index_to   = (block + 1) * size / total_blocks;

	lr[block] = 0;

	for (; index_from < index_to; ++index_from)
		lr[block] += values[index_from] * x[column_indices[index_from]];

	barrier(CLK_LOCAL_MEM_FENCE);

	// Binary reduction (one of two - then - one of four - etc work-items)
	for (uint dist = 1; dist < total_blocks; dist <<= 1)
	{
		if (!(block & dist))
		{
			uint distant = block + dist;
			if (distant < total_blocks) lr[block] += lr[distant];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	if (!block)
		if (clear) y[row] = lr[0]; else y[row] += lr[0];
}


__kernel void geometric_multigrid_with_CSR(
	__global const uint *row_indices,	// Indices to first element of matrix's each row
	__global const uint *column_indices,// Column index for each element
	__global const double *values,		// Value for each element
	__global double *b,					// Values of dense vector elements
	__global double *x,					// Values of dense vector elements
	__global uint *offsets,
	__global uchar *instructions,
	__global double *r,
	__local double *lr,					// Array with get_local_size(0) uninitialized elements
	char level,
	uint instr_begin,
	uint instr_end,
	bool jacobi)						// true: Jacobi iteration, false: hybrid Gauss-Seidel iteration
{
	for (; instr_begin < instr_end; ++instr_begin)
	{
		switch(instructions[instr_begin])
		{
			case 0:
				// multiply e with interpolation matrix and add it to e?
				--level;
			case 255:
				// multiply r = b - A*x with restriction matrix
				++level;
			default:
				// do iterations
				for (
				iteration_step_with_CSR
				
	__global const uint *row_indices,
	__global const uint *column_indices,
	__global const double *values,
	__global const double *b,
	__global double *x,
	__local double *lr,
	bool jacobi
	}
}














/* It must called with:
- global(0) discretization as the number of rows of matrix.
- local(0) discretization as a number smaller than average blocks per row. * /
__kernel void iteration_step_with_CSRB(
	__global const uint *row_indices,			// Indices to first block of matrix's each row
	__global const uint *column_indices,		// Column index for the first element of each row's block
	__global const uint *delta_distance_indices,// Distance array index of each row's block distances in block (elements - 1 in block) 
	__global const uchar *delta_distances,		// Distance between block's elements (0 is 1) - First distance of block is actually the number of distances in block (elements - 1 in block) 
	__global const double *values,				// Values of matrix elements. They make pairs with delta_distances
	__global double *vec)							// Values of dense vector elements
{
	uint row = get_global_id(0);
	uint block_index_from = row_indices[row];
	uint block_index_to   = row_indices[row + 1];
	uint work_item = get_local_id(0);
	uint total_work_items = get_local_size(0);
	uint size = block_index_to - block_index_from;
	block_index_from =  work_item      * size / total_work_items;
	block_index_to   = (work_item + 1) * size / total_work_items;

	__local double r[total_work_items];
	__local double diag = 0;
	r[work_item] = 0;

	for (; block_index_from < block_index_to; ++block_index_from) // iterate blocks in a row
	{
		uint column = column_indices[block_index_from];
		uint delta_distance_index = delta_distance_indices[block_index_from];
		uchar total_elements_in_block = delta_distances[delta_distance_index];
		if (row == column) diag = values[delta_distance_index];
		else r[work_item] += values[delta_distance_index] * vec[column];
		uint delta_distance_index_end = delta_distance_index + total_elements_in_block;
		for (++delta_distance_index; delta_distance_index < delta_distance_index_end; ++delta_distance_index)
		{
			column += 1 + delta_distances[delta_distance_index]
			if (row == column) diag = values[delta_distance_index];
			else r[work_item] += values[delta_distance_index] * vec[column];
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);


	// Binary reduction (one of two - then - one of four - etc work-items)
	for (uint dist = 1; dist < total_work_items; dist <<= 1)
	{
		if (!(work_item & dist))
		{
			uint distant = work_item + dist;
			if (distant < total_work_items) r[work_item] += r[distant];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}


/ * It must called with:
- global(0) discretization as the number of rows of matrix.
- local(0) discretization as a number smaller than average blocks per row. * /
__kernel void hybrid_gauss_seidel_step_with_CSRB(
	__global const uint *row_indices,			// Indices to first block of matrix's each row
	__global const uint *column_indices,		// Column index for the first element of each row's block
	__global const uint *delta_distance_indices,// Distance array index of each row's block distances in block (elements - 1 in block) 
	__global const uchar *delta_distances,		// Distance between block's elements (0 is 1) - First distance of block is actually the number of distances in block (elements - 1 in block) 
	__global const double *values,				// Values of matrix elements. They make pairs with delta_distances
	__global double *vec)							// Values of dense vector elements
{
	hybrid_gauss_seidel_step_with_CSRB(row_indices, column_indices, delta_distance_indices, delta_distances, values, vec);

	// division with diagonal element (only first work-item)
	// and placement on vector
	if (!work_item) vec[row] = r[0] / diag;
}


/ * It must called with:
- global(0) discretization as the number of rows of matrix.
- local(0) discretization as a number smaller than average blocks per row. * /
__kernel void jacobi_step_with_CSRB(
	__global const uint *row_indices,			// Indices to first block of matrix's each row
	__global const uint *column_indices,		// Column index for the first element of each row's block
	__global const uint *delta_distance_indices,// Distance array index of each row's block distances in block (elements - 1 in block) 
	__global const uchar *delta_distances,		// Distance between block's elements (0 is 1) - First distance of block is actually the number of distances in block (elements - 1 in block) 
	__global const double *values,				// Values of matrix elements. They make pairs with delta_distances
	__global double *vec)							// Values of dense vector elements
{
	hybrid_gauss_seidel_step_with_CSRB(row_indices, column_indices, delta_distance_indices, delta_distances, values, vec);

	// division with diagonal element (only first work-item)
	// and placement on vector
	barrier(CLK_GLOBAL_MEM_FENCE);		// Jacobi
	if (!work_item) vec[row] = r[0] / diag;
}





/ * It must called with:
- global(0) discretization as the number of rows of matrix.
- local(0) discretization as a number smaller than average blocks per row. * /
__kernel void iteration_step_with_CSRBVI(
	__global const uint *row_indices,			// Indices to first block of matrix's each row
	__global const uint *column_indices,		// Column index for the first element of each row's block
	__global const uint *delta_distance_indices,// Distance array index of each row's block distances in block (elements - 1 in block) 
	__global const uchar *delta_distances,		// Distance between block's elements (0 is 1) - First distance of block is actually the number of distances in block (elements - 1 in block) 
	__global const uchar *value_indices,		// Indices to distinct_values. They make pairs with delta_distances
	__global const double *distinct_values,		// Distinct values of matrix elements
	__global double *vec)							// Values of dense vector elements
{
	uint row = get_global_id(0);
	uint block_index_from = row_indices[row];
	uint block_index_to   = row_indices[row + 1];
	uint work_item = get_local_id(0);
	uint total_work_items = get_local_size(0);
	uint size = block_index_to - block_index_from;
	block_index_from =  work_item      * size / total_work_items;
	block_index_to   = (work_item + 1) * size / total_work_items;

	__local double r[total_work_items];
	__local double diag = 0;
	r[work_item] = 0;

	for (; block_index_from < block_index_to; ++block_index_from) // iterate blocks in a row
	{
		uint column = column_indices[block_index_from];
		uint delta_distance_index = delta_distance_indices[block_index_from];
		uchar total_elements_in_block = delta_distances[delta_distance_index];
		if (row == column) diag = distinct_values[value_indices[delta_distance_index]];
		else r[work_item] += distinct_values[value_indices[delta_distance_index]] * vec[column];
		uint delta_distance_index_end = delta_distance_index + total_elements_in_block;
		for (++delta_distance_index; delta_distance_index < delta_distance_index_end; ++delta_distance_index)
		{
			column += 1 + delta_distances[delta_distance_index]
			if (row == column) diag = distinct_values[value_indices[delta_distance_index]];
			else r[work_item] += distinct_values[value_indices[delta_distance_index]] * vec[column];
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);


	// Binary reduction (one of two - then - one of four - etc work-items)
	for (uint dist = 1; dist < total_work_items; dist <<= 1)
	{
		if (!(work_item & dist))
		{
			uint distant = work_item + dist;
			if (distant < total_work_items) r[work_item] += r[distant];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}


/ * It must called with:
- global(0) discretization as the number of rows of matrix.
- local(0) discretization as a number smaller than average blocks per row. * /
__kernel void hybrid_gauss_seidel_step_with_CSRBVI(
	__global const uint *row_indices,			// Indices to first block of matrix's each row
	__global const uint *column_indices,		// Column index for the first element of each row's block
	__global const uint *delta_distance_indices,// Distance array index of each row's block distances in block (elements - 1 in block) 
	__global const uchar *delta_distances,		// Distance between block's elements (0 is 1) - First distance of block is actually the number of distances in block (elements - 1 in block) 
	__global const double *value_indices,		// Indices to distinct_values. They make pairs with delta_distances
	__global const double *distinct_values,		// Distinct values of matrix elements
	__global double *vec)							// Values of dense vector elements
{
	hybrid_gauss_seidel_step_with_CSRBVI(row_indices, column_indices, delta_distance_indices, delta_distances, value_indices, distinct_values, vec);

	// division with diagonal element (only first work-item)
	// and placement on vector
	if (!work_item) vec[row] = r[0] / diag;
}


/ * It must called with:
- global(0) discretization as the number of rows of matrix.
- local(0) discretization as a number smaller than average blocks per row. * /
__kernel void jacobi_step_with_CSRBVI(
	__global const uint *row_indices,			// Indices to first block of matrix's each row
	__global const uint *column_indices,		// Column index for the first element of each row's block
	__global const uint *delta_distance_indices,// Distance array index of each row's block distances in block (elements - 1 in block) 
	__global const uchar *delta_distances,		// Distance between block's elements (0 is 1) - First distance of block is actually the number of distances in block (elements - 1 in block) 
	__global const double *value_indices,		// Indices to distinct_values. They make pairs with delta_distances
	__global const double *distinct_values,		// Distinct values of matrix elements
	__global double *vec)							// Values of dense vector elements
{
	hybrid_gauss_seidel_step_with_CSRBVI(row_indices, column_indices, delta_distance_indices, delta_distances, value_indices, distinct_values, vec);

	// division with diagonal element (only first work-item)
	// and placement on vector
	barrier(CLK_GLOBAL_MEM_FENCE);		// Jacobi
	if (!work_item) vec[row] = r[0] / diag;
}*/