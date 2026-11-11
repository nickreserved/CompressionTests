namespace MGroup.Solvers.LinearAlgebraExtensions.Matrices.Builders
{
	using System;
	using System.Collections.Generic;
	using System.Diagnostics;
	using System.Linq;
	using System.Text;

	using MGroup.LinearAlgebra.Commons;
	using MGroup.LinearAlgebra.Exceptions;
	using MGroup.LinearAlgebra.Implementations;
	using MGroup.LinearAlgebra.Matrices;
	using MGroup.Solvers.LinearAlgebraExtensions.Matrices.Builders;
	using MGroup.LinearAlgebra.Output.Formatting;
	using MGroup.LinearAlgebra.Vectors;

	/// <summary>
	/// Use this class for building large sparse matrices, e.g. <see cref="CsrMatrix"/>, not for operations. Convert to other 
	/// matrix formats once finished and use them instead for matrix operations. The large matrices and their properties will be 
	/// characterized as "global" in this namespace. This class is optimized for building global matrices with at least 1 entry 
	/// per column and row major storage formats.
	/// </summary>
	public class DokRowMajor : DefaultMatrix, ISparseMatrix
	{
		/// <summary>
		/// See the rant in <see cref="DokSymmetric.columns"/> about performance.
		/// </summary>
		private Dictionary<int, double>[] rows;

		private DokRowMajor(int numRows, int numCols, Dictionary<int, double>[] rows)
		{
			this.rows = rows;
			NumRows = numRows;
			NumColumns = numCols;
		}

		/// <summary>
		/// See <see cref="IIndexable2D.MatrixSymmetry"/>.
		/// </summary>
		public MatrixSymmetry MatrixSymmetry { get; set; }

		/// <summary>
		/// See <see cref="IIndexable2D.MatrixSymmetry"/>.
		/// </summary>
		MatrixSymmetry IIndexable2D.MatrixSymmetry => MatrixSymmetry;

		/// <summary>
		/// The number of columns of the matrix.
		/// </summary>
		public override int NumColumns { get; }

		/// <summary>
		/// The number of rows of the matrix.
		/// </summary>
		public override int NumRows { get; }

		/// <summary>
		/// The internal representation of the non-zero entries of the matrix. Each entry of the array Dictionary[int, double][] 
		/// is a row of the matrix and each entry Dictionary[int, double] corresponds to Dictionary[column, value].
		/// </summary>
		public Dictionary<int, double>[] RawRows => rows;

		/// <summary>
		/// See <see cref="IIndexable2D.this[int, int]"/>, <see cref="IMatrixBuilder.this[int, int]"/>.
		/// </summary>
		public override double this[int rowIdx, int colIdx]
		{
			get
			{
				Preconditions.CheckIndices(this, rowIdx, colIdx);
				if (rows[rowIdx].TryGetValue(colIdx, out var val)) return val;
				else return 0.0;
			}
			set //not thread safe
			{
				Preconditions.CheckIndices(this, rowIdx, colIdx);
				rows[rowIdx][colIdx] = value;
			}
		}

		/// <summary>
		/// Initializes a new instance of <see cref="DokRowMajor"/> with the specified matrix dimensions and all entries being
		/// equal to 0.
		/// </summary>
		/// <param name="numRows">The number of rows of the matrix to build.</param>
		/// <param name="numCols">The number of columns of the matrix to build.</param>
		public static DokRowMajor CreateEmpty(int numRows, int numCols)
		{
			var rows = new Dictionary<int, double>[numRows];
			for (var i = 0; i < numRows; ++i) rows[i] = new Dictionary<int, double>(); //Initial capacity may be optimized.
			return new DokRowMajor(numRows, numCols, rows);
		}

		/// <summary>
		/// Initializes a new instance of <see cref="DokRowMajor"/> with the specified matrix dimensions and entries being 
		/// the same as the identity matrix.
		/// </summary>
		/// <param name="order">The number of rows/columns of the square matrix to build.</param>
		public static DokRowMajor CreateIdentity(int order)
		{
			var rows = new Dictionary<int, double>[order];
			for (var j = 0; j < order; ++j)
			{
				var idenityRow = new Dictionary<int, double>(); //Initial capacity may be optimized.
				idenityRow[j] = 1.0;
				rows[j] = idenityRow;
			}
			return new DokRowMajor(order, order, rows);
		}

		/// <summary>
		/// Initializes a new instance of <see cref="DokRowMajor"/> with the specified matrix dimensions and the non-zero 
		/// entries of the provided sparse matrix.
		/// </summary>
		/// <param name="matrix">A sparse matrix whose dimensions and non-zero entries will be used to intialize the new
		///     <see cref="DokRowMajor"/>.</param>
		public static DokRowMajor CreateFromSparseMatrix(ISparseMatrix matrix)
			=> CreateFromSparsePattern(matrix.NumRows, matrix.NumColumns, matrix.EnumerateNonZeros());

		/// <summary>
		/// Initializes a new instance of <see cref="DokRowMajor"/> with the specified matrix dimensions and the non-zero 
		/// entries defined by the provided pattern.
		/// </summary>
		/// <param name="numRows">The number of rows of the matrix to build.</param>
		/// <param name="numCols">The number of columns of the matrix to build.</param>
		/// <param name="nonZeroEntries">The non-zero entries of the matrix to build.</param>
		public static DokRowMajor CreateFromSparsePattern(int numRows, int numColumns,
			IEnumerable<(int row, int col, double value)> nonZeroEntries)
		{
			var dok = CreateEmpty(numRows, numColumns);
			foreach (var (row, col, val) in nonZeroEntries) dok.rows[row].Add(col, val);
			return dok;
		}

		/// <summary>
		/// See <see cref="IMatrixBuilder.AddToEntry(int, int, double)"/>
		/// </summary>
		public void AddToEntry(int rowIdx, int colIdx, double value)
		{
			if (rows[rowIdx].TryGetValue(colIdx, out var oldValue))
			{
				rows[rowIdx][colIdx] = value + oldValue;
			}
			else rows[rowIdx][colIdx] = value;
			//The Dictionary rowss[rowIdx] is indexed twice in both cases. Is it possible to only index it once?
		}

		/// <summary>
		/// See <see cref="IGeneralMatrixBuilder.AddSubmatrix(IIndexable2D, IReadOnlyDictionary{int, int}, 
		/// IReadOnlyDictionary{int, int})"/>.
		/// </summary>
		public void AddSubmatrix(IIndexable2D subMatrix,
			IReadOnlyDictionary<int, int> subRowsToGlobalRows, IReadOnlyDictionary<int, int> subColsToGlobalCols)
		{
			foreach (var rowPair in subRowsToGlobalRows)
			{
				var subRow = rowPair.Key;
				var globalRow = rowPair.Value;
				//Debug.Assert((globalRow >= 0) && (globalRow < NumRows));
				foreach (var colPair in subColsToGlobalCols)
				{
					var subCol = colPair.Key;
					var globalCol = colPair.Value;
					//Debug.Assert((globalCol >= 0) && (globalCol < NumColumns));
					var subValue = subMatrix[subRow, subCol];
					if (rows[globalRow].TryGetValue(globalCol, out var oldGlobalValue))
					{
						rows[globalRow][globalCol] = subValue + oldGlobalValue;
					}
					else rows[globalRow][globalCol] = subValue;
				}
			}
		}

		/// <summary>
		/// See <see cref="IGeneralMatrixBuilder.AddSubmatrix(IIndexable2D, int[], int[], int[], int[])"/>.
		/// </summary>
		public void AddSubmatrix(IIndexable2D subMatrix, int[] subMatrixRows, int[] globalMatrixRows,
			int[] subMatrixCols, int[] globalMatrixCols)
		{
			var numRows = subMatrixRows.Length;
			var numCols = subMatrixCols.Length;
			Debug.Assert(numRows == globalMatrixRows.Length);
			Debug.Assert(numCols == globalMatrixCols.Length);

			for (var i = 0; i < numRows; ++i)
			{
				var subRow = subMatrixRows[i];
				var globalRow = globalMatrixRows[i];
				Debug.Assert(globalRow >= 0 && globalRow < NumRows);

				for (var j = 0; j < numCols; ++j)
				{
					var subCol = subMatrixCols[j];
					var globalCol = globalMatrixCols[j];
					Debug.Assert(globalCol >= 0 && globalCol < NumColumns);

					var subVal = subMatrix[subRow, subCol];
					rows[globalRow].TryGetValue(globalCol, out var oldGlobalVal); // default value = 0.0, if the entry is new
					rows[globalRow][globalCol] = subVal + oldGlobalVal;
				}
			}
		}

		/// <summary>
		/// See <see cref="IGeneralMatrixBuilder.AddSubmatrixSymmetric(IIndexable2D, IReadOnlyDictionary{int, int})"/>.
		/// </summary>
		public void AddSubmatrixSymmetric(IIndexable2D subMatrix, IReadOnlyDictionary<int, int> subIndicesToGlobalIndices)
		{
			foreach (var rowPair in subIndicesToGlobalIndices)
			{
				var subRow = rowPair.Key;
				var globalRow = rowPair.Value;
				foreach (var colPair in subIndicesToGlobalIndices)
				{
					var subCol = colPair.Key;
					var globalCol = colPair.Value;

					if (globalCol > globalRow)
					{
						var subValue = subMatrix[subRow, subCol];
						rows[globalRow].TryGetValue(globalCol, out var oldGlobalValue); // default value = 0.0
						var newGlobalValue = oldGlobalValue + subValue;
						rows[globalRow][globalCol] = newGlobalValue;
						rows[globalCol][globalRow] = newGlobalValue;
					}
					else if (globalCol == globalRow)
					{
						var subValue = subMatrix[subRow, subCol];
						rows[globalRow].TryGetValue(globalCol, out var oldGlobalValue); // default value = 0.0
						rows[globalRow][globalCol] = oldGlobalValue + subValue;
					}
				}
			}
		}

		/// <summary>
		/// See <see cref="IGeneralMatrixBuilder.AddSubmatrixSymmetric(IIndexable2D, int[], int[])"/>.
		/// </summary>
		public void AddSubmatrixSymmetric(IIndexable2D subMatrix, int[] subMatrixIndices, int[] globalIndices)
		{
			Debug.Assert(subMatrix.NumRows == subMatrix.NumColumns);
			Debug.Assert(globalIndices.Length == subMatrixIndices.Length);

			var numRelevantRows = subMatrixIndices.Length;
			for (var i = 0; i < numRelevantRows; ++i)
			{
				var subRow = subMatrixIndices[i];
				var globalRow = globalIndices[i];
				for (var j = 0; j < numRelevantRows; ++j)
				{
					var subCol = subMatrixIndices[j];
					var globalCol = globalIndices[j];

					if (globalCol > globalRow)
					{
						var subValue = subMatrix[subRow, subCol];
						rows[globalRow].TryGetValue(globalCol, out var oldGlobalValue); // default value = 0.0
						var newGlobalValue = oldGlobalValue + subValue;
						rows[globalRow][globalCol] = newGlobalValue;
						rows[globalCol][globalRow] = newGlobalValue;
					}
					else if (globalCol == globalRow)
					{
						var subValue = subMatrix[subRow, subCol];
						rows[globalRow].TryGetValue(globalCol, out var oldGlobalValue); // default value = 0.0
						rows[globalRow][globalCol] = oldGlobalValue + subValue;
					}
				}
			}
		}

		/// <summary>
		/// Creates the values and indexing arrays in CSR storage format of the current matrix. This method should be 
		/// called after fully defining the matrix in <see cref="DokRowMajor"/> format.
		/// </summary>
		/// <param name="sortColsOfEachCol">True to sort the column indices of the CSR matrix between rowOffsets[i] and 
		///     rowOffsets[i+1] in ascending order. False to leave them unordered. Ordered columns might result in better  
		///     performance during multiplications or they might be required by 3rd party libraries. Conversely, leaving them 
		///     unordered will be faster during creation of the CSR matrix.</param>
		/// <exception cref="EmptyMatrixBuilderException">Thrown if no non-zero entries have been defined yet.</exception>
		public (double[] values, int[] colIndices, int[] rowOffsets) BuildCsrArrays(bool sortColsOfEachCol)
		{
			var rowOffsets = new int[NumRows + 1];
			var nnz = 0;
			for (var i = 0; i < NumRows; ++i)
			{
				rowOffsets[i] = nnz;
				nnz += rows[i].Count;
			}
			if (nnz == 0) throw new EmptyMatrixBuilderException("Cannot build CSR arrays from a DOK with nnz = 0.");
			rowOffsets[NumRows] = nnz; //The last CSR entry is nnz.

			var colIndices = new int[nnz];
			var values = new double[nnz];
			var counter = 0;
			for (var i = 0; i < NumRows; ++i)
			{
				if (sortColsOfEachCol)
				{
					foreach (var rowVal in rows[i].OrderBy(pair => pair.Key))
					{
						colIndices[counter] = rowVal.Key;
						values[counter] = rowVal.Value;
						++counter;
					}
				}
				else
				{
					foreach (var rowVal in rows[i])
					{
						colIndices[counter] = rowVal.Key;
						values[counter] = rowVal.Value;
						++counter;
					}
				}
			}

			return (values, colIndices, rowOffsets);
		}

		/// <summary>
		/// Initializes a <see cref="CsrMatrix"/> representation of the current matrix. This method should be 
		/// called after fully defining the matrix in <see cref="DokRowMajor"/> format.
		/// </summary>
		/// <param name="sortColsOfEachRow">True to sort the column indices of the CSR matrix between rowOffsets[i] and 
		///     rowOffsets[i+1] in ascending order. False to leave them unordered. Ordered columns might result in better  
		///     performance during multiplications or they might be required by 3rd party libraries. Conversely, leaving them 
		///     unordered will be faster during creation of the CSR matrix.</param>
		/// <exception cref="EmptyMatrixBuilderException">Thrown if no non-zero entries have been defined yet.</exception>
		public CsrMatrix BuildCsrMatrix(bool sortColsOfEachRow)
		{
			(var values, var colIndices, var rowOffsets) = BuildCsrArrays(sortColsOfEachRow);
			return CsrMatrix.CreateFromArrays(NumRows, NumColumns, values, colIndices, rowOffsets, false);
		}

		/// <summary>
		/// Frees all memory held by this <see cref="DokRowMajor"/> instance. Afterwards this object cannot be reused. Therefore 
		/// this method should be called to save up space after building the matrix or its internal arrays.
		/// </summary>
		public override void Clear() => rows = null;

		/// <summary>
		/// See <see cref="ISparseMatrix.CountNonZeros"/>.
		/// </summary>
		public int CountNonZeros()
		{
			var count = 0;
			for (var i = 0; i < NumRows; ++i) count += rows[i].Count;
			return count;
		}

		/// <summary>
		/// Removes all entries Aij with |Aij| &lt;= <paramref name="tolerance"/>. Diagonal entries will be retained.
		/// </summary>
		/// <param name="tolerance">Must be positive or 0.</param>
		/// <param name="canDropDiagonalEntries">
		/// If false, entries that would fall bellow <paramref name="tolerance"/> will not be dropped.
		/// </param>
		public void DropEntriesBelowMagnitude(double tolerance, bool canDropDiagonalEntries)
		{
			if (tolerance < 0) throw new ArgumentException("Tolerance must be positive or 0");

			var colsToDrop = new List<int>(); 
			for (int row = 0; row < NumRows; row++)
			{
				colsToDrop.Clear();
				Dictionary<int, double> wholeRow = rows[row];
				foreach (var entry in wholeRow)
				{
					double abs = Math.Abs(entry.Value);
					if (abs <= tolerance)
					{
						int col = entry.Key;
						if (canDropDiagonalEntries || (col != row))
						{
							colsToDrop.Add(entry.Key);
						}
					}
				}

				foreach (int col in colsToDrop)
				{
					wholeRow.Remove(col);
				}
			}
		}

		/// <summary>
		/// See <see cref="ISparseMatrix.EnumerateNonZeros"/>.
		/// </summary>
		public IEnumerable<(int row, int col, double value)> EnumerateNonZeros()
		{
			for (var i = 0; i < NumRows; ++i)
			{
				foreach (var colVal in rows[i])
				{
					yield return (i, colVal.Key, colVal.Value);
				}
			}
		}

		public IEnumerable<(int col, double value)> EnumerateNonZerosOfRow(int rowIdx)
		{
			foreach (var colVal in rows[rowIdx])
			{
				yield return (colVal.Key, colVal.Value);
			}
		}

		/// <summary>
		/// See <see cref="IIndexable2D.Equals(IIndexable2D, double)"/>.
		/// </summary>
		public bool Equals(IIndexable2D other, double tolerance = 1e-13)
		{
			return DenseStrategies.AreEqual(this, other, tolerance);
		}

		/// <summary>
		/// Returns a <see cref="Vector"/> with the entries of the matrix's main diagonal and the index of the first zero entry.
		/// If there are no zero entries, -1 is returned as the index. The <see cref="DokRowMajor"/> matrix must be square.
		/// </summary>
		/// <exception cref="NonMatchingDimensionsException">Thrown if the matrix is not square.</exception>
		public (Vector diagonal, int firstZeroIdx) GetDiagonal()
		{
			(var diagonal, var firstZeroIdx) = GetDiagonalAsArray();
			return (Vector.CreateFromArray(diagonal, false), firstZeroIdx);
		}

		/// <summary>
		/// Returns an array with the entries of the matrix's main diagonal and the index of the first zero entry.
		/// If there are no zero entries, -1 is returned as the index. The <see cref="DokRowMajor"/> matrix must be square.
		/// </summary>
		/// <exception cref="NonMatchingDimensionsException">Thrown if the matrix is not square.</exception>
		public (double[] diagonal, int firstZeroIdx) GetDiagonalAsArray() //TODO: the -1 sentinel value should be a constant somewhere
		{
			Preconditions.CheckSquare(this);
			var diag = new double[NumRows];
			var firstZeroIdx = -1;
			for (var i = 0; i < NumRows; ++i)
			{
				var isStored = rows[i].TryGetValue(i, out var val);
				if (isStored) diag[i] = val;
				else
				{
					diag[i] = 0.0;
					firstZeroIdx = i;
				}
			}
			return (diag, firstZeroIdx);
		}

		/// <summary>
		/// See <see cref="ISparseMatrix.GetSparseFormat"/>.
		/// </summary>
		public SparseFormat GetSparseFormat()
		{
			(var values, var colIndices, var rowOffsets) = BuildCsrArrays(false);
			var format = new SparseFormat();
			format.RawValuesTitle = "Values";
			format.RawValuesArray = values;
			format.RawIndexArrays.Add("Column indices", colIndices);
			format.RawIndexArrays.Add("Row offsets", rowOffsets);
			return format;
		}

		public DokRowMajor GetSubmatrix(int[] rowsToKeep, int[] colsToKeep)
		{
			int numRowsResult = rowsToKeep.Length;
			int numColsResult = colsToKeep.Length;
			var oldToNewCols = new int[NumColumns];
			Array.Fill(oldToNewCols, -1);
			for (int j = 0; j < numColsResult; j++)
			{
				int J = colsToKeep[j];
				oldToNewCols[J] = j;
			}
			
			DokRowMajor result = CreateEmpty(numRowsResult, numColsResult);
			for (int i = 0; i < rowsToKeep.Length; i++)
			{
				int I = rowsToKeep[i];
				Dictionary<int, double> resultRows = result.rows[i];
				foreach (var colValPair in this.rows[I])
				{
					int J = colValPair.Key;
					int j = oldToNewCols[J];
					if (j >= 0)
					{
						double val = colValPair.Value;
						resultRows[j] = val;
					}
				}
			}

			return result;
		}

		//TODO: Diff this with the existing one. I made changes elsewhere
		public DokRowMajor KroneckerProduct(DokRowMajor other)
		{
			var mA = NumRows;
			var nA = NumColumns;
			var mB = other.NumRows;
			var nB = other.NumColumns;
			var mC = mA * mB;
			var nC = nA * nB;

			// Preallocate data structures for the resulting matrix
			var resultRows = new Dictionary<int, double>[mC];
			var nnzRowB = new int[mB];
			for (var iB = 0; iB < mB; iB++)
			{
				nnzRowB[iB] = other.rows[iB].Count;
			}

			for (var iA = 0; iA < mA; iA++)
			{
				var rowOffset = iA * mB;
				var nnzRowA = rows[iA].Count;
				for (var iB = 0; iB < mB; iB++)
				{
					var iC = rowOffset + iB;
					var nnzRowC = nnzRowA * nnzRowB[iB];
					resultRows[iC] = new Dictionary<int, double>(nnzRowC);
				}
			}

			// Perform the tensor product
			for (var iA = 0; iA < mA; iA++)
			{
				var rowA = rows[iA];
				var rowOffset = iA * mB;
				foreach ((var jA, var a) in rowA)
				{
					var colOffset = jA * nB;
					for (var iB = 0; iB < mB; iB++)
					{
						var rowB = other.rows[iB];
						var iC = rowOffset + iB;
						foreach ((var jB, var b) in rowB)
						{
							var jC = colOffset + jB;
							resultRows[iC].Add(jC, a * b);
						}
					}
				}
			}

			return new DokRowMajor(mC, nC, resultRows);
		}

		public DokRowMajor KroneckerProductIdentityTimesThis(int identityMatrixOrder)
		{
			if (identityMatrixOrder < 1)
			{
				throw new ArgumentOutOfRangeException(nameof(identityMatrixOrder), "The identity matrix order must be 1 or greater");
			}

			var mA = identityMatrixOrder;
			var mB = NumRows;
			var nB = NumColumns;
			var mC = mA * mB;
			var nC = mA * nB;
			var resultRows = new Dictionary<int, double>[mC];

			// Optimized case: first block can just be copied
			for (var iB = 0; iB < mB; iB++)
			{
				var rowB = rows[iB];
				var rowC = new Dictionary<int, double>(rowB);
				resultRows[iB] = rowC;
			}

			for (var iA = 1; iA < mA; iA++)
			{
				// Copy the original DOK to this block
				var rowOffset = iA * mB;
				var colOffset = iA * nB;
				for (var iB = 0; iB < mB; iB++)
				{
					var rowB = rows[iB];
					var rowC = new Dictionary<int, double>(rowB.Count);
					foreach ((var jB, var b) in rowB)
					{
						var jC = colOffset + jB;
						rowC.Add(jC, b);
					}

					var iC = rowOffset + iB;
					resultRows[iC] = rowC;
				}
			}

			return new DokRowMajor(mC, nC, resultRows);
		}

		public DokRowMajor KroneckerProductThisTimesIdentity(int identityMatrixOrder)
		{
			if (identityMatrixOrder < 1)
			{
				throw new ArgumentOutOfRangeException(nameof(identityMatrixOrder), "The identity matrix order must be 1 or greater");
			}

			var mA = NumRows;
			var nA = NumColumns;
			var mB = identityMatrixOrder;
			var mC = mA * mB;
			var nC = nA * mB;

			// Preallocate data structures for the resulting matrix
			var resultRows = new Dictionary<int, double>[mC];
			for (var iA = 0; iA < mA; iA++)
			{
				var rowOffset = iA * mB;
				var nnzRowA = rows[iA].Count;
				for (var iB = 0; iB < mB; iB++)
				{
					var iC = rowOffset + iB;
					resultRows[iC] = new Dictionary<int, double>(nnzRowA);
				}
			}

			// Perform the tensor product
			for (var iA = 0; iA < mA; iA++)
			{
				var rowA = rows[iA];
				var rowOffset = iA * mB;
				foreach ((var jA, var a) in rowA)
				{
					var colOffset = jA * mB;
					for (var iB = 0; iB < mB; iB++)
					{
						var iC = rowOffset + iB;
						var jC = colOffset + iB; // jB=iB
						resultRows[iC].Add(jC, a); // a*b = a*1
					}
				}
			}

			return new DokRowMajor(mC, nC, resultRows);
		}

		/// <summary>
		/// Performs the matrix-vector multiplication: this * <paramref name="vector"/>. This method is at least as fast as 
		/// creating a CSR matrix and multiplying it with 1 vector. If more than 1 multiplications are needed, then the CSR 
		/// matrix should be created.
		/// </summary>
		/// <param name="vector">A vector with <see cref="IIndexable1D.Length"/> being equal to this.<see cref="NumColumns"/>.
		///     </param>
		/// <param name="avoidBuilding">If true, no matrices will be built internally. If false, such matrices may be built if 
		///     the method decides that using them is faster (e.g. fast native library), however that requires memory for 
		///     building them, which may not be available.</param>
		/// <exception cref="NonMatchingDimensionsException">Thrown if 
		///     <paramref name="vector"/>.<see cref="IIndexable1D.Length"/> != this.<see cref="NumColumns"/>.</exception>
		public Vector MultiplyRight(Vector vector, bool avoidBuilding = false)
		{
			// SparseBLAS functions are way faster than managed code. Just don't sort the CSR.
			if (!avoidBuilding) return BuildCsrMatrix(false).Multiply(vector);

			Preconditions.CheckMultiplicationDimensions(NumColumns, vector.Length);
			var result = new double[NumRows];
			for (var i = 0; i < NumRows; ++i)
			{
				var dot = 0.0;
				foreach (var colValPair in rows[i]) dot += colValPair.Value * vector[colValPair.Key];
				result[i] = dot;
			}
			return Vector.CreateFromArray(result, false);
		}

		public double ReduceMaxAbs()
		{
			double max = -1;
			foreach (Dictionary<int, double> wholeRow in rows)
			{
				foreach (double val in wholeRow.Values)
				{
					double abs = Math.Abs(val);
					if (abs > max) max = abs;
				}
			}
			return max;
		}

		public void ScaleIntoThis(double scalar)
		{
			for (var i = 0; i < NumRows; i++)
			{
				var wholeRow = rows[i];
				foreach (var colValPair in wholeRow)
				{
					var j = colValPair.Key;
					wholeRow[j] *= scalar;
				}
			}
		}

		public DokRowMajor Transpose()
		{
			var result = CreateEmpty(NumColumns, NumRows); //TODO: Preallocate the dictionaries
			for (var i = 0; i < NumRows; i++)
			{
				foreach (var colValPair in rows[i])
				{
					var j = colValPair.Key;
					var val = colValPair.Value;
					result.rows[j][i] = val;
				}
			}

			return result;
		}

		public DokRowMajor_SubmatrixView ViewSubmatrix(int[] rowsToKeep)
			=> new DokRowMajor_SubmatrixView(rows, rowsToKeep, NumColumns);

		/// <summary>
		/// Use this method if 1) both the global DOK and the element matrix are symmetric and 2) the rows and columns correspond
		/// to the same degrees of freedom. The caller is responsible for making sure that both matrices are symmetric and that 
		/// the dimensions and dofs of the element matrix and dof mappings match.
		/// </summary>
		/// <param name="subMatrix">The element submatrix, entries of which will be added to the global DOK. It must be 
		///     symmetric and its <see cref="IIndexable2D.NumColumns"/> = <see cref="IIndexable2D.NumRows"/> must be equal to
		///     elemenDofs.Length = globalDofs.Length.</param>
		/// <param name="subDofs">The entries in the element matrix to be added to the global matrix. Specificaly, pairs of 
		///     (elementDofs[i], elementDofs[j]) will be added to (globalDofs[i], globalDofs[j]).</param>
		/// <param name="globalDofs">The entries in the global matrix where element matrix entries will be added to. Specificaly,
		///     pairs of (elementDofs[i], elementDofs[j]) will be added to (globalDofs[i], globalDofs[j]).</param>
		private void AddSubmatrixSymmetricOLD(IIndexable2D subMatrix, int[] subDofs, int[] globalDofs) //TODO: this should be reworked
		{
			var n = subDofs.Length;
			for (var i = 0; i < n; ++i)
			{
				var subRow = subDofs[i];
				var globalRow = globalDofs[i];

				//Diagonal entry
				if (rows[globalRow].TryGetValue(globalRow, out var oldGlobalDiagValue))
				{
					rows[globalRow][globalRow] = subMatrix[subRow, subRow] + oldGlobalDiagValue;
				}
				else rows[globalRow][globalRow] = subMatrix[subRow, subRow];

				//Non diagonal entries
				for (var j = 0; j < i; ++j)
				{
					var subCol = subDofs[j];
					var globalCol = globalDofs[j];
					var newGlobalValue = subMatrix[subRow, subRow];
					// Only check the lower triangle. If the DOK matrix is not symmetric, this will cause errors
					if (rows[globalRow].TryGetValue(globalCol, out var oldGlobalValue))
					{
						newGlobalValue += oldGlobalValue;
					}
					rows[globalRow][globalCol] = newGlobalValue;
					rows[globalCol][globalRow] = newGlobalValue;
				}
			}
		}

		public override IMatrix CreateZeroMatrixWithSameFormat() => DokRowMajor.CreateEmpty(NumRows, NumColumns);

		public override bool HasSameFormat(IReadOnlyMatrix otherMatrix) => otherMatrix == this;
	}
}
