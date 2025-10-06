namespace MGroup.LinearAlgebra.Matrices
{
    using MGroup.LinearAlgebra.Implementations;
    using MGroup.LinearAlgebra.Matrices.Builders;
    using MGroup.LinearAlgebra.Reduction;
    using MGroup.LinearAlgebra.Vectors;
    using System;
    using System.Collections.Generic;

    public class DuViCompressedSparseMatrix : IMatrix // This is the correct: ILinearTransformation, but the whole library built upon IMatrix
	{
		public class Unit
		{
			public Unit(bool newRow, DeltaBytes db, int nextColumn, int[] deltaCols)
			{
				const int low2 = 1 << 7;
                const int low3 = (1 << 14) + low2;
                const int low4 = (1 << 21) + low3;
                const int low5 = (1 << 29) + low4;

                int totalBytes = 2 + 1 + (1 << (int)db) * deltaCols.Length;
				// variable integer size - 1
				     if (nextColumn >= low5) throw new NotImplementedException();
				else if (nextColumn >= low4) totalBytes += 3;
				else if (nextColumn >= low3) totalBytes += 2;
				else if (nextColumn >= low2) totalBytes += 1;

				data = new byte[totalBytes];

				data[0] = (byte) ((newRow ? 1 : 0) | ((int)db << 1));
				data[1] = (byte)deltaCols.Length;

				// variable integer storage (big endian)
				if (nextColumn >= low4)
				{
					nextColumn -= low4;
					data[2] = (byte)((nextColumn >> 24) | 128 | 64 | 32);
					data[3] = (byte)(nextColumn >> 16);
					data[4] = (byte)(nextColumn >> 8);
					data[5] = (byte)nextColumn;
					totalBytes = 2 + 4;
				}
				else if (nextColumn >= low3)
				{
                    nextColumn -= low3;
                    data[2] = (byte)((nextColumn >> 16) | 128 | 64);
					data[3] = (byte)(nextColumn >> 8);
					data[4] = (byte)nextColumn;
					totalBytes = 2 + 3;
				}
				else if (nextColumn >= low2)
				{
                    nextColumn -= low2;
                    data[2] = (byte)((nextColumn >> 8) | 128);
					data[3] = (byte)nextColumn;
					totalBytes = 2 + 2;
				}
				else
				{
					data[2] = (byte)nextColumn;
					totalBytes = 2 + 1;
				}

				// delta bytes storage, if they are one, two or four bytes
				switch(db)
				{
					case DeltaBytes.one:
						for (int i = 0; i < deltaCols.Length; ++i)
							data[totalBytes++] = (byte)deltaCols[i];
						break;
					case DeltaBytes.two:
						for (int i = 0; i < deltaCols.Length; ++i)
						{
							data[totalBytes++] = (byte)deltaCols[i];
							data[totalBytes++] = (byte)(deltaCols[i] >> 8);
						}
						break;
					case DeltaBytes.four:
						for (int i = 0; i < deltaCols.Length; ++i)
						{
							data[totalBytes++] = (byte)deltaCols[i];
							data[totalBytes++] = (byte)(deltaCols[i] >> 8);
							data[totalBytes++] = (byte)(deltaCols[i] >> 16);
							data[totalBytes++] = (byte)(deltaCols[i] >> 24);
						}
						break;
					default: throw new NotImplementedException();
				}
			}
			public enum DeltaBytes { one, two, four, eight };
			private readonly byte[] data;

			/// <summary>
			/// This is the first unit in matrix row. So first element of the unit is first element of the row.
			/// </summary>
			public bool IsNewRow() { return (data[0] & 1) == 1; }
			
			/// <summary>
			/// Column delta between consecutive non-zero elements can be represented in specific number of bytes.
			/// </summary>
			/// <returns>The number of bytes which can represent the difference between columns of 2 consecutive non-zero elements of row</returns>
			public DeltaBytes GetDeltaBytes() { return (DeltaBytes)(data[0] >> 1); }

            /// <summary>
            /// The number of column distances (delta) between consecutive non-zero elements in unit. This is the number of elements - 1.
            /// </summary>
            /// <returns>Number of elements in unit - 1.</returns>
            public int GetTotalEntries() { return data[1]; }
			public static (int, int) ExtractVariableInteger(byte[] data, int index)
			{
                const int low2 = 1 << 7;
                const int low3 = (1 << 14) + low2;
                const int low4 = (1 << 21) + low3;
                const int low5 = (1 << 29) + low4;
                
				if ((data[index] & 128) == 0) return ( data[index], 1 );
				if ((data[index] & 64) == 0) return ( low2 + (((data[index] & 63) << 8) | data[index + 1]), 2 );
				if ((data[index] & 32) == 0) return ( low3 + (((data[index] & 31) << 16) | (data[index + 1] << 8) | data[index + 2]), 3 );
				return ( low4 + (((data[index] & 31) << 24) | (data[index + 1] << 16) | (data[index + 2] << 8) | data[index + 3]), 4);
			}

            /// <summary>
            /// The column distance from last element of previous unit, or the beginning of row if the unit is the first in row, with the first element of this unit.
            /// </summary>
            /// <returns>A tuple of 2 integers.
            /// First integer is the column distance from last element of previous unit, or the beginning of row if the unit is the first in row, with the first element of this unit.
			/// Second integer is the number of bytes to represent first integer in unit. This is useful in order to proceed with information next.</returns>
            public (int, int) GetNextColumn() { return ExtractVariableInteger(data, 2); }

            /// <summary>
            /// Get the column distance of a non-zero element in row with next non-zero element in row in unit.
            /// </summary>
            /// <param name="index">Index of non-zero element in row in unit</param>
            /// <param name="db">Number of bytes required to represent the column distance between row elements in unit</param>
            /// <param name="byte_array_index">The byte index of the beginning of the column deltas. This is always the value:
			/// 2 + GetNextColumn().Item2</param>
            /// <returns>The column distance of given with <paramref name="index"/> non-zero element in unit with next non-zero element in unit.</returns>
            /// <exception cref="NotImplementedException"></exception>
            public int GetIndex(int index, DeltaBytes db, int byte_array_index)
			{
				switch(db)
				{
					case DeltaBytes.one: return data[byte_array_index + index];
					case DeltaBytes.two:
						byte_array_index += index * 2;
						return data[byte_array_index] | (data[byte_array_index + 1] << 8);
					case DeltaBytes.four:
						byte_array_index += index * 4;
						return data[byte_array_index] | (data[byte_array_index + 1] << 8) | (data[byte_array_index + 2] << 16) | (data[byte_array_index + 3] << 24);
					default: throw new NotImplementedException();
				}
			}
		}

		public int NumColumns { get; }
		public int NumRows { get; }

        private readonly double[] uniqueValues;
        private readonly int[] valueIndices;
		public Unit[] units;

		/// <summary>
		/// If matrix was not VI compressed but CSR then returns the value with index <paramref name="idx"/>.
		/// </summary>
		/// <param name="idx">The index of value. If matrix was CSR this is the idx in the columnIndices</param>
		/// <returns>The corresponding value of entry</returns>
		public double GetValueAtIndex(int idx) => uniqueValues[valueIndices[idx]];

		internal class ToleranceComparer : IEqualityComparer<double>
		{
			private readonly double tolerance;
			public ToleranceComparer(double tolerance) => this.tolerance = tolerance;
			public bool Equals(double x, double y) => Math.Abs(x - y) < tolerance;
			public int GetHashCode(double x) => Math.Round(x / (2 * tolerance)).GetHashCode();
		}

		public DuViCompressedSparseMatrix(int height, int width, double[] values, double tolerance = 1e-10)
		{
			NumRows = height; NumColumns = width;

			Dictionary<double, int> map = tolerance > 0 ? new Dictionary<double, int>(new ToleranceComparer(tolerance))
														: new Dictionary<double, int>();
			List<Unit> units = new();
			List<int> valueIndices = new();
			List<double> uniqueValues = new();

			// find the units with the same integer type of delta (uint8, uint16, uint32)
			for (int y = 0, indexIn = 0; y < height; ++y)
			{//TODO: Δεν λαμβάνει την περίπτωση όλα τα στοιχεία μιας γραμμής να είναι μηδέν ή όλος ο πίνακας να είναι μηδέν
				Unit.DeltaBytes db = Unit.DeltaBytes.one;
				bool newRow = true;
				int deltaNextUnit = -1, xOld = -1; // dummy values
				int x;
				for (x = 0; x < width; ++x)
				{
					double v = values[indexIn++];
					if (Math.Abs(v) > tolerance)
					{
						// value storage
						if (map.TryGetValue(v, out int index))
							valueIndices.Add(index);
						else
						{
							valueIndices.Add(map.Count);
							uniqueValues.Add(v);
							map[v] = map.Count;
						}

						// index storage
						xOld = x; deltaNextUnit = x;
						++x;
						break;
					}
				}
				List<int> deltaCols = new();
				for (; x < width; ++x)
				{
					double v = values[indexIn++];
					if (Math.Abs(v) > tolerance)
					{

						// value storage
						if (map.TryGetValue(v, out int index))
							valueIndices.Add(index);
						else
						{
							valueIndices.Add(map.Count);
							uniqueValues.Add(v);
							map[v] = map.Count;
						}

						// index storage
						int delta = x - xOld;
						Action func = () =>	// eliminate code duplication
						{
							if (deltaCols.Count > 0)
							{
								units.Add(new Unit(newRow, db, deltaNextUnit, deltaCols.ToArray()));
								newRow = false;
								deltaNextUnit = delta;
								deltaCols.Clear();
							}
							else deltaCols.Add(delta);
						};
						if (delta < 256)
						{
							if (db != Unit.DeltaBytes.one)
							{
								func();
								db = Unit.DeltaBytes.one;
							}
							else deltaCols.Add(delta);
						}
						else if (delta < 65536)
						{
							if (db != Unit.DeltaBytes.two)
							{
								func();
								db = Unit.DeltaBytes.two;
							}
							else deltaCols.Add(delta);
						}
						else
						{
							if (db != Unit.DeltaBytes.four)
							{
								func();
								db = Unit.DeltaBytes.four;
							}
							else deltaCols.Add(delta);
						}

						xOld = x;
					}
				}
				// close open unit
				if (deltaNextUnit != -1) // -1 is impossible actually : All of row elements are zero
					units.Add(new Unit(newRow, db, deltaNextUnit, deltaCols.ToArray()));
			}
		
			this.uniqueValues = uniqueValues.ToArray();
			this.valueIndices = valueIndices.ToArray();
			this.units = units.ToArray();
		}

		public DuViCompressedSparseMatrix(int height, int width, double[] values, int[] colIndices, int[] rowIndices, double tolerance = 1e-10)
		{
			NumRows = height; NumColumns = width;

			Dictionary<double, int> map = tolerance > 0 ? new Dictionary<double, int>(new ToleranceComparer(tolerance))
														: new Dictionary<double, int>();
			List<Unit> units = new();
			List<int> valueIndices = new();
			List<double> uniqueValues = new();

			// find the units with the same integer type of delta (uint8, uint16, uint32)
			for (int y = 0; y < height; ++y)
			{
				int rowIndex = rowIndices[y];
				bool empty = rowIndices[y + 1] - rowIndex == 0;
				if (empty)
				{
                    // value storage
					// required because a new unit implies an index and a corresponding value
                    if (map.TryGetValue(0.0, out int indeX))
                        valueIndices.Add(indeX);
                    else
                    {
                        valueIndices.Add(map.Count);
                        uniqueValues.Add(0.0);
                        map[0.0] = map.Count;
                    }

                    // index storage
                    // at least one unit per row because it signs "new row"
                    units.Add(new Unit(true, Unit.DeltaBytes.one, 0, Array.Empty<int>()));
                    continue;
				}

				Unit.DeltaBytes db = Unit.DeltaBytes.one;
				bool newRow = true;
				int xOld = colIndices[rowIndex];
				int deltaNextUnit = xOld;

				// value storage
				double v = values[rowIndex];
				if (map.TryGetValue(v, out int index))
					valueIndices.Add(index);
				else
				{
					valueIndices.Add(map.Count);
					uniqueValues.Add(v);
					map[v] = map.Count;
				}

				List<int> deltaCols = new();
				for (int idx = rowIndex + 1; idx < rowIndices[y + 1]; ++idx)
				{
					v = values[idx];
					// value storage
					if (map.TryGetValue(v, out index))
						valueIndices.Add(index);
					else
					{
						valueIndices.Add(map.Count);
						uniqueValues.Add(v);
						map[v] = map.Count;
					}

					// index storage
					int x = colIndices[idx];
					int delta = x - xOld;
					Action func = () => // eliminate code duplication
					{
						if (deltaCols.Count > 0)
						{
							units.Add(new Unit(newRow, db, deltaNextUnit, deltaCols.ToArray()));
							newRow = false;
							deltaNextUnit = delta;
							deltaCols.Clear();
						}
						else deltaCols.Add(delta);
					};
					if (delta < 256)
					{
						if (db != Unit.DeltaBytes.one)
						{
							func();
							db = Unit.DeltaBytes.one;
						}
						else deltaCols.Add(delta);
					}
					else if (delta < 65536)
					{
						if (db != Unit.DeltaBytes.two)
						{
							func();
							db = Unit.DeltaBytes.two;
						}
						else deltaCols.Add(delta);
					}
					else
					{
						if (db != Unit.DeltaBytes.four)
						{
							func();
							db = Unit.DeltaBytes.four;
						}
						else deltaCols.Add(delta);
					}

					xOld = x;
				}
				// close open unit
				units.Add(new Unit(newRow, db, deltaNextUnit, deltaCols.ToArray()));
			}

			this.uniqueValues = uniqueValues.ToArray();
			this.valueIndices = valueIndices.ToArray();
			this.units = units.ToArray();
		}

		public DuViCompressedSparseMatrix(int height, int width)
		{
			NumRows = height; NumColumns = width;
			uniqueValues = new double[1];   //     0 by def
			valueIndices = new int[height]; // all 0 by def
			Unit u = new(true, Unit.DeltaBytes.one, 0, Array.Empty<int>());
			units = new Unit[height];
			for (int i = 0; i < units.Length; ++i)
				units[i] = u;
		}

		public DuViCompressedSparseMatrix(int height, int width, (double[], int[], int[]) csrArrays, double tolerance = 1e-10)
			: this(height, width, csrArrays.Item1, csrArrays.Item2, csrArrays.Item3, tolerance) { }

		public DuViCompressedSparseMatrix(DokRowMajor mat, double tolerance = 1e-10)
			: this(mat.NumRows, mat.NumColumns, mat.BuildCsrArrays(true), tolerance) { }
        
		public void Multiply(IVectorView lhsVector, IVector rhsVector) => Multiply(lhsVector, (Vector)rhsVector);

		public void Multiply(IVectorView lhsVector, Vector rhsVector)
		{
			int row = -1, col = 0, valueIndex = 0;
			foreach (Unit unit in units)
			{
				if (unit.IsNewRow()) { ++row; col = 0; rhsVector[row] = 0; }
				int totalEntries = unit.GetTotalEntries();
				(int delta, int size) = unit.GetNextColumn();
				size += 1 /* header */ + 1 /* num entries */;
				col += delta;
				double value = uniqueValues[valueIndices[valueIndex++]];
				rhsVector[row] += value * lhsVector[col];
				Unit.DeltaBytes db = unit.GetDeltaBytes();
				for (int i = 0; i < totalEntries; ++i)
				{
					col += unit.GetIndex(i, db, size);
					value = uniqueValues[valueIndices[valueIndex++]];
					rhsVector[row] += value * lhsVector[col];
				}
			}
		}
        public void MultiplyIntoResult(IVectorView lhsVector, IVector rhsVector, bool transposeThis = false)
        {
            if (transposeThis) throw new NotImplementedException();
            Multiply(lhsVector, rhsVector);
        }

        public MatrixSymmetry MatrixSymmetry => throw new NotImplementedException();

        public double this[int rowIdx, int colIdx]
		{
			get
			{
                int row = -1, col = 0, valueIndex = 0;
                foreach (Unit unit in units)
                {
                    if (unit.IsNewRow()) { ++row; col = 0; }
                    int totalEntries = unit.GetTotalEntries();
                    (int delta, int size) = unit.GetNextColumn();
                    size += 1 /*header*/ +1 /*num entries*/;
                    col += delta;
					if (row == rowIdx)
					{
						if (col == colIdx) return uniqueValues[valueIndices[valueIndex]];
						else if (col > colIdx) return 0;
					}
					else if (row > rowIdx) return 0;
                        ++valueIndex;
                    Unit.DeltaBytes db = unit.GetDeltaBytes();
                    for (int i = 0; i < totalEntries; ++i)
                    {
                        col += unit.GetIndex(i, db, size);
                        if (row == rowIdx)
                        {
                            if (col == colIdx) return uniqueValues[valueIndices[valueIndex]];
                            else if (col > colIdx) return 0;
                        }
                        else if (row > rowIdx) return 0;
                        ++valueIndex;
                    }
                }
                return 0;
            }
        }
        public void AxpyIntoThis(IMatrixView otherMatrix, double otherCoefficient) => throw new NotImplementedException();
        public void Clear() => throw new NotImplementedException();
        public void LinearCombinationIntoThis(double thisCoefficient, IMatrixView otherMatrix, double otherCoefficient) => throw new NotImplementedException();
        public void ScaleIntoThis(double scalar) => throw new NotImplementedException();
		public void SetEntryRespectingPattern(int rowIdx, int colIdx, double value) => throw new NotImplementedException();
        public Matrix CopyToFullMatrix() => throw new NotImplementedException();
		public Matrix MultiplyLeft(IMatrixView other, bool transposeThis = false, bool transposeOther = false) => throw new NotImplementedException();
        public Matrix MultiplyRight(IMatrixView other, bool transposeThis = false, bool transposeOther = false) => throw new NotImplementedException();

        public IVector Multiply(IVectorView vector, bool transposeThis = false)
		{
			if (transposeThis) throw new NotImplementedException();
			IVector result = Vector.CreateZero(NumRows);
			Multiply(vector, result);
			return result;
		}
		
        public IMatrix Transpose() => throw new NotImplementedException();
        public double Reduce(double identityValue, ProcessEntry processEntry, ProcessZeros processZeros, Finalize finalize) => throw new NotImplementedException();
        public IMatrix DoEntrywise(IMatrixView matrix, Func<double, double, double> binaryOperation) => throw new NotImplementedException();
        public IMatrix DoToAllEntries(Func<double, double> unaryOperation) => throw new NotImplementedException();
        public Vector GetColumn(int colIndex) => throw new NotImplementedException();
        public Vector GetRow(int rowIndex) => throw new NotImplementedException();
        public IMatrix GetSubmatrix(int[] rowIndices, int[] colIndices) => throw new NotImplementedException();
        public IMatrix GetSubmatrix(int rowStartInclusive, int rowEndExclusive, int colStartInclusive, int colEndExclusive) => throw new NotImplementedException();
        public bool Equals(IIndexable2D other, double tolerance = 1E-13) => throw new NotImplementedException();
        public void DoEntrywiseIntoThis(IMatrixView matrix, Func<double, double, double> binaryOperation) => throw new NotImplementedException();
        public void DoToAllEntriesIntoThis(Func<double, double> unaryOperation) => throw new NotImplementedException();
    }
}
