namespace MGroup.Solvers.LinearAlgebraExtensions.Matrices.Builders
{
	using System;
	using System.Collections.Generic;
	using System.Text;

	public class DokRowMajor_SubmatrixView
	{
		private readonly Dictionary<int, double>[] rows;

		public DokRowMajor_SubmatrixView(Dictionary<int, double>[] allRows, int[] rowsToKeep, int numColumns)
		{
			NumColumns = numColumns;
			NumRows = rowsToKeep.Length;
			this.rows = new Dictionary<int, double>[NumRows];
			for (int i = 0; i < NumRows; i++)
			{
				rows[i] = allRows[rowsToKeep[i]];
			}
		}

		public int NumColumns { get; }

		public int NumRows { get; }

		public int[] CountNonZerosOfColumns()
		{
			var result = new int[NumColumns];
			foreach (Dictionary<int, double> dataOfRow in rows)
			{
				foreach (int col in dataOfRow.Keys)
				{
					result[col]++;
				}
			}

			return result;
		}
	}
}
