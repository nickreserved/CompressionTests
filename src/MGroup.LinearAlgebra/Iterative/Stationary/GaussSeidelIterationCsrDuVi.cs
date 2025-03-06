using MGroup.LinearAlgebra.Iterative.Stationary;
using MGroup.LinearAlgebra.Matrices;
using MGroup.LinearAlgebra.Vectors;

namespace Compression.src.MGroup.LinearAlgebra.Iterative.Stationary
{
    public class GaussSeidelIterationCsrDuVi : IStationaryIteration
    {
        private DuViCompressedSparseMatrix mat;

        public GaussSeidelIterationCsrDuVi() { }
        public GaussSeidelIterationCsrDuVi(DuViCompressedSparseMatrix m) => mat = m;

        public string Name => "Delta Unit Value Indexed Row Major Sparse Matrix Gauss-Seidel Iteration";

        public IStationaryIteration CopyWithInitialSettings() => this;

        public void LinkWith(IStationaryIteration other) { }

        public void UpdateMatrix(IMatrixView m, bool isPatternModified) => mat = (DuViCompressedSparseMatrix) m;

        public void Execute(Vector b, Vector x)
        {
            int row = -1, col = 0, non_zero_entry_index = 0;
            double divisor = 0;
            foreach (DuViCompressedSparseMatrix.Unit unit in mat.units)
            {
                DuViCompressedSparseMatrix.Unit.DeltaBytes db = unit.GetDeltaBytes();
                if (unit.IsNewRow())
                {
                    if (row != -1) x[row] /= divisor;
                    ++row; col = 0;
                    x[row] = b[row];
                }
                (int delta, int offset) = unit.GetNextColumn();
                offset += 2;
                col += delta;

                int total = unit.GetTotalEntries(), i = 0;
                for(; ; )
                {
                    double a = mat.GetValueAtIndex(non_zero_entry_index++);
                    if (row != col) x[row] -= a * x[col];
                    else divisor = a;
                    if (i == total) break;
                    col += unit.GetIndex(i++, db, offset);
                }
            }
            x[row] /= divisor;
        }
    }
}
